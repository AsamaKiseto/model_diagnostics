"""管理 checkpoint-conditioned 诊断的 versioned 标量产物、resume 与报告。

阶段: post-training diagnostics artifact owner。hook 和指标代码从不写盘；引擎在一个
condition 收口后把 JSON-safe 记录交给 ``DiagnosticsArtifactStore``。rank-local shard
先独立落盘，rank 0 合并去重后才写最终 ``manifest.json`` commit marker。
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping
import uuid

from .common import file_sha256, stable_json_hash, write_json_atomic
from .checkpoint_validation import (
    CHECKPOINT_TRANSACTION_STREAMS,
    canonical_analyzer_evidence_inventory,
    canonical_transaction_payload,
    finalized_stream_matches_descriptor,
    resolve_checkpoint_path,
    transaction_has_valid_terminal,
    transaction_payload_hash,
    transaction_payload_line,
)
from ..checkpoint.contracts import (
    ANALYSIS_SEMANTICS,
    CONDITION_TRANSACTION_CONTRACT,
    FORMAT_VERSION,
    CheckpointRef,
    CheckpointRuntimeDescriptor,
    CohortSelection,
    DiagnosticsRecipe,
)
from ..registry import AnalyzerCatalog


_SHARD_NAMES = CHECKPOINT_TRANSACTION_STREAMS


class DiagnosticsArtifactStore:
    """拥有单次诊断 run 的 shard writers、resume 校验与最终汇总。"""

    def __init__(self, output_dir: str | Path, *, rank: int, world_size: int) -> None:
        self.output_dir = Path(output_dir)
        existing_documents = (
            self.output_dir / "manifest.json",
            self.output_dir / "run_state.json",
        )
        for document in existing_documents:
            if not document.is_file():
                continue
            try:
                existing = json.loads(document.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            if (
                isinstance(existing, Mapping)
                and existing.get("format_version") != FORMAT_VERSION
            ):
                raise RuntimeError(
                    "unsupported checkpoint format version in "
                    f"{document}: {existing.get('format_version')!r}; "
                    f"expected {FORMAT_VERSION!r}"
                )
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.commit_dir = self.output_dir / "commits"
        self.expected_commit_dir = self.commit_dir / "expected"
        self.analyzer_dir = self.output_dir / "analyzers"
        self.core_analyzer_dir = self.analyzer_dir / "_base"
        self.shard_dir = self.core_analyzer_dir / "shards"
        self.stream_dir = self.core_analyzer_dir / "streams"
        self._generation = uuid.uuid4().hex
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        self.commit_dir.mkdir(parents=True, exist_ok=True)
        self.expected_commit_dir.mkdir(parents=True, exist_ok=True)
        self._handles: dict[str, Any] = {}
        self._transactions: dict[str, dict[str, Any]] = {}
        self.already_complete = False
        self.completed_status: str | None = None
        self.component_catalog_digest: str | None = None
        self.output_component_ids: tuple[str, ...] = ()

    def register_expected_condition_ids(
        self,
        condition_ids: Iterable[str],
    ) -> None:
        """在执行 condition 前持久化本 rank 的完整计划集合。"""

        path = self.expected_commit_dir / f"rank{self.rank:05d}.json"
        expected: set[str] = set()
        if path.exists():
            try:
                previous = json.loads(path.read_text(encoding="utf-8"))
                if (
                    isinstance(previous, Mapping)
                    and previous.get("format_version") == FORMAT_VERSION
                    and int(previous.get("rank", -1)) == self.rank
                    and isinstance(previous.get("condition_ids"), list)
                ):
                    expected.update(
                        str(value)
                        for value in previous["condition_ids"]
                        if str(value)
                    )
            except (
                OSError,
                UnicodeDecodeError,
                json.JSONDecodeError,
                TypeError,
                ValueError,
            ):
                pass
        expected.update(
            str(value) for value in condition_ids if str(value)
        )
        write_json_atomic(
            path,
            {
                "format_version": FORMAT_VERSION,
                "rank": self.rank,
                "condition_ids": sorted(expected),
            },
        )

    def initialize(
        self,
        *,
        run_id: str,
        recipe: DiagnosticsRecipe,
        checkpoints: Iterable[CheckpointRef],
        cohort: CohortSelection,
        analysis_id: str,
        runtime_descriptor: CheckpointRuntimeDescriptor,
    ) -> tuple[str, str, set[str]]:
        """校验或建立 resume identity，并返回 recipe/cohort hash 与终态 condition IDs。"""

        recipe_payload = recipe.to_dict()
        recipe_hash = stable_json_hash(recipe_payload)
        cohort_metadata = dict(cohort.metadata)
        component_catalog = cohort_metadata.pop("component_catalog", None)
        component_catalog_digest = cohort_metadata.get(
            "component_catalog_digest"
        )
        if not isinstance(component_catalog, list) or not isinstance(
            component_catalog_digest,
            str,
        ):
            raise ValueError(
                "checkpoint cohort must provide a component catalog and digest"
            )
        if stable_json_hash(component_catalog) != component_catalog_digest:
            raise ValueError("component catalog does not match its digest")
        self.output_component_ids = tuple(
            str(component["component_id"])
            for component in component_catalog
            if isinstance(component, Mapping)
            and component.get("component_kind") == "output"
        )
        cohort_metadata["component_catalog_path"] = "component_catalog.json"
        self.component_catalog_digest = component_catalog_digest
        cohort_payload = {
            "format_version": FORMAT_VERSION,
            "selection_policy": (
                "fixed_group_stratified_sweep"
                if recipe.stage == "checkpoint_sweep"
                else "fixed_group_stratified_final"
            ),
            "status": cohort.status,
            "skip_reason": cohort.skip_reason,
            "identity": cohort.identity,
            "sample_refs": [sample.to_dict() for sample in cohort.samples],
            "metadata": cohort_metadata,
        }
        cohort_hash = stable_json_hash(cohort_payload)
        checkpoint_payload = [
            {
                "path": ref.path,
                "identity": ref.identity,
                "update": ref.update,
                "kind": ref.kind,
                "metadata": dict(ref.metadata),
            }
            for ref in checkpoints
        ]
        runtime_identity = runtime_descriptor.to_dict()
        identity = {
            "format_version": FORMAT_VERSION,
            "run_id": run_id,
            "analysis_id": analysis_id,
            "recipe_hash": recipe_hash,
            "cohort_hash": cohort_hash,
            "component_catalog_digest": component_catalog_digest,
            "checkpoint_hash": stable_json_hash(checkpoint_payload),
            **runtime_identity,
        }
        state_path = self.output_dir / "run_state.json"
        final_manifest = self.output_dir / "manifest.json"
        if final_manifest.exists():
            previous = json.loads(final_manifest.read_text(encoding="utf-8"))
            for field in (
                "format_version",
                "run_id",
                "analysis_id",
                "recipe_hash",
                "cohort_hash",
                "component_catalog_digest",
                "checkpoint_hash",
                "runtime_descriptor_digest",
                "task_definition_id",
                "task_definition_version",
                "objective_executor_version",
                "capability_descriptors",
            ):
                if previous.get(field) != identity[field]:
                    raise RuntimeError(f"Cannot resume checkpoint diagnostics: {field} mismatch")
            if previous.get("status") in {"complete", "degraded"}:
                manifest_expected = {
                    str(value)
                    for value in previous.get("expected_condition_ids", ())
                }
                planned, planning_valid = self._load_expected_condition_ids()
                validated = set(self._validated_commits())
                if (
                    previous.get("condition_transaction_contract")
                    == CONDITION_TRANSACTION_CONTRACT
                    and previous.get("condition_planning_status") == "complete"
                    and planning_valid
                    and manifest_expected
                    and planned == manifest_expected
                    and validated == manifest_expected
                    and not previous.get("missing_expected_condition_ids")
                    and not previous.get("unexpected_valid_condition_ids")
                    and int(previous.get("expected_condition_count", -1))
                    == len(manifest_expected)
                    and int(previous.get("committed_condition_count", -1))
                    == len(manifest_expected)
                    and {
                        str(value)
                        for value in previous.get(
                            "committed_condition_ids",
                            (),
                        )
                    }
                    == manifest_expected
                    and self._finalized_streams_match(previous)
                ):
                    self.already_complete = True
                    self.completed_status = str(previous.get("status"))
                    if self.rank == 0 and state_path.exists():
                        state_path.unlink()
                elif self.rank == 0:
                    write_json_atomic(
                        final_manifest,
                        {
                            **previous,
                            "status": "corrupt",
                            "validation_error": (
                                "complete_manifest_commit_validation_failed"
                            ),
                            "planned_condition_ids": sorted(planned),
                            "condition_planning_valid": planning_valid,
                            "validated_committed_condition_ids": sorted(validated),
                        },
                    )
        elif state_path.exists():
            previous = json.loads(state_path.read_text(encoding="utf-8"))
            for field in (
                "format_version",
                "run_id",
                "analysis_id",
                "recipe_hash",
                "cohort_hash",
                "component_catalog_digest",
                "checkpoint_hash",
                "runtime_descriptor_digest",
                "task_definition_id",
                "task_definition_version",
                "objective_executor_version",
                "capability_descriptors",
            ):
                if previous.get(field) != identity[field]:
                    raise RuntimeError(f"Cannot resume checkpoint diagnostics: {field} mismatch")
        if self.rank == 0 and not state_path.exists() and not self.already_complete:
            write_json_atomic(
                state_path,
                {**identity, "status": "running"},
            )
        recipe_path = self.output_dir / "recipe.json"
        if self.rank == 0 and not recipe_path.exists():
            write_json_atomic(recipe_path, recipe_payload)
        cohort_path = self.output_dir / "cohort.json"
        if self.rank == 0 and not cohort_path.exists():
            write_json_atomic(cohort_path, cohort_payload)
        component_catalog_path = self.output_dir / "component_catalog.json"
        if self.rank == 0:
            if component_catalog_path.exists():
                existing_catalog = json.loads(
                    component_catalog_path.read_text(encoding="utf-8")
                )
                if stable_json_hash(existing_catalog) != component_catalog_digest:
                    raise RuntimeError(
                        "Cannot resume checkpoint diagnostics: "
                        "component catalog mismatch"
                    )
            else:
                write_json_atomic(component_catalog_path, component_catalog)
        if not self.already_complete:
            self.register_expected_condition_ids(())
        completed = set(self._validated_commits(rank=self.rank))
        return recipe_hash, cohort_hash, completed

    def _load_expected_condition_ids(self) -> tuple[set[str], bool]:
        """读取所有 rank 的 durable execution plan；缺失或损坏时 fail closed。"""

        expected: set[str] = set()
        condition_owners: dict[str, int] = {}
        seen_ranks: set[int] = set()
        for path in sorted(self.expected_commit_dir.glob("rank*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if (
                    not isinstance(payload, Mapping)
                    or payload.get("format_version") != FORMAT_VERSION
                    or not isinstance(payload.get("condition_ids"), list)
                ):
                    return set(), False
                plan_rank = int(payload.get("rank", -1))
                if (
                    plan_rank < 0
                    or plan_rank >= self.world_size
                    or plan_rank in seen_ranks
                ):
                    return set(), False
                seen_ranks.add(plan_rank)
                for value in payload["condition_ids"]:
                    condition_id = (
                        ""
                        if value is None
                        else str(value)
                    )
                    if not condition_id:
                        continue
                    previous_owner = condition_owners.get(condition_id)
                    if (
                        previous_owner is not None
                        and previous_owner != plan_rank
                    ):
                        return set(), False
                    condition_owners[condition_id] = plan_rank
                    expected.add(condition_id)
            except (
                OSError,
                UnicodeDecodeError,
                json.JSONDecodeError,
                TypeError,
                ValueError,
            ):
                return set(), False
        return expected, seen_ranks == set(range(self.world_size))

    def append(self, stream: str, record: Mapping[str, Any]) -> None:
        """把 condition detail/terminal 记录写入 rank-local transaction shard。

        core condition 由 ``checkpoints``/``conditions`` 的终态行提交；adapter analyzer
        的 detail 即使带 ``status=success`` 也只累计充分统计，只有显式
        ``record_kind=analyzer_terminal`` 才提交。
        """

        if stream not in _SHARD_NAMES:
            raise KeyError(f"Unknown diagnostics artifact stream: {stream}")
        handle = self._handles.get(stream)
        if handle is None:
            path = self.shard_dir / (
                f"{stream}.rank{self.rank:05d}.{self._generation}.jsonl"
            )
            handle = path.open("a", encoding="utf-8", buffering=1024 * 1024)
            self._handles[stream] = handle
        handle.write(
            json.dumps(
                dict(record),
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
                default=_json_default,
            )
        )
        handle.write("\n")
        condition_id = str(record.get("condition_id") or "")
        if condition_id:
            transaction = self._transactions.setdefault(
                condition_id,
                {
                    "row_counts": {},
                    "payloads": {},
                    "single_stream_hashers": {},
                    "stream_files": {},
                },
            )
            payload = canonical_transaction_payload(
                dict(record),
                default=_json_default,
            )
            if stream == "analyzers":
                hasher = transaction["single_stream_hashers"].setdefault(
                    stream,
                    hashlib.sha256(),
                )
                hasher.update(transaction_payload_line(payload))
            else:
                transaction["payloads"].setdefault(stream, []).append(payload)
            transaction["stream_files"][stream] = str(
                (
                    self.shard_dir
                    / f"{stream}.rank{self.rank:05d}.{self._generation}.jsonl"
                ).relative_to(self.output_dir)
            )
            counts = transaction["row_counts"]
            counts[stream] = int(counts.get(stream, 0)) + 1
            if (
                (
                    stream in {"checkpoints", "conditions"}
                    or (
                        stream == "analyzers"
                        and record.get("record_kind") == "analyzer_terminal"
                    )
                )
                and record.get("status") in {"success", "failed", "skipped", "insufficient_evidence"}
            ):
                for transaction_handle in self._handles.values():
                    transaction_handle.flush()
                    os.fsync(transaction_handle.fileno())
                self._commit_transaction(condition_id, str(record["status"]))

    def flush(self) -> None:
        """刷新当前 rank 的所有 buffered writers。"""

        for handle in self._handles.values():
            handle.flush()

    def close(self) -> None:
        """关闭所有 writers；异常路径也可幂等调用。"""

        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def mark_corrupt(self, *, error: BaseException) -> None:
        """在不可恢复异常后关闭 writer，并留下明确的 corrupt 终态。"""

        self.close()
        if self.rank != 0 or self.already_complete:
            return
        state_path = self.output_dir / "run_state.json"
        state: dict[str, Any] = {"format_version": FORMAT_VERSION}
        if state_path.exists():
            try:
                payload = json.loads(state_path.read_text(encoding="utf-8"))
                if isinstance(payload, Mapping):
                    state.update(payload)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                pass
        failure = f"{type(error).__name__}: {error}"
        corrupt_state = {
            **state,
            "status": "corrupt",
            "validation_error": "checkpoint_execution_failed",
            "failure": failure,
        }
        write_json_atomic(state_path, corrupt_state)
        write_json_atomic(self.output_dir / "manifest.json", corrupt_state)

    def finalize(
        self,
        *,
        run_id: str,
        recipe: DiagnosticsRecipe,
        recipe_hash: str,
        cohort_hash: str,
        checkpoints: Iterable[CheckpointRef],
        analysis_id: str,
        analyzer_catalog: AnalyzerCatalog,
        runtime_descriptor: CheckpointRuntimeDescriptor,
    ) -> str:
        """rank 0 从合法 commit 合并最终 streams，并原子完成 manifest。"""

        self.close()
        if self.rank != 0:
            return
        expected_condition_ids, planning_valid = (
            self._load_expected_condition_ids()
        )
        validated_commits = self._validated_commits()
        all_valid_committed_ids = set(validated_commits)
        missing_expected_condition_ids = (
            expected_condition_ids - all_valid_committed_ids
            if planning_valid
            else set()
        )
        unexpected_valid_condition_ids = (
            all_valid_committed_ids - expected_condition_ids
            if planning_valid
            else set(all_valid_committed_ids)
        )
        valid_committed_ids = (
            all_valid_committed_ids & expected_condition_ids
            if planning_valid
            else set()
        )
        committed_rows: dict[str, list[dict[str, Any]]] = {
            stream: [] for stream in _SHARD_NAMES
        }
        for condition_id in sorted(valid_committed_ids):
            for stream, rows in validated_commits[condition_id].items():
                committed_rows[stream].extend(rows)
        merged: dict[str, list[dict[str, Any]]] = {
            stream: _deduplicate_rows(
                committed_rows.get(stream, []),
                stream=stream,
            )
            for stream in _SHARD_NAMES
        }
        self.stream_dir.mkdir(parents=True, exist_ok=True)
        finalized_streams: list[dict[str, Any]] = []
        condition_analyzers = [
            name
            for name in recipe.analyzers
            if analyzer_catalog.definition(name).execution_mode == "condition"
        ]
        for stream in _SHARD_NAMES:
            if stream == "analyzers":
                continue
            stream_path = self.stream_dir / f"{stream}.jsonl"
            _write_jsonl(stream_path, merged[stream])
            finalized_streams.append(
                _finalized_stream_descriptor(
                    self.output_dir,
                    stream_path,
                    stream_id=f"base/{stream}",
                    row_count=len(merged[stream]),
                    analyzer_names=condition_analyzers,
                )
            )
        self.analyzer_dir.mkdir(parents=True, exist_ok=True)
        analyzer_names = sorted(
            {
                str(row.get("analyzer"))
                for row in merged["analyzers"]
                if row.get("analyzer")
            }
        )
        for analyzer in analyzer_names:
            analyzer_stream_dir = self.analyzer_dir / analyzer / "streams"
            analyzer_stream_dir.mkdir(parents=True, exist_ok=True)
            analyzer_rows = [
                row
                for row in merged["analyzers"]
                if row.get("analyzer") == analyzer
                and row.get("record_kind") != "analyzer_terminal"
            ]
            analyzer_stream_path = analyzer_stream_dir / "records.jsonl"
            _write_jsonl(
                analyzer_stream_path,
                analyzer_rows,
            )
            finalized_streams.append(
                _finalized_stream_descriptor(
                    self.output_dir,
                    analyzer_stream_path,
                    stream_id=f"{analyzer}/records",
                    row_count=len(analyzer_rows),
                    analyzer_names=(analyzer,),
                )
            )
        checkpoint_rows = [
            {
                "path": ref.path,
                "identity": ref.identity,
                "update": ref.update,
                "kind": ref.kind,
                "metadata": dict(ref.metadata),
            }
            for ref in checkpoints
        ]
        requested_adapter_analyzers = [
            name
            for name in recipe.analyzers
            if analyzer_catalog.definition(name).execution_mode
            in {"aggregate", "stream"}
        ]
        analyzer_terminal_rows = [
            row
            for row in merged["analyzers"]
            if row.get("record_kind") == "analyzer_terminal"
        ]
        analyzer_statuses = {
            analyzer: sorted(
                {
                    str(row.get("status", "failed"))
                    for row in analyzer_terminal_rows
                    if row.get("analyzer") == analyzer
                }
            )
            for analyzer in requested_adapter_analyzers
        }
        missing_analyzers = [
            analyzer for analyzer, statuses in analyzer_statuses.items() if not statuses
        ]
        if (
            not planning_valid
            or not expected_condition_ids
            or missing_expected_condition_ids
            or unexpected_valid_condition_ids
        ):
            condition_planning_status = "invalid"
            final_status = "corrupt"
        else:
            condition_planning_status = "complete"
            final_status = (
                "degraded"
                if merged["failures"]
                or any(
                    row.get("status") == "failed"
                    for row in merged["analyzers"]
                )
                or missing_analyzers
                or any(
                    status != "success"
                    for statuses in analyzer_statuses.values()
                    for status in statuses
                )
                else "complete"
            )
        manifest = {
            "format_version": FORMAT_VERSION,
            "run_id": run_id,
            "analysis_id": analysis_id,
            "status": final_status,
            "analysis_semantics": ANALYSIS_SEMANTICS,
            "condition_transaction_contract": CONDITION_TRANSACTION_CONTRACT,
            "recipe_hash": recipe_hash,
            "cohort_hash": cohort_hash,
            "component_catalog_digest": self.component_catalog_digest,
            "component_catalog_path": "component_catalog.json",
            "checkpoint_hash": stable_json_hash(checkpoint_rows),
            **runtime_descriptor.to_dict(),
            "recipe": recipe.to_dict(),
            "checkpoints": checkpoint_rows,
            "ignored_payload_keys_by_checkpoint": {
                str(row.get("checkpoint_path")): list(row.get("ignored_payload_keys", []))
                for row in merged["checkpoints"]
            },
            "condition_count": len(merged["conditions"]),
            "failure_count": len(merged["failures"]),
            "analyzers": list(recipe.analyzers),
            "analyzer_definitions": analyzer_catalog.selected_descriptors(
                recipe.analyzers
            ),
            "analyzer_record_count": sum(
                row.get("record_kind") != "analyzer_terminal"
                for row in merged["analyzers"]
            ),
            "analyzer_terminal_statuses": analyzer_statuses,
            "analyzer_terminal_condition_ids": {
                analyzer: sorted(
                    {
                        str(row["condition_id"])
                        for row in analyzer_terminal_rows
                        if row.get("analyzer") == analyzer
                        and row.get("condition_id")
                    }
                )
                for analyzer in requested_adapter_analyzers
            },
            "analyzer_evidence_inventories": {
                analyzer: sorted(
                    (
                        canonical_analyzer_evidence_inventory(row)
                        for row in analyzer_terminal_rows
                        if row.get("analyzer") == analyzer
                    ),
                    key=lambda item: str(
                        item.get("condition_id") or ""
                    ),
                )
                for analyzer in requested_adapter_analyzers
            },
            "missing_analyzers": missing_analyzers,
            "finalized_streams": finalized_streams,
            "condition_planning_status": condition_planning_status,
            "expected_condition_count": len(expected_condition_ids),
            "expected_condition_ids": sorted(expected_condition_ids),
            "missing_expected_condition_ids": sorted(
                missing_expected_condition_ids
            ),
            "unexpected_valid_condition_ids": sorted(
                unexpected_valid_condition_ids
            ),
            "committed_condition_count": len(valid_committed_ids),
            "committed_condition_ids": sorted(valid_committed_ids),
            "historical_trajectory_reproduced": False,
            "historical_optimizer_state_restored": False,
            "historical_scheduler_state_restored": False,
            "historical_scaler_state_restored": False,
            "historical_rng_state_restored": False,
            "historical_loader_cursor_restored": False,
            "optimizer_state_origin": "fresh",
        }
        write_json_atomic(self.output_dir / "manifest.json", manifest)
        state_path = self.output_dir / "run_state.json"
        if state_path.exists():
            state_path.unlink()
        return final_status

    def _commit_transaction(self, condition_id: str, status: str) -> None:
        transaction = self._transactions.pop(condition_id)
        payloads = transaction["payloads"]
        single_stream_hashers = transaction["single_stream_hashers"]
        if payloads and single_stream_hashers:
            raise RuntimeError(
                f"Cannot commit condition {condition_id}: mixed buffered payload modes"
            )
        if payloads:
            payload_sha256 = transaction_payload_hash(payloads)
        elif len(single_stream_hashers) == 1:
            payload_sha256 = next(iter(single_stream_hashers.values())).hexdigest()
        else:
            raise RuntimeError(
                f"Cannot commit condition {condition_id}: unsupported payload stream layout"
            )
        commit = {
            "format_version": FORMAT_VERSION,
            "condition_id": condition_id,
            "rank": self.rank,
            "status": "committed",
            "terminal_status": status,
            "row_counts": dict(transaction["row_counts"]),
            "stream_files": dict(transaction["stream_files"]),
            "payload_sha256": payload_sha256,
        }
        write_json_atomic(
            self.commit_dir / f"{condition_id}.json",
            commit,
        )

    def _validated_commits(
        self,
        rank: int | None = None,
    ) -> dict[str, dict[str, list[dict[str, Any]]]]:
        """按 shard 单次扫描校验 commit，并返回可直接合并的 rows。

        多个 condition 共用 append-only shard。校验器先收集 commit 引用，再让每个
        shard 只顺序读取一次，避免按 condition 重复扫描同一大文件。
        """

        completed: dict[str, dict[str, list[dict[str, Any]]]] = {}
        seen_condition_ids: set[str] = set()
        duplicate_condition_ids: set[str] = set()
        documents: list[
            tuple[str, Mapping[str, Any], dict[str, Path]]
        ] = []
        for path in sorted(self.commit_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(payload, Mapping):
                    continue
                condition_id = str(payload.get("condition_id") or "")
                if condition_id:
                    if condition_id in seen_condition_ids:
                        duplicate_condition_ids.add(condition_id)
                    seen_condition_ids.add(condition_id)
                commit_rank = int(payload.get("rank", -1))
                structurally_valid = (
                    payload.get("format_version") == FORMAT_VERSION
                    and payload.get("status") == "committed"
                    and condition_id
                    and isinstance(payload.get("row_counts"), dict)
                    and isinstance(payload.get("stream_files"), dict)
                    and isinstance(payload.get("payload_sha256"), str)
                    and len(payload["payload_sha256"]) == 64
                    and (rank is None or commit_rank == int(rank))
                )
                if not structurally_valid:
                    continue
                stream_files = payload["stream_files"]
                row_counts = payload["row_counts"]
                if set(map(str, stream_files)) != set(map(str, row_counts)):
                    continue
                resolved_paths: dict[str, Path] = {}
                for stream, raw_path in stream_files.items():
                    stream_name = str(stream)
                    resolved = resolve_checkpoint_path(
                        self.output_dir,
                        str(raw_path),
                    )
                    if (
                        stream_name not in _SHARD_NAMES
                        or resolved is None
                        or not resolved.is_file()
                    ):
                        resolved_paths = {}
                        break
                    resolved_paths[stream_name] = resolved
                if resolved_paths:
                    documents.append(
                        (condition_id, payload, resolved_paths)
                    )
            except (
                OSError,
                UnicodeDecodeError,
                json.JSONDecodeError,
                TypeError,
                ValueError,
            ):
                continue

        requested_by_path: dict[Path, set[str]] = {}
        for condition_id, _payload, paths in documents:
            if condition_id in duplicate_condition_ids:
                continue
            for shard_path in paths.values():
                requested_by_path.setdefault(shard_path, set()).add(
                    condition_id
                )

        cached_rows: dict[
            tuple[Path, str],
            tuple[list[dict[str, Any]], list[bytes]],
        ] = {}
        invalid_paths: set[Path] = set()
        for shard_path, requested_ids in requested_by_path.items():
            buckets = {
                condition_id: ([], [])
                for condition_id in requested_ids
            }
            try:
                handle = shard_path.open("r", encoding="utf-8")
            except OSError:
                invalid_paths.add(shard_path)
                continue
            with handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(row, Mapping):
                        invalid_paths.add(shard_path)
                        break
                    condition_id = str(row.get("condition_id") or "")
                    bucket = buckets.get(condition_id)
                    if bucket is None:
                        continue
                    parsed_row = dict(row)
                    try:
                        canonical = canonical_transaction_payload(
                            parsed_row,
                            default=_json_default,
                        )
                    except (TypeError, ValueError):
                        invalid_paths.add(shard_path)
                        break
                    bucket[0].append(parsed_row)
                    bucket[1].append(canonical)
            if shard_path in invalid_paths:
                continue
            for condition_id, bucket in buckets.items():
                cached_rows[(shard_path, condition_id)] = bucket

        for condition_id, payload, paths in documents:
            if condition_id in duplicate_condition_ids or any(
                shard_path in invalid_paths
                for shard_path in paths.values()
            ):
                continue
            rows_by_stream: dict[str, list[dict[str, Any]]] = {}
            canonical_payloads: dict[str, list[bytes]] = {}
            for stream, shard_path in paths.items():
                rows, canonical = cached_rows.get(
                    (shard_path, condition_id),
                    ([], []),
                )
                rows_by_stream[stream] = rows
                canonical_payloads[stream] = canonical
            try:
                actual_counts = {
                    stream: len(rows)
                    for stream, rows in rows_by_stream.items()
                }
                expected_counts = {
                    str(stream): int(count)
                    for stream, count in payload["row_counts"].items()
                }
                if actual_counts != expected_counts:
                    continue
                if (
                    transaction_payload_hash(canonical_payloads)
                    != payload["payload_sha256"]
                ):
                    continue
                if not transaction_has_valid_terminal(
                    rows_by_stream,
                    terminal_status=str(
                        payload.get("terminal_status") or ""
                    ),
                ):
                    continue
                completed[condition_id] = rows_by_stream
            except (
                TypeError,
                ValueError,
            ):
                continue
        return {
            condition_id: rows
            for condition_id, rows in completed.items()
            if condition_id not in duplicate_condition_ids
        }

    def _finalized_streams_match(self, manifest: Mapping[str, Any]) -> bool:
        """验证 complete manifest 暴露的最终 streams，避免幂等返回损坏结果。"""

        descriptors = manifest.get("finalized_streams")
        if not isinstance(descriptors, list) or not descriptors:
            return False
        seen_paths: set[str] = set()
        seen_stream_ids: set[str] = set()
        for descriptor in descriptors:
            if not isinstance(descriptor, Mapping):
                return False
            relative = str(descriptor.get("path") or "")
            stream_id = str(descriptor.get("stream_id") or "")
            if (
                not relative
                or relative in seen_paths
                or not stream_id
                or stream_id in seen_stream_ids
            ):
                return False
            seen_paths.add(relative)
            seen_stream_ids.add(stream_id)
            path = resolve_checkpoint_path(self.output_dir, relative)
            if path is None:
                return False
            if not finalized_stream_matches_descriptor(path, descriptor):
                return False
        return True

def _deduplicate_rows(rows: list[dict[str, Any]], *, stream: str) -> list[dict[str, Any]]:
    """按各 stream 的真实证据身份合并 rank-local committed rows。

    一个 condition 可以同时产生多个输出组件或 metric 的记录，因此 detail stream
    优先使用显式 ``evidence_key``；缺少该键的上下文记录只去除字节语义相同的副本。
    """

    deduplicated: dict[str, dict[str, Any]] = {}
    for row in rows:
        if stream == "conditions":
            key = str(row.get("condition_id"))
        elif stream == "checkpoints":
            key = str(row.get("checkpoint_identity"))
        elif stream == "analyzers":
            key = str(
                row.get("analyzer_record_id")
                or stable_json_hash(row)
            )
        else:
            evidence_key = str(row.get("evidence_key") or "").strip()
            key = evidence_key or stable_json_hash(row)
        deduplicated[key] = row
    return sorted(
        deduplicated.values(),
        key=lambda row: (
            -1 if row.get("checkpoint_update") is None else int(row["checkpoint_update"]),
            str(row.get("sample_id", "")),
            str(row.get("condition_id", "")),
            int(row.get("probe_step", 0) or 0),
            str(row.get("node_id", "")),
            str(row.get("response_component_id", "")),
            str(row.get("metric_id", "")),
        ),
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    dict(row),
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                    default=_json_default,
                )
            )
            handle.write("\n")


def _finalized_stream_descriptor(
    output_dir: Path,
    path: Path,
    *,
    stream_id: str,
    row_count: int,
    analyzer_names: Iterable[str] = (),
) -> dict[str, Any]:
    """把最终可消费 stream 绑定到 analysis manifest。"""

    return {
        "stream_id": stream_id,
        "path": str(path.relative_to(output_dir)),
        "row_count": int(row_count),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "analyzer_names": [
            str(name) for name in analyzer_names if str(name)
        ],
    }


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


__all__ = ["DiagnosticsArtifactStore"]
