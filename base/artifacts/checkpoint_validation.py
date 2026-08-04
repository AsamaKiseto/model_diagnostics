"""校验 checkpoint-conditioned transaction、identity 与 finalized streams。

该模块只依赖 Python 标准库和 parent package 的 artifact 原语。checkpoint writer
与统一 report 共用 canonical payload、terminal、路径及 digest 规则，但各自保留 resume
决策和 fail-closed source status。
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .common import ValidationIssue, file_sha256, stable_json_hash
from .checkpoint_identity import (
    CHECKPOINT_FORMAT_VERSION,
    RUNTIME_DESCRIPTOR_FIELDS,
    build_checkpoint_analysis_identity_payload,
    normalize_runtime_descriptor,
)


CHECKPOINT_TRANSACTION_CONTRACT = "explicit_analyzer_terminal_v2"
CHECKPOINT_TRANSACTION_STREAMS = (
    "checkpoints",
    "conditions",
    "module_effects",
    "local_taylor",
    "failures",
    "analyzers",
)
CHECKPOINT_STREAM_NAMES = frozenset(CHECKPOINT_TRANSACTION_STREAMS)
@dataclass(frozen=True)
class CheckpointValidation:
    """一个 checkpoint analysis 的 fail-closed 完整性结论。"""

    analysis_id: str | None
    status: str
    valid: bool
    manifest_status: str | None
    committed_condition_count: int
    issues: tuple[ValidationIssue, ...]
    manifest: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "analysis_id": self.analysis_id,
            "status": self.status,
            "valid": self.valid,
            "manifest_status": self.manifest_status,
            "committed_condition_count": self.committed_condition_count,
            "issues": [issue.as_dict() for issue in self.issues],
        }


@dataclass(frozen=True)
class _CheckpointCommitValidation:
    """保存 payload-valid commits 及 analyzer terminal 的实际观测。"""

    commits: Mapping[str, Mapping[str, Any]]
    analyzer_terminal_statuses: Mapping[str, set[str]]
    analyzer_terminal_condition_ids: Mapping[str, set[str]]
    analyzer_evidence_inventories: Mapping[
        str,
        tuple[Mapping[str, Any], ...],
    ]


_EVIDENCE_INVENTORY_FIELDS = (
    "condition_id",
    "checkpoint_identity",
    "evidence_inventory_contract",
    "expected_evidence_keys",
    "observed_evidence_keys",
    "skipped_evidence_keys",
    "failed_evidence_keys",
    "missing_evidence_keys",
    "expected_evidence_count",
    "observed_evidence_count",
    "skipped_evidence_count",
    "failed_evidence_count",
    "missing_evidence_count",
    "evidence_inventory_digest",
)


def canonical_analyzer_evidence_inventory(
    terminal: Mapping[str, Any],
) -> dict[str, Any]:
    """提取 manifest 与 validator 共用的 analyzer inventory 摘要。"""

    return {
        field: terminal.get(field)
        for field in _EVIDENCE_INVENTORY_FIELDS
    }


def canonical_transaction_payload(
    value: Mapping[str, Any],
    *,
    default: Callable[[Any], Any] | None = None,
) -> bytes:
    """把一条 transaction row 编码为不含换行的 canonical JSON bytes。"""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=default,
    ).encode("utf-8")


def transaction_payload_hash(
    payloads: Mapping[str, Iterable[bytes]],
) -> str:
    """按 stream 名和 stream 内原始顺序计算 transaction SHA-256。"""

    digest = hashlib.sha256()
    for stream in sorted(payloads):
        for payload in payloads[stream]:
            digest.update(transaction_payload_line(payload))
    return digest.hexdigest()


def transaction_payload_line(payload: bytes) -> bytes:
    """补齐 transaction hash contract 使用的单个 LF delimiter。"""

    return payload + b"\n"


def resolve_checkpoint_path(
    analysis_root: Path,
    relative: str | Path,
) -> Path | None:
    """解析 analysis 内相对路径；路径逃逸时返回 ``None``。"""

    root = analysis_root.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        return None
    return path


def finalized_stream_matches_descriptor(
    path: Path,
    descriptor: Mapping[str, Any],
    *,
    row_count: int | None = None,
) -> bool:
    """校验 finalized stream 的 row count、bytes 与 SHA-256。"""

    try:
        expected_rows = int(descriptor["row_count"])
        expected_bytes = int(descriptor["bytes"])
        if row_count is None:
            with path.open("r", encoding="utf-8") as handle:
                actual_rows = sum(1 for line in handle if line.strip())
        else:
            actual_rows = int(row_count)
        return (
            path.is_file()
            and actual_rows == expected_rows
            and path.stat().st_size == expected_bytes
            and file_sha256(path) == descriptor.get("sha256")
        )
    except (
        KeyError,
        TypeError,
        ValueError,
        OSError,
        UnicodeDecodeError,
    ):
        return False


class CheckpointTransactionTerminalValidator:
    """流式校验 core terminal 或 analyzer detail→terminal 契约。"""

    def __init__(
        self,
        stream_names: Iterable[str],
        *,
        terminal_status: str,
    ) -> None:
        self._stream_names = {str(stream) for stream in stream_names}
        self._terminal_status = str(terminal_status)
        self._core_terminal_found = False
        self._core_terminal_rows: list[Mapping[str, Any]] = []
        self._core_detail_keys_by_status: dict[str, set[str]] = {}
        self._core_duplicate_or_missing_evidence_key = False
        self._analyzer_terminal_rows: list[Mapping[str, Any]] = []
        self._analyzer_detail_count = 0
        self._analyzer_detail_status_counts: dict[str, int] = {}
        self._analyzer_detail_keys_by_status: dict[str, set[str]] = {}
        self._duplicate_or_missing_evidence_key = False
        self._analyzer_terminal_was_last = True

    def observe(self, stream: str, row: Mapping[str, Any]) -> None:
        """按原始 row 顺序累积 terminal 充分统计。"""

        if stream == "analyzers":
            if row.get("record_kind") == "analyzer_terminal":
                self._analyzer_terminal_rows.append(row)
                return
            if self._analyzer_terminal_rows:
                self._analyzer_terminal_was_last = False
            self._analyzer_detail_count += 1
            status = str(row.get("status") or "")
            self._analyzer_detail_status_counts[status] = (
                self._analyzer_detail_status_counts.get(status, 0) + 1
            )
            evidence_key = str(row.get("evidence_key") or "").strip()
            status_keys = self._analyzer_detail_keys_by_status.setdefault(
                status,
                set(),
            )
            if not evidence_key or evidence_key in set().union(
                *self._analyzer_detail_keys_by_status.values()
            ):
                self._duplicate_or_missing_evidence_key = True
            else:
                status_keys.add(evidence_key)
        else:
            if (
                stream in {"checkpoints", "conditions"}
                and row.get("record_kind")
                in {"checkpoint_terminal", "condition_terminal"}
            ):
                self._core_terminal_rows.append(row)
                self._core_terminal_found = (
                    str(row.get("status") or "") == self._terminal_status
                )
                return
            evidence_key = str(row.get("evidence_key") or "").strip()
            if evidence_key:
                status = str(row.get("status") or "success")
                status_keys = self._core_detail_keys_by_status.setdefault(
                    status,
                    set(),
                )
                if evidence_key in set().union(
                    *self._core_detail_keys_by_status.values()
                ):
                    self._core_duplicate_or_missing_evidence_key = True
                else:
                    status_keys.add(evidence_key)

    def result(self) -> tuple[bool, Mapping[str, Any] | None]:
        """返回 terminal 是否有效及唯一 analyzer terminal row。"""

        if "analyzers" not in self._stream_names:
            if (
                not self._core_terminal_found
                or len(self._core_terminal_rows) != 1
            ):
                return False, None
            terminal = self._core_terminal_rows[0]
            inventory_valid = (
                self._validate_evidence_inventory(
                    terminal,
                    detail_keys_by_status=(
                        self._core_detail_keys_by_status
                    ),
                    duplicate_or_missing_evidence_key=(
                        self._core_duplicate_or_missing_evidence_key
                    ),
                )
                if "evidence_inventory_contract" in terminal
                else True
            )
            return inventory_valid, None
        if (
            self._stream_names != {"analyzers"}
            or len(self._analyzer_terminal_rows) != 1
            or not self._analyzer_terminal_was_last
        ):
            return False, None
        terminal = self._analyzer_terminal_rows[0]
        inventory_valid = self._validate_evidence_inventory(terminal)
        valid = (
            str(terminal.get("status") or "") == self._terminal_status
            and terminal.get("detail_row_count")
            == self._analyzer_detail_count
            and terminal.get("detail_status_counts")
            == self._analyzer_detail_status_counts
            and inventory_valid
        )
        return valid, terminal if valid else None

    def _validate_evidence_inventory(
        self,
        terminal: Mapping[str, Any],
        *,
        detail_keys_by_status: Mapping[str, set[str]] | None = None,
        duplicate_or_missing_evidence_key: bool | None = None,
    ) -> bool:
        """核对 aggregate producer 声明的 expected/skip/missing inventory。"""

        keys_by_status = (
            self._analyzer_detail_keys_by_status
            if detail_keys_by_status is None
            else detail_keys_by_status
        )
        duplicate = (
            self._duplicate_or_missing_evidence_key
            if duplicate_or_missing_evidence_key is None
            else duplicate_or_missing_evidence_key
        )
        if (
            duplicate
            or terminal.get("evidence_inventory_contract")
            != "explicit_evidence_keys_v1"
        ):
            return False
        names = (
            "expected",
            "observed",
            "skipped",
            "failed",
            "missing",
        )
        inventories: dict[str, list[str]] = {}
        for name in names:
            values = terminal.get(f"{name}_evidence_keys")
            if (
                not isinstance(values, list)
                or any(not isinstance(value, str) or not value for value in values)
                or values != sorted(set(values))
                or terminal.get(f"{name}_evidence_count") != len(values)
            ):
                return False
            inventories[name] = values
        expected = set(inventories["expected"])
        observed = set(inventories["observed"])
        skipped = set(inventories["skipped"])
        failed = set(inventories["failed"])
        missing = set(inventories["missing"])
        if not expected or any(
            left & right
            for index, left in enumerate(
                (observed, skipped, failed, missing)
            )
            for right in (observed, skipped, failed, missing)[index + 1 :]
        ):
            return False
        actual_observed = keys_by_status.get(
            "success",
            set(),
        )
        actual_skipped = (
            keys_by_status.get("skipped", set())
            | keys_by_status.get(
                "insufficient_evidence",
                set(),
            )
        )
        actual_failed = keys_by_status.get(
            "failed",
            set(),
        )
        expected_terminal_status = (
            "failed"
            if actual_failed
            else "insufficient_evidence"
            if (
                keys_by_status.get(
                    "insufficient_evidence",
                    set(),
                )
                or missing
                or actual_skipped and actual_observed
            )
            else "skipped"
            if actual_skipped and not actual_observed
            else "success"
        )
        return (
            observed == actual_observed
            and skipped == actual_skipped
            and failed == actual_failed
            and observed | skipped | failed | missing == expected
            and missing == expected - observed - skipped - failed
            and terminal.get("evidence_inventory_digest")
            == stable_json_hash(sorted(expected))
            and str(terminal.get("status") or "")
            == expected_terminal_status
        )


def transaction_has_valid_terminal(
    rows_by_stream: Mapping[str, Iterable[Mapping[str, Any]]],
    *,
    terminal_status: str,
) -> bool:
    """对已解析 transaction rows 应用统一 terminal contract。"""

    validator = CheckpointTransactionTerminalValidator(
        rows_by_stream,
        terminal_status=terminal_status,
    )
    for stream, rows in rows_by_stream.items():
        for row in rows:
            validator.observe(str(stream), row)
    valid, _terminal = validator.result()
    return valid


def _load_object(path: Path) -> tuple[dict[str, Any], ValidationIssue | None]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}, ValidationIssue("missing_file", "error", "required file is missing", str(path))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        return {}, ValidationIssue("invalid_json", "error", str(error), str(path))
    if not isinstance(value, dict):
        return {}, ValidationIssue("invalid_document", "error", "expected JSON object", str(path))
    return value, None


def validate_checkpoint_analysis(
    analysis_dir: str | Path,
) -> CheckpointValidation:
    """把单 source 的任意 schema 错误隔离为 corrupt，不中断其他 source 报告。"""

    analysis_path = Path(analysis_dir)
    try:
        return _validate_checkpoint_analysis_impl(analysis_path)
    except Exception as error:
        return CheckpointValidation(
            analysis_path.name,
            "corrupt",
            False,
            None,
            0,
            (
                ValidationIssue(
                    "checkpoint_validator_internal_error",
                    "error",
                    f"{type(error).__name__}: {error}",
                    str(analysis_path),
                ),
            ),
            {},
        )


def _record_claim_sensitive_commit_issue(
    issues: list[ValidationIssue],
    *,
    claimed_ids: set[str] | None,
    condition_id: str,
    path: Path,
    issue: ValidationIssue,
    replace_since: int | None = None,
) -> None:
    """未被 manifest 声明的坏 commit 只作为 orphan warning，不污染合法恢复。"""

    if claimed_ids is not None and condition_id not in claimed_ids:
        if replace_since is not None:
            del issues[replace_since:]
        issues.append(
            ValidationIssue(
                "orphan_invalid_commit",
                "warning",
                condition_id,
                str(path),
            )
        )
        return
    issues.append(issue)


def _collect_checkpoint_commit_candidates(
    analysis_dir: Path,
    *,
    claimed_ids: set[str] | None,
    issues: list[ValidationIssue],
) -> dict[str, Mapping[str, Any]]:
    """读取 commit envelope，并在读取 payload 前完成身份和 schema 筛选。"""

    candidates: dict[str, Mapping[str, Any]] = {}
    for path in sorted((analysis_dir / "commits").glob("*.json")):
        commit, commit_issue = _load_object(path)
        if commit_issue is not None:
            _record_claim_sensitive_commit_issue(
                issues,
                claimed_ids=claimed_ids,
                condition_id=path.stem,
                path=path,
                issue=commit_issue,
            )
            continue

        condition_id = str(commit.get("condition_id", ""))
        if not condition_id or condition_id in candidates:
            condition_hint = condition_id or path.stem
            _record_claim_sensitive_commit_issue(
                issues,
                claimed_ids=claimed_ids,
                condition_id=condition_hint,
                path=path,
                issue=ValidationIssue(
                    "duplicate_or_missing_condition_id",
                    "error",
                    condition_id or path.name,
                    str(path),
                ),
            )
            continue
        if commit.get("status") != "committed":
            _record_claim_sensitive_commit_issue(
                issues,
                claimed_ids=claimed_ids,
                condition_id=condition_id,
                path=path,
                issue=ValidationIssue(
                    "uncommitted_condition",
                    "error",
                    condition_id,
                    str(path),
                ),
            )
            continue

        try:
            commit_rank = int(commit.get("rank", 0))
        except (TypeError, ValueError):
            commit_rank = -1
        if (
            commit.get("format_version") != CHECKPOINT_FORMAT_VERSION
            or commit_rank < 0
            or commit.get("terminal_status")
            not in {"success", "failed", "skipped", "insufficient_evidence"}
        ):
            _record_claim_sensitive_commit_issue(
                issues,
                claimed_ids=claimed_ids,
                condition_id=condition_id,
                path=path,
                issue=ValidationIssue(
                    "invalid_commit_schema",
                    "error",
                    condition_id,
                    str(path),
                ),
            )
            continue
        candidates[condition_id] = commit
    return candidates


def _validate_checkpoint_commit_payload(
    analysis_dir: Path,
    *,
    analysis_root: Path,
    condition_id: str,
    commit: Mapping[str, Any],
    issues: list[ValidationIssue],
) -> tuple[bool, Mapping[str, Any] | None]:
    """重算一个 condition transaction 的 rows、hash 与 terminal contract。"""

    issue_start = len(issues)
    commit_path = analysis_dir / "commits" / f"{condition_id}.json"
    raw_streams = commit.get("stream_files")
    raw_counts = commit.get("row_counts")
    if not isinstance(raw_streams, Mapping) or not isinstance(raw_counts, Mapping):
        issues.append(
            ValidationIssue(
                "missing_stream_contract",
                "error",
                f"{condition_id}: commit cannot be payload-verified",
                str(commit_path),
            )
        )
        return False, None

    normalized_streams = {
        str(stream): relative for stream, relative in raw_streams.items()
    }
    stream_names = set(normalized_streams)
    unknown_streams = stream_names - CHECKPOINT_STREAM_NAMES
    if unknown_streams:
        issues.append(
            ValidationIssue(
                "unknown_condition_stream",
                "error",
                ",".join(sorted(unknown_streams)),
                str(commit_path),
            )
        )
    if stream_names != {str(stream) for stream in raw_counts}:
        issues.append(
            ValidationIssue(
                "condition_stream_set_mismatch",
                "error",
                condition_id,
                str(commit_path),
            )
        )
        return False, None

    digest = hashlib.sha256()
    actual_counts: dict[str, int] = {}
    terminal_validator = CheckpointTransactionTerminalValidator(
        stream_names,
        terminal_status=str(commit.get("terminal_status") or ""),
    )
    for stream in sorted(stream_names):
        relative = normalized_streams[stream]
        path = resolve_checkpoint_path(analysis_root, str(relative))
        if path is None:
            issues.append(
                ValidationIssue(
                    "stream_path_escape",
                    "error",
                    str(relative),
                    str((analysis_root / str(relative)).resolve()),
                )
            )
            continue
        try:
            handle = path.open("r", encoding="utf-8")
        except OSError as error:
            issues.append(
                ValidationIssue(
                    "stream_unreadable",
                    "error",
                    str(error),
                    str(path),
                )
            )
            continue

        count = 0
        with handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    issues.append(
                        ValidationIssue(
                            "invalid_stream_row",
                            "error",
                            f"line {line_number}: {error}",
                            str(path),
                        )
                    )
                    break
                if not isinstance(row, dict):
                    issues.append(
                        ValidationIssue(
                            "invalid_stream_row",
                            "error",
                            f"line {line_number}: expected object",
                            str(path),
                        )
                    )
                    continue
                if str(row.get("condition_id", "")) != condition_id:
                    continue

                terminal_validator.observe(stream, row)
                try:
                    digest.update(
                        transaction_payload_line(
                            canonical_transaction_payload(row)
                        )
                    )
                except (TypeError, ValueError) as error:
                    issues.append(
                        ValidationIssue(
                            "noncanonical_stream_row",
                            "error",
                            f"line {line_number}: {error}",
                            str(path),
                        )
                    )
                    continue
                count += 1
        actual_counts[stream] = count

    try:
        normalized_counts = {
            str(stream): int(count) for stream, count in raw_counts.items()
        }
    except (TypeError, ValueError):
        normalized_counts = {}
        issues.append(
            ValidationIssue(
                "invalid_condition_row_counts",
                "error",
                condition_id,
                str(commit_path),
            )
        )
    if normalized_counts != actual_counts:
        issues.append(
            ValidationIssue(
                "condition_row_counts_mismatch",
                "error",
                condition_id,
                str(commit_path),
            )
        )
    if commit.get("payload_sha256") != digest.hexdigest():
        issues.append(
            ValidationIssue(
                "condition_payload_hash_mismatch",
                "error",
                condition_id,
                str(commit_path),
            )
        )

    terminal_row_found, analyzer_terminal = terminal_validator.result()
    if "analyzers" in stream_names:
        if stream_names != {"analyzers"}:
            issues.append(
                ValidationIssue(
                    "analyzer_transaction_stream_set_mismatch",
                    "error",
                    condition_id,
                    str(commit_path),
                )
            )
    if not terminal_row_found:
        issues.append(
            ValidationIssue(
                "missing_condition_terminal_row",
                "error",
                condition_id,
                str(commit_path),
            )
        )

    valid = not any(
        issue.severity == "error" for issue in issues[issue_start:]
    )
    return valid, analyzer_terminal


def _validate_checkpoint_commit_payloads(
    analysis_dir: Path,
    candidates: Mapping[str, Mapping[str, Any]],
    *,
    claimed_ids: set[str] | None,
    issues: list[ValidationIssue],
) -> _CheckpointCommitValidation:
    """逐 commit 校验独立 shard，并汇总合法 analyzer terminal。"""

    # 同名 stream 在不同 rank/generation commit 中指向不同 shard 是合法状态。
    analysis_root = analysis_dir.resolve()
    commits: dict[str, Mapping[str, Any]] = {}
    terminal_statuses: dict[str, set[str]] = {}
    terminal_condition_ids: dict[str, set[str]] = {}
    terminal_inventories: dict[str, list[Mapping[str, Any]]] = {}
    for condition_id, commit in candidates.items():
        issue_start = len(issues)
        valid, analyzer_terminal = _validate_checkpoint_commit_payload(
            analysis_dir,
            analysis_root=analysis_root,
            condition_id=condition_id,
            commit=commit,
            issues=issues,
        )
        if not valid:
            _record_claim_sensitive_commit_issue(
                issues,
                claimed_ids=claimed_ids,
                condition_id=condition_id,
                path=analysis_dir / "commits" / f"{condition_id}.json",
                issue=ValidationIssue(
                    "invalid_condition_commit",
                    "error",
                    condition_id,
                    str(analysis_dir / "commits" / f"{condition_id}.json"),
                ),
                replace_since=issue_start,
            )
            continue

        commits[condition_id] = commit
        if analyzer_terminal is None:
            continue
        analyzer_name = str(analyzer_terminal.get("analyzer") or "")
        if not analyzer_name:
            continue
        terminal_statuses.setdefault(analyzer_name, set()).add(
            str(analyzer_terminal.get("status") or "")
        )
        terminal_condition_ids.setdefault(analyzer_name, set()).add(
            condition_id
        )
        terminal_inventories.setdefault(analyzer_name, []).append(
            canonical_analyzer_evidence_inventory(analyzer_terminal)
        )

    return _CheckpointCommitValidation(
        commits=commits,
        analyzer_terminal_statuses=terminal_statuses,
        analyzer_terminal_condition_ids=terminal_condition_ids,
        analyzer_evidence_inventories={
            analyzer: tuple(
                sorted(
                    inventories,
                    key=lambda item: str(item.get("condition_id") or ""),
                )
            )
            for analyzer, inventories in terminal_inventories.items()
        },
    )


def _validate_checkpoint_identity_documents(
    analysis_dir: Path,
    *,
    manifest: Mapping[str, Any],
    analysis_id: Any,
    issues: list[ValidationIssue],
) -> tuple[dict[str, Any], bool]:
    """校验 identity 文档及其显式 runtime 绑定。"""

    recipe, recipe_issue = _load_object(analysis_dir / "recipe.json")
    cohort, cohort_issue = _load_object(analysis_dir / "cohort.json")
    for document_issue in (recipe_issue, cohort_issue):
        if document_issue is not None:
            issues.append(document_issue)

    if recipe_issue is None and manifest.get("recipe_hash") != stable_json_hash(recipe):
        issues.append(
            ValidationIssue(
                "recipe_hash_mismatch",
                "error",
                "recipe.json does not match manifest recipe_hash",
                str(analysis_dir / "recipe.json"),
            )
        )
    if (
        recipe_issue is None
        and manifest.get("condition_transaction_contract")
        == CHECKPOINT_TRANSACTION_CONTRACT
        and manifest.get("recipe") != recipe
    ):
        issues.append(
            ValidationIssue(
                "manifest_recipe_mismatch",
                "error",
                "manifest recipe must exactly match recipe.json",
                str(analysis_dir / "manifest.json"),
            )
        )
    if cohort_issue is None and manifest.get("cohort_hash") != stable_json_hash(cohort):
        issues.append(
            ValidationIssue(
                "cohort_hash_mismatch",
                "error",
                "cohort.json does not match manifest cohort_hash",
                str(analysis_dir / "cohort.json"),
            )
        )

    checkpoints = manifest.get("checkpoints")
    if (
        not isinstance(checkpoints, list)
        or manifest.get("checkpoint_hash") != stable_json_hash(checkpoints)
    ):
        issues.append(
            ValidationIssue(
                "checkpoint_hash_mismatch",
                "error",
                "manifest checkpoints do not match checkpoint_hash",
                str(analysis_dir / "manifest.json"),
            )
        )
    if (
        isinstance(checkpoints, list)
        and recipe_issue is None
        and cohort_issue is None
    ):
        missing_runtime_fields = [
            field
            for field in RUNTIME_DESCRIPTOR_FIELDS
            if field not in manifest
        ]
        if missing_runtime_fields:
            issues.append(
                ValidationIssue(
                    "missing_runtime_descriptor_fields",
                    "error",
                    ",".join(missing_runtime_fields),
                    str(analysis_dir / "manifest.json"),
                )
            )
        else:
            runtime_descriptor = {
                field: manifest[field]
                for field in RUNTIME_DESCRIPTOR_FIELDS
            }
            try:
                identity_payload = build_checkpoint_analysis_identity_payload(
                    recipe=recipe,
                    analyzer_definitions=manifest.get(
                        "analyzer_definitions",
                        (),
                    ),
                    checkpoint_identities=[
                        checkpoint.get("identity")
                        for checkpoint in checkpoints
                        if isinstance(checkpoint, Mapping)
                    ],
                    cohort_status=cohort.get("status"),
                    cohort_identity=cohort.get("identity"),
                    cohort_samples=cohort.get("sample_refs", []),
                    runtime_descriptor=runtime_descriptor,
                )
            except (TypeError, ValueError) as error:
                issues.append(
                    ValidationIssue(
                        "invalid_runtime_descriptor",
                        "error",
                        str(error),
                        str(analysis_dir / "manifest.json"),
                    )
                )
            else:
                if stable_json_hash(identity_payload) != str(analysis_id):
                    issues.append(
                        ValidationIssue(
                            "analysis_id_derivation_mismatch",
                            "error",
                            "analysis_id is not derived from canonical "
                            "recipe/checkpoint/cohort/runtime identity",
                            str(analysis_dir / "manifest.json"),
                        )
                    )
    return recipe, recipe_issue is None


def _validate_component_catalog_document(
    analysis_dir: Path,
    *,
    manifest: Mapping[str, Any],
    issues: list[ValidationIssue],
) -> set[str]:
    """校验 cohort 绑定的唯一 analysis-level component catalog sidecar。"""

    cohort, cohort_issue = _load_object(analysis_dir / "cohort.json")
    if cohort_issue is not None:
        return set()
    metadata = cohort.get("metadata")
    if not isinstance(metadata, Mapping):
        issues.append(
            ValidationIssue(
                "invalid_component_catalog_metadata",
                "error",
                "cohort metadata must be an object",
                str(analysis_dir / "cohort.json"),
            )
        )
        return set()
    catalog_path_value = metadata.get("component_catalog_path")
    digest = metadata.get("component_catalog_digest")
    capability_descriptors = manifest.get("capability_descriptors", ())
    capability_declared = (
        isinstance(capability_descriptors, list)
        and any(
            isinstance(descriptor, Mapping)
            and descriptor.get("capability_id") == "component_catalog"
            for descriptor in capability_descriptors
        )
    )
    if (
        catalog_path_value is None
        and digest is None
        and not capability_declared
    ):
        return set()
    if (
        "component_catalog" in metadata
        or catalog_path_value != "component_catalog.json"
        or not isinstance(digest, str)
    ):
        issues.append(
            ValidationIssue(
                "missing_component_catalog",
                "error",
                "cohort metadata must bind component_catalog.json and digest",
                str(analysis_dir / "cohort.json"),
            )
        )
        return set()
    if (
        manifest.get("component_catalog_path") != catalog_path_value
        or manifest.get("component_catalog_digest") != digest
    ):
        issues.append(
            ValidationIssue(
                "component_catalog_manifest_mismatch",
                "error",
                "manifest and cohort must bind the same component catalog",
                str(analysis_dir / "manifest.json"),
            )
        )
    catalog_path = resolve_checkpoint_path(
        analysis_dir,
        catalog_path_value,
    )
    expected_catalog_path = (analysis_dir / "component_catalog.json").resolve()
    if catalog_path != expected_catalog_path:
        issues.append(
            ValidationIssue(
                "component_catalog_path_escape",
                "error",
                str(catalog_path_value),
                str(analysis_dir / "cohort.json"),
            )
        )
        return set()
    try:
        rows = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        issues.append(
            ValidationIssue(
                "invalid_component_catalog",
                "error",
                str(error),
                str(catalog_path),
            )
        )
        return set()
    if not isinstance(rows, list):
        issues.append(
            ValidationIssue(
                "invalid_component_catalog",
                "error",
                "component catalog must be a JSON array",
                str(catalog_path),
            )
        )
        return set()
    if stable_json_hash(rows) != digest:
        issues.append(
            ValidationIssue(
                "component_catalog_digest_mismatch",
                "error",
                "component catalog does not match its digest",
                str(catalog_path),
            )
        )
    expected_fields = {
        "component_id",
        "semantic_id",
        "component_kind",
        "display_label",
        "component_group",
        "component_index",
        "tensor_path",
        "tensor_axis",
        "tensor_index",
        "component_shape",
        "normalization_id",
        "reduction_semantics",
        "metadata",
    }
    component_ids: set[str] = set()
    invalid_rows: list[int] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != expected_fields:
            invalid_rows.append(index)
            continue
        component_id = str(row.get("component_id") or "")
        kind = str(row.get("component_kind") or "")
        normalization_id = row.get("normalization_id")
        coordinates = (
            row.get("tensor_axis"),
            row.get("tensor_index"),
        )
        component_index = row.get("component_index")
        component_shape = row.get("component_shape")
        reduction_semantics = row.get("reduction_semantics")
        valid = (
            component_id
            and component_id not in component_ids
            and str(row.get("semantic_id") or "")
            and str(row.get("display_label") or "")
            and str(row.get("tensor_path") or "")
            and kind in {"input", "output"}
            and all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in coordinates
            )
            and (
                component_index is None
                or isinstance(component_index, int)
                and not isinstance(component_index, bool)
            )
            and (
                component_shape is None
                or isinstance(component_shape, list)
                and all(
                    isinstance(value, int)
                    and not isinstance(value, bool)
                    and value >= 0
                    for value in component_shape
                )
            )
            and (
                reduction_semantics is None
                or isinstance(reduction_semantics, str)
                and bool(reduction_semantics)
            )
            and isinstance(row.get("metadata"), Mapping)
            and (
                kind != "output"
                or isinstance(normalization_id, str)
                and bool(normalization_id)
            )
        )
        if not valid:
            invalid_rows.append(index)
            continue
        component_ids.add(component_id)
    if invalid_rows:
        issues.append(
            ValidationIssue(
                "invalid_component_catalog_rows",
                "error",
                ",".join(str(index) for index in invalid_rows[:16]),
                str(catalog_path),
            )
        )
    component_bound_identity = stable_json_hash(
        {
            "host_cohort_identity": metadata.get("host_cohort_identity"),
            "component_catalog_digest": digest,
        }
    )
    hierarchy_rows = metadata.get("hierarchy_catalog")
    hierarchy_digest = metadata.get("hierarchy_catalog_digest")
    hierarchy_status = metadata.get("hierarchy_discovery_status")
    if hierarchy_rows is None and hierarchy_digest is None:
        expected_identity = component_bound_identity
    elif (
        isinstance(hierarchy_rows, list)
        and isinstance(hierarchy_digest, str)
        and stable_json_hash(hierarchy_rows) == hierarchy_digest
        and isinstance(hierarchy_status, str)
        and bool(hierarchy_status)
        and all(
            isinstance(row, Mapping)
            and isinstance(row.get("node_id"), str)
            and bool(row.get("node_id"))
            and isinstance(row.get("hierarchy_level"), str)
            and bool(row.get("hierarchy_level"))
            for row in hierarchy_rows
        )
    ):
        expected_identity = stable_json_hash(
            {
                "component_bound_cohort_identity": component_bound_identity,
                "hierarchy_catalog_digest": hierarchy_digest,
                "hierarchy_discovery_status": hierarchy_status,
            }
        )
    else:
        issues.append(
            ValidationIssue(
                "invalid_hierarchy_catalog",
                "error",
                "checkpoint hierarchy catalog or digest is invalid",
                str(analysis_dir / "cohort.json"),
            )
        )
        expected_identity = None
    if cohort.get("identity") != expected_identity:
        issues.append(
            ValidationIssue(
                "component_catalog_cohort_identity_mismatch",
                "error",
                "cohort identity does not bind the component catalog digest",
                str(analysis_dir / "cohort.json"),
            )
        )
    return component_ids


def _validate_analyzer_definitions(
    manifest: Mapping[str, Any],
    requested_analyzers: list[str],
    *,
    manifest_path: Path,
    issues: list[ValidationIssue],
) -> list[str]:
    """校验 catalog descriptors，并返回需要 aggregate terminal 的 analyzer。

    validator 只依赖 manifest 中持久化的 execution mode，不维护 analyzer 名称 allow-list；
    因而宿主扩展 analyzer 不需要修改 artifact integrity 代码。
    """

    raw_definitions = manifest.get("analyzer_definitions")
    if not isinstance(raw_definitions, list):
        issues.append(
            ValidationIssue(
                "missing_analyzer_definitions",
                "error",
                "manifest must persist analyzer catalog descriptors",
                str(manifest_path),
            )
        )
        return []
    expected_keys = {
        "name",
        "evidence_kind",
        "execution_mode",
        "option_keys",
        "definition_version",
        "claim_boundaries",
    }
    names: list[str] = []
    terminal_analyzers: list[str] = []
    for index, raw_definition in enumerate(raw_definitions):
        if not isinstance(raw_definition, Mapping):
            issues.append(
                ValidationIssue(
                    "invalid_analyzer_definition",
                    "error",
                    f"descriptor {index} is not an object",
                    str(manifest_path),
                )
            )
            continue
        if set(raw_definition) != expected_keys:
            issues.append(
                ValidationIssue(
                    "invalid_analyzer_definition_fields",
                    "error",
                    f"descriptor {index} fields do not match contract",
                    str(manifest_path),
                )
            )
        name = str(raw_definition.get("name") or "")
        execution_mode = str(raw_definition.get("execution_mode") or "")
        evidence_kind = str(raw_definition.get("evidence_kind") or "")
        version = raw_definition.get("definition_version")
        if (
            not name
            or not evidence_kind
            or execution_mode not in {"condition", "aggregate", "stream"}
            or not isinstance(raw_definition.get("option_keys"), list)
            or not isinstance(raw_definition.get("claim_boundaries"), list)
            or not isinstance(version, int)
            or isinstance(version, bool)
            or version < 1
        ):
            issues.append(
                ValidationIssue(
                    "invalid_analyzer_definition_value",
                    "error",
                    name or f"descriptor {index}",
                    str(manifest_path),
                )
            )
            continue
        names.append(name)
        if execution_mode in {"aggregate", "stream"}:
            terminal_analyzers.append(name)
    if names != requested_analyzers or len(set(names)) != len(names):
        issues.append(
            ValidationIssue(
                "analyzer_definitions_recipe_mismatch",
                "error",
                "descriptor names must uniquely match recipe analyzer order",
                str(manifest_path),
            )
        )
    return terminal_analyzers


def _validate_finalized_checkpoint_streams(
    analysis_dir: Path,
    *,
    manifest: Mapping[str, Any],
    commits: Mapping[str, Mapping[str, Any]],
    component_ids: set[str],
    issues: list[ValidationIssue],
) -> None:
    """验证 report-visible allow-list，拒绝 orphan row、路径逃逸与 digest 漂移。"""

    finalized = manifest.get("finalized_streams")
    if not isinstance(finalized, list) or not finalized:
        issues.append(
            ValidationIssue(
                "missing_finalized_streams",
                "error",
                "complete analysis must bind report-visible streams to the manifest",
                str(analysis_dir / "manifest.json"),
            )
        )
        return

    analysis_root = analysis_dir.resolve()
    seen_stream_ids: set[str] = set()
    seen_paths: set[str] = set()
    referenced_component_ids: set[str] = set()
    for descriptor in finalized:
        if not isinstance(descriptor, Mapping):
            issues.append(
                ValidationIssue(
                    "invalid_finalized_stream_descriptor",
                    "error",
                    "expected object",
                    str(analysis_dir / "manifest.json"),
                )
            )
            continue
        stream_id = str(descriptor.get("stream_id") or "")
        relative = str(descriptor.get("path") or "")
        if (
            not stream_id
            or not relative
            or stream_id in seen_stream_ids
            or relative in seen_paths
        ):
            issues.append(
                ValidationIssue(
                    "duplicate_or_missing_finalized_stream_identity",
                    "error",
                    stream_id or relative,
                    str(analysis_dir / "manifest.json"),
                )
            )
            continue
        seen_stream_ids.add(stream_id)
        seen_paths.add(relative)

        path = resolve_checkpoint_path(analysis_root, relative)
        if path is None:
            issues.append(
                ValidationIssue(
                    "stream_path_escape",
                    "error",
                    relative,
                    str((analysis_root / relative).resolve()),
                )
            )
            continue
        try:
            actual_rows = 0
            orphan_condition_ids: set[str] = set()
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if not isinstance(row, Mapping):
                        raise ValueError(
                            "finalized stream row must be an object"
                        )
                    actual_rows += 1
                    row_condition_id = row.get("condition_id")
                    if (
                        row_condition_id is not None
                        and str(row_condition_id) not in commits
                    ):
                        orphan_condition_ids.add(str(row_condition_id))
                    for field in (
                        "intervened_component_id",
                        "response_component_id",
                    ):
                        component_id = row.get(field)
                        if component_id is not None:
                            referenced_component_ids.add(str(component_id))
        except (
            KeyError,
            TypeError,
            ValueError,
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as error:
            issues.append(
                ValidationIssue(
                    "finalized_stream_unreadable",
                    "error",
                    str(error),
                    str(path),
                )
            )
            continue
        if orphan_condition_ids:
            issues.append(
                ValidationIssue(
                    "finalized_stream_contains_uncommitted_rows",
                    "error",
                    ",".join(sorted(orphan_condition_ids)[:8]),
                    str(path),
                )
            )
        if not finalized_stream_matches_descriptor(
            path,
            descriptor,
            row_count=actual_rows,
        ):
            issues.append(
                ValidationIssue(
                    "finalized_stream_integrity_mismatch",
                    "error",
                    stream_id,
                    str(path),
                )
            )
    unresolved = sorted(referenced_component_ids - component_ids)
    if unresolved:
        issues.append(
            ValidationIssue(
                "unresolved_component_references",
                "error",
                ",".join(unresolved[:16]),
                str(analysis_dir / "manifest.json"),
            )
        )


def _reconcile_checkpoint_manifest(
    analysis_dir: Path,
    *,
    manifest: Mapping[str, Any],
    recipe: Mapping[str, Any],
    recipe_valid: bool,
    commit_validation: _CheckpointCommitValidation,
    issues: list[ValidationIssue],
) -> None:
    """对账 manifest 的 condition inventory、analyzer 请求与 terminal 摘要。"""

    commits = commit_validation.commits
    manifest_path = analysis_dir / "manifest.json"
    if manifest.get("committed_condition_count") != len(commits):
        issues.append(
            ValidationIssue(
                "manifest_condition_count_mismatch",
                "error",
                f"manifest={manifest.get('committed_condition_count')} actual={len(commits)}",
                str(manifest_path),
            )
        )
    committed_ids = manifest.get("committed_condition_ids")
    if committed_ids is not None and (
        not isinstance(committed_ids, list)
        or {str(condition_id) for condition_id in committed_ids}
        != set(commits)
    ):
        issues.append(
            ValidationIssue(
                "manifest_condition_ids_mismatch",
                "error",
                "manifest committed_condition_ids do not match commit files",
                str(manifest_path),
            )
        )

    expected_ids = manifest.get("expected_condition_ids")
    if (
        manifest.get("condition_transaction_contract")
        == CHECKPOINT_TRANSACTION_CONTRACT
    ):
        expected_set = (
            {str(condition_id) for condition_id in expected_ids}
            if isinstance(expected_ids, list)
            else set()
        )
        if (
            manifest.get("condition_planning_status") != "complete"
            or not expected_set
            or manifest.get("expected_condition_count") != len(expected_set)
            or expected_set != set(commits)
            or manifest.get("missing_expected_condition_ids") not in ([], ())
            or manifest.get("unexpected_valid_condition_ids") not in ([], ())
        ):
            issues.append(
                ValidationIssue(
                    "condition_plan_integrity_mismatch",
                    "error",
                    "expected condition inventory does not match valid commits",
                    str(manifest_path),
                )
            )
    if manifest.get("status") in {"complete", "degraded"} and not commits:
        issues.append(
            ValidationIssue(
                "terminal_analysis_without_commits",
                "error",
                "terminal analysis must contain at least one committed condition",
                str(manifest_path),
            )
        )

    raw_requested_analyzers = (
        recipe.get("analyzers", {})
        if recipe_valid and isinstance(recipe, Mapping)
        else {}
    )
    requested_analyzers = (
        list(raw_requested_analyzers)
        if isinstance(raw_requested_analyzers, (Mapping, list))
        else []
    )
    if (
        recipe_valid
        and isinstance(recipe, Mapping)
        and manifest.get("condition_transaction_contract")
        == CHECKPOINT_TRANSACTION_CONTRACT
        and manifest.get("analyzers") != requested_analyzers
    ):
        issues.append(
            ValidationIssue(
                "manifest_analyzers_mismatch",
                "error",
                "manifest analyzers must match validated recipe.json",
                str(manifest_path),
            )
        )

    terminal_statuses = manifest.get("analyzer_terminal_statuses", {})
    terminal_condition_ids = manifest.get(
        "analyzer_terminal_condition_ids",
        {},
    )
    terminal_inventories = manifest.get(
        "analyzer_evidence_inventories",
        {},
    )
    terminal_analyzers = _validate_analyzer_definitions(
        manifest,
        [str(analyzer) for analyzer in requested_analyzers],
        manifest_path=manifest_path,
        issues=issues,
    )
    if (
        terminal_analyzers
        and manifest.get("condition_transaction_contract")
        != CHECKPOINT_TRANSACTION_CONTRACT
    ):
        issues.append(
            ValidationIssue(
                "analyzer_transaction_contract_mismatch",
                "error",
                ",".join(terminal_analyzers),
                str(manifest_path),
            )
        )
    for analyzer_name in terminal_analyzers:
        statuses = (
            terminal_statuses.get(analyzer_name)
            if isinstance(terminal_statuses, Mapping)
            else None
        )
        if not isinstance(statuses, list) or not statuses:
            issues.append(
                ValidationIssue(
                    "missing_analyzer_terminal_status",
                    "error",
                    analyzer_name,
                    str(manifest_path),
                )
            )
        elif {str(status) for status in statuses} != (
            commit_validation.analyzer_terminal_statuses.get(
                analyzer_name,
                set(),
            )
        ):
            issues.append(
                ValidationIssue(
                    "analyzer_terminal_status_mismatch",
                    "error",
                    analyzer_name,
                    str(manifest_path),
                )
            )
        condition_ids = (
            terminal_condition_ids.get(analyzer_name)
            if isinstance(terminal_condition_ids, Mapping)
            else None
        )
        if not isinstance(condition_ids, list) or {
            str(condition_id) for condition_id in condition_ids
        } != commit_validation.analyzer_terminal_condition_ids.get(
            analyzer_name,
            set(),
        ):
            issues.append(
                ValidationIssue(
                    "analyzer_terminal_condition_ids_mismatch",
                    "error",
                    analyzer_name,
                    str(manifest_path),
                )
            )
        expected_inventories = [
            dict(item)
            for item in commit_validation.analyzer_evidence_inventories.get(
                analyzer_name,
                (),
            )
        ]
        manifest_inventories = (
            terminal_inventories.get(analyzer_name)
            if isinstance(terminal_inventories, Mapping)
            else None
        )
        if manifest_inventories != expected_inventories:
            issues.append(
                ValidationIssue(
                    "analyzer_evidence_inventory_mismatch",
                    "error",
                    analyzer_name,
                    str(manifest_path),
                )
            )


def _validate_checkpoint_analysis_impl(analysis_dir: Path) -> CheckpointValidation:
    """校验 condition commit，并按 stream 顺序重新计算 payload SHA-256。"""

    issues: list[ValidationIssue] = []
    manifest_path = analysis_dir / "manifest.json"
    run_state_path = analysis_dir / "run_state.json"
    if not manifest_path.is_file() and run_state_path.is_file():
        run_state, run_state_issue = _load_object(run_state_path)
        if run_state_issue is not None:
            return CheckpointValidation(
                analysis_dir.name,
                "corrupt",
                False,
                None,
                0,
                (run_state_issue,),
                run_state,
            )
        issues.append(
            ValidationIssue(
                "checkpoint_analysis_not_finalized",
                "warning",
                "run_state exists without a terminal manifest",
                str(run_state_path),
            )
        )
        run_state_format = str(run_state.get("format_version") or "")
        if run_state_format != CHECKPOINT_FORMAT_VERSION:
            issues.append(
                ValidationIssue(
                    "format_version_mismatch",
                    "error",
                    f"expected {CHECKPOINT_FORMAT_VERSION}",
                    str(run_state_path),
                )
            )
            return CheckpointValidation(
                str(run_state.get("analysis_id") or analysis_dir.name),
                "corrupt",
                False,
                str(run_state.get("status") or "running"),
                0,
                tuple(issues),
                run_state,
            )
        try:
            normalize_runtime_descriptor(
                {
                    field: run_state[field]
                    for field in RUNTIME_DESCRIPTOR_FIELDS
                }
            )
        except (KeyError, TypeError, ValueError) as error:
            issues.append(
                ValidationIssue(
                    "invalid_runtime_descriptor",
                    "error",
                    str(error),
                    str(run_state_path),
                )
            )
        return CheckpointValidation(
            str(run_state.get("analysis_id") or analysis_dir.name),
            "partial",
            False,
            str(run_state.get("status") or "running"),
            len(tuple((analysis_dir / "commits").glob("*.json"))),
            tuple(issues),
            run_state,
        )
    manifest, issue = _load_object(manifest_path)
    if issue is not None:
        issues.append(issue)
        return CheckpointValidation(None, "corrupt", False, None, 0, tuple(issues), manifest)
    analysis_id = manifest.get("analysis_id")
    format_version = str(manifest.get("format_version") or "")
    if str(analysis_id) != analysis_dir.name:
        issues.append(
            ValidationIssue(
                "analysis_identity_mismatch",
                "error",
                "manifest analysis_id does not match directory",
                str(analysis_dir),
            )
        )
    if format_version != CHECKPOINT_FORMAT_VERSION:
        issues.append(
            ValidationIssue(
                "format_version_mismatch",
                "error",
                f"expected {CHECKPOINT_FORMAT_VERSION}",
                str(analysis_dir / "manifest.json"),
            )
        )
        return CheckpointValidation(
            str(analysis_id) if analysis_id is not None else None,
            "corrupt",
            False,
            str(manifest.get("status"))
            if manifest.get("status") is not None
            else None,
            0,
            tuple(issues),
            manifest,
        )

    raw_claimed_ids = manifest.get("committed_condition_ids")
    raw_expected_ids = manifest.get("expected_condition_ids")
    claimed_ids = (
        {str(condition_id) for condition_id in raw_claimed_ids}
        if isinstance(raw_claimed_ids, list)
        else None
    )
    if isinstance(raw_expected_ids, list):
        expected_claims = {
            str(condition_id) for condition_id in raw_expected_ids
        }
        claimed_ids = (
            expected_claims
            if claimed_ids is None
            else claimed_ids | expected_claims
        )
    candidate_commits = _collect_checkpoint_commit_candidates(
        analysis_dir,
        claimed_ids=claimed_ids,
        issues=issues,
    )

    commit_validation = _validate_checkpoint_commit_payloads(
        analysis_dir,
        candidate_commits,
        claimed_ids=claimed_ids,
        issues=issues,
    )
    commits = commit_validation.commits

    recipe, recipe_valid = _validate_checkpoint_identity_documents(
        analysis_dir,
        manifest=manifest,
        analysis_id=analysis_id,
        issues=issues,
    )
    component_ids = _validate_component_catalog_document(
        analysis_dir,
        manifest=manifest,
        issues=issues,
    )

    _validate_finalized_checkpoint_streams(
        analysis_dir,
        manifest=manifest,
        commits=commits,
        component_ids=component_ids,
        issues=issues,
    )
    _reconcile_checkpoint_manifest(
        analysis_dir,
        manifest=manifest,
        recipe=recipe,
        recipe_valid=recipe_valid,
        commit_validation=commit_validation,
        issues=issues,
    )
    valid = not any(issue.severity == "error" for issue in issues)
    manifest_status = str(manifest.get("status")) if manifest.get("status") is not None else None
    if not valid or manifest_status == "corrupt":
        status = "corrupt"
    elif manifest_status == "degraded":
        status = "degraded"
    elif manifest_status == "complete":
        status = "success"
    else:
        status = "partial"
    return CheckpointValidation(
        str(analysis_id) if analysis_id is not None else None,
        status,
        valid,
        manifest_status,
        len(commits),
        tuple(issues),
        manifest,
    )


__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "CHECKPOINT_STREAM_NAMES",
    "CHECKPOINT_TRANSACTION_STREAMS",
    "CHECKPOINT_TRANSACTION_CONTRACT",
    "CheckpointTransactionTerminalValidator",
    "CheckpointValidation",
    "canonical_analyzer_evidence_inventory",
    "canonical_transaction_payload",
    "finalized_stream_matches_descriptor",
    "resolve_checkpoint_path",
    "transaction_has_valid_terminal",
    "transaction_payload_hash",
    "transaction_payload_line",
    "validate_checkpoint_analysis",
]
