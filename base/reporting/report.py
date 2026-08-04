"""从已验证的 diagnostics/v2 sources 生成 descriptor-driven 统一报告。"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..artifacts.checkpoint_validation import (
    CheckpointValidation,
    validate_checkpoint_analysis,
)
from ..artifacts.common import (
    V2_ROOT_FORMAT_VERSION,
    discover_v2_artifacts,
    file_sha256,
    stable_json_hash,
    update_v2_index,
    write_json_atomic,
)
from ..artifacts.failure import validate_failure_bundle
from ..artifacts.online import (
    iter_committed_transactions,
    validate_online_session,
)
from .renderers import (
    BASE_EVIDENCE_RENDERER_CATALOG,
    EvidenceRendererCatalog,
    build_metric_guide_catalog,
    render_evidence_figure,
    safe_figure_id,
)
from .web import (
    ReportArtifact,
    STANDALONE_REPORT_FILENAME,
    write_interactive_report,
)


REPORT_FORMAT_VERSION = 18


def generate_diagnostics_report(
    run_dir: str | Path,
    *,
    renderer_catalog: EvidenceRendererCatalog | None = None,
) -> Path:
    """校验 manifests、commits 与 digests 后流式汇总通用 analyzer evidence。"""

    root = Path(run_dir).resolve()
    target = (root / "diagnostics" / "v2" / "report").resolve()
    target.mkdir(parents=True, exist_ok=True)
    (target / "figures").mkdir(exist_ok=True)
    (target / "data").mkdir(exist_ok=True)
    started_at = _utc_now()
    write_json_atomic(
        target / "manifest.json",
        {
            "format_version": REPORT_FORMAT_VERSION,
            "artifact_kind": "diagnostics_v2_report",
            "status": "running",
            "started_at": started_at,
        },
    )

    inventory = discover_v2_artifacts(root)
    checkpoint_paths = _active_checkpoint_analyses(
        root,
        inventory.checkpoint_analyses,
    )
    online_results = [
        validate_online_session(path) for path in inventory.online_sessions
    ]
    checkpoint_results = [
        validate_checkpoint_analysis(path)
        for path in checkpoint_paths
    ]
    failure_results = [
        validate_failure_bundle(path)
        for path in inventory.failure_bundles
    ]
    source_statuses = [
        result.status
        for result in (
            *online_results,
            *checkpoint_results,
            *failure_results,
        )
    ]
    status = _report_status(source_statuses)
    valid_online = tuple(
        path
        for path, result in zip(
            inventory.online_sessions,
            online_results,
            strict=True,
        )
        if result.valid
        and result.status in {"complete", "success", "degraded"}
    )
    valid_checkpoint = tuple(
        path
        for path, result in zip(
            checkpoint_paths,
            checkpoint_results,
            strict=True,
        )
        if result.valid
        and result.status in {"complete", "success", "degraded"}
    )
    online_summaries = [
        {
            **result.as_dict(),
            "content": (
                dict(result.content_summary)
                if result.valid and result.status != "corrupt"
                else {"status": "not_consumed_after_validation_failure"}
            ),
        }
        for result in online_results
    ]
    checkpoint_summaries = [
        {
            **result.as_dict(),
            "analyzer_evidence": _analyzer_evidence_summary(path, result),
        }
        for path, result in zip(
            checkpoint_paths,
            checkpoint_results,
            strict=True,
        )
    ]
    failure_summaries = [
        {
            "bundle_id": result.bundle_id,
            "path": str(path),
            "status": result.status,
            "valid": result.valid,
            "expected_rank_count": result.expected_rank_count,
            "captured_rank_count": result.captured_rank_count,
            "evidence_scope": sorted(
                {
                    rank.evidence_scope
                    for rank in result.rank_validations
                    if rank.evidence_scope is not None
                }
            ),
            "issue_codes": sorted(
                {issue.code for issue in result.issues}
            ),
            # 报告只展示完整性与安全等级，不把 failure attachments 暴露到 HTML。
            "attachments_exposed": False,
        }
        for path, result in zip(
            inventory.failure_bundles,
            failure_results,
            strict=True,
        )
    ]
    hierarchy = _hierarchy_index(valid_online, valid_checkpoint)
    component_catalogs = _component_catalog_index(
        valid_checkpoint,
        valid_online,
    )
    active_renderer_catalog = (
        BASE_EVIDENCE_RENDERER_CATALOG
        if renderer_catalog is None
        else renderer_catalog
    )
    if not isinstance(active_renderer_catalog, EvidenceRendererCatalog):
        raise TypeError(
            "renderer_catalog must be an EvidenceRendererCatalog"
        )
    evidence_figures = _render_evidence_figures(
        target=target,
        online_sessions=valid_online,
        checkpoint_analyses=valid_checkpoint,
        renderer_catalog=active_renderer_catalog,
    )
    summary: dict[str, Any] = {
        "format_version": REPORT_FORMAT_VERSION,
        "root_format_version": V2_ROOT_FORMAT_VERSION,
        "status": status,
        "run_dir": str(root),
        "generated_at": _utc_now(),
        "online_sessions": online_summaries,
        "checkpoint_analyses": checkpoint_summaries,
        "failure_bundles": failure_summaries,
        "source_count": (
            len(online_summaries)
            + len(checkpoint_summaries)
            + len(failure_summaries)
        ),
        "hierarchy_node_count": len(hierarchy),
        "component_catalogs": component_catalogs,
        "evidence_figures": evidence_figures,
        "renderer_evidence_kinds": sorted(
            active_renderer_catalog.definitions
        ),
        "metric_guides": build_metric_guide_catalog(
            active_renderer_catalog
        ),
        "execution_semantics": {
            "artifact_scope": "diagnostics/v2 only",
            "transaction_visibility": "committed transactions only",
            "analyzer_interpretation": "manifest descriptors and claim boundaries",
            "local_taylor_is_conductance": False,
            "local_taylor_is_integrated_gradients": False,
            "local_taylor_is_activation_patching": False,
            "physical_causality_claimed": False,
            "failure_attachments_exposed": False,
        },
    }
    artifacts = _report_artifacts(
        root,
        valid_online,
        valid_checkpoint,
        tuple(
            path
            for path, result in zip(
                inventory.failure_bundles,
                failure_results,
                strict=True,
            )
            if result.valid and result.status in {"success", "degraded"}
        ),
    )
    delivery = write_interactive_report(
        root,
        target,
        report_format_version=REPORT_FORMAT_VERSION,
        summary=summary,
        artifacts=artifacts,
        hierarchy_index=hierarchy,
        metric_guides=summary["metric_guides"],
    )
    write_json_atomic(target / "summary.json", summary)
    (target / "report.md").write_text(
        _markdown(summary),
        encoding="utf-8",
    )
    files = {
        "summary": target / "summary.json",
        "markdown": target / "report.md",
        "html": root / "diagnostics" / "v2" / STANDALONE_REPORT_FILENAME,
    }
    write_json_atomic(
        target / "manifest.json",
        {
            "format_version": REPORT_FORMAT_VERSION,
            "artifact_kind": "diagnostics_v2_report",
            "status": status,
            "started_at": started_at,
            "generated_at": summary["generated_at"],
            "source_count": summary["source_count"],
            "delivery": delivery,
            "files": {
                name: {
                    "path": os.path.relpath(path, target),
                    "sha256": file_sha256(path),
                    "bytes": path.stat().st_size,
                }
                for name, path in files.items()
            },
            "figures": [
                {
                    "figure_id": figure["figure_id"],
                    "source_id": figure["source_id"],
                    "evidence_kind": figure["evidence_kind"],
                    "status": figure["status"],
                    "path": figure["figure_path"],
                    "sha256": file_sha256(
                        target / str(figure["figure_path"])
                    ),
                    "bytes": (
                        target / str(figure["figure_path"])
                    ).stat().st_size,
                }
                for figure in evidence_figures
            ],
        },
    )
    update_v2_index(
        root,
        "reports",
        "latest",
        {
            "path": str(target.relative_to(root / "diagnostics" / "v2")),
            "standalone_path": STANDALONE_REPORT_FILENAME,
            "status": status,
            "generated_at": summary["generated_at"],
            "format_version": REPORT_FORMAT_VERSION,
        },
    )
    return target


def _active_checkpoint_analyses(
    root: Path,
    analyses: Iterable[Path],
) -> tuple[Path, ...]:
    """只消费 index 声明的当前 runtime generation，避免新旧证据混图。"""

    candidates = tuple(analyses)
    index_path = root / "diagnostics" / "v2" / "index.json"
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return candidates
    active_digest = (
        index.get("active_checkpoint_runtime_descriptor_digest")
        if isinstance(index, Mapping)
        else None
    )
    if not isinstance(active_digest, str) or not active_digest.strip():
        return candidates
    selected: list[Path] = []
    for analysis in candidates:
        try:
            manifest = json.loads(
                (analysis / "manifest.json").read_text(encoding="utf-8")
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if (
            isinstance(manifest, Mapping)
            and manifest.get("runtime_descriptor_digest")
            == active_digest.strip()
        ):
            selected.append(analysis)
    if not selected:
        raise ValueError(
            "active checkpoint runtime descriptor has no analysis artifacts"
        )
    return tuple(selected)


def _render_evidence_figures(
    *,
    target: Path,
    online_sessions: Iterable[Path],
    checkpoint_analyses: Iterable[Path],
    renderer_catalog: EvidenceRendererCatalog,
) -> list[dict[str, Any]]:
    """从 validator-approved sources 流式生成 figures，并清理同 owner 的陈旧 SVG。"""

    figure_directory = target / "figures"
    for stale in figure_directory.glob("*.svg"):
        stale.unlink()
    figures: list[dict[str, Any]] = []
    for session in online_sessions:
        try:
            manifest = json.loads(
                (session / "manifest.json").read_text(encoding="utf-8")
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        descriptors = manifest.get("analyzer_definitions", ())
        if not isinstance(descriptors, list):
            continue
        for descriptor in descriptors:
            if not isinstance(descriptor, Mapping):
                continue
            analyzer_name = str(descriptor.get("name") or "")
            if not analyzer_name:
                continue
            source_id = f"online:{session.name}:analyzer:{analyzer_name}"
            figure = render_evidence_figure(
                source_id=source_id,
                analyzer_name=analyzer_name,
                descriptor=descriptor,
                rows=_iter_online_evidence_rows(
                    session,
                    descriptor,
                    renderer_catalog,
                ),
                renderer_catalog=renderer_catalog,
                figure_id=_stable_figure_id(source_id),
            )
            figures.append(
                _write_figure(figure_directory, figure)
            )
    for analysis in checkpoint_analyses:
        try:
            manifest = json.loads(
                (analysis / "manifest.json").read_text(encoding="utf-8")
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        descriptors = manifest.get("analyzer_definitions", ())
        if not isinstance(descriptors, list):
            continue
        for descriptor in descriptors:
            if not isinstance(descriptor, Mapping):
                continue
            analyzer_name = str(descriptor.get("name") or "")
            if not analyzer_name:
                continue
            source_id = (
                f"checkpoint:{analysis.name}:analyzer:{analyzer_name}"
            )
            figure = render_evidence_figure(
                source_id=source_id,
                analyzer_name=analyzer_name,
                descriptor=descriptor,
                rows=_iter_checkpoint_evidence_rows(
                    analysis,
                    manifest,
                    analyzer_name,
                ),
                renderer_catalog=renderer_catalog,
                figure_id=_stable_figure_id(source_id),
            )
            figures.append(
                _write_figure(figure_directory, figure)
            )
    return figures


def _iter_online_evidence_rows(
    session: Path,
    descriptor: Mapping[str, Any],
    renderer_catalog: EvidenceRendererCatalog,
) -> Iterable[Mapping[str, Any]]:
    """用当前 renderer contract 字段筛选在线 stream，不合成 analyzer descriptor。"""

    evidence_kind = str(descriptor.get("evidence_kind") or "")
    definition = renderer_catalog.get(evidence_kind)
    definition_version = descriptor.get("definition_version")
    compatible = (
        definition is not None
        and isinstance(definition_version, int)
        and not isinstance(definition_version, bool)
        and definition_version == definition.analyzer_definition_version
    )
    selected_paths = (
        tuple(metric.path for metric in definition.metrics)
        + definition.null_control_fields
        + definition.status_fields
        if compatible and definition is not None
        else ()
    )
    for transaction in iter_committed_transactions(session):
        for row in transaction.rows:
            if not selected_paths or any(
                _mapping_has_path(row, path)
                for path in selected_paths
            ):
                yield row


def _mapping_has_path(value: Mapping[str, Any], path: str) -> bool:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return False
        current = current[part]
    return True


def _iter_checkpoint_evidence_rows(
    analysis: Path,
    manifest: Mapping[str, Any],
    analyzer_name: str,
) -> Iterable[Mapping[str, Any]]:
    """通过 finalized stream descriptor 连接 analyzer evidence，不按名称分发。"""

    analysis_root = analysis.resolve()
    streams = manifest.get("finalized_streams", ())
    if not isinstance(streams, list):
        return
    for stream in streams:
        if not isinstance(stream, Mapping):
            continue
        raw_analyzer_names = stream.get("analyzer_names", ())
        if not isinstance(raw_analyzer_names, (list, tuple)):
            continue
        analyzer_names = {
            str(value) for value in raw_analyzer_names
        }
        if analyzer_name not in analyzer_names:
            continue
        path = (analysis_root / str(stream.get("path") or "")).resolve()
        try:
            path.relative_to(analysis_root)
        except ValueError:
            continue
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if not isinstance(row, Mapping):
                        continue
                    row_analyzer = row.get("analyzer")
                    if (
                        row_analyzer is not None
                        and str(row_analyzer) != analyzer_name
                    ):
                        continue
                    yield row
        except (
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ):
            continue


def _stable_figure_id(source_id: str) -> str:
    digest = hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:10]
    return safe_figure_id(f"{source_id}-{digest}")


def _write_figure(
    figure_directory: Path,
    figure: Mapping[str, Any],
) -> dict[str, Any]:
    """写 standalone SVG，同时把同一安全 markup 保留给 self-contained HTML。"""

    materialized = dict(figure)
    path = figure_directory / f"{materialized['figure_id']}.svg"
    path.write_text(str(materialized["svg"]), encoding="utf-8")
    materialized["figure_path"] = str(
        Path("figures") / path.name
    )
    return materialized


def _analyzer_evidence_summary(
    analysis_dir: Path,
    validation: CheckpointValidation,
) -> list[dict[str, Any]]:
    """只按 manifest descriptor 汇总，不维护 analyzer 名称 allow-list。"""

    if not validation.valid or validation.status == "corrupt":
        return []
    descriptors = validation.manifest.get("analyzer_definitions", ())
    streams = validation.manifest.get("finalized_streams", ())
    if not isinstance(descriptors, list) or not isinstance(streams, list):
        return []
    rows: list[dict[str, Any]] = []
    for descriptor in descriptors:
        if not isinstance(descriptor, Mapping):
            continue
        name = str(descriptor.get("name") or "")
        matching = [
            stream
            for stream in streams
            if isinstance(stream, Mapping)
            and isinstance(
                stream.get("analyzer_names", ()),
                (list, tuple),
            )
            and name
            in {
                str(value)
                for value in stream.get("analyzer_names", ())
            }
        ]
        inventories = validation.manifest.get(
            "analyzer_evidence_inventories",
            {},
        )
        analyzer_inventories = (
            inventories.get(name, ())
            if isinstance(inventories, Mapping)
            else ()
        )
        if not isinstance(analyzer_inventories, list):
            analyzer_inventories = ()
        rows.append(
            {
                "name": name,
                "evidence_kind": descriptor.get("evidence_kind"),
                "execution_mode": descriptor.get("execution_mode"),
                "definition_version": descriptor.get("definition_version"),
                "claim_boundaries": list(
                    descriptor.get("claim_boundaries", ())
                ),
                "stream_count": len(matching),
                "row_count": sum(
                    int(stream.get("row_count", 0) or 0)
                    for stream in matching
                ),
                "expected_evidence_count": sum(
                    int(item.get("expected_evidence_count", 0) or 0)
                    for item in analyzer_inventories
                    if isinstance(item, Mapping)
                ),
                "observed_evidence_count": sum(
                    int(item.get("observed_evidence_count", 0) or 0)
                    for item in analyzer_inventories
                    if isinstance(item, Mapping)
                ),
                "skipped_evidence_count": sum(
                    int(item.get("skipped_evidence_count", 0) or 0)
                    for item in analyzer_inventories
                    if isinstance(item, Mapping)
                ),
                "failed_evidence_count": sum(
                    int(item.get("failed_evidence_count", 0) or 0)
                    for item in analyzer_inventories
                    if isinstance(item, Mapping)
                ),
                "missing_evidence_count": sum(
                    int(item.get("missing_evidence_count", 0) or 0)
                    for item in analyzer_inventories
                    if isinstance(item, Mapping)
                ),
                "source_integrity": validation.status,
            }
        )
    return rows


def _component_catalog_index(
    analyses: Iterable[Path],
    online_sessions: Iterable[Path],
) -> dict[str, dict[str, Any]]:
    """读取已验证 source 的唯一 component catalog，供网页解析身份。"""

    catalogs: dict[str, dict[str, Any]] = {}
    for session in online_sessions:
        try:
            hierarchy = json.loads(
                (session / "hierarchy.json").read_text(encoding="utf-8")
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(hierarchy, Mapping):
            continue
        components = hierarchy.get("component_catalog")
        digest = hierarchy.get("component_catalog_digest")
        if (
            hierarchy.get("component_catalog_status") != "frozen"
            or not isinstance(components, list)
            or not components
            or not isinstance(digest, str)
            or stable_json_hash(components) != digest
        ):
            continue
        catalogs[session.name] = {
            "component_catalog_digest": digest,
            "components": [
                dict(component)
                for component in components
                if isinstance(component, Mapping)
            ],
        }
    for analysis in analyses:
        try:
            cohort = json.loads(
                (analysis / "cohort.json").read_text(encoding="utf-8")
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        metadata = (
            cohort.get("metadata", {})
            if isinstance(cohort, Mapping)
            else {}
        )
        catalog_path = (
            metadata.get("component_catalog_path")
            if isinstance(metadata, Mapping)
            else None
        )
        digest = (
            metadata.get("component_catalog_digest")
            if isinstance(metadata, Mapping)
            else None
        )
        if catalog_path != "component_catalog.json" or not isinstance(
            digest,
            str,
        ):
            continue
        try:
            components = json.loads(
                (analysis / catalog_path).read_text(encoding="utf-8")
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(components, list):
            continue
        checkpoint_catalog = {
            "component_catalog_digest": digest,
            "components": [
                dict(component)
                for component in components
                if isinstance(component, Mapping)
            ],
        }
        existing = catalogs.get(analysis.name)
        if (
            existing is None
            or existing.get("component_catalog_digest") == digest
        ):
            catalogs[analysis.name] = checkpoint_catalog
    return catalogs


def _report_artifacts(
    root: Path,
    online_sessions: Iterable[Path],
    checkpoint_analyses: Iterable[Path],
    failure_bundles: Iterable[Path],
) -> list[ReportArtifact]:
    """列出报告可读 source，并携带 analyzer descriptor 驱动的栏目归属。"""

    artifacts: list[ReportArtifact] = []
    for session in online_sessions:
        try:
            session_manifest = json.loads(
                (session / "manifest.json").read_text(encoding="utf-8")
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            session_manifest = {}
        online_evidence_kinds = tuple(
            sorted(
                {
                    str(descriptor.get("evidence_kind"))
                    for descriptor in session_manifest.get(
                        "analyzer_definitions",
                        (),
                    )
                    if isinstance(descriptor, Mapping)
                    and descriptor.get("evidence_kind")
                }
            )
        )
        for path in (session / "manifest.json", session / "hierarchy.json"):
            if path.is_file():
                artifacts.append(
                    ReportArtifact(
                        name=f"online-{session.name}-{path.stem}",
                        label=f"online {session.name} · {path.name}",
                        group="online",
                        path=path,
                        artifact_format="json",
                        content_kind="metadata",
                        scope_id=session.name,
                    )
                )
        for path in sorted((session / "segments").glob("*.jsonl")):
            artifacts.append(
                ReportArtifact(
                    name=f"online-{session.name}-{path.stem}",
                    label=f"online {session.name} · {path.name}",
                    group="online",
                    path=path,
                    artifact_format="jsonl",
                    content_kind="evidence_stream",
                    scope_id=session.name,
                    evidence_kinds=online_evidence_kinds,
                )
            )
    for analysis in checkpoint_analyses:
        try:
            manifest = json.loads(
                (analysis / "manifest.json").read_text(encoding="utf-8")
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        analyzer_evidence_kinds = {
            str(descriptor.get("name")): str(
                descriptor.get("evidence_kind")
            )
            for descriptor in manifest.get("analyzer_definitions", ())
            if isinstance(descriptor, Mapping)
            and descriptor.get("name")
            and descriptor.get("evidence_kind")
        }
        paths: dict[Path, tuple[str, tuple[str, ...]]] = {
            analysis / "manifest.json": ("metadata", ()),
            analysis / "recipe.json": ("metadata", ()),
            analysis / "cohort.json": ("metadata", ()),
            analysis / "component_catalog.json": ("metadata", ()),
        }
        for descriptor in manifest.get("finalized_streams", ()):
            if not isinstance(descriptor, Mapping):
                continue
            relative = str(descriptor.get("path") or "")
            candidate = (analysis / relative).resolve()
            try:
                candidate.relative_to(analysis.resolve())
            except ValueError:
                continue
            analyzer_names = descriptor.get("analyzer_names", ())
            evidence_kinds = tuple(
                sorted(
                    {
                        *(
                            analyzer_evidence_kinds[str(name)]
                            for name in analyzer_names
                            if str(name) in analyzer_evidence_kinds
                        ),
                        *_explicit_evidence_kinds(candidate),
                    }
                )
            )
            paths[candidate] = ("evidence_stream", evidence_kinds)
        for path in sorted(paths):
            if not path.is_file() or path.suffix not in {".json", ".jsonl"}:
                continue
            relative = path.relative_to(analysis)
            content_kind, evidence_kinds = paths[path]
            artifacts.append(
                ReportArtifact(
                    name=(
                        "checkpoint-"
                        + analysis.name
                        + "-"
                        + hashlib.sha256(
                            str(relative).encode()
                        ).hexdigest()[:12]
                    ),
                    label=f"checkpoint {analysis.name} · {relative}",
                    group="checkpoint",
                    path=path,
                    artifact_format=(
                        "json" if path.suffix == ".json" else "jsonl"
                    ),
                    content_kind=content_kind,
                    scope_id=analysis.name,
                    evidence_kinds=evidence_kinds,
                )
            )
    for bundle in failure_bundles:
        # Failure payload 可含模型权重、梯度、RNG 与原始输入；统一报告只链接经过
        # validator 的 manifest/commit 元数据，不复制或嵌入任何 attachment。
        paths = [
            path
            for path in bundle.glob("ranks/rank_*/manifest.json")
            if path.is_file()
        ]
        paths.extend(
            path
            for path in bundle.glob("ranks/rank_*/commit.json")
            if path.is_file()
        )
        for path in sorted(paths):
            relative = path.relative_to(bundle)
            artifacts.append(
                ReportArtifact(
                    name=(
                        "failure-"
                        + bundle.name
                        + "-"
                        + hashlib.sha256(
                            str(relative).encode()
                        ).hexdigest()[:12]
                    ),
                    label=f"failure {bundle.name} · {relative}",
                    group="failure",
                    path=path,
                    artifact_format="json",
                    content_kind="metadata",
                    scope_id=bundle.name,
                )
            )
    return artifacts


def _explicit_evidence_kinds(path: Path) -> set[str]:
    """读取 stream 行级 evidence owner，支持一次执行产生多种证据。"""

    if path.suffix != ".jsonl" or not path.is_file():
        return set()
    kinds: set[str] = set()
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, Mapping):
                    continue
                evidence_kind = str(row.get("evidence_kind") or "").strip()
                if evidence_kind:
                    kinds.add(evidence_kind)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return set()
    return kinds


def _hierarchy_index(
    session_dirs: Iterable[Path],
    checkpoint_dirs: Iterable[Path],
) -> list[dict[str, Any]]:
    """合并已验证 online 与 checkpoint-bound hierarchy，不从 evidence 猜节点。"""

    rows: list[dict[str, Any]] = []
    for session in session_dirs:
        try:
            hierarchy = json.loads(
                (session / "hierarchy.json").read_text(encoding="utf-8")
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(hierarchy, Mapping):
            continue
        for node in hierarchy.get("nodes", ()):
            if isinstance(node, Mapping):
                rows.append(
                    {
                        "source_kind": "online_session",
                        "session_id": session.name,
                        "node_id": node.get("node_id"),
                        "parent_id": node.get("parent_id"),
                        "hierarchy_level": node.get("hierarchy_level"),
                        "model_name": node.get("model_name"),
                        "module_path": node.get("module_path"),
                        "module_type": node.get("module_type"),
                    }
                )
    for analysis in checkpoint_dirs:
        try:
            cohort = json.loads(
                (analysis / "cohort.json").read_text(encoding="utf-8")
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        metadata = cohort.get("metadata", {}) if isinstance(cohort, Mapping) else {}
        nodes = (
            metadata.get("hierarchy_catalog", ())
            if isinstance(metadata, Mapping)
            else ()
        )
        digest = (
            metadata.get("hierarchy_catalog_digest")
            if isinstance(metadata, Mapping)
            else None
        )
        if (
            not isinstance(nodes, list)
            or not isinstance(digest, str)
            or stable_json_hash(nodes) != digest
        ):
            continue
        for node in nodes:
            if not isinstance(node, Mapping):
                continue
            rows.append(
                {
                    "source_kind": "checkpoint_analysis",
                    "analysis_id": analysis.name,
                    "node_id": node.get("node_id"),
                    "parent_id": node.get("parent_id"),
                    "hierarchy_level": node.get("hierarchy_level"),
                    "model_name": node.get("model_name"),
                    "module_path": node.get("module_path"),
                    "module_type": node.get("module_type"),
                    "discovery_source": node.get("discovery_source"),
                }
            )
    return rows


def _report_status(statuses: Iterable[str]) -> str:
    values = tuple(str(value) for value in statuses)
    if not values:
        return "partial"
    if any(value == "corrupt" for value in values):
        return "corrupt"
    if any(value == "degraded" for value in values):
        return "degraded"
    if all(value in {"complete", "success"} for value in values):
        return "success"
    return "partial"


def _markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Diagnostics v2 报告",
        "",
        f"状态：`{summary['status']}`。",
        "",
        "报告只消费通过 transaction、commit、hash 与 descriptor 校验的 `diagnostics/v2` source。",
        "",
        "每个 analyzer 的解释范围只来自 manifest descriptor 的 claim_boundaries；"
        "统一报告不从指标名称推断物理因果。",
        "",
        "## Analyzer evidence",
        "",
    ]
    found = False
    for analysis in summary.get("checkpoint_analyses", ()):
        for evidence in analysis.get("analyzer_evidence", ()):
            found = True
            boundaries = "；".join(evidence.get("claim_boundaries", ())) or "未声明"
            lines.append(
                f"- `{evidence.get('name')}`：`{evidence.get('evidence_kind')}`，"
                f"{evidence.get('row_count', 0)} rows；边界：{boundaries}。"
            )
    if not found:
        lines.append("- 无完整 analyzer evidence。")
    lines.extend(["", "## Failure bundles", ""])
    failures = list(summary.get("failure_bundles", ()))
    if not failures:
        lines.append("- 无 failure bundle。")
    for failure in failures:
        evidence_scope = (
            "、".join(failure.get("evidence_scope", ())) or "未声明"
        )
        lines.append(
            f"- `{failure.get('bundle_id') or 'unknown'}`："
            f"`{failure.get('status')}`，"
            f"ranks={failure.get('captured_rank_count', 0)}/"
            f"{failure.get('expected_rank_count') or '?'}，"
            f"证据范围：{evidence_scope}；附件未嵌入报告。"
        )
    lines.extend(["", "## Evidence figures", ""])
    figures = list(summary.get("evidence_figures", ()))
    if not figures:
        lines.append("- 无可渲染 evidence。")
    for figure in figures:
        lines.extend(
            [
                f"### {figure.get('title')}",
                "",
                (
                    f"- evidence：`{figure.get('evidence_kind')}`；"
                    f"status：`{figure.get('status')}`；"
                    f"rows：{figure.get('row_count', 0)}；"
                    f"renderer：`{figure.get('renderer_kind')}`。"
                ),
                (
                    "- null control："
                    f"`{figure.get('null_control_status', 'not_reported')}`；"
                    "insufficient evidence rows："
                    f"{figure.get('insufficient_evidence_count', 0)}。"
                ),
            ]
        )
        for boundary in figure.get("claim_boundaries", ()):
            lines.append(f"- claim boundary：{boundary}")
        lines.extend(
            [
                "",
                (
                    f"![{figure.get('title')}]"
                    f"({figure.get('figure_path')})"
                ),
                "",
            ]
        )
    lines.extend(["", "[打开交互报告](../diagnostics-report.html)。", ""])
    return "\n".join(lines)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


__all__ = ["REPORT_FORMAT_VERSION", "generate_diagnostics_report"]
