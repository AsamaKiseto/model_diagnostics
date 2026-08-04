"""
打包 validator-approved artifacts，并生成唯一自包含交互报告。
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any, Iterable, Mapping
import uuid

from .dashboard import render_dashboard_document


STANDALONE_REPORT_FILENAME = "diagnostics-report.html"
_EMBEDDED_CHUNK_MAX_BYTES = 4 * 1024 * 1024
_RAW_CHUNK_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class ReportArtifact:
    """一个允许进入网页报告的显式 v2 source。"""

    name: str
    label: str
    group: str
    path: Path
    artifact_format: str
    content_kind: str = "metadata"
    scope_id: str | None = None
    evidence_kinds: tuple[str, ...] = ()


def _script_safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace(
        "<",
        "\\u003c",
    )


def _collect_field_paths(
    value: Any,
    numeric_paths: set[str],
    categorical_paths: set[str],
    prefix: str = "",
    depth: int = 0,
) -> None:
    """收集可交互查看的 numeric/categorical leaf 路径。"""

    if depth > 6:
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            _collect_field_paths(
                item,
                numeric_paths,
                categorical_paths,
                path,
                depth + 1,
            )
        return
    if isinstance(value, list):
        if prefix and value and all(
            item is None
            or isinstance(item, (int, float))
            and not isinstance(item, bool)
            for item in value
        ):
            numeric_paths.add(prefix)
        elif prefix and value and all(
            item is None or isinstance(item, (str, bool)) for item in value
        ):
            categorical_paths.add(prefix)
        return
    if (
        prefix
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    ):
        numeric_paths.add(prefix)
    elif prefix and isinstance(value, (str, bool)):
        categorical_paths.add(prefix)


def _inspect_artifact(
    path: Path,
    artifact_format: str,
) -> tuple[int, list[str], list[str], list[str]]:
    """流式扫描行数及 leaf 类型，供栏目和完整字段 explorer 使用。"""

    if artifact_format == "json":
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return 0, [], [], []
        numeric_paths: set[str] = set()
        categorical_paths: set[str] = set()
        if isinstance(value, dict):
            _collect_field_paths(
                value,
                numeric_paths,
                categorical_paths,
            )
        return (
            1,
            sorted(value) if isinstance(value, dict) else [],
            sorted(numeric_paths),
            sorted(categorical_paths),
        )
    columns: set[str] = set()
    numeric_paths: set[str] = set()
    categorical_paths: set[str] = set()
    count = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            if artifact_format == "csv":
                header = handle.readline().rstrip("\r\n")
                return (
                    sum(1 for line in handle if line.strip()),
                    header.split(",") if header else [],
                    [],
                    [],
                )
            for line in handle:
                if not line.strip():
                    continue
                count += 1
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    columns.update(str(key) for key in value)
                    _collect_field_paths(
                        value,
                        numeric_paths,
                        categorical_paths,
                    )
    except (OSError, UnicodeDecodeError):
        return 0, [], [], []
    return (
        count,
        sorted(columns),
        sorted(numeric_paths),
        sorted(categorical_paths),
    )


def _compress_file_chunks(
    source: Path,
    staging: Path,
    stem: str,
    *,
    raw_chunk_bytes: int,
    linked_chunk_max_bytes: int,
) -> list[Path]:
    """把源字节切成独立 gzip members，并限制每个内嵌 member。

    文本 evidence 通常有很高压缩率，固定按 2 MiB 原文切分会频繁重置 gzip
    字典。这里先使用较大的原文窗口；只有压缩结果超过单 member 上限时才按
    原始字节递归二分，因此仍保持既有上限和字节顺序。
    """

    chunk_size = max(64 * 1024, raw_chunk_bytes)
    chunks: list[Path] = []
    with source.open("rb") as input_handle:
        index = 0
        while True:
            raw = input_handle.read(chunk_size)
            if not raw and index:
                break
            pending = [raw or b""]
            while pending:
                piece = pending.pop()
                compressed = gzip.compress(piece, compresslevel=9, mtime=0)
                if len(compressed) > linked_chunk_max_bytes:
                    if len(piece) <= 1:
                        raise ValueError(
                            "compressed report chunk exceeds limit: "
                            f"{source} ({len(compressed)} bytes)"
                        )
                    midpoint = len(piece) // 2
                    pending.extend((piece[midpoint:], piece[:midpoint]))
                    continue
                destination = staging / f"{stem}-{index:06d}.gz"
                destination.write_bytes(compressed)
                chunks.append(destination)
                index += 1
            if not raw:
                break
    return chunks


def _compact_metadata_for_embedding(
    source: Path,
    staging: Path,
    stem: str,
) -> Path:
    """建立只供网页使用的可逆验证摘要，不改写权威 manifest。

    成功 evidence key 已逐行存在于 finalized streams，网页再嵌入完整 expected
    与 observed key 列表只会形成第二份高熵事实。报告副本保留 inventory digest、
    全部计数以及 missing/failed/skipped key；原始 manifest 仍留在 analysis 目录，
    validator 也始终在压缩前对它完成严格校验。
    """

    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return source
    if not isinstance(value, dict):
        return source
    inventories = value.get("analyzer_evidence_inventories")
    if not isinstance(inventories, dict):
        return source
    compacted: dict[str, list[dict[str, Any]]] = {}
    changed = False
    for analyzer, raw_items in inventories.items():
        if not isinstance(raw_items, list):
            compacted[str(analyzer)] = []
            continue
        items: list[dict[str, Any]] = []
        for raw_item in raw_items:
            if not isinstance(raw_item, dict):
                continue
            item = dict(raw_item)
            for key in ("expected_evidence_keys", "observed_evidence_keys"):
                if key in item:
                    item.pop(key)
                    changed = True
            items.append(item)
        compacted[str(analyzer)] = items
    if not changed:
        return source
    value["analyzer_evidence_inventories"] = compacted
    value["report_embedding"] = {
        "successful_evidence_keys_compacted": True,
        "authoritative_manifest_sha256": hashlib.sha256(
            source.read_bytes()
        ).hexdigest(),
        "retained_key_sets": [
            "missing_evidence_keys",
            "failed_evidence_keys",
            "skipped_evidence_keys",
        ],
    }
    destination = staging / f"{stem}.report.json"
    destination.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    return destination


def _artifact_descriptor(
    artifact: ReportArtifact,
    root: Path,
) -> dict[str, Any]:
    """建立浏览器所需 source metadata；不读取或复制 task-specific 语义。"""

    available = artifact.path.is_file()
    row_count, columns, numeric_paths, categorical_paths = (
        _inspect_artifact(artifact.path, artifact.artifact_format)
        if available
        else (0, [], [], [])
    )
    return {
        "name": artifact.name,
        "label": artifact.label,
        "group": artifact.group,
        "content_kind": artifact.content_kind,
        "scope_id": artifact.scope_id,
        "evidence_kinds": list(artifact.evidence_kinds),
        "source": str(artifact.path.relative_to(root)),
        "format": artifact.artifact_format,
        "available": available,
        "row_count": row_count,
        "columns": columns,
        "numeric_paths": numeric_paths,
        "categorical_paths": categorical_paths,
        "source_bytes": artifact.path.stat().st_size if available else 0,
        "compressed_bytes": 0,
        "chunks": [],
    }


def write_interactive_report(
    run_dir: str | Path,
    output_dir: str | Path,
    *,
    report_format_version: int,
    summary: dict[str, Any],
    artifacts: Iterable[ReportArtifact] = (),
    hierarchy_index: Iterable[Mapping[str, Any]] = (),
    metric_guides: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """生成按功能分栏、保持 observation identity 的交互报告。"""

    root = Path(run_dir).resolve()
    target = Path(output_dir).resolve()
    standalone_output = (
        root / "diagnostics" / "v2" / STANDALONE_REPORT_FILENAME
    )
    target.mkdir(parents=True, exist_ok=True)
    normalized: list[ReportArtifact] = []
    seen_names: set[str] = set()
    for artifact in artifacts:
        path = Path(artifact.path).resolve()
        try:
            path.relative_to(root / "diagnostics" / "v2")
        except ValueError as error:
            raise ValueError(
                f"report source is outside diagnostics/v2: {path}"
            ) from error
        if artifact.name in seen_names:
            raise ValueError(f"duplicate report artifact name: {artifact.name}")
        seen_names.add(artifact.name)
        normalized.append(
            ReportArtifact(
                name=artifact.name,
                label=artifact.label,
                group=artifact.group,
                path=path,
                artifact_format=artifact.artifact_format,
                content_kind=artifact.content_kind,
                scope_id=artifact.scope_id,
                evidence_kinds=tuple(artifact.evidence_kinds),
            )
        )

    staging = target / f".web-staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=True)
    descriptors: list[dict[str, Any]] = []
    chunk_paths: dict[str, list[Path]] = {}
    try:
        total_compressed_bytes = 0
        for artifact in normalized:
            descriptor = _artifact_descriptor(artifact, root)
            if descriptor["available"]:
                safe_name = re.sub(
                    r"[^a-zA-Z0-9_.-]+",
                    "-",
                    artifact.name,
                ).strip("-")[:96] or "artifact"
                stem = (
                    f"{safe_name}-"
                    f"{hashlib.sha256(str(artifact.path).encode()).hexdigest()[:10]}"
                )
                embedded_source = (
                    _compact_metadata_for_embedding(
                        artifact.path,
                        staging,
                        stem,
                    )
                    if artifact.content_kind == "metadata"
                    and artifact.artifact_format == "json"
                    else artifact.path
                )
                chunks = _compress_file_chunks(
                    embedded_source,
                    staging,
                    stem,
                    raw_chunk_bytes=_RAW_CHUNK_BYTES,
                    linked_chunk_max_bytes=_EMBEDDED_CHUNK_MAX_BYTES,
                )
                chunk_paths[artifact.name] = chunks
                descriptor["embedded_source_bytes"] = (
                    embedded_source.stat().st_size
                )
                descriptor["compressed_bytes"] = sum(
                    path.stat().st_size for path in chunks
                )
                total_compressed_bytes += descriptor["compressed_bytes"]
            descriptors.append(descriptor)

        summary_delivery = {
            "file": str(standalone_output.relative_to(root)),
            "delivery": "self_contained_gzip_embedded",
            "artifact_count": len(descriptors) + 1,
            "automatic_conclusions_included": False,
        }
        summary["interactive_report"] = summary_delivery
        source_compressed_bytes = total_compressed_bytes

        def compress_summary() -> tuple[bytes, list[Path]]:
            embedded_summary = {
                **summary,
                "evidence_figures": [
                    {
                        key: item
                        for key, item in figure.items()
                        if key != "svg"
                    }
                    for figure in summary.get("evidence_figures", ())
                    if isinstance(figure, Mapping)
                ],
            }
            raw = json.dumps(
                embedded_summary,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            source = staging / "report-summary.json"
            source.write_bytes(raw)
            chunks = _compress_file_chunks(
                source,
                staging,
                "report-summary",
                raw_chunk_bytes=_RAW_CHUNK_BYTES,
                linked_chunk_max_bytes=_EMBEDDED_CHUNK_MAX_BYTES,
            )
            source.unlink()
            return raw, chunks

        summary_raw, summary_chunks = compress_summary()
        total_compressed_bytes = source_compressed_bytes + sum(
            path.stat().st_size for path in summary_chunks
        )
        delivery_kind = "self_contained_gzip_embedded"
        if delivery_kind != summary_delivery["delivery"]:
            for path in summary_chunks:
                path.unlink()
            summary_delivery["delivery"] = delivery_kind
            summary_raw, summary_chunks = compress_summary()
            total_compressed_bytes = source_compressed_bytes + sum(
                path.stat().st_size for path in summary_chunks
            )
        chunk_paths["report_summary"] = summary_chunks
        descriptors.append(
            {
                "name": "report_summary",
                "label": "统一报告 summary",
                "group": "report",
                "content_kind": "metadata",
                "scope_id": None,
                "evidence_kinds": [],
                "source": str((target / "summary.json").relative_to(root)),
                "format": "json",
                "available": True,
                "row_count": 1,
                "columns": sorted(summary),
                "numeric_paths": [],
                "categorical_paths": [],
                "source_bytes": len(summary_raw),
                "compressed_bytes": sum(
                    path.stat().st_size for path in summary_chunks
                ),
                "chunks": [],
            }
        )

        standalone_descriptors = [
            {**descriptor, "chunks": []} for descriptor in descriptors
        ]
        payload_elements: list[str] = []
        payload_index = 0
        for descriptor in standalone_descriptors:
            references = []
            for path in (
                chunk_paths.get(str(descriptor["name"]), ())
            ):
                key = f"payload-chunk-{payload_index:06d}"
                encoded = base64.b64encode(path.read_bytes()).decode("ascii")
                payload_elements.append(
                    f'<script id="{key}" '
                    'type="application/octet-stream">'
                    f"{encoded}</script>"
                )
                references.append({"embedded_key": key})
                payload_index += 1
            descriptor["chunks"] = references

        data_directory = target / "data"
        if data_directory.exists():
            shutil.rmtree(data_directory)
        descriptors = standalone_descriptors

        delivery = {
            "file": str(standalone_output.relative_to(root)),
            "delivery": delivery_kind,
            "external_runtime_dependencies": [],
            "artifact_count": len(descriptors),
            "available_artifact_count": sum(
                bool(descriptor["available"]) for descriptor in descriptors
            ),
            "compressed_source_bytes": total_compressed_bytes,
            "embedded_chunk_max_bytes": _EMBEDDED_CHUNK_MAX_BYTES,
            "automatic_conclusions_included": False,
        }
        hierarchy_rows = [dict(row) for row in hierarchy_index]
        guide_rows = [dict(row) for row in metric_guides]
        evidence_figures = [
            {
                key: item
                for key, item in figure.items()
                if key != "svg"
            }
            for figure in summary.get("evidence_figures", ())
            if isinstance(figure, Mapping)
        ]
        executed_analyzers = sorted(
            {
                str(evidence.get("name"))
                for analysis in summary.get("checkpoint_analyses", ())
                for evidence in analysis.get("analyzer_evidence", ())
                if evidence.get("name")
            }
            | {
                str(figure.get("analyzer_name"))
                for figure in evidence_figures
                if figure.get("analyzer_name")
            }
        )

        def bootstrap_for(
            artifact_descriptors: list[dict[str, Any]],
            delivery_name: str,
        ) -> dict[str, Any]:
            return {
                "report_format_version": report_format_version,
                "run_dir": str(root),
                "status": summary.get("status"),
                "delivery": delivery_name,
                "artifacts": artifact_descriptors,
                "hierarchy_index": hierarchy_rows,
                "component_catalogs": summary.get(
                    "component_catalogs",
                    {},
                ),
                "metric_guides": guide_rows,
                "evidence_figures": evidence_figures,
                "executed_analyzers": executed_analyzers,
                "semantics": summary.get("execution_semantics", {}),
                "automatic_conclusions_included": False,
            }

        document = render_dashboard_document(
            report_format_version=report_format_version,
            bootstrap_json=_script_safe_json(
                bootstrap_for(descriptors, delivery_kind)
            ),
            payload_elements="\n".join(payload_elements),
        )
        def write_document(path: Path, content: str) -> None:
            temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
            try:
                with temporary.open("w", encoding="utf-8") as handle:
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
            finally:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass

        write_document(standalone_output, document)
        delivery["standalone_bytes"] = standalone_output.stat().st_size
        delivery["standalone_sha256"] = hashlib.sha256(
            standalone_output.read_bytes()
        ).hexdigest()
        return delivery
    finally:
        shutil.rmtree(staging, ignore_errors=True)


__all__ = [
    "ReportArtifact",
    "STANDALONE_REPORT_FILENAME",
    "write_interactive_report",
]
