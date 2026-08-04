"""定义 evidence-kind 驱动的有界统计、显式 renderer catalog 与纯 SVG 图形。"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import html
import math
import re
from types import MappingProxyType
from typing import Any, Iterable, Mapping


_VALID_METRIC_ROLES = frozenset(
    {"primary", "control", "intervention", "context"}
)
_VALID_METRIC_KINDS = frozenset({"scalar", "series"})
_VALID_VISUALIZATION_KINDS = frozenset(
    {
        "timeline",
        "scatter",
        "distribution",
        "heatmap",
        "matrix",
        "series",
        "bar",
    }
)
_VALID_METRIC_CATEGORIES = frozenset(
    {
        "training",
        "module",
        "parameter",
        "input",
        "rollout",
        "multi_objective",
        "runtime",
    }
)
_VALID_METRIC_PRIORITIES = frozenset({"P0", "P1", "P2"})
_GENERIC_EXCLUDED_FIELDS = frozenset(
    {
        "analyzer",
        "analyzer_record_id",
        "checkpoint_update",
        "comparison_index",
        "condition_id",
        "definition_version",
        "physical_causality_claimed",
        "probe_step",
        "rank",
        "recipe_hash",
        "run_id",
        "update",
    }
)


@dataclass(frozen=True, slots=True)
class EvidenceMetric:
    """声明数值路径及其面向人工判断的可视化契约。"""

    path: str
    label: str
    role: str = "primary"
    value_kind: str = "scalar"
    visualization: str = "scatter"
    category: str = "runtime"
    priority: str = "P1"
    description: str = "任务无关的数值诊断字段。"
    reading: str = "固定其它身份维度，比较不同 update、模块或条件下的相对变化。"
    reference: str = "没有跨任务通用的绝对好坏阈值。"
    invalid_when: str = "来源不完整、support 不足或比较条件不一致时不作横向解释。"

    def __post_init__(self) -> None:
        path = str(self.path).strip()
        label = str(self.label).strip()
        role = str(self.role).strip().lower()
        value_kind = str(self.value_kind).strip().lower()
        visualization = str(self.visualization).strip().lower()
        category = str(self.category).strip().lower()
        priority = str(self.priority).strip().upper()
        explanation_fields = {
            "description": str(self.description).strip(),
            "reading": str(self.reading).strip(),
            "reference": str(self.reference).strip(),
            "invalid_when": str(self.invalid_when).strip(),
        }
        if not path or not label:
            raise ValueError("evidence metric path and label must not be empty")
        if role not in _VALID_METRIC_ROLES:
            raise ValueError(
                f"unsupported evidence metric role: {self.role}"
            )
        if value_kind not in _VALID_METRIC_KINDS:
            raise ValueError(
                f"unsupported evidence metric kind: {self.value_kind}"
            )
        if visualization not in _VALID_VISUALIZATION_KINDS:
            raise ValueError(
                f"unsupported metric visualization: {self.visualization}"
            )
        if category not in _VALID_METRIC_CATEGORIES:
            raise ValueError(f"unsupported metric category: {self.category}")
        if priority not in _VALID_METRIC_PRIORITIES:
            raise ValueError(f"unsupported metric priority: {self.priority}")
        if any(not value for value in explanation_fields.values()):
            raise ValueError("metric explanation fields must not be empty")
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "value_kind", value_kind)
        object.__setattr__(self, "visualization", visualization)
        object.__setattr__(self, "category", category)
        object.__setattr__(self, "priority", priority)
        for field, value in explanation_fields.items():
            object.__setattr__(self, field, value)

    def to_guide(self, *, evidence_kind: str | None = None) -> dict[str, Any]:
        """返回浏览器可直接展示、但不产生自动结论的字段说明。"""

        return {
            "path": self.path,
            "label": self.label,
            "role": self.role,
            "value_kind": self.value_kind,
            "visualization": self.visualization,
            "category": self.category,
            "priority": self.priority,
            "description": self.description,
            "reading": self.reading,
            "reference": self.reference,
            "invalid_when": self.invalid_when,
            "evidence_kind": evidence_kind,
        }


@dataclass(frozen=True, slots=True)
class EvidenceRendererDefinition:
    """描述一种 evidence 的展示字段，不绑定 analyzer name 或任务类型。"""

    evidence_kind: str
    title: str
    metrics: tuple[EvidenceMetric, ...] = ()
    null_control_fields: tuple[str, ...] = ()
    status_fields: tuple[str, ...] = ()
    discover_scalar_metrics: bool = False
    definition_version: int = 1
    analyzer_definition_version: int = 1

    def __post_init__(self) -> None:
        evidence_kind = str(self.evidence_kind).strip().lower()
        title = str(self.title).strip()
        if not evidence_kind or not title:
            raise ValueError(
                "renderer evidence_kind and title must not be empty"
            )
        if int(self.definition_version) < 1:
            raise ValueError("renderer definition_version must be positive")
        if (
            isinstance(self.analyzer_definition_version, bool)
            or int(self.analyzer_definition_version) < 1
        ):
            raise ValueError(
                "analyzer_definition_version must be a positive integer"
            )
        metrics = tuple(self.metrics)
        if any(not isinstance(metric, EvidenceMetric) for metric in metrics):
            raise TypeError(
                "renderer metrics must contain EvidenceMetric values"
            )
        metric_paths = [metric.path for metric in metrics]
        if len(set(metric_paths)) != len(metric_paths):
            raise ValueError("renderer metric paths must be unique")
        null_fields = tuple(
            str(field).strip() for field in self.null_control_fields
        )
        status_fields = tuple(str(field).strip() for field in self.status_fields)
        if any(not field for field in (*null_fields, *status_fields)):
            raise ValueError(
                "renderer null/status fields must not contain empty values"
            )
        object.__setattr__(self, "evidence_kind", evidence_kind)
        object.__setattr__(self, "title", title)
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "null_control_fields", null_fields)
        object.__setattr__(self, "status_fields", status_fields)
        object.__setattr__(
            self,
            "analyzer_definition_version",
            int(self.analyzer_definition_version),
        )

    def to_summary(self) -> dict[str, Any]:
        """返回 report summary 可记录的 renderer provenance。"""

        return {
            "evidence_kind": self.evidence_kind,
            "definition_version": int(self.definition_version),
            "analyzer_definition_version": self.analyzer_definition_version,
            "configured_metric_paths": [
                metric.path for metric in self.metrics
            ],
            "metric_guides": [
                metric.to_guide(evidence_kind=self.evidence_kind)
                for metric in self.metrics
            ],
            "generic_metric_discovery": bool(
                self.discover_scalar_metrics
            ),
        }


@dataclass(frozen=True, slots=True)
class EvidenceRendererCatalog:
    """保存 evidence-kind 到 renderer definition 的显式不可变映射。"""

    definitions: Mapping[str, EvidenceRendererDefinition]

    def __post_init__(self) -> None:
        normalized: dict[str, EvidenceRendererDefinition] = {}
        for raw_kind, definition in self.definitions.items():
            if not isinstance(definition, EvidenceRendererDefinition):
                raise TypeError(
                    "renderer catalog values must be "
                    "EvidenceRendererDefinition instances"
                )
            evidence_kind = str(raw_kind).strip().lower()
            if evidence_kind != definition.evidence_kind:
                raise ValueError(
                    "renderer catalog key does not match evidence_kind: "
                    f"{raw_kind}"
                )
            if evidence_kind in normalized:
                raise ValueError(
                    f"duplicate evidence renderer: {evidence_kind}"
                )
            normalized[evidence_kind] = definition
        object.__setattr__(
            self,
            "definitions",
            MappingProxyType(normalized),
        )

    def get(self, evidence_kind: str) -> EvidenceRendererDefinition | None:
        """按 evidence kind 取 renderer；未知类型交给 generic fallback。"""

        return self.definitions.get(
            str(evidence_kind).strip().lower()
        )


def compose_renderer_catalogs(
    *catalogs: (
        EvidenceRendererCatalog
        | Iterable[EvidenceRendererDefinition]
    ),
) -> EvidenceRendererCatalog:
    """显式组合 Base 与 extension renderers，重复 evidence kind 立即失败。"""

    merged: dict[str, EvidenceRendererDefinition] = {}
    for source in catalogs:
        definitions = (
            source.definitions.values()
            if isinstance(source, EvidenceRendererCatalog)
            else source
        )
        for definition in definitions:
            if not isinstance(definition, EvidenceRendererDefinition):
                raise TypeError(
                    "compose_renderer_catalogs inputs must contain "
                    "EvidenceRendererDefinition values"
                )
            if definition.evidence_kind in merged:
                raise ValueError(
                    "evidence renderer already registered: "
                    f"{definition.evidence_kind}"
                )
            merged[definition.evidence_kind] = definition
    return EvidenceRendererCatalog(merged)


class _CategoricalCounter:
    """完整保留 categorical labels 及计数，不折叠或截断。"""

    def __init__(self) -> None:
        self._counts: Counter[str] = Counter()

    def observe(self, value: Any) -> None:
        label = str(value).strip() or "unspecified"
        self._counts[label] += 1

    def get(self, key: str, default: int = 0) -> int:
        return int(self._counts.get(key, default))

    def most_common(self) -> list[tuple[str, int]]:
        return [
            (str(label), int(count))
            for label, count in self._counts.most_common()
        ]

    def as_dict(self) -> dict[str, int]:
        return {
            str(label): int(count)
            for label, count in sorted(self._counts.items())
        }

    def __bool__(self) -> bool:
        return bool(self._counts)


class _EvidenceAccumulator:
    """以充分统计流式汇总 rows，同时保留完整类别与完整 series。"""

    def __init__(self, definition: EvidenceRendererDefinition) -> None:
        self.definition = definition
        self.row_count = 0
        self.status_counts = _CategoricalCounter()
        self.skip_reasons = _CategoricalCounter()
        self.scalar_statistics: dict[str, list[float]] = {}
        self.series_sums: dict[str, list[float]] = {}
        self.series_counts: dict[str, list[int]] = {}
        self.null_controls: dict[str, _CategoricalCounter] = {
            field: _CategoricalCounter()
            for field in definition.null_control_fields
        }
        self.secondary_statuses: dict[str, _CategoricalCounter] = {
            field: _CategoricalCounter()
            for field in definition.status_fields
        }
        self._metrics = {
            metric.path: metric for metric in definition.metrics
        }

    def observe(self, row: Mapping[str, Any]) -> None:
        """吸收一条 row，保留充分统计、完整类别和完整 series。"""

        self.row_count += 1
        raw_status = row.get("status")
        if raw_status is not None:
            self.status_counts.observe(raw_status)
        reason = row.get("skip_reason")
        if reason not in {None, ""}:
            self.skip_reasons.observe(reason)

        for field, counts in self.null_controls.items():
            value = _resolve_path(row, field)
            if value is not None:
                counts.observe(
                    "passed"
                    if value is True
                    else "failed"
                    if value is False
                    else str(value)
                )
        for field, counts in self.secondary_statuses.items():
            value = _resolve_path(row, field)
            if value not in {None, ""}:
                counts.observe(value)

        for metric in self._metrics.values():
            value = _resolve_path(row, metric.path)
            if metric.value_kind == "series":
                self._observe_series(metric.path, value)
            else:
                self._observe_scalar(metric.path, value)
        if self.definition.discover_scalar_metrics:
            for path, value in _numeric_leaf_items(row):
                self._observe_scalar(path, value)

    def _observe_scalar(self, path: str, value: Any) -> None:
        numeric = _finite_number(value)
        if numeric is None:
            return
        statistics = self.scalar_statistics.setdefault(
            path,
            [0.0, 0.0, math.inf, -math.inf],
        )
        statistics[0] += 1.0
        statistics[1] += numeric
        statistics[2] = min(statistics[2], numeric)
        statistics[3] = max(statistics[3], numeric)

    def _observe_series(self, path: str, value: Any) -> None:
        if (
            not isinstance(value, (list, tuple))
            or isinstance(value, (str, bytes))
        ):
            return
        limit = len(value)
        sums = self.series_sums.setdefault(path, [])
        counts = self.series_counts.setdefault(path, [])
        if len(sums) < limit:
            sums.extend([0.0] * (limit - len(sums)))
            counts.extend([0] * (limit - len(counts)))
        for index, raw in enumerate(value[:limit]):
            numeric = _finite_number(raw)
            if numeric is not None:
                sums[index] += numeric
                counts[index] += 1

    def finish(self) -> dict[str, Any]:
        """生成 JSON-safe figure statistics、状态与 null-control 结论。"""

        metric_summaries: list[dict[str, Any]] = []
        for path in sorted(self.scalar_statistics):
            count, total, minimum, maximum = self.scalar_statistics[path]
            configured = self._metrics.get(path)
            metric_summaries.append(
                {
                    **(
                        configured.to_guide()
                        if configured is not None
                        else {
                            "path": path,
                            "label": path,
                            "role": "context",
                            "value_kind": "scalar",
                            "visualization": "scatter",
                            "category": "runtime",
                            "priority": "P2",
                            "description": "自动发现的有限数值字段。",
                            "reading": (
                                "固定其它身份维度，仅比较该字段的相对变化；"
                                "报告未注册专用语义。"
                            ),
                            "reference": "没有注册跨任务通用参考值。",
                            "invalid_when": (
                                "字段语义未知、support 不足或身份维度不一致时"
                                "不作横向解释。"
                            ),
                            "evidence_kind": None,
                        }
                    ),
                    "count": int(count),
                    "mean": total / count,
                    "min": minimum,
                    "max": maximum,
                }
            )
        for path in sorted(self.series_sums):
            configured = self._metrics[path]
            sums = self.series_sums[path]
            counts = self.series_counts[path]
            metric_summaries.append(
                {
                    **configured.to_guide(),
                    "point_count": len(sums),
                    "values": [
                        total / count if count else None
                        for total, count in zip(
                            sums,
                            counts,
                            strict=True,
                        )
                    ],
                    "support": counts,
                }
            )
        null_summary = {
            field: counts.as_dict()
            for field, counts in self.null_controls.items()
            if counts
        }
        status_summary = {
            field: counts.as_dict()
            for field, counts in self.secondary_statuses.items()
            if counts
        }
        status_counts = self.status_counts.as_dict()
        labelled_status_count = sum(status_counts.values())
        return {
            "row_count": self.row_count,
            "status": _figure_status(
                self.row_count,
                status_counts,
                bool(metric_summaries),
            ),
            "status_counts": status_counts,
            "unlabelled_status_row_count": max(
                self.row_count - labelled_status_count,
                0,
            ),
            "insufficient_evidence_count": int(
                self.status_counts.get("insufficient_evidence", 0)
            ),
            "skip_reasons": [
                {"reason": reason, "count": count}
                for reason, count in self.skip_reasons.most_common()
            ],
            "metrics": metric_summaries,
            "null_controls": null_summary,
            "null_control_status": _null_control_status(
                null_summary,
                expected_fields=self.definition.null_control_fields,
            ),
            "secondary_statuses": status_summary,
            "aggregation_policy": {
                "categorical_values": "complete",
                "categorical_labels": "complete",
                "series_points": "complete",
                "scalar_fields": "complete",
            },
        }


def render_evidence_figure(
    *,
    source_id: str,
    analyzer_name: str,
    descriptor: Mapping[str, Any],
    rows: Iterable[Mapping[str, Any]],
    renderer_catalog: EvidenceRendererCatalog,
    figure_id: str,
) -> dict[str, Any]:
    """按 descriptor evidence_kind 选择 renderer，并流式生成结构化 figure。"""

    evidence_kind = str(descriptor.get("evidence_kind") or "").strip().lower()
    if not evidence_kind:
        raise ValueError("evidence descriptor must include evidence_kind")
    raw_analyzer_version = descriptor.get("definition_version")
    analyzer_version = (
        int(raw_analyzer_version)
        if isinstance(raw_analyzer_version, int)
        and not isinstance(raw_analyzer_version, bool)
        and raw_analyzer_version >= 1
        else None
    )
    definition = renderer_catalog.get(evidence_kind)
    renderer_kind = "configured"
    renderer_fallback: dict[str, Any] | None = None
    if (
        definition is not None
        and analyzer_version != definition.analyzer_definition_version
    ):
        raise ValueError(
            "analyzer definition version is not supported by the configured "
            f"renderer: evidence_kind={evidence_kind!r}, "
            f"observed={analyzer_version!r}, "
            f"expected={definition.analyzer_definition_version}"
        )
    if definition is None:
        renderer_kind = "generic"
        renderer_fallback = {
            "reason": "renderer_not_registered",
            "configured_renderer_definition": None,
            "observed_analyzer_definition_version": analyzer_version,
        }
        definition = EvidenceRendererDefinition(
            evidence_kind=evidence_kind,
            title=str(analyzer_name or evidence_kind),
            discover_scalar_metrics=True,
            analyzer_definition_version=analyzer_version or 1,
        )
    accumulator = _EvidenceAccumulator(definition)
    for row in rows:
        if isinstance(row, Mapping):
            accumulator.observe(row)
    statistics = accumulator.finish()
    raw_boundaries = descriptor.get("claim_boundaries", ())
    claim_boundaries = (
        [
            str(value)
            for value in raw_boundaries
            if str(value).strip()
        ]
        if isinstance(raw_boundaries, (list, tuple))
        else []
    )
    figure = {
        "figure_id": str(figure_id),
        "source_id": str(source_id),
        "analyzer_name": str(analyzer_name),
        "evidence_kind": evidence_kind,
        "title": definition.title,
        "renderer_kind": renderer_kind,
        "renderer_definition": definition.to_summary(),
        "renderer_fallback": renderer_fallback,
        "analyzer_definition_version": raw_analyzer_version,
        "claim_boundaries": claim_boundaries,
        **statistics,
    }
    figure["svg"] = _render_svg(figure)
    return figure


def safe_figure_id(value: str) -> str:
    """把 source identity 收口为可预测且不含路径分隔符的 SVG stem。"""

    normalized = re.sub(
        r"[^a-zA-Z0-9_.-]+",
        "-",
        str(value),
    ).strip("-")
    return (normalized[:120] or "evidence-figure").lower()


def _resolve_path(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _numeric_leaf_items(
    value: Mapping[str, Any],
) -> Iterable[tuple[str, float]]:
    """按稳定 key 顺序枚举至多三层 numeric leaves，排除 engine identity。"""

    pending: list[tuple[str, Any, int]] = [
        (str(key), value[key], 1) for key in sorted(value, reverse=True)
    ]
    while pending:
        path, item, depth = pending.pop()
        leaf = path.rsplit(".", 1)[-1]
        if leaf in _GENERIC_EXCLUDED_FIELDS:
            continue
        numeric = _finite_number(item)
        if numeric is not None:
            yield path, numeric
        elif isinstance(item, Mapping) and depth < 3:
            pending.extend(
                (
                    f"{path}.{key}",
                    item[key],
                    depth + 1,
                )
                for key in sorted(item, reverse=True)
            )


def _figure_status(
    row_count: int,
    statuses: Mapping[str, int],
    has_metrics: bool,
) -> str:
    if not row_count:
        return "insufficient_evidence"
    observed = {
        str(status)
        for status, count in statuses.items()
        if int(count) > 0
    }
    labelled_count = sum(max(int(count), 0) for count in statuses.values())
    if labelled_count < row_count:
        observed.add("<unlabelled>")
    if len(observed) > 1:
        return "partial"
    if observed == {"success"}:
        return "success"
    if observed & {"failed", "corrupt"}:
        return "failed"
    if observed == {"insufficient_evidence"}:
        return "insufficient_evidence"
    if observed == {"skipped"}:
        return "skipped"
    if observed:
        return "partial"
    return "partial" if has_metrics else "insufficient_evidence"


def _null_control_status(
    summary: Mapping[str, Mapping[str, int]],
    *,
    expected_fields: Iterable[str],
) -> str:
    if not summary:
        return "not_reported"
    expected = tuple(expected_fields)
    if set(summary) != set(expected):
        return "mixed"
    observed_labels = {
        label
        for counts in summary.values()
        for label, count in counts.items()
        if int(count) > 0
    }
    if observed_labels == {"failed"}:
        return "failed"
    if observed_labels == {"passed"}:
        return "passed"
    return "mixed"


def _render_svg(figure: Mapping[str, Any]) -> str:
    """把有界 metric summary 渲染为无第三方依赖的独立 SVG。"""

    width, height = 960, 540
    chart_left, chart_top = 74, 132
    chart_width, chart_height = 830, 238
    scalar_metrics = [
        metric
        for metric in figure.get("metrics", ())
        if metric.get("value_kind") == "scalar"
    ][:10]
    series_metrics = [
        metric
        for metric in figure.get("metrics", ())
        if metric.get("value_kind") == "series"
        and any(value is not None for value in metric.get("values", ()))
    ][:3]
    elements = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="0 0 {width} {height}" role="img" '
            f'aria-labelledby="title desc">'
        ),
        "<style>"
        ".bg{fill:#fff}.ink{fill:#172033}.muted{fill:#657084}"
        ".grid{stroke:#d9e0e8;stroke-width:1}.axis{stroke:#657084;stroke-width:1.2}"
        ".primary{fill:#087f79;stroke:#087f79}.control{fill:#d99123;stroke:#d99123}"
        ".intervention{fill:#7957b8;stroke:#7957b8}.context{fill:#4d7fb8;stroke:#4d7fb8}"
        ".label{font:12px ui-sans-serif,system-ui,sans-serif}"
        ".small{font:11px ui-sans-serif,system-ui,sans-serif}"
        ".title{font:700 21px ui-sans-serif,system-ui,sans-serif}"
        ".subtitle{font:12px ui-monospace,SFMono-Regular,monospace}"
        "</style>",
        '<rect class="bg" width="960" height="540" rx="16"/>',
        f'<title id="title">{html.escape(str(figure.get("title")))}</title>',
        (
            '<desc id="desc">'
            + html.escape(
                f"{figure.get('evidence_kind')}; "
                f"status={figure.get('status')}; "
                f"rows={figure.get('row_count')}"
            )
            + "</desc>"
        ),
        (
            f'<text x="34" y="42" class="title ink">'
            f"{html.escape(str(figure.get('title')))}</text>"
        ),
        (
            f'<text x="34" y="68" class="subtitle muted">'
            f"{html.escape(str(figure.get('evidence_kind')))}</text>"
        ),
        (
            f'<text x="926" y="42" text-anchor="end" class="label ink">'
            f"status: {html.escape(str(figure.get('status')))}</text>"
        ),
        (
            f'<text x="926" y="65" text-anchor="end" class="small muted">'
            f"rows: {int(figure.get('row_count', 0))} · "
            f"null: {html.escape(str(figure.get('null_control_status')))}</text>"
        ),
    ]
    if series_metrics:
        all_values = [
            float(value)
            for metric in series_metrics
            for value in metric["values"]
            if value is not None
        ]
        lower, upper = min(all_values), max(all_values)
        if math.isclose(lower, upper):
            margin = max(abs(lower) * 0.1, 1.0)
            lower, upper = lower - margin, upper + margin
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = chart_top + chart_height * fraction
            value = upper - (upper - lower) * fraction
            elements.extend(
                [
                    (
                        f'<line x1="{chart_left}" y1="{y:.2f}" '
                        f'x2="{chart_left + chart_width}" y2="{y:.2f}" '
                        'class="grid"/>'
                    ),
                    (
                        f'<text x="{chart_left - 8}" y="{y + 4:.2f}" '
                        f'text-anchor="end" class="small muted">'
                        f"{value:.3g}</text>"
                    ),
                ]
            )
        for metric_index, metric in enumerate(series_metrics):
            values = metric["values"]
            denominator = max(len(values) - 1, 1)
            points = [
                (
                    chart_left + chart_width * index / denominator,
                    chart_top
                    + chart_height
                    * (upper - float(value))
                    / (upper - lower),
                )
                for index, value in enumerate(values)
                if value is not None
            ]
            role = str(metric.get("role") or "primary")
            point_text = " ".join(
                f"{x:.2f},{y:.2f}" for x, y in points
            )
            elements.append(
                f'<polyline points="{point_text}" fill="none" '
                f'class="{html.escape(role)}" stroke-width="2.5"/>'
            )
            elements.append(
                f'<text x="{chart_left + metric_index * 240}" y="405" '
                f'class="small {html.escape(role)}">'
                f"{html.escape(str(metric.get('label')))}</text>"
            )
    elif scalar_metrics:
        values = [float(metric["mean"]) for metric in scalar_metrics]
        lower = min(
            min(float(metric["min"]) for metric in scalar_metrics),
            0.0,
        )
        upper = max(
            max(float(metric["max"]) for metric in scalar_metrics),
            0.0,
        )
        if math.isclose(lower, upper):
            upper = lower + 1.0
        zero_y = chart_top + chart_height * upper / (upper - lower)
        elements.extend(
            [
                (
                    f'<line x1="{chart_left}" y1="{zero_y:.2f}" '
                    f'x2="{chart_left + chart_width}" y2="{zero_y:.2f}" '
                    'class="axis"/>'
                ),
                (
                    f'<line x1="{chart_left}" y1="{chart_top}" '
                    f'x2="{chart_left}" y2="{chart_top + chart_height}" '
                    'class="axis"/>'
                ),
            ]
        )
        slot = chart_width / max(len(scalar_metrics), 1)
        bar_width = max(min(slot * 0.58, 64.0), 8.0)
        for index, (metric, value) in enumerate(
            zip(scalar_metrics, values, strict=True)
        ):
            value_y = (
                chart_top
                + chart_height * (upper - value) / (upper - lower)
            )
            minimum_y = (
                chart_top
                + chart_height
                * (upper - float(metric["min"]))
                / (upper - lower)
            )
            maximum_y = (
                chart_top
                + chart_height
                * (upper - float(metric["max"]))
                / (upper - lower)
            )
            x = chart_left + slot * (index + 0.5) - bar_width / 2
            role = html.escape(str(metric.get("role") or "primary"))
            elements.extend(
                [
                    (
                        f'<line x1="{x + bar_width / 2:.2f}" '
                        f'y1="{maximum_y:.2f}" '
                        f'x2="{x + bar_width / 2:.2f}" '
                        f'y2="{minimum_y:.2f}" class="{role}" '
                        'stroke-width="4" opacity="0.45"/>'
                    ),
                    (
                        f'<circle cx="{x + bar_width / 2:.2f}" '
                        f'cy="{value_y:.2f}" r="5" '
                        f'class="{role}" opacity="0.9"/>'
                    ),
                    (
                        f'<text x="{x + bar_width / 2:.2f}" y="392" '
                        'text-anchor="middle" class="small muted">'
                        f"{html.escape(str(metric.get('label'))[:22])}</text>"
                    ),
                    (
                        f'<text x="{x + bar_width / 2:.2f}" '
                        f'y="{max(maximum_y - 7, 112):.2f}" text-anchor="middle" '
                        'class="small ink">'
                        f"{value:.3g} [{float(metric['min']):.3g}, "
                        f"{float(metric['max']):.3g}]</text>"
                    ),
                ]
            )
    else:
        reason = (
            str(figure.get("skip_reasons", [{}])[0].get("reason"))
            if figure.get("skip_reasons")
            else "no finite configured metrics"
        )
        elements.extend(
            [
                (
                    f'<rect x="{chart_left}" y="{chart_top}" '
                    f'width="{chart_width}" height="{chart_height}" '
                    'rx="10" fill="#edf1f5"/>'
                ),
                (
                    f'<text x="{chart_left + chart_width / 2}" y="245" '
                    'text-anchor="middle" class="label muted">'
                    f"insufficient evidence: {html.escape(reason)}</text>"
                ),
            ]
        )
    statuses = ", ".join(
        f"{key}={value}"
        for key, value in figure.get("status_counts", {}).items()
    ) or "unlabelled rows"
    elements.append(
        f'<text x="34" y="438" class="small muted">'
        f"status counts: {html.escape(statuses)}</text>"
    )
    boundaries = list(figure.get("claim_boundaries", ()))
    if boundaries:
        elements.append(
            f'<text x="34" y="468" class="label ink">'
            f"Claim boundary: {html.escape(str(boundaries[0])[:135])}</text>"
        )
    if len(boundaries) > 1:
        elements.append(
            f'<text x="34" y="493" class="label ink">'
            f"{html.escape(str(boundaries[1])[:150])}</text>"
        )
    elements.append(
        '<text x="34" y="520" class="small muted">'
        "Descriptor-driven summary; no physical-causality inference.</text>"
    )
    elements.append("</svg>")
    return "".join(elements)


BASE_RUNTIME_METRIC_GUIDES = (
    EvidenceMetric(
        "component_objective_value",
        "逐输出训练目标",
        visualization="timeline",
        category="training",
        priority="P0",
        description=(
            "从同一次真实训练 ObjectiveUnit 的 exact ledger 汇总出的逐输出 "
            "objective；每个输出在一个 update 只形成一个趋势点。"
        ),
        reading=(
            "按输出通道分别查看随 update 的趋势、突变和相对收敛速度，并同时核对 "
            "support、partition coverage 与 objective observation coverage。"
        ),
        reference=(
            "只在 objective identity、normalization、support 定义和输出身份一致时"
            "比较；没有跨任务通用阈值。"
        ),
        invalid_when=(
            "status 不是 success、ledger 不能 exact 拆分、microbatch 被回滚、"
            "component catalog 漂移或 unit coverage 不完整。"
        ),
    ),
    EvidenceMetric(
        "raw_numerator_sum",
        "归一化前逐输出目标",
        visualization="timeline",
        category="training",
        priority="P1",
        description=(
            "逐输出目标在 task loss 自身 reduction/weight 完成后、除以 runtime "
            "normalization divisor 前的可加 contribution；不是原始平方误差和。"
        ),
        reading=(
            "用于区分输出目标本身的变化与 accumulation/runtime divisor 变化；"
            "应和逐输出训练目标、normalization divisor、support 一起核对。"
        ),
        reference=(
            "只在 raw numerator semantics、objective identity、support 和 task "
            "reduction 一致时比较。"
        ),
        invalid_when=(
            "raw_numerator_status 不是 available、unit coverage 不完整，或不同 "
            "semantics 被混合；shared terms 仍保持未分摊。"
        ),
    ),
    EvidenceMetric(
        "raw_objective_sum",
        "raw objective",
        visualization="timeline",
        category="training",
        priority="P0",
        description="未应用 backward normalization 前的 objective 汇总。",
        reading="沿 update 查看趋势和突变，并与 backward objective、cap 状态对照。",
        reference="只在 objective identity、reduction 和样本支持一致时比较。",
        invalid_when="objective_trace_status 不完整或不同 objective identity 被混在一起。",
    ),
    EvidenceMetric(
        "backward_objective_sum",
        "backward objective",
        visualization="timeline",
        category="training",
        priority="P0",
        description="实际传入 autograd.backward 的 objective 汇总。",
        reading="与 raw objective、normalization divisor 和 cap hit fraction 联合观察。",
        reference="同一 objective contract 下应与训练实际 backward 路径一致。",
        invalid_when="objective provenance 未物化、microbatch 被回滚或 schedule 不一致。",
    ),
    EvidenceMetric(
        "loss_cap_hit_fraction",
        "loss-cap hit fraction",
        role="context",
        visualization="timeline",
        category="training",
        priority="P0",
        description="本次逻辑 update 中命中 loss cap 的 objective observation 比例。",
        reading="查看是否集中发生在某段训练或特定样本组；不要单独解释模型质量。",
        reference="0 表示未命中；非零只表示训练 objective 被 cap 改写。",
        invalid_when="cap 未配置或 objective observation coverage 不完整。",
    ),
    EvidenceMetric(
        "gradient_norm",
        "全局梯度范数",
        visualization="timeline",
        category="training",
        priority="P0",
        description="训练执行器已计算的全局梯度范数，不额外扫描或反向传播。",
        reading="沿训练步观察数量级突变、持续衰减和失败前后的变化。",
        reference="没有跨模型统一阈值；与本次训练自身历史和裁剪配置比较。",
        invalid_when="梯度尚未 unscale、梯度汇总不完整或该 update 未执行 backward。",
    ),
    EvidenceMetric(
        "rejected_microbatch_count",
        "被拒绝的 microbatch 数",
        role="context",
        visualization="timeline",
        category="training",
        priority="P0",
        description="逻辑 update 中因非有限 objective 等原因被回滚的 microbatch 数。",
        reading="非零点表示该 update 没有完整使用计划内训练信号，应与失败原因对齐。",
        reference="正常完成且未丢弃 microbatch 时参考值为 0。",
        invalid_when="训练执行器未报告 microbatch 终态。",
    ),
)


def build_metric_guide_catalog(
    renderer_catalog: EvidenceRendererCatalog,
) -> list[dict[str, Any]]:
    """组合通用 runtime 与显式 evidence guides，不按 analyzer 名称分发。"""

    guides = [
        metric.to_guide(evidence_kind=None)
        for metric in BASE_RUNTIME_METRIC_GUIDES
    ]
    guides.extend(
        metric.to_guide(evidence_kind=definition.evidence_kind)
        for definition in renderer_catalog.definitions.values()
        for metric in definition.metrics
    )
    return sorted(
        guides,
        key=lambda guide: (
            str(guide["category"]),
            str(guide["priority"]),
            str(guide["path"]),
            str(guide.get("evidence_kind") or ""),
        ),
    )


BASE_EVIDENCE_RENDERER_DEFINITIONS = (
    EvidenceRendererDefinition(
        evidence_kind="checkpoint_training_health",
        title="逐 checkpoint 训练健康",
        analyzer_definition_version=1,
        metrics=(
            EvidenceMetric(
                "raw_objective",
                "检查点原始目标",
                visualization="timeline",
                category="training",
                priority="P0",
                description="固定小样本集合上、未应用 backward normalization 前的目标。",
                reading="按检查点训练步查看总体收敛趋势和突变，并与逐输出指标对照。",
                reference="只比较 objective identity、cohort 和 reduction 完全一致的检查点。",
                invalid_when="raw_objective_finite 为 false 或 objective contract 变化。",
            ),
            EvidenceMetric(
                "backward_objective",
                "检查点反向目标",
                visualization="timeline",
                category="training",
                priority="P0",
                description="异常 sweep 实际用于 backward 的标量目标。",
                reading="与 raw objective 联合查看 normalization 或 cap 是否改变训练信号。",
                reference="只在 objective identity 和 normalization 一致时比较。",
                invalid_when="backward_objective_finite 为 false 或目标不是标量。",
            ),
            EvidenceMetric(
                "response_value",
                "逐输出验证指标",
                visualization="timeline",
                category="training",
                priority="P0",
                description="固定小 cohort 上每个输出通道的验证指标。",
                reading="横轴使用 checkpoint update，每个输出通道一条线，观察持续偏离或突变。",
                reference="只与同一输出、同一 normalization 和同一 cohort 的历史值比较。",
                invalid_when="support、cohort 或 normalization 在 checkpoint 间变化。",
            ),
            EvidenceMetric(
                "output_gradient_rms",
                "逐输出梯度 RMS",
                visualization="timeline",
                category="training",
                priority="P0",
                description="同一次总目标 backward 对各输出预测张量的逐通道梯度 RMS。",
                reading="每个输出通道一条线，查看训练过程中哪些输出的直接监督梯度持续偏大、偏小或突变。",
                reference="不增加 backward 次数；只与同一输出、objective 和 reduction 的历史比较。",
                invalid_when="输出张量未参与该 objective、gradient status 不可用或通道坐标变化。",
            ),
            EvidenceMetric(
                "output_gradient_zero_fraction",
                "逐输出零梯度比例",
                visualization="timeline",
                category="training",
                priority="P0",
                description="各输出预测张量的有限梯度中精确为零的元素比例。",
                reading="关注某个输出是否突然升高或长期接近 1，并与该通道 loss/response 联合查看。",
                reference="不存在跨任务统一阈值；精确 1 表示本次观测的有限元素均无直接梯度。",
                invalid_when="mask、support、loss reduction 或输出坐标不一致。",
            ),
            EvidenceMetric(
                "output_gradient_nonfinite_fraction",
                "逐输出非有限梯度比例",
                visualization="timeline",
                category="training",
                priority="P0",
                description="各输出预测张量梯度中的 NaN/Inf 比例。",
                reading="任何非零点都应与同一 checkpoint 的 objective、参数梯度和分布图对齐查看。",
                reference="有限 backward 的参考值为 0。",
                invalid_when="输出梯度未保留或该输出没有参与 objective。",
            ),
            EvidenceMetric(
                "relative_grad_rms",
                "相对梯度 RMS",
                visualization="timeline",
                category="parameter",
                priority="P0",
                description="参数叶子层梯度 RMS 相对参数 RMS 的比例。",
                reading="每个参数叶子层一条线，关注跨数量级突增、持续接近零和非有限值。",
                reference="没有跨模型统一阈值，应与该层自身历史和相邻层对照。",
                invalid_when="梯度不是 post-backward/pre-clip 或参数 owner 发生变化。",
            ),
            EvidenceMetric(
                "grad_abs_max",
                "参数梯度绝对最大值",
                visualization="timeline",
                category="parameter",
                priority="P0",
                description="参数叶子层有限梯度元素的最大绝对值。",
                reading=(
                    "每个参数叶子层一条线，用于补充 RMS 对孤立梯度尖峰"
                    "不敏感的问题。"
                ),
                reference="没有跨模型统一阈值；重点比较同一层的历史数量级。",
                invalid_when="梯度支持为空、尚未 unscale 或采集发生在梯度清除之后。",
            ),
            EvidenceMetric(
                "gradient_energy_share",
                "梯度能量占比",
                visualization="timeline",
                category="parameter",
                priority="P0",
                description="该参数叶子层有限梯度平方和占模型全部有限梯度平方和的比例。",
                reading="每个参数叶子层一条线，关注能量是否突然集中到少数层或长期消失。",
                reference="同一 checkpoint 各层占比之和约为 1；应与层自身历史比较。",
                invalid_when="参数 owner、有效梯度支持或模型结构在 checkpoint 间变化。",
            ),
            EvidenceMetric(
                "grad_nonfinite_fraction",
                "梯度非有限比例",
                visualization="timeline",
                category="parameter",
                priority="P0",
                description="参数叶子层梯度中 NaN/Inf 的比例。",
                reading="任何非零点都应在时间轴上突出显示，并回查相邻 objective 与分布。",
                reference="健康参考值为 0。",
                invalid_when="该层没有有效梯度或采集发生在梯度清除之后。",
            ),
            EvidenceMetric(
                "grad_zero_fraction",
                "零梯度比例",
                visualization="timeline",
                category="parameter",
                priority="P0",
                description="参数叶子层已观测梯度中精确为零的元素比例。",
                reading="每个参数叶子层一条线，关注突然升高或长期接近 1 的位置。",
                reference="不存在通用理想值；应结合稀疏结构和相对梯度 RMS 判断。",
                invalid_when="该层为 no-grad、梯度支持为空或参数 owner 改变。",
            ),
            EvidenceMetric(
                "no_grad_parameter_tensor_count",
                "无梯度参数张量数",
                role="context",
                visualization="timeline",
                category="parameter",
                priority="P0",
                description="参数叶子层中本次 backward 后 grad 仍为 None 的参数张量数。",
                reading="关注某层是否从 0 突然升高或长期缺少梯度，并结合模型路由解释。",
                reference="对预期始终参与目标的参数层，参考值为 0。",
                invalid_when="该层属于条件路由、冻结参数或本 cohort 没有触发相应分支。",
            ),
            EvidenceMetric(
                "output_rms",
                "激活输出 RMS",
                visualization="timeline",
                category="runtime",
                priority="P0",
                description=(
                    "每个被发现的 Activation/Norm nn.Module 输出分布的 RMS；"
                    "module_path 与 tap_id 共同标识实际层。"
                ),
                reading=(
                    "每个实际激活层或归一化层各一条线，查看随 checkpoint "
                    "持续漂移或突然跃迁。"
                ),
                reference="没有统一阈值；与同一位置的 input RMS、std 和历史基线比较。",
                invalid_when="观测点身份或模型 mode 在 checkpoint 间变化。",
            ),
            EvidenceMetric(
                "output_std",
                "激活输出标准差",
                visualization="timeline",
                category="runtime",
                priority="P0",
                description=(
                    "每个被发现的 Activation/Norm nn.Module 输出的标准差；"
                    "各 module_path/tap_id 独立统计。"
                ),
                reading=(
                    "每个实际激活层或归一化层各一条线，结合 RMS 判断"
                    "整体偏移与离散程度漂移。"
                ),
                reference="没有统一阈值，只比较同一位置的检查点历史。",
                invalid_when="观测点身份、样本集合或模型 mode 变化。",
            ),
            EvidenceMetric(
                "output_abs_max",
                "激活输出绝对最大值",
                visualization="timeline",
                category="runtime",
                priority="P0",
                description=(
                    "每个被发现的 Activation/Norm nn.Module 采样输出的"
                    "绝对最大值，各实际层独立统计。"
                ),
                reading="用于突出 RMS 可能掩盖的孤立尖峰；与同一位置的 RMS 和历史值联合查看。",
                reference="没有跨层统一阈值，持续跨数量级上升比单次轻微波动更值得关注。",
                invalid_when="采样支持、观测点身份或输入 cohort 变化。",
            ),
            EvidenceMetric(
                "output_zero_fraction",
                "激活输出零值比例",
                visualization="timeline",
                category="runtime",
                priority="P0",
                description=(
                    "每个被发现的 Activation/Norm nn.Module 输出中精确"
                    "为零的元素比例，各实际层独立统计。"
                ),
                reading="查看同一观测点是否突然饱和、失活或改变稀疏模式。",
                reference="不存在通用理想值；与激活函数和本层历史比较。",
                invalid_when="padding、mask 或观测支持在检查点间不一致。",
            ),
            EvidenceMetric(
                "output_nonfinite_fraction",
                "激活非有限比例",
                visualization="timeline",
                category="runtime",
                priority="P0",
                description=(
                    "每个被发现的 Activation/Norm nn.Module 输出中 NaN/Inf "
                    "的比例，各实际层独立统计。"
                ),
                reading="非零值应直接突出，并与梯度和逐输出指标的同一 checkpoint 对齐。",
                reference="健康参考值为 0。",
                invalid_when="观测点没有输出 tensor 或样本支持为空。",
            ),
            EvidenceMetric(
                "normalization_running_mean_rms",
                "归一化层运行均值 RMS",
                visualization="timeline",
                category="runtime",
                priority="P0",
                description="具有 running statistics 的归一化层运行均值 RMS。",
                reading="每个归一化层一条线，观察训练后期是否持续漂移或发生突跳。",
                reference="只比较同一层、同一训练/评估 mode 和相同 checkpoint 语义。",
                invalid_when="该归一化层没有 running_mean，或 buffer 未从 checkpoint 恢复。",
            ),
            EvidenceMetric(
                "normalization_running_var_mean",
                "归一化层运行方差均值",
                visualization="timeline",
                category="runtime",
                priority="P0",
                description="具有 running statistics 的归一化层运行方差平均值。",
                reading="观察方差是否持续塌缩、膨胀或在相邻 checkpoint 间突变。",
                reference="没有跨层统一阈值，应与该层自身历史和输出标准差对照。",
                invalid_when="该层没有 running_var，或 buffer provenance 不完整。",
            ),
            EvidenceMetric(
                "elapsed_seconds",
                "单个检查点诊断耗时",
                visualization="timeline",
                category="runtime",
                priority="P0",
                description="单个 checkpoint 的模型加载与健康 sweep 实际墙钟耗时之和。",
                reading="按 checkpoint 查看耗时趋势，并用总和直接核对批量诊断预算。",
                reference="100 个 checkpoint 的总耗时目标不超过 7,200 秒。",
                invalid_when="不同 checkpoint 使用了不同 cohort 大小、设备或执行精度。",
            ),
        ),
        status_fields=("status",),
    ),
    EvidenceRendererDefinition(
        evidence_kind="checkpoint_conditioned_activation_intervention",
        title="最终模型模块影响",
        analyzer_definition_version=2,
        metrics=(
            EvidenceMetric(
                "effect_value",
                "模块对输出的影响",
                role="intervention",
                visualization="matrix",
                category="module",
                priority="P0",
                description=(
                    "记未干预模块时的输出评价指标为 B，干预该模块后的同一指标为"
                    " C。对于越小越好的指标，effect_value = C - B；对于越大越好"
                    "的指标，effect_value = B - C。因此正值表示干预后任务表现"
                    "变差，负值表示改善，0 表示没有测得响应。"
                ),
                reading=(
                    "固定干预方法、样本、输出指标和归一化方式，比较模块对各输出"
                    "的响应方向和幅度；必须与恒等干预对照分开查看。"
                ),
                reference=(
                    "identity 与 output_scale=1 恒等对照应接近 0；实际干预响应"
                    "只有明显超过对照波动时才具有解释价值。"
                ),
                invalid_when=(
                    "component catalog、module site、condition isolation、response "
                    "normalization 或样本配对不完整。"
                ),
            ),
            EvidenceMetric(
                "normalized_effect",
                "模块对输出的相对影响",
                role="context",
                visualization="matrix",
                category="module",
                priority="P1",
                description=(
                    "normalized_effect = effect_value / "
                    "max(abs(B), 1e-12)，B 是同一输出指标的基线值。"
                ),
                reading=(
                    "仅用于在指标语义和 cohort 一致时补充比较不同输出"
                    "量级；默认放在补充证据中。"
                ),
                reference="恒等干预应接近 0；绝对值越大只表示相对基线响应越强。",
                invalid_when="基线接近 0 时比值会被放大；不同指标语义不能直接比较。",
            ),
            EvidenceMetric(
                "objective_normalized_local_taylor",
                "模块局部 Taylor 敏感度",
                role="context",
                visualization="bar",
                category="module",
                priority="P0",
                description=(
                    "设模块输出为 a，当前 checkpoint 的反向目标为 J。该指标为 "
                    "sum(|a × ∂J/∂a|) / max(|J|, 1e-12)，求和范围是已捕获的模块"
                    "输出元素。分子是局部一阶 Taylor 敏感度，分母把它换算成相对"
                    "当前目标函数量级。"
                ),
                reading=(
                    "用于筛查当前参数点附近，哪些模块输出的微小相对变化更可能"
                    "影响目标函数。数值较大只表示局部一阶敏感度较强，必须再与真实"
                    "模块干预的 effect_value 对照，不能直接当作模块贡献。"
                ),
                reference=(
                    "只比较相同 J、相同样本、相同捕获范围和相同模块身份；没有跨"
                    "模型统一阈值。与恒等干预或小幅 output_scale 的真实响应一致时"
                    "才支持局部近似。"
                ),
                invalid_when=(
                    "J 接近 0、梯度饱和、捕获元素不完整或干预幅度较大时会失真。"
                    "它不是 Integrated Gradients、Conductance、activation patching，"
                    "也不表示物理因果。"
                ),
            ),
        ),
        null_control_fields=("identity_control_exact",),
        status_fields=("buffer_restore_status",),
    ),
)

BASE_EVIDENCE_RENDERER_CATALOG = compose_renderer_catalogs(
    BASE_EVIDENCE_RENDERER_DEFINITIONS
)

__all__ = [
    "BASE_EVIDENCE_RENDERER_CATALOG",
    "BASE_EVIDENCE_RENDERER_DEFINITIONS",
    "BASE_RUNTIME_METRIC_GUIDES",
    "EvidenceMetric",
    "EvidenceRendererCatalog",
    "EvidenceRendererDefinition",
    "build_metric_guide_catalog",
    "compose_renderer_catalogs",
    "render_evidence_figure",
    "safe_figure_id",
]
