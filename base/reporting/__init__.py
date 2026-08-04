"""公开 validator-first diagnostics/v2 统一报告入口。"""

from .report import REPORT_FORMAT_VERSION, generate_diagnostics_report
from .renderers import (
    BASE_EVIDENCE_RENDERER_CATALOG,
    BASE_EVIDENCE_RENDERER_DEFINITIONS,
    BASE_RUNTIME_METRIC_GUIDES,
    EvidenceMetric,
    EvidenceRendererCatalog,
    EvidenceRendererDefinition,
    build_metric_guide_catalog,
    compose_renderer_catalogs,
    render_evidence_figure,
)

__all__ = [
    "BASE_EVIDENCE_RENDERER_CATALOG",
    "BASE_EVIDENCE_RENDERER_DEFINITIONS",
    "BASE_RUNTIME_METRIC_GUIDES",
    "EvidenceMetric",
    "EvidenceRendererCatalog",
    "EvidenceRendererDefinition",
    "REPORT_FORMAT_VERSION",
    "build_metric_guide_catalog",
    "compose_renderer_catalogs",
    "generate_diagnostics_report",
    "render_evidence_figure",
]
