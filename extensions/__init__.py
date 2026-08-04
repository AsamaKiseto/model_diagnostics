"""提供可独立导入并显式组合的高级诊断 extension packages。"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from model_diagnostics.base.reporting import EvidenceRendererCatalog


def build_report_renderer_catalog() -> EvidenceRendererCatalog:
    """显式组合 Base 与当前提供 renderer 的 extensions，不修改全局状态。"""

    from model_diagnostics.base.reporting import (
        BASE_EVIDENCE_RENDERER_CATALOG,
        compose_renderer_catalogs,
    )

    from .multi_objective import (
        REPORT_RENDERER_CATALOG as MULTI_OBJECTIVE_RENDERERS,
    )
    from .input_dependence import (
        REPORT_RENDERER_CATALOG as INPUT_DEPENDENCE_RENDERERS,
    )
    from .rollout import REPORT_RENDERER_CATALOG as ROLLOUT_RENDERERS

    return compose_renderer_catalogs(
        BASE_EVIDENCE_RENDERER_CATALOG,
        MULTI_OBJECTIVE_RENDERERS,
        INPUT_DEPENDENCE_RENDERERS,
        ROLLOUT_RENDERERS,
    )


__all__ = ["build_report_renderer_catalog"]
