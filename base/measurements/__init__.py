"""公开 checkpoint probe、分布统计和最终模型局部测量。"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_PUBLIC_EXPORTS = {
    "DistributionObservation": (
        ".probes",
        "DistributionObservation",
    ),
    "DistributionTap": (".probes", "DistributionTap"),
    "ModuleDiagnosticsProbe": (
        ".probes",
        "ModuleDiagnosticsProbe",
    ),
    "TensorSiteProbe": (".tensor_site", "TensorSiteProbe"),
    "aggregate_tensor_site_records": (
        ".tensor_site",
        "aggregate_tensor_site_records",
    ),
    "tensor_site_identity": (".tensor_site", "tensor_site_identity"),
    "GradientSketch": (".checkpoint", "GradientSketch"),
    "GradientSnapshot": (".checkpoint", "GradientSnapshot"),
    "ParameterSnapshot": (".checkpoint", "ParameterSnapshot"),
    "aggregate_gradients": (".checkpoint", "aggregate_gradients"),
    "build_gradient_sketches": (".checkpoint", "build_gradient_sketches"),
    "compare_gradient_sketches": (
        ".checkpoint",
        "compare_gradient_sketches",
    ),
    "compare_gradients": (".checkpoint", "compare_gradients"),
    "compare_outputs": (".checkpoint", "compare_outputs"),
    "hierarchy_nodes": (".checkpoint", "hierarchy_nodes"),
    "parameter_delta_metrics": (
        ".checkpoint",
        "parameter_delta_metrics",
    ),
    "parameter_name_in_module_path": (
        ".checkpoint",
        "parameter_name_in_module_path",
    ),
    "scalar_terms": (".checkpoint", "scalar_terms"),
    "capture_distribution_observations": (
        ".probes",
        "capture_distribution_observations",
    ),
    "capture_normalization_state": (
        ".probes",
        "capture_normalization_state",
    ),
    "discover_distribution_taps": (
        ".probes",
        "discover_distribution_taps",
    ),
    "finalize_normalization_state": (
        ".probes",
        "finalize_normalization_state",
    ),
    "merge_distribution_summaries": (
        ".probes",
        "merge_distribution_summaries",
    ),
    "summarize_distribution_observations": (
        ".probes",
        "summarize_distribution_observations",
    ),
}

__all__ = sorted(_PUBLIC_EXPORTS)


def __getattr__(name: str) -> Any:
    """按实际 owner 延迟加载 checkpoint 测量实现。"""

    try:
        module_name, attribute_name = _PUBLIC_EXPORTS[name]
    except KeyError as error:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from error
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
