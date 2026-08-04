"""延迟公开任务无关的 PyTorch diagnostics Base contracts。"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_SYMBOL_OWNERS = {
    "AnalyzerBinding": (".registry", "AnalyzerBinding"),
    "AnalyzerBindingCatalog": (".registry", "AnalyzerBindingCatalog"),
    "AnalyzerCapabilityUnavailable": (
        ".registry",
        "AnalyzerCapabilityUnavailable",
    ),
    "AnalyzerCatalog": (".registry", "AnalyzerCatalog"),
    "AnalyzerDefinition": (".registry", "AnalyzerDefinition"),
    "AnalyzerExecutionResult": (
        ".registry",
        "AnalyzerExecutionResult",
    ),
    "AnalyzerRunContext": (".registry", "AnalyzerRunContext"),
    "BASE_ANALYZER_CATALOG": (".registry", "BASE_ANALYZER_CATALOG"),
    "bind_adapter_capability": (".registry", "bind_adapter_capability"),
    "compose_bindings": (".registry", "compose_bindings"),
    "compose_catalogs": (".registry", "compose_catalogs"),
    "BASE_EVIDENCE_RENDERER_CATALOG": (
        ".reporting",
        "BASE_EVIDENCE_RENDERER_CATALOG",
    ),
    "EvidenceMetric": (".reporting", "EvidenceMetric"),
    "EvidenceRendererCatalog": (
        ".reporting",
        "EvidenceRendererCatalog",
    ),
    "EvidenceRendererDefinition": (
        ".reporting",
        "EvidenceRendererDefinition",
    ),
    "compose_renderer_catalogs": (
        ".reporting",
        "compose_renderer_catalogs",
    ),
    "ArtifactTransaction": (".artifacts.common", "ArtifactTransaction"),
    "CheckpointRef": (".checkpoint.contracts", "CheckpointRef"),
    "CheckpointRuntimeDescriptor": (
        ".checkpoint.contracts",
        "CheckpointRuntimeDescriptor",
    ),
    "DiagnosticComponent": (
        ".checkpoint.contracts",
        "DiagnosticComponent",
    ),
    "SampleRef": (".checkpoint.contracts", "SampleRef"),
    "ModuleSite": (".interventions", "ModuleSite"),
    "TensorSite": (".interventions", "TensorSite"),
    "apply_tensor_intervention": (
        ".interventions",
        "apply_tensor_intervention",
    ),
    "InterventionSpec": (".interventions", "InterventionSpec"),
    "ObjectiveTrace": (".runtime.contracts", "ObjectiveTrace"),
}

__all__ = sorted(_SYMBOL_OWNERS)


def __getattr__(name: str) -> Any:
    """按 owner 加载 DTO，Base cold import 不加载 runtime 或 extensions。"""

    try:
        module_name, attribute_name = _SYMBOL_OWNERS[name]
    except KeyError as error:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from error
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
