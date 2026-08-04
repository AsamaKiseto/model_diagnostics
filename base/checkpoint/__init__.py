"""延迟公开任务无关的 checkpoint contracts 与执行入口。"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_CONTRACT_SYMBOLS = {
    "ANALYSIS_SEMANTICS",
    "AnalyzerBindingCapability",
    "CONDITION_TRANSACTION_CONTRACT",
    "CheckpointDiagnosticsAdapter",
    "CheckpointRef",
    "CheckpointRuntimeDescriptor",
    "ComponentResponse",
    "CohortSelection",
    "ConditionUnavailable",
    "DeviceCapability",
    "DiagnosticComponent",
    "DiagnosticComponentCatalogCapability",
    "DiagnosticsRecipe",
    "DistributedCheckpointCapability",
    "FORMAT_VERSION",
    "LoadedModel",
    "ModulePatchCapability",
    "ModulePatchReference",
    "ModuleSite",
    "ModuleSiteCapability",
    "ObjectiveResult",
    "SampleRef",
    "SamplePayloadCloneCapability",
    "TensorSite",
    "build_activation_record_identity",
    "build_activation_site_identity",
    "canonical_activation_output_path",
}
_SYMBOL_OWNERS = {
    **{name: ".contracts" for name in _CONTRACT_SYMBOLS},
    "ActivationCaptureSample": ".activation_capture",
    "ModuleActivationCapture": ".activation_capture",
    "capture_canonical_model_state": ".branch_state",
    "paired_branch_rng": ".branch_state",
    "restore_canonical_model_state": ".branch_state",
    "run_checkpoint_diagnostics": ".engine",
}

__all__ = sorted(_SYMBOL_OWNERS)


def __getattr__(name: str) -> Any:
    owner = _SYMBOL_OWNERS.get(name)
    if owner is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(owner, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
