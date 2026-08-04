"""公开任务无关的 module/tensor site 与 intervention 机制。"""

from .contracts import ModuleSite, TensorSite, canonical_activation_output_path
from .module import (
    InterventionSpec,
    ModuleOutputCapture,
    ModuleOutputIntervention,
    UnsupportedModuleOutput,
    apply_tensor_intervention,
    capture_module_modes,
    clone_tensor_structure,
    iter_module_output_tensors,
    preserve_module_modes,
    resolve_module,
    restore_module_modes,
    select_module_output_path,
    transform_module_output,
)

__all__ = [
    "InterventionSpec",
    "ModuleOutputCapture",
    "ModuleOutputIntervention",
    "ModuleSite",
    "TensorSite",
    "UnsupportedModuleOutput",
    "apply_tensor_intervention",
    "canonical_activation_output_path",
    "capture_module_modes",
    "clone_tensor_structure",
    "iter_module_output_tensors",
    "preserve_module_modes",
    "resolve_module",
    "restore_module_modes",
    "select_module_output_path",
    "transform_module_output",
]
