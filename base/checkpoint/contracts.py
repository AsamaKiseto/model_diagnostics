"""定义任务无关的 checkpoint 分析 recipe、DTO 与宿主协议。"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import torch

from ..artifacts.common import stable_json_hash
from ..artifacts.checkpoint_identity import (
    CHECKPOINT_FORMAT_VERSION,
    normalize_runtime_descriptor,
    runtime_descriptor_payload,
)
from ..interventions.contracts import (
    ModuleSite,
    TensorSite,
    canonical_activation_output_path,
)
from ..registry import AnalyzerBindingCatalog, AnalyzerCatalog


FORMAT_VERSION = CHECKPOINT_FORMAT_VERSION
ANALYSIS_SEMANTICS = "checkpoint_conditioned_analysis"
CONDITION_TRANSACTION_CONTRACT = "explicit_analyzer_terminal_v2"
ACTIVATION_SITE_IDENTITY_VERSION = 1

def _freeze_json_value(value: Any) -> Any:
    """递归冻结已验证 JSON 值，防止 digest 建立后 descriptor 被原地修改。"""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                str(key): _freeze_json_value(item)
                for key, item in value.items()
            }
        )
    if isinstance(value, list):
        return tuple(_freeze_json_value(item) for item in value)
    return value


def _thaw_json_value(value: Any) -> Any:
    """把内部只读 JSON tree 还原为可序列化副本。"""

    if isinstance(value, Mapping):
        return {
            str(key): _thaw_json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return [_thaw_json_value(item) for item in value]
    return value


class ConditionUnavailable(RuntimeError):
    """表示 analyzer 缺少可靠输入，应记录 structured skip。"""


@dataclass(frozen=True, slots=True)
class DiagnosticComponent:
    """任务中立的输入或输出 component 身份及 tensor 坐标。"""

    component_id: str
    semantic_id: str
    component_kind: str
    display_label: str
    tensor_path: str
    tensor_axis: int
    tensor_index: int
    component_group: str | None = None
    component_index: int | None = None
    normalization_id: str | None = None
    component_shape: tuple[int, ...] | None = None
    reduction_semantics: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        required = {
            "component_id": self.component_id,
            "semantic_id": self.semantic_id,
            "display_label": self.display_label,
            "tensor_path": self.tensor_path,
        }
        empty = sorted(
            name for name, value in required.items() if not str(value).strip()
        )
        if empty:
            raise ValueError(
                f"DiagnosticComponent fields must not be empty: {empty}"
            )
        kind = str(self.component_kind).strip().lower()
        if kind not in {"input", "output"}:
            raise ValueError(
                "DiagnosticComponent.component_kind must be input or output"
            )
        if kind == "output" and not str(
            self.normalization_id or ""
        ).strip():
            raise ValueError(
                "output DiagnosticComponent requires normalization_id"
            )
        coordinates = {
            "tensor_axis": self.tensor_axis,
            "tensor_index": self.tensor_index,
        }
        if self.component_index is not None:
            coordinates["component_index"] = self.component_index
        if any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in coordinates.values()
        ):
            raise TypeError(
                "DiagnosticComponent indices must be integers"
            )
        component_shape = (
            None
            if self.component_shape is None
            else tuple(self.component_shape)
        )
        if component_shape is not None and any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            for value in component_shape
        ):
            raise ValueError(
                "DiagnosticComponent.component_shape must contain "
                "non-negative integers"
            )
        reduction_semantics = (
            None
            if self.reduction_semantics is None
            else str(self.reduction_semantics).strip()
        )
        if self.reduction_semantics is not None and not reduction_semantics:
            raise ValueError(
                "DiagnosticComponent.reduction_semantics must be non-empty "
                "when provided"
            )
        metadata = dict(self.metadata)
        try:
            json.dumps(metadata, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "DiagnosticComponent.metadata must be finite JSON"
            ) from error
        object.__setattr__(self, "component_id", str(self.component_id).strip())
        object.__setattr__(self, "semantic_id", str(self.semantic_id).strip())
        object.__setattr__(self, "component_kind", kind)
        object.__setattr__(self, "display_label", str(self.display_label).strip())
        object.__setattr__(self, "tensor_path", str(self.tensor_path).strip())
        object.__setattr__(
            self,
            "component_group",
            (
                str(self.component_group).strip()
                if self.component_group is not None
                else None
            ),
        )
        object.__setattr__(
            self,
            "normalization_id",
            (
                str(self.normalization_id).strip()
                if self.normalization_id is not None
                else None
            ),
        )
        object.__setattr__(self, "component_shape", component_shape)
        object.__setattr__(
            self,
            "reduction_semantics",
            reduction_semantics,
        )
        object.__setattr__(self, "metadata", MappingProxyType(metadata))

    def to_dict(self) -> dict[str, Any]:
        """返回 artifact component catalog 使用的唯一 schema。"""

        return {
            "component_id": self.component_id,
            "semantic_id": self.semantic_id,
            "component_kind": self.component_kind,
            "display_label": self.display_label,
            "component_group": self.component_group,
            "component_index": self.component_index,
            "tensor_path": self.tensor_path,
            "tensor_axis": self.tensor_axis,
            "tensor_index": self.tensor_index,
            "normalization_id": self.normalization_id,
            "component_shape": (
                list(self.component_shape)
                if self.component_shape is not None
                else None
            ),
            "reduction_semantics": self.reduction_semantics,
            "metadata": dict(self.metadata),
        }


def build_activation_site_identity(
    *,
    node_id: str,
    stage_id: str | None,
    parent_node_id: str | None,
    invocation_index: int | None,
    output_path: str | None,
    alias_node_ids: Sequence[str] = (),
    output_path_is_aggregate_root: bool = False,
) -> dict[str, Any]:
    """构造跨 probe 与 intervention 可连接的 activation site identity。"""

    observed_node_id = str(node_id)
    aliases = tuple(
        sorted(
            {
                observed_node_id,
                *(str(alias) for alias in alias_node_ids if str(alias)),
            }
        )
    )
    normalized_path = canonical_activation_output_path(output_path)
    if normalized_path is None and output_path_is_aggregate_root:
        normalized_path = "$"
    ambiguity: list[str] = []
    normalized_invocation: int | None
    if invocation_index is None:
        normalized_invocation = None
        ambiguity.append("invocation_index")
    else:
        normalized_invocation = int(invocation_index)
        if normalized_invocation < 0:
            raise ValueError("activation invocation_index must be non-negative")
    if normalized_path is None:
        ambiguity.append("output_path")
    payload = {
        "identity_version": ACTIVATION_SITE_IDENTITY_VERSION,
        "observed_node_id": observed_node_id,
        "stage_id": None if stage_id is None else str(stage_id),
        "parent_node_id": None if parent_node_id is None else str(parent_node_id),
        "invocation_index": normalized_invocation,
        "output_path": normalized_path,
        "alias_node_ids": list(aliases),
    }
    return {
        **payload,
        "activation_site_id": stable_json_hash(payload),
        "activation_site_status": (
            "exact"
            if not ambiguity
            else "ambiguous_" + "_and_".join(ambiguity)
        ),
        "activation_site_ambiguity_fields": ambiguity,
    }


def build_activation_record_identity(
    *,
    observed_site: Mapping[str, Any],
    condition_target_site: Mapping[str, Any] | None = None,
    record_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """把 observed site、condition site 和 observation context 绑定为记录身份。"""

    site = dict(observed_site)
    target = dict(condition_target_site or {})
    target_site_id = target.get("activation_site_id")
    record_payload = {
        "observed_activation_site_id": site.get("activation_site_id"),
        "condition_target_activation_site_id": target_site_id,
        "context": dict(record_context or {}),
    }
    return {
        **site,
        "condition_target_node_id": target.get("observed_node_id"),
        "condition_target_activation_site_id": target_site_id,
        "condition_target_activation_site_status": target.get(
            "activation_site_status"
        ),
        "condition_target_stage_id": target.get("stage_id"),
        "condition_target_parent_node_id": target.get("parent_node_id"),
        "condition_target_invocation_index": target.get("invocation_index"),
        "condition_target_output_path": target.get("output_path"),
        "condition_target_alias_node_ids": target.get("alias_node_ids"),
        "activation_record_id": stable_json_hash(record_payload),
    }


@dataclass(frozen=True, slots=True)
class CheckpointRef:
    """描述由宿主解析的 checkpoint 身份，不假设目录或文件命名。"""

    path: str
    identity: str
    update: int | None = None
    kind: str = "checkpoint"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.path).strip() or not str(self.identity).strip():
            raise ValueError("CheckpointRef path and identity must not be empty")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class CheckpointRuntimeDescriptor:
    """绑定 task、objective executor 与可选 capability 实现的运行时身份。"""

    runtime_descriptor_digest: str
    task_definition_id: str
    task_definition_version: str
    objective_executor_version: str
    capability_descriptors: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        normalized = normalize_runtime_descriptor(self.to_dict())
        object.__setattr__(
            self,
            "runtime_descriptor_digest",
            normalized["runtime_descriptor_digest"],
        )
        object.__setattr__(
            self,
            "task_definition_id",
            normalized["task_definition_id"],
        )
        object.__setattr__(
            self,
            "task_definition_version",
            normalized["task_definition_version"],
        )
        object.__setattr__(
            self,
            "objective_executor_version",
            normalized["objective_executor_version"],
        )
        object.__setattr__(
            self,
            "capability_descriptors",
            tuple(
                _freeze_json_value(descriptor)
                for descriptor in normalized["capability_descriptors"]
            ),
        )

    @classmethod
    def create(
        cls,
        *,
        task_definition_id: str,
        task_definition_version: str,
        objective_executor_version: str,
        capability_descriptors: Sequence[Mapping[str, Any]] = (),
    ) -> "CheckpointRuntimeDescriptor":
        """由完整 canonical payload 派生 digest，避免调用方维护第二套哈希公式。"""

        payload = runtime_descriptor_payload(
            task_definition_id=task_definition_id,
            task_definition_version=task_definition_version,
            objective_executor_version=objective_executor_version,
            capability_descriptors=capability_descriptors,
        )
        return cls(
            runtime_descriptor_digest=stable_json_hash(payload),
            task_definition_id=payload["task_definition_id"],
            task_definition_version=payload["task_definition_version"],
            objective_executor_version=payload["objective_executor_version"],
            capability_descriptors=tuple(payload["capability_descriptors"]),
        )

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "CheckpointRuntimeDescriptor":
        """严格解析调用方 mapping；缺字段、未知字段或 digest 漂移立即失败。"""

        normalized = normalize_runtime_descriptor(value)
        return cls(
            runtime_descriptor_digest=normalized["runtime_descriptor_digest"],
            task_definition_id=normalized["task_definition_id"],
            task_definition_version=normalized["task_definition_version"],
            objective_executor_version=normalized["objective_executor_version"],
            capability_descriptors=tuple(
                normalized["capability_descriptors"]
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """返回 analysis identity、run state 与 manifest 共用的唯一字段集合。"""

        return {
            "runtime_descriptor_digest": self.runtime_descriptor_digest,
            "task_definition_id": self.task_definition_id,
            "task_definition_version": self.task_definition_version,
            "objective_executor_version": self.objective_executor_version,
            "capability_descriptors": [
                _thaw_json_value(descriptor)
                for descriptor in self.capability_descriptors
            ],
        }


@dataclass(frozen=True, slots=True)
class SampleRef:
    """描述可由宿主重新物化的稳定样本身份。"""

    sample_id: str
    partition: str
    sample_index: int | None = None
    group_id: str | None = None
    position: int | float | str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.sample_id).strip() or not str(self.partition).strip():
            raise ValueError("SampleRef sample_id and partition must not be empty")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_dict(self) -> dict[str, Any]:
        """返回不含 tensor 的 JSON-safe cohort 记录。"""

        return {
            "sample_id": self.sample_id,
            "partition": self.partition,
            "sample_index": self.sample_index,
            "group_id": self.group_id,
            "position": self.position,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class LoadedModel:
    """宿主加载 model-only checkpoint 后交给 Base 的运行时对象。"""

    model: torch.nn.Module
    checkpoint_update: int | None
    model_name: str
    structure_metadata: Mapping[str, Any] = field(default_factory=dict)
    ignored_payload_keys: tuple[str, ...] = ()
    adapter_context: Any = None


@dataclass(slots=True)
class ComponentResponse:
    """描述同一次 forward 中一个任务中立输出组件的标量响应。

    `response_component_id` 只引用 analysis 级 component catalog；显示名称、组别与
    tensor 坐标不在每条 response 中重复。该对象既可承载误差，也可承载任务声明的
    其它标量 score，因此数值方向必须由 `higher_is_better` 显式给出。
    """

    response_component_id: str
    metric_id: str
    value: torch.Tensor
    support_count: int | float
    normalization_id: str
    higher_is_better: bool
    gradient_tensor: torch.Tensor | None = field(
        default=None,
        repr=False,
    )
    gradient_axis: int | None = None
    gradient_index: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        component_id = str(self.response_component_id).strip()
        metric_id = str(self.metric_id).strip()
        normalization_id = str(self.normalization_id).strip()
        if not component_id or not metric_id or not normalization_id:
            raise ValueError(
                "component response identity fields must not be empty"
            )
        if not isinstance(self.value, torch.Tensor) or self.value.numel() != 1:
            raise ValueError("component response value must be a scalar Tensor")
        if self.gradient_tensor is not None and not isinstance(
            self.gradient_tensor,
            torch.Tensor,
        ):
            raise TypeError("gradient_tensor must be a torch.Tensor")
        if (self.gradient_axis is None) != (self.gradient_index is None):
            raise ValueError(
                "gradient_axis and gradient_index must be provided together"
            )
        support = float(self.support_count)
        if not math.isfinite(support) or support <= 0.0:
            raise ValueError(
                "component response support_count must be finite and positive"
            )
        object.__setattr__(self, "response_component_id", component_id)
        object.__setattr__(self, "metric_id", metric_id)
        object.__setattr__(self, "normalization_id", normalization_id)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(slots=True)
class ObjectiveResult:
    """一次 forward 的实际诊断 objective、分项、输出与 provenance。"""

    total: torch.Tensor
    objective_identity: str
    terms: Mapping[str, torch.Tensor] = field(default_factory=dict)
    outputs: Mapping[str, torch.Tensor] = field(default_factory=dict)
    raw_total: torch.Tensor | None = None
    backward_total: torch.Tensor | None = None
    backward_cotangent: torch.Tensor | None = None
    cotangent_identity: str | None = None
    component_responses: tuple[ComponentResponse, ...] = ()
    aux: Mapping[str, Any] = field(default_factory=dict)
    objective_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        identity = str(self.objective_identity).strip()
        if not identity or identity.lower() == "unspecified":
            raise ValueError(
                "ObjectiveResult.objective_identity must identify the actual "
                "backward objective"
            )
        self.objective_identity = identity
        objective = (
            self.backward_total
            if self.backward_total is not None
            else self.total
        )
        if not isinstance(objective, torch.Tensor) or objective.numel() < 1:
            raise ValueError("backward objective must be a non-empty Tensor")
        cotangent = self.backward_cotangent
        cotangent_identity = (
            None
            if self.cotangent_identity is None
            else str(self.cotangent_identity).strip()
        )
        if cotangent is None:
            if cotangent_identity:
                raise ValueError(
                    "cotangent_identity requires backward_cotangent"
                )
        else:
            if not cotangent_identity:
                raise ValueError(
                    "cotangent_identity must be non-empty when cotangent is provided"
                )
            if cotangent.shape != objective.shape:
                raise ValueError(
                    "backward_cotangent shape must exactly match objective"
                )
            if (
                cotangent.device != objective.device
                or cotangent.dtype != objective.dtype
            ):
                raise ValueError(
                    "backward_cotangent must match objective device and dtype"
                )
            if cotangent.requires_grad or not bool(
                torch.isfinite(cotangent).all().item()
            ):
                raise ValueError(
                    "backward_cotangent must be detached and finite"
                )
        self.cotangent_identity = cotangent_identity
        responses = tuple(self.component_responses)
        if any(
            not isinstance(response, ComponentResponse)
            for response in responses
        ):
            raise TypeError(
                "component_responses must contain ComponentResponse values"
            )
        response_keys = [
            (response.response_component_id, response.metric_id)
            for response in responses
        ]
        if len(response_keys) != len(set(response_keys)):
            raise ValueError(
                "component_responses must be unique by component and metric"
            )
        self.component_responses = responses

    def effective_backward_objective(self) -> torch.Tensor:
        """返回 checkpoint branch 应执行的 scalar/VJP 等价目标。"""

        objective = (
            self.backward_total
            if self.backward_total is not None
            else self.total
        )
        if self.backward_cotangent is None:
            if objective.numel() != 1:
                raise ConditionUnavailable(
                    "non_scalar_backward_objective_requires_explicit_cotangent"
                )
            return objective.reshape(())
        return torch.sum(
            objective * self.backward_cotangent.detach()
        ).reshape(())


@dataclass(frozen=True, slots=True)
class CohortSelection:
    """adapter 返回的 cohort 及其来源说明。"""

    status: str
    samples: tuple[SampleRef, ...] = ()
    skip_reason: str | None = None
    identity: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ModulePatchReference:
    """显式 module patch 值及其有界来源说明。"""

    replacement: Any
    provenance: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, init=False)
class DiagnosticsRecipe:
    """Base 内部生成的两阶段 checkpoint 执行计划。

    执行精度、objective provenance、buffer freeze、paired RNG、resume 与资源分批
    都由 runtime/engine 固定拥有。调用方只能选择 sweep 或 final checkpoint，
    不能覆盖 analyzer、cohort、checkpoint 集合或算法参数。
    """

    stage: str
    analyzers: tuple[str, ...]
    final_checkpoint: str | None

    @classmethod
    def checkpoint_sweep(
        cls,
        *,
        analyzer_catalog: AnalyzerCatalog,
    ) -> "DiagnosticsRecipe":
        """建立覆盖全部 trajectory checkpoint 的固定训练健康扫描。"""

        return cls._create(
            stage="checkpoint_sweep",
            analyzers=("checkpoint_sweep",),
            final_checkpoint=None,
            analyzer_catalog=analyzer_catalog,
        )

    @classmethod
    def final_selected(
        cls,
        checkpoint: str | Path,
        *,
        analyzer_catalog: AnalyzerCatalog,
    ) -> "DiagnosticsRecipe":
        """建立对一个显式最终 checkpoint 的完整固定分析。"""

        normalized = str(Path(checkpoint).resolve())
        return cls._create(
            stage="final_selected",
            analyzers=tuple(
                name
                for name, definition in analyzer_catalog.definitions.items()
                if name != "checkpoint_sweep"
                and definition.execution_mode in {"condition", "aggregate"}
            ),
            final_checkpoint=normalized,
            analyzer_catalog=analyzer_catalog,
        )

    @classmethod
    def _create(
        cls,
        *,
        stage: str,
        analyzers: Sequence[str],
        final_checkpoint: str | None,
        analyzer_catalog: AnalyzerCatalog,
    ) -> "DiagnosticsRecipe":
        normalized = tuple(str(name).strip().lower() for name in analyzers)
        if not normalized:
            raise ValueError(f"{stage} selected no compatible analyzers")
        analyzer_catalog.validate(normalized, {})
        instance = object.__new__(cls)
        object.__setattr__(instance, "stage", stage)
        object.__setattr__(instance, "analyzers", normalized)
        object.__setattr__(instance, "final_checkpoint", final_checkpoint)
        return instance

    def to_dict(self) -> dict[str, Any]:
        """返回用于 identity 与 manifest 的 canonical mapping。"""

        return {
            "stage": self.stage,
            "analyzers": list(self.analyzers),
            "final_checkpoint": self.final_checkpoint,
        }


@runtime_checkable
class CheckpointDiagnosticsAdapter(Protocol):
    """宿主必须实现的最小 checkpoint、sample 与 objective 执行边界。

    payload clone、module-site discovery、distributed barrier、module replacement
    和 aggregate analyzer runner 都是可选 capability；Base 对任务无关情形提供默认
    行为，缺少任务语义时 structured skip。
    """

    analyzer_catalog: AnalyzerCatalog
    execution_precision: str

    def configure(self, recipe: DiagnosticsRecipe) -> None:
        """接收 Base 已严格解析的唯一 effective recipe。"""

    def resolve_checkpoints(
        self,
        *,
        run_dir: Path,
    ) -> Sequence[CheckpointRef]:
        """按已配置阶段和宿主布局解析 checkpoint，不允许 Base 猜测文件名。"""

    def load_model(self, checkpoint_path: str, *, precision: str) -> LoadedModel:
        """只恢复 canonical model state 与 registered buffers。"""

    def build_cohort(
        self,
        loaded: LoadedModel,
        *,
        checkpoints: Sequence[CheckpointRef],
    ) -> CohortSelection:
        """按宿主固定统计策略构造跨 checkpoint 稳定 cohort。"""

    def materialize_sample(self, loaded: LoadedModel, sample: SampleRef) -> Any:
        """把稳定 sample identity 物化为设备 payload。"""

    def forward_objective(
        self,
        loaded: LoadedModel,
        payload: Any,
        *,
        objective: Mapping[str, Any],
        precision: str,
        return_aux: bool,
    ) -> ObjectiveResult:
        """执行宿主 forward 并返回明确的诊断 objective。"""

    def close_loaded_model(self, loaded: LoadedModel) -> None:
        """释放 checkpoint-local 宿主资源。"""


@runtime_checkable
class SamplePayloadCloneCapability(Protocol):
    """允许宿主为自定义 payload 提供 storage-isolated clone。"""

    def clone_sample_payload(self, payload: Any) -> Any:
        """返回不共享可变 tensor storage 的 condition payload。"""


@runtime_checkable
class ModuleSiteCapability(Protocol):
    """允许宿主用已验证 hierarchy artifact 覆盖通用结构发现。"""

    def list_module_sites(self, loaded: LoadedModel) -> Sequence[ModuleSite]:
        """列出可以执行 output intervention 的 module sites。"""


@runtime_checkable
class DiagnosticComponentCatalogCapability(Protocol):
    """允许宿主把任务 component 映射为分析级稳定身份。"""

    def list_diagnostic_components(
        self,
        loaded: LoadedModel,
        samples: Sequence[SampleRef],
    ) -> Sequence[DiagnosticComponent]:
        """列出本 analysis 使用的全部输入与输出 component。"""


@runtime_checkable
class ModulePatchCapability(Protocol):
    """允许宿主物化 task-owned mean 或 explicit donor replacement。"""

    def module_patch_reference(
        self,
        loaded: LoadedModel,
        *,
        site: ModuleSite,
        method: str,
        sample: SampleRef,
        payload: Any,
    ) -> ModulePatchReference:
        """为 mean/explicit donor patch 提供 task-owned replacement。"""


@runtime_checkable
class AnalyzerBindingCapability(Protocol):
    """允许 adapter 显式提供不依赖名称分发表的 analyzer bindings。"""

    analyzer_bindings: AnalyzerBindingCatalog


@runtime_checkable
class DistributedCheckpointCapability(Protocol):
    """为多进程 condition sharding 提供显式 rank 与 barrier。"""

    rank: int
    world_size: int

    def barrier(self) -> None:
        """多进程 condition sharding 的同步边界；单进程为 no-op。"""


@runtime_checkable
class DeviceCapability(Protocol):
    """允许宿主声明 paired-RNG 与资源回收使用的执行设备。"""

    device: torch.device


__all__ = [
    "ACTIVATION_SITE_IDENTITY_VERSION",
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
]
