"""定义任务中立的宿主运行时 DTO 与 provider 协议。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import torch
from torch import nn

if TYPE_CHECKING:
    from .composition import ExecutionSession
    from .execution import TrainingAttemptRequest, TrainingAttemptResult


def _nonempty(value: str, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty")
    return normalized


def _frozen_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(value))


@dataclass(frozen=True)
class RunRef:
    """标识一次宿主 run；`locator` 的解释完全归宿主 provider。"""

    run_id: str
    locator: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _nonempty(self.run_id, "run_id"))
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))


@dataclass(frozen=True)
class ModelSpec:
    """描述构造模型所需的任务中立 identity 与不可解释配置。"""

    model_id: str
    config: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", _nonempty(self.model_id, "model_id"))
        object.__setattr__(self, "config", _frozen_mapping(self.config))
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))


@dataclass(frozen=True)
class RuntimeCheckpointRef:
    """标识 checkpoint；不规定目录、文件名或序列化格式。"""

    checkpoint_id: str
    locator: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "checkpoint_id",
            _nonempty(self.checkpoint_id, "checkpoint_id"),
        )
        object.__setattr__(self, "locator", _nonempty(self.locator, "locator"))
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))


@dataclass(frozen=True)
class CheckpointTrainingProvenance:
    """描述 checkpoint 权重对应的训练位置，不规定宿主序列化格式。"""

    definition_version: int
    local_completed_update: int | None = None
    objective_schedule_update: int | float | str | None = None
    next_objective_schedule_update: int | float | str | None = None
    training_updates: int | float | None = None
    weight_boundary: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            isinstance(self.definition_version, bool)
            or not isinstance(self.definition_version, int)
            or self.definition_version <= 0
        ):
            raise ValueError("definition_version must be a positive integer")
        for field_name in ("local_completed_update",):
            value = getattr(self, field_name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(
                    f"{field_name} must be a non-negative integer when provided"
                )
        if self.training_updates is not None and (
            isinstance(self.training_updates, bool)
            or not isinstance(self.training_updates, (int, float))
            or not math.isfinite(float(self.training_updates))
            or float(self.training_updates) <= 0
        ):
            raise ValueError(
                "training_updates must be finite and positive when provided"
            )
        if (
            self.local_completed_update is not None
            and self.training_updates is not None
            and float(self.local_completed_update) > float(self.training_updates)
        ):
            raise ValueError(
                "local_completed_update must not exceed training_updates"
            )
        boundary = (
            None
            if self.weight_boundary is None
            else str(self.weight_boundary).strip()
        )
        if self.weight_boundary is not None and not boundary:
            raise ValueError("weight_boundary must not be empty when provided")
        object.__setattr__(self, "weight_boundary", boundary)
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))


@dataclass(frozen=True)
class CheckpointObjectiveSpec:
    """定义 checkpoint/eval 分支复用真实 objective 时的默认上下文。"""

    objective_id: str
    precision: str = "fp32"
    training: bool = False
    normalization_divisor: float = 1.0
    loss_cap: float | None = None
    amp_scale: float = 1.0
    schedule_total: int | float | None = None
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "objective_id",
            _nonempty(self.objective_id, "objective_id"),
        )
        object.__setattr__(
            self,
            "precision",
            _nonempty(self.precision, "precision").lower(),
        )
        if (
            not math.isfinite(float(self.normalization_divisor))
            or float(self.normalization_divisor) <= 0
        ):
            raise ValueError("normalization_divisor must be finite and positive")
        if self.loss_cap is not None and (
            not math.isfinite(float(self.loss_cap))
            or float(self.loss_cap) <= 0
        ):
            raise ValueError("loss_cap must be finite and positive when provided")
        if not math.isfinite(float(self.amp_scale)) or float(self.amp_scale) <= 0:
            raise ValueError("amp_scale must be finite and positive")
        if self.schedule_total is not None and (
            isinstance(self.schedule_total, bool)
            or not isinstance(self.schedule_total, (int, float))
            or not math.isfinite(float(self.schedule_total))
            or float(self.schedule_total) <= 0
        ):
            raise ValueError(
                "schedule_total must be finite and positive when provided"
            )
        object.__setattr__(self, "options", _frozen_mapping(self.options))


@dataclass(frozen=True)
class CheckpointState:
    """checkpoint provider 解析出的模型构造信息与不透明状态载荷。"""

    checkpoint: RuntimeCheckpointRef
    model_spec: ModelSpec
    state_payload: Any
    build_payload: Any = None
    training_provenance: CheckpointTrainingProvenance | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.training_provenance is not None and not isinstance(
            self.training_provenance,
            CheckpointTrainingProvenance,
        ):
            raise TypeError(
                "training_provenance must be CheckpointTrainingProvenance"
            )
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))


@dataclass(frozen=True)
class ModelHandle:
    """统一承载 `nn.Module` 及宿主执行时可能需要的不透明上下文。"""

    model_id: str
    model: nn.Module
    context: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", _nonempty(self.model_id, "model_id"))
        if not isinstance(self.model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))


@dataclass(frozen=True)
class RuntimeSampleRef:
    """标识一个宿主样本，不假设样本来自任何特定数据模态。"""

    sample_id: str
    partition: str
    sample_index: int | None = None
    group_id: str | None = None
    position: int | float | str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sample_id", _nonempty(self.sample_id, "sample_id"))
        object.__setattr__(self, "partition", _nonempty(self.partition, "partition"))
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))


@dataclass(frozen=True)
class RuntimeComponentRef:
    """标识任务中的一个输入或输出 component，不解释具体数据模态。

    `semantic_id` 连接语义相同但执行位置不同的 component；例如同一变量在输入
    history 与输出 supervision 中可以共享 semantic identity，同时保留不同的
    `component_id`。显示名称不参与 identity，避免仅改标签就破坏跨 analyzer 对齐。
    """

    component_id: str
    semantic_id: str
    component_kind: str
    display_label: str
    component_group: str | None = None
    component_index: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "component_id",
            _nonempty(self.component_id, "component_id"),
        )
        object.__setattr__(
            self,
            "semantic_id",
            _nonempty(self.semantic_id, "semantic_id"),
        )
        kind = _nonempty(self.component_kind, "component_kind").lower()
        if kind not in {"input", "output"}:
            raise ValueError("component_kind must be 'input' or 'output'")
        object.__setattr__(self, "component_kind", kind)
        object.__setattr__(
            self,
            "display_label",
            _nonempty(self.display_label, "display_label"),
        )
        group = (
            None
            if self.component_group is None
            else _nonempty(self.component_group, "component_group")
        )
        object.__setattr__(self, "component_group", group)
        if self.component_index is not None and (
            isinstance(self.component_index, bool)
            or not isinstance(self.component_index, int)
            or self.component_index < 0
        ):
            raise ValueError(
                "component_index must be a non-negative integer when provided"
            )
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """返回可直接进入 canonical catalog 的 JSON-like payload。"""

        return {
            "component_id": self.component_id,
            "semantic_id": self.semantic_id,
            "component_kind": self.component_kind,
            "display_label": self.display_label,
            "component_group": self.component_group,
            "component_index": self.component_index,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RuntimeInputRef:
    """把 task-neutral input identity 绑定到宿主 tensor 坐标。"""

    component: RuntimeComponentRef
    tensor_path: str
    tensor_axis: int
    tensor_index: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.component, RuntimeComponentRef)
            or self.component.component_kind != "input"
        ):
            raise TypeError(
                "component must be an input RuntimeComponentRef"
            )
        object.__setattr__(
            self,
            "tensor_path",
            _nonempty(self.tensor_path, "tensor_path"),
        )
        if (
            isinstance(self.tensor_axis, bool)
            or not isinstance(self.tensor_axis, int)
        ):
            raise TypeError("tensor_axis must be an integer")
        if (
            isinstance(self.tensor_index, bool)
            or not isinstance(self.tensor_index, int)
            or self.tensor_index < 0
        ):
            raise ValueError("tensor_index must be a non-negative integer")

    def to_dict(self) -> dict[str, Any]:
        """返回带宿主 tensor 坐标的 canonical input payload。"""

        return {
            **self.component.to_dict(),
            "tensor_path": self.tensor_path,
            "tensor_axis": self.tensor_axis,
            "tensor_index": self.tensor_index,
            "normalization_id": None,
        }


@dataclass(frozen=True)
class RuntimeOutputRef:
    """把 task-neutral output identity 绑定到监督 tensor 与归一化口径。"""

    component: RuntimeComponentRef
    tensor_path: str
    tensor_axis: int
    tensor_index: int
    normalization_id: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.component, RuntimeComponentRef)
            or self.component.component_kind != "output"
        ):
            raise TypeError(
                "component must be an output RuntimeComponentRef"
            )
        object.__setattr__(
            self,
            "tensor_path",
            _nonempty(self.tensor_path, "tensor_path"),
        )
        if (
            isinstance(self.tensor_axis, bool)
            or not isinstance(self.tensor_axis, int)
        ):
            raise TypeError("tensor_axis must be an integer")
        if (
            isinstance(self.tensor_index, bool)
            or not isinstance(self.tensor_index, int)
            or self.tensor_index < 0
        ):
            raise ValueError("tensor_index must be a non-negative integer")
        object.__setattr__(
            self,
            "normalization_id",
            _nonempty(self.normalization_id, "normalization_id"),
        )

    def to_dict(self) -> dict[str, Any]:
        """返回带监督坐标与归一化 identity 的 canonical output payload。"""

        return {
            **self.component.to_dict(),
            "tensor_path": self.tensor_path,
            "tensor_axis": self.tensor_axis,
            "tensor_index": self.tensor_index,
            "normalization_id": self.normalization_id,
        }


@dataclass(frozen=True)
class RuntimeOutputResponse:
    """保存同一次宿主执行中一个显式 output 的 scalar response。"""

    output: RuntimeOutputRef
    response_id: str
    metric_id: str
    value: torch.Tensor
    support_count: int
    normalization: str
    higher_is_better: bool
    gradient_tensor: torch.Tensor | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.output, RuntimeOutputRef):
            raise TypeError("output must be RuntimeOutputRef")
        object.__setattr__(
            self,
            "response_id",
            _nonempty(self.response_id, "response_id"),
        )
        object.__setattr__(
            self,
            "metric_id",
            _nonempty(self.metric_id, "metric_id"),
        )
        if not isinstance(self.value, torch.Tensor) or self.value.numel() != 1:
            raise TypeError("value must be a scalar torch.Tensor")
        if (
            isinstance(self.support_count, bool)
            or not isinstance(self.support_count, int)
            or self.support_count <= 0
        ):
            raise ValueError("support_count must be a positive integer")
        object.__setattr__(
            self,
            "normalization",
            _nonempty(self.normalization, "normalization"),
        )
        if self.gradient_tensor is not None and not isinstance(
            self.gradient_tensor,
            torch.Tensor,
        ):
            raise TypeError("gradient_tensor must be a torch.Tensor")
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))


@dataclass(frozen=True)
class RuntimeInputPerturbation:
    """保存一次隔离输入扰动分支及其实际输入差分能量。

    Host Runtime 不解释输入模态。宿主负责保证 ``batch`` 只改变声明的输入
    位置，并在声明的模型输入坐标中报告实际扰动平方和；该输入能量用于验证
    干预执行，不作为输出相对响应的分母。portable analyzer 不读取宿主 payload。
    """

    batch: BatchEnvelope
    input: RuntimeInputRef
    scale: float
    direction: int
    input_delta_square_sum: float
    input_support_count: int
    affected_value_element_count: int
    modified_paths: Sequence[str]
    preserved_paths_verified: bool
    normalization: str
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.batch, BatchEnvelope):
            raise TypeError("batch must be BatchEnvelope")
        if not isinstance(self.input, RuntimeInputRef):
            raise TypeError("input must be RuntimeInputRef")
        scale = float(self.scale)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("scale must be finite and positive")
        if int(self.direction) not in {-1, 1}:
            raise ValueError("direction must be -1 or 1")
        square_sum = float(self.input_delta_square_sum)
        if not math.isfinite(square_sum) or square_sum <= 0:
            raise ValueError(
                "input_delta_square_sum must be finite and positive"
            )
        if (
            isinstance(self.input_support_count, bool)
            or not isinstance(self.input_support_count, int)
            or self.input_support_count <= 0
        ):
            raise ValueError("input_support_count must be a positive integer")
        if (
            isinstance(self.affected_value_element_count, bool)
            or not isinstance(self.affected_value_element_count, int)
            or self.affected_value_element_count <= 0
        ):
            raise ValueError(
                "affected_value_element_count must be a positive integer"
            )
        paths = tuple(_nonempty(path, "modified_path") for path in self.modified_paths)
        if not paths:
            raise ValueError("modified_paths must not be empty")
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "direction", int(self.direction))
        object.__setattr__(
            self,
            "input_delta_square_sum",
            square_sum,
        )
        object.__setattr__(self, "modified_paths", paths)
        object.__setattr__(
            self,
            "normalization",
            _nonempty(self.normalization, "normalization"),
        )
        object.__setattr__(self, "provenance", _frozen_mapping(self.provenance))


@dataclass(frozen=True)
class RuntimeOutputPerturbation:
    """保存同一有效 support 上的输出能量与扰动差分能量。

    三个平方和必须来自相同输出坐标、validity mask 和 measurement space；
    portable analyzer 据此计算相对于输出自身幅度的对称相对 RMS 响应。Host
    不在这里引入训练集标准差、逐样本特征或输入扰动幅度分母。
    """

    output: RuntimeOutputRef
    baseline_output_square_sum: float
    condition_output_square_sum: float
    output_difference_square_sum: float
    support_count: int
    normalization: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.output, RuntimeOutputRef):
            raise TypeError("output must be RuntimeOutputRef")
        square_sums = {
            "baseline_output_square_sum": float(
                self.baseline_output_square_sum
            ),
            "condition_output_square_sum": float(
                self.condition_output_square_sum
            ),
            "output_difference_square_sum": float(
                self.output_difference_square_sum
            ),
        }
        for name, value in square_sums.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if (
            isinstance(self.support_count, bool)
            or not isinstance(self.support_count, int)
            or self.support_count <= 0
        ):
            raise ValueError("support_count must be a positive integer")
        for name, value in square_sums.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "normalization",
            _nonempty(self.normalization, "normalization"),
        )
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))


@dataclass(frozen=True)
class RuntimeOutputObjective:
    """保存 actual objective ledger 中一个 output 的可微 objective slice。

    `raw_numerator` 是 task loss 已完成自身 reduction/weight、但尚未除以 runtime
    accumulation/normalization divisor 的通道 contribution；它不是把不同 loss
    term 的原始误差平方和强行相加。`value` 则与本次实际 backward unit 的口径一致。
    """

    output: RuntimeOutputRef
    objective_identity: str
    value: torch.Tensor
    support_count: float
    normalization: str
    partition_status: str
    partition_coverage: float
    raw_numerator: torch.Tensor | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.output, RuntimeOutputRef):
            raise TypeError("output must be RuntimeOutputRef")
        object.__setattr__(
            self,
            "objective_identity",
            _nonempty(self.objective_identity, "objective_identity"),
        )
        if not isinstance(self.value, torch.Tensor) or self.value.numel() != 1:
            raise TypeError("value must be a scalar torch.Tensor")
        if self.raw_numerator is not None and (
            not isinstance(self.raw_numerator, torch.Tensor)
            or self.raw_numerator.numel() != 1
        ):
            raise TypeError(
                "raw_numerator must be a scalar torch.Tensor when provided"
            )
        if (
            not math.isfinite(float(self.support_count))
            or float(self.support_count) <= 0
        ):
            raise ValueError("support_count must be finite and positive")
        object.__setattr__(
            self,
            "normalization",
            _nonempty(self.normalization, "normalization"),
        )
        object.__setattr__(
            self,
            "partition_status",
            _nonempty(self.partition_status, "partition_status"),
        )
        coverage = float(self.partition_coverage)
        if not math.isfinite(coverage) or not 0.0 <= coverage <= 1.0:
            raise ValueError("partition_coverage must be within [0, 1]")
        object.__setattr__(self, "partition_coverage", coverage)
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))


@dataclass(frozen=True)
class SampleSelector:
    """向宿主声明样本选择约束；确定性策略由具体任务实现。"""

    partition: str
    limit: int | None = None
    criteria: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "partition", _nonempty(self.partition, "partition"))
        if self.limit is not None and int(self.limit) <= 0:
            raise ValueError("limit must be positive when provided")
        object.__setattr__(self, "criteria", _frozen_mapping(self.criteria))


@dataclass(frozen=True)
class BatchEnvelope:
    """把宿主 batch payload 与稳定样本 identity 组合为统一执行输入。"""

    batch_id: str
    payload: Any
    samples: Sequence[RuntimeSampleRef] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "batch_id", _nonempty(self.batch_id, "batch_id"))
        object.__setattr__(self, "samples", tuple(self.samples))
        if any(not isinstance(sample, RuntimeSampleRef) for sample in self.samples):
            raise TypeError("samples must contain RuntimeSampleRef values")
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))

@dataclass(frozen=True)
class RuntimeDescriptor:
    """描述一个已组合 runtime 的 task definition 与 run identity。"""

    runtime_id: str
    task_definition_id: str
    task_definition_version: int
    run: RunRef
    objective_executor_version: int
    checkpoint_objective: CheckpointObjectiveSpec
    device: str = "cpu"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "runtime_id", _nonempty(self.runtime_id, "runtime_id"))
        object.__setattr__(
            self,
            "task_definition_id",
            _nonempty(self.task_definition_id, "task_definition_id"),
        )
        if int(self.task_definition_version) <= 0:
            raise ValueError("task_definition_version must be positive")
        if (
            isinstance(self.objective_executor_version, bool)
            or not isinstance(self.objective_executor_version, int)
            or self.objective_executor_version <= 0
        ):
            raise ValueError(
                "objective_executor_version must be a positive integer"
            )
        if not isinstance(self.checkpoint_objective, CheckpointObjectiveSpec):
            raise TypeError(
                "checkpoint_objective must be CheckpointObjectiveSpec"
            )
        object.__setattr__(self, "device", _nonempty(self.device, "device"))
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))


@dataclass(frozen=True)
class CapabilityDescriptor:
    """声明可选 runtime capability 的稳定名称与 contract 版本。"""

    capability_id: str
    definition_version: int = 1
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "capability_id",
            _nonempty(self.capability_id, "capability_id"),
        )
        if int(self.definition_version) <= 0:
            raise ValueError("definition_version must be positive")
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))


@dataclass(frozen=True)
class ObjectiveContext:
    """描述 objective 的执行位置、变换口径与宿主 policy。"""

    objective_id: str
    training: bool
    update_index: int | None = None
    microbatch_index: int | None = None
    schedule_coordinate: int | float | str | None = None
    schedule_total: int | float | None = None
    normalization_divisor: float = 1.0
    loss_cap: float | None = None
    amp_scale: float = 1.0
    options: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "objective_id",
            _nonempty(self.objective_id, "objective_id"),
        )
        if (
            not math.isfinite(float(self.normalization_divisor))
            or float(self.normalization_divisor) <= 0
        ):
            raise ValueError("normalization_divisor must be finite and positive")
        if self.loss_cap is not None and (
            not math.isfinite(float(self.loss_cap)) or float(self.loss_cap) <= 0
        ):
            raise ValueError("loss_cap must be finite and positive when provided")
        if not math.isfinite(float(self.amp_scale)) or float(self.amp_scale) <= 0:
            raise ValueError("amp_scale must be finite and positive")
        object.__setattr__(self, "options", _frozen_mapping(self.options))
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))


@dataclass(frozen=True)
class ObjectiveUnit:
    """一次可独立 backward 的完整 objective provenance。

    普通任务通常只产生一个单元；需要分块反传的任务可按真实执行顺序产生多个
    单元。`backward_total` 是在 AMP scale 前实际交给 autograd 的 tensor；非标量
    值必须同时提供真实 VJP 使用的显式 cotangent。executor 不再从 `raw_total`
    重建 cap、normalization 或 reduction。
    """

    unit_id: str
    objective_identity: str
    raw_total: torch.Tensor
    backward_total: torch.Tensor
    backward_cotangent: torch.Tensor | None = None
    cotangent_identity: str | None = None
    ledger: Any = None
    terms: Mapping[str, torch.Tensor | float] = field(default_factory=dict)
    outputs: Mapping[str, Any] = field(default_factory=dict)
    schedule_coordinate: int | float | str | None = None
    normalization_divisor: float = 1.0
    loss_cap: float | None = None
    loss_cap_applied: bool = False
    amp_scale: float = 1.0
    chunk_identity: str | None = None
    chunk_index: int | None = None
    chunk_count: int | None = None
    segment_identity: str | None = None
    retain_graph: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "unit_id", _nonempty(self.unit_id, "unit_id"))
        object.__setattr__(
            self,
            "objective_identity",
            _nonempty(self.objective_identity, "objective_identity"),
        )
        for field_name, value in (
            ("raw_total", self.raw_total),
            ("backward_total", self.backward_total),
        ):
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{field_name} must be a torch.Tensor")
        if self.raw_total.numel() != 1:
            raise ValueError("raw_total must contain exactly one scalar value")
        if self.backward_total.numel() < 1:
            raise ValueError("backward_total must not be empty")
        cotangent = self.backward_cotangent
        identity = (
            None
            if self.cotangent_identity is None
            else str(self.cotangent_identity).strip()
        )
        if cotangent is None:
            if self.backward_total.numel() != 1:
                raise ValueError(
                    "non-scalar backward_total requires an explicit cotangent"
                )
            if identity:
                raise ValueError(
                    "cotangent_identity requires backward_cotangent"
                )
        else:
            if not isinstance(cotangent, torch.Tensor):
                raise TypeError("backward_cotangent must be a torch.Tensor")
            if not identity:
                raise ValueError(
                    "cotangent_identity must be non-empty when cotangent is provided"
                )
            if cotangent.shape != self.backward_total.shape:
                raise ValueError(
                    "backward_cotangent shape must exactly match backward_total"
                )
            if cotangent.device != self.backward_total.device:
                raise ValueError(
                    "backward_cotangent and backward_total must use the same device"
                )
            if cotangent.dtype != self.backward_total.dtype:
                raise ValueError(
                    "backward_cotangent and backward_total must use the same dtype"
                )
            if cotangent.requires_grad:
                raise ValueError("backward_cotangent must be detached")
            if not bool(torch.isfinite(cotangent).all().item()):
                raise ValueError("backward_cotangent must be finite")
        object.__setattr__(self, "cotangent_identity", identity)
        if (
            not math.isfinite(float(self.normalization_divisor))
            or float(self.normalization_divisor) <= 0
        ):
            raise ValueError("normalization_divisor must be finite and positive")
        if self.loss_cap is not None and (
            not math.isfinite(float(self.loss_cap)) or float(self.loss_cap) <= 0
        ):
            raise ValueError("loss_cap must be finite and positive when provided")
        if not math.isfinite(float(self.amp_scale)) or float(self.amp_scale) <= 0:
            raise ValueError("amp_scale must be finite and positive")
        if self.chunk_index is not None and int(self.chunk_index) < 0:
            raise ValueError("chunk_index must be non-negative when provided")
        if self.chunk_count is not None and int(self.chunk_count) <= 0:
            raise ValueError("chunk_count must be positive when provided")
        if (
            self.chunk_index is not None
            and self.chunk_count is not None
            and int(self.chunk_index) >= int(self.chunk_count)
        ):
            raise ValueError("chunk_index must be smaller than chunk_count")
        for term_name, term in self.terms.items():
            if isinstance(term, torch.Tensor) and term.numel() != 1:
                raise ValueError(
                    f"objective term {term_name!r} must contain one scalar value"
                )
        object.__setattr__(self, "terms", _frozen_mapping(self.terms))
        object.__setattr__(self, "outputs", _frozen_mapping(self.outputs))
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))

    def effective_backward_objective(self) -> torch.Tensor:
        """返回与实际 scalar/VJP backward 数学等价的 graph-connected scalar。"""

        if self.backward_cotangent is None:
            return self.backward_total.reshape(())
        return torch.sum(
            self.backward_total * self.backward_cotangent.detach()
        ).reshape(())


@dataclass(frozen=True)
class ObjectiveExecution:
    """封装一次 objective 调用产生的单次可消费 unit stream。"""

    execution_id: str
    units: Iterable[ObjectiveUnit]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "execution_id",
            _nonempty(self.execution_id, "execution_id"),
        )
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))


@runtime_checkable
class ModelProvider(Protocol):
    """构造和释放模型；不拥有 checkpoint 的解析语义。"""

    def create(
        self,
        model_spec: ModelSpec,
        build_payload: Any,
        *,
        device: torch.device,
    ) -> ModelHandle: ...

    def close(self, handle: ModelHandle) -> None: ...


@runtime_checkable
class CheckpointProvider(Protocol):
    """发现、加载并恢复宿主 checkpoint。"""

    def resolve(self, run: RunRef) -> Sequence[RuntimeCheckpointRef]: ...

    def load(
        self,
        checkpoint: RuntimeCheckpointRef,
        *,
        map_location: torch.device,
    ) -> CheckpointState: ...

    def restore(self, handle: ModelHandle, state: CheckpointState) -> None: ...


@runtime_checkable
class BatchProvider(Protocol):
    """选择样本并物化任务 batch envelope。"""

    def select(
        self,
        run: RunRef,
        selector: SampleSelector,
    ) -> Sequence[RuntimeSampleRef]: ...

    def materialize(
        self,
        samples: Sequence[RuntimeSampleRef],
        *,
        device: torch.device,
    ) -> BatchEnvelope: ...

    def clone(self, batch: BatchEnvelope) -> BatchEnvelope: ...


@runtime_checkable
class SessionAwareBatchProvider(Protocol):
    """可选的多 model-session batch context 切换协议。

    普通任务的 batch provider 不需要实现本协议。若 sample selection 或
    materialization 依赖当前 checkpoint model session，调用方必须在相应操作前
    显式激活 session，不能依赖“最后打开的 checkpoint”这一隐式全局状态。
    """

    def activate_session(
        self,
        session: "ExecutionSession | ModelHandle",
    ) -> None: ...


@runtime_checkable
class ExecutionPreparationCapability(Protocol):
    """在资源构造前幂等激活 capability 已声明的执行需求。"""

    def prepare_execution(self) -> None: ...


@runtime_checkable
class ComponentCatalogCapability(Protocol):
    """显式列出一次执行 payload 中全部输入与输出 component。"""

    def list_input_components(
        self,
        *,
        session: "ExecutionSession",
        batch: BatchEnvelope,
    ) -> Sequence[RuntimeInputRef]: ...

    def list_output_components(
        self,
        *,
        session: "ExecutionSession",
        batch: BatchEnvelope,
    ) -> Sequence[RuntimeOutputRef]: ...


@runtime_checkable
class OutputEvaluationCapability(Protocol):
    """从同一次 objective execution 评价一组显式输出。"""

    def evaluate_outputs(
        self,
        *,
        batch: BatchEnvelope,
        unit: ObjectiveUnit,
        outputs: Sequence[RuntimeOutputRef],
    ) -> Sequence[RuntimeOutputResponse]: ...


@runtime_checkable
class OutputObjectiveCapability(Protocol):
    """从 actual objective provenance 提取一组显式输出 objective。"""

    def objective_slices(
        self,
        *,
        unit: ObjectiveUnit,
        outputs: Sequence[RuntimeOutputRef],
    ) -> Sequence[RuntimeOutputObjective]: ...


@runtime_checkable
class InputSensitivityCapability(Protocol):
    """物化输入扰动和逐输出同 support 能量，不计算诊断指标。"""

    scales: Sequence[float]

    def perturb(
        self,
        *,
        batch: BatchEnvelope,
        input: RuntimeInputRef,
        scale: float,
        direction: int,
        direction_seed: int,
    ) -> RuntimeInputPerturbation: ...

    def compare_outputs(
        self,
        *,
        batch: BatchEnvelope,
        baseline_unit: ObjectiveUnit,
        condition_unit: ObjectiveUnit,
        outputs: Sequence[RuntimeOutputRef],
    ) -> Sequence[RuntimeOutputPerturbation]: ...


@runtime_checkable
class ObjectiveExecutor(Protocol):
    """调用宿主唯一 objective 实现并按真实 backward 顺序返回单元。

    `execute()` 本身只准备 execution；需要 forward/backward 邻接或 no-sync scope
    的计算必须延迟到 `ObjectiveExecution.units` 被逐项消费时执行。
    """

    definition_version: int

    def execute(
        self,
        handle: ModelHandle,
        batch: BatchEnvelope,
        context: ObjectiveContext,
    ) -> ObjectiveExecution: ...


@runtime_checkable
class AttemptExecutor(Protocol):
    """执行一个 update attempt；具体实现不得拥有任务 objective 数学。"""

    def execute(
        self,
        session: "ExecutionSession",
        request: "TrainingAttemptRequest",
        *,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any = None,
        scaler: Any = None,
    ) -> "TrainingAttemptResult": ...


CheckpointRef = RuntimeCheckpointRef
SampleRef = RuntimeSampleRef

__all__ = [
    "AttemptExecutor",
    "BatchEnvelope",
    "BatchProvider",
    "CapabilityDescriptor",
    "ComponentCatalogCapability",
    "CheckpointObjectiveSpec",
    "CheckpointProvider",
    "CheckpointRef",
    "CheckpointState",
    "CheckpointTrainingProvenance",
    "ExecutionPreparationCapability",
    "ModelHandle",
    "ModelProvider",
    "ModelSpec",
    "ObjectiveContext",
    "ObjectiveExecution",
    "ObjectiveExecutor",
    "ObjectiveUnit",
    "OutputEvaluationCapability",
    "OutputObjectiveCapability",
    "RunRef",
    "RuntimeCheckpointRef",
    "RuntimeComponentRef",
    "RuntimeDescriptor",
    "RuntimeInputRef",
    "RuntimeOutputObjective",
    "RuntimeOutputRef",
    "RuntimeOutputResponse",
    "RuntimeSampleRef",
    "SampleRef",
    "SampleSelector",
    "SessionAwareBatchProvider",
]
