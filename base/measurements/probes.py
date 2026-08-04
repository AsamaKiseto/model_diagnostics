"""从模块边界采集完整 activation/local Taylor 与输入输出分布。

Block 探针保留 detach 后的 activation 副本，不保存原 output 或计算图；
Norm/Activation Tap 在 forward hook 内立即归约为标量充分统计，因此不会把输入输出
Tensor 保留到 logical update 结束。两类探针都支持常见 structured output。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from hashlib import sha1
import math
from typing import Any

import torch
from torch import nn

_EPS = 1e-12

_NORMALIZATION_TYPES = (
    nn.LayerNorm,
    nn.GroupNorm,
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
)
_ACTIVATION_TYPES = (
    nn.ReLU,
    nn.ReLU6,
    nn.GELU,
    nn.SiLU,
    nn.ELU,
    nn.LeakyReLU,
    nn.Sigmoid,
    nn.Tanh,
    nn.Softmax,
)


@dataclass(slots=True)
class DistributionTap:
    """一个稳定的 Norm/Activation 分布观测点；module 引用不进入 artifact。"""

    tap_id: str
    owner_node_id: str
    model_name: str
    module_path: str
    module_type: str
    category: str
    module: nn.Module = field(repr=False)

    def manifest_record(self) -> dict[str, Any]:
        """返回 JSON-safe Tap identity，并显式关联所属 hierarchy node。"""

        return {
            "tap_id": self.tap_id,
            "owner_node_id": self.owner_node_id,
            "model_name": self.model_name,
            "module_path": self.module_path,
            "module_type": self.module_type,
            "category": self.category,
            "discovery_source": "deterministic_module_type",
        }


@dataclass(slots=True)
class DistributionObservation:
    """一次 Tensor 边界的 device-scalar 充分统计，不拥有原 Tensor storage。"""

    source: str
    path: str
    source_numel: int
    sample_numel: int
    finite_count: torch.Tensor
    value_sum: torch.Tensor
    square_sum: torch.Tensor
    abs_max: torch.Tensor
    zero_count: torch.Tensor
    positive_count: torch.Tensor
    negative_count: torch.Tensor
    saturation_count: torch.Tensor | None


@dataclass(slots=True)
class _TensorSample:
    path: str
    shape: tuple[int, ...]
    source_numel: int
    activation: torch.Tensor
    gradient: torch.Tensor | None = None
    loss_scale: float = 1.0
    handle: Any | None = None


@dataclass(slots=True)
class _ProbeRuntimeState:
    """由同一 invocation 的 tensor hooks 共享，确保首次失败后整体停止捕获。"""

    disabled: bool = False
    failure_reason: str | None = None


class ModuleDiagnosticsProbe:
    """拥有一次 module invocation 的完整 activation/gradient probes。"""

    def __init__(
        self,
        samples: list[_TensorSample],
        *,
        runtime_state: _ProbeRuntimeState | None = None,
    ) -> None:
        self._samples = samples
        self._runtime_state = runtime_state or _ProbeRuntimeState()
        self._closed = False

    @property
    def sample_element_count(self) -> int:
        """返回该 invocation 实际持有的完整 activation 元素数。"""

        return sum(int(sample.activation.numel()) for sample in self._samples)

    @property
    def disabled(self) -> bool:
        """返回 tensor hook 是否因采样失败而停用整个 invocation。"""

        return self._runtime_state.disabled

    @classmethod
    def from_output(
        cls,
        output: Any,
        *,
        with_grad: bool,
        failure_callback: Callable[[str, Exception], None] | None = None,
        propagate_hook_failures: bool = True,
    ) -> "ModuleDiagnosticsProbe | None":
        """遍历全部 structured output，并为每个 Tensor 建立完整统计 probe。

        backward tensor hook 只观察 gradient，永远返回 ``None``。degrade 模式下，
        首次采集异常会停用该 invocation 的全部 tensor hooks 并通知 controller；
        strict 模式由 ``propagate_hook_failures`` 保留原异常。
        """

        tensors = _structured_tensors(output, root="output")
        if not tensors:
            return None
        samples: list[_TensorSample] = []
        runtime_state = _ProbeRuntimeState()
        try:
            for path, tensor in tensors:
                count = int(tensor.numel())
                sample = _TensorSample(
                    path=path,
                    shape=tuple(int(dim) for dim in tensor.shape),
                    source_numel=count,
                    activation=_sample_values(tensor, count),
                )
                # 先纳入 cleanup ownership；即使 register_hook 在中途失败，也不会遗留
                # 已注册的早期 tensor hooks 或对 activation sample 的引用。
                samples.append(sample)
                if with_grad and tensor.requires_grad:
                    def capture_gradient(
                        gradient: torch.Tensor,
                        target: _TensorSample = sample,
                    ) -> None:
                        if gradient is None or runtime_state.disabled:
                            return None
                        try:
                            complete = _sample_values(
                                gradient,
                                int(gradient.numel()),
                            )
                            target.gradient = complete / float(target.loss_scale)
                        except Exception as exc:
                            runtime_state.disabled = True
                            try:
                                runtime_state.failure_reason = (
                                    f"{type(exc).__name__}: {exc}"
                                )
                            except Exception:
                                # 失败的异常格式化不能越过 observer degrade 边界。
                                runtime_state.failure_reason = type(exc).__name__
                            if failure_callback is not None:
                                try:
                                    failure_callback("activation_gradient_sampling", exc)
                                except Exception:
                                    # failure callback 属于诊断控制面；degrade 时不能让它
                                    # 覆盖原训练 backward，strict 仍传播原采样异常。
                                    pass
                            if propagate_hook_failures:
                                raise
                        return None

                    sample.handle = tensor.register_hook(capture_gradient)
        except Exception:
            for sample in samples:
                if sample.handle is not None:
                    try:
                        sample.handle.remove()
                    except Exception:
                        pass
                    sample.handle = None
            raise
        return cls(samples, runtime_state=runtime_state)

    def bind_amp_scale(self, amp_scale: float) -> None:
        """记录 GradScaler scale，以便把 ``d(scaled_loss)/dh`` 还原为 ``dLoss/dh``。"""

        value = float(amp_scale)
        if math.isfinite(value) and value > 0.0:
            for sample in self._samples:
                if sample.gradient is None:
                    sample.loss_scale = value

    def finalize(
        self,
        *,
        update: int,
        rank: int,
        objective_raw_sum: float | None,
        backward_objective_sum: float | None,
        objective_normalization_divisor: float | None,
        loss_cap_hit_fraction: float | None,
        objective_trace_status: str,
    ) -> list[dict[str, Any]]:
        """生成可加的 activation/local-Taylor 充分统计并移除 tensor hooks。

        gradient 已在 tensor hook 中按当次 `amp_scale` 还原，因此不会误用最后一次
        backward 的 scale。loss-normalized 派生量只使用同一个 backward objective；
        raw loss 仅作为 provenance。
        """

        if self._runtime_state.disabled:
            self.close()
            return []
        records = [
            _record(
                sample,
                update=int(update),
                rank=int(rank),
                objective_raw_sum=objective_raw_sum,
                backward_objective_sum=backward_objective_sum,
                objective_normalization_divisor=objective_normalization_divisor,
                loss_cap_hit_fraction=loss_cap_hit_fraction,
                objective_trace_status=objective_trace_status,
            )
            for sample in self._samples
        ]
        self.close()
        return records

    def close(self) -> None:
        """幂等移除所有 autograd hook，并清除梯度抽样引用。"""

        if self._closed:
            return
        self._closed = True
        for sample in self._samples:
            if sample.handle is not None:
                try:
                    sample.handle.remove()
                except Exception:
                    pass
                sample.handle = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def discover_distribution_taps(
    models: Mapping[str, nn.Module],
    hierarchy_nodes: Iterable[Any],
) -> tuple[DistributionTap, ...]:
    """发现全部 Activation/Norm，并关联到最深 hierarchy owner。"""

    nodes = tuple(hierarchy_nodes)
    taps: list[DistributionTap] = []
    for model_name, model in sorted(models.items()):
        for module_path, module in model.named_modules():
            if not module_path:
                continue
            if isinstance(module, _NORMALIZATION_TYPES):
                category = "normalization"
            elif isinstance(module, _ACTIVATION_TYPES):
                category = "activation"
            else:
                continue
            owner = max(
                (
                    node
                    for node in nodes
                    if getattr(node, "model_name", None) == model_name
                    and getattr(node, "hierarchy_level", None) in {"Model", "Stage", "Block"}
                    and (
                        not str(getattr(node, "module_path", ""))
                        or module_path == str(node.module_path)
                        or module_path.startswith(f"{node.module_path}.")
                    )
                ),
                key=lambda node: (
                    len(str(getattr(node, "module_path", ""))),
                    {"Model": 0, "Stage": 1, "Block": 2}.get(str(node.hierarchy_level), -1),
                ),
            )
            digest = sha1(
                f"{model_name}:{module_path}".encode("utf-8"),
                usedforsecurity=False,
            ).hexdigest()[:10]
            taps.append(
                DistributionTap(
                    tap_id=f"tap/{model_name}/{digest}",
                    owner_node_id=str(owner.node_id),
                    model_name=model_name,
                    module_path=module_path,
                    module_type=module.__class__.__name__,
                    category=category,
                    module=module,
                )
            )
    return tuple(sorted(taps, key=lambda tap: (tap.model_name, tap.module_path, tap.tap_id)))


def capture_distribution_observations(
    value: Any,
    *,
    source: str,
    module: nn.Module,
) -> list[DistributionObservation]:
    """在 hook 内把全部 structured Tensor 立即归约为 device scalar 充分统计。"""

    tensors = _structured_tensors(value, root=source)
    if not tensors:
        return []
    observations: list[DistributionObservation] = []
    for path, tensor in tensors:
        source_numel = int(tensor.numel())
        values = _sample_values(tensor, source_numel)
        finite = torch.isfinite(values)
        safe = torch.where(finite, values, torch.zeros((), dtype=values.dtype, device=values.device))
        saturation = None
        if isinstance(module, nn.Sigmoid):
            saturation = finite & ((safe <= 0.01) | (safe >= 0.99))
        elif isinstance(module, nn.Tanh):
            saturation = finite & (safe.abs() >= 0.99)
        observations.append(
            DistributionObservation(
                source=source,
                path=path,
                source_numel=source_numel,
                sample_numel=int(values.numel()),
                finite_count=finite.sum().detach(),
                value_sum=safe.double().sum().detach(),
                square_sum=safe.double().square().sum().detach(),
                abs_max=safe.abs().max().detach(),
                zero_count=(finite & (safe == 0)).sum().detach(),
                positive_count=(finite & (safe > 0)).sum().detach(),
                negative_count=(finite & (safe < 0)).sum().detach(),
                saturation_count=None if saturation is None else saturation.sum().detach(),
            )
        )
    return observations


def capture_normalization_state(module: nn.Module) -> dict[str, Any]:
    """把 Norm running/affine 状态归约成 device scalar，不保留完整 buffer/parameter。"""

    result: dict[str, Any] = {"module_training": bool(module.training)}
    running_mean = getattr(module, "running_mean", None)
    if torch.is_tensor(running_mean) and running_mean.numel() > 0:
        values = running_mean.detach().float().reshape(-1)
        finite = torch.isfinite(values)
        safe = torch.where(finite, values, torch.zeros((), dtype=values.dtype, device=values.device))
        result["normalization_running_mean_rms"] = torch.sqrt(
            safe.double().square().sum() / finite.sum().clamp_min(1)
        ).detach()
    running_var = getattr(module, "running_var", None)
    if torch.is_tensor(running_var) and running_var.numel() > 0:
        values = running_var.detach().float().reshape(-1)
        finite = torch.isfinite(values)
        safe = torch.where(finite, values, torch.zeros((), dtype=values.dtype, device=values.device))
        result["normalization_running_var_mean"] = (
            safe.double().sum() / finite.sum().clamp_min(1)
        ).detach()
        result["normalization_running_var_min"] = torch.where(
            finite,
            values,
            torch.full((), float("inf"), dtype=values.dtype, device=values.device),
        ).min().detach()
        result["normalization_running_var_nonfinite_fraction"] = (
            (~finite).double().mean().detach()
        )
    for attribute, field_name in (
        ("weight", "normalization_weight_rms"),
        ("bias", "normalization_bias_rms"),
    ):
        tensor = getattr(module, attribute, None)
        if torch.is_tensor(tensor) and tensor.numel() > 0:
            values = tensor.detach().float().reshape(-1)
            finite = torch.isfinite(values)
            safe = torch.where(finite, values, torch.zeros((), dtype=values.dtype, device=values.device))
            result[field_name] = torch.sqrt(
                safe.double().square().sum() / finite.sum().clamp_min(1)
            ).detach()
    return result


def finalize_normalization_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """在 update 收口后把 Norm 的 device scalar state 转成 JSON-safe 值。"""

    output: dict[str, Any] = {}
    for key, value in state.items():
        if torch.is_tensor(value):
            candidate = float(value.item())
            output[key] = candidate if math.isfinite(candidate) else None
        else:
            output[key] = value
    return output


def summarize_distribution_observations(
    observations: Iterable[DistributionObservation],
    *,
    category: str,
    module_type: str,
) -> dict[str, Any]:
    """在 update 收口同步后把 device scalar 转为可跨 rank 合并的 JSON 标量。"""

    grouped: dict[str, list[DistributionObservation]] = {"input": [], "output": []}
    for observation in observations:
        grouped.setdefault(observation.source, []).append(observation)
    raw: dict[str, Any] = {}
    for source in ("input", "output"):
        items = grouped.get(source, [])
        raw.update(
            {
                f"_{source}_value_sum": sum(float(item.value_sum.item()) for item in items),
                f"_{source}_square_sum": sum(float(item.square_sum.item()) for item in items),
                f"_{source}_finite_count": sum(int(item.finite_count.item()) for item in items),
                f"_{source}_sample_count": sum(item.sample_numel for item in items),
                f"_{source}_zero_count": sum(int(item.zero_count.item()) for item in items),
                f"_{source}_positive_count": sum(int(item.positive_count.item()) for item in items),
                f"_{source}_negative_count": sum(int(item.negative_count.item()) for item in items),
                f"_{source}_abs_max": max(
                    (float(item.abs_max.item()) for item in items),
                    default=None,
                ),
                f"_{source}_saturation_count": sum(
                    int(item.saturation_count.item())
                    for item in items
                    if item.saturation_count is not None
                ),
                f"_{source}_saturation_sample_count": sum(
                    item.sample_numel for item in items if item.saturation_count is not None
                ),
                f"{source}_source_numel": sum(item.source_numel for item in items),
                f"{source}_sample_numel": sum(item.sample_numel for item in items),
                f"{source}_tensor_paths": sorted({item.path for item in items}),
            }
        )
    return _distribution_metrics(raw, category=category, module_type=module_type)


def merge_distribution_summaries(
    rows: Iterable[Mapping[str, Any]],
    *,
    category: str,
    module_type: str,
) -> dict[str, Any]:
    """按充分统计合并各 rank，避免简单平均局部 mean/std/fraction。"""

    items = list(rows)
    raw: dict[str, Any] = {}
    for source in ("input", "output"):
        for field_name in (
            "value_sum",
            "square_sum",
            "finite_count",
            "sample_count",
            "zero_count",
            "positive_count",
            "negative_count",
            "saturation_count",
            "saturation_sample_count",
        ):
            key = f"_{source}_{field_name}"
            raw[key] = sum(float(row.get(key) or 0.0) for row in items)
        abs_values = [row.get(f"_{source}_abs_max") for row in items]
        raw[f"_{source}_abs_max"] = max(
            (float(value) for value in abs_values if value is not None),
            default=None,
        )
        raw[f"{source}_source_numel"] = sum(int(row.get(f"{source}_source_numel") or 0) for row in items)
        raw[f"{source}_sample_numel"] = sum(int(row.get(f"{source}_sample_numel") or 0) for row in items)
        raw[f"{source}_tensor_paths"] = sorted(
            {
                str(path)
                for row in items
                for path in row.get(f"{source}_tensor_paths", []) or []
            }
        )
    return _distribution_metrics(raw, category=category, module_type=module_type)


def _structured_tensors(output: Any, *, root: str) -> list[tuple[str, torch.Tensor]]:
    """稳定遍历常见 structured output；循环对象和同一 Tensor 只访问一次。"""

    result: list[tuple[str, torch.Tensor]] = []
    stack: list[tuple[str, Any]] = [(root, output)]
    visited_containers: set[int] = set()
    visited_tensors: set[int] = set()
    while stack and len(visited_containers) < 256:
        path, value = stack.pop()
        if torch.is_tensor(value):
            if id(value) not in visited_tensors and value.numel() > 0:
                visited_tensors.add(id(value))
                result.append((path, value))
            continue
        identity = id(value)
        if identity in visited_containers:
            continue
        visited_containers.add(identity)
        if isinstance(value, Mapping):
            for key in sorted(value, key=lambda item: str(item), reverse=True):
                stack.append((f"{path}.{key}", value[key]))
        elif is_dataclass(value) and not isinstance(value, type):
            for item in reversed(fields(value)):
                stack.append((f"{path}.{item.name}", getattr(value, item.name)))
        elif isinstance(value, tuple) and hasattr(value, "_fields"):
            for name in reversed(value._fields):
                stack.append((f"{path}.{name}", getattr(value, name)))
        elif isinstance(value, (tuple, list)):
            for index in range(len(value) - 1, -1, -1):
                stack.append((f"{path}.{index}", value[index]))
    result.sort(key=lambda item: item[0])
    return result


def _sample_values(tensor: torch.Tensor, size: int, *, offset: int = 0) -> torch.Tensor:
    """轮转确定性复制至多 ``size`` 个元素；不推进全局 RNG。"""

    # 必须先切断 autograd，再执行 index_select/类型转换。若先采样后 detach，
    # 这些 observer 算子会被 non-reentrant activation checkpoint 记入原
    # forward 的 saved-tensor 集合，而 cadence/recompute 路径未必执行同一
    # observer，最终破坏训练图的重计算一致性。
    detached = tensor.detach()
    count = int(detached.numel())
    size = min(max(int(size), 1), count)
    if size == count:
        indices = torch.arange(count, device=detached.device)
    else:
        positions = torch.arange(
            size,
            device=detached.device,
            dtype=torch.float64,
        )
        indices = torch.floor((positions + 0.5) * count / size).long()
        if offset:
            indices = torch.remainder(indices + int(offset), count)
    return detached.reshape(-1).index_select(0, indices).float()


def _distribution_metrics(raw: Mapping[str, Any], *, category: str, module_type: str) -> dict[str, Any]:
    """从可加充分统计恢复 mean/std/RMS/fraction，并保留 raw fields 供 DDP。"""

    output = dict(raw)
    for source in ("input", "output"):
        finite_count = int(raw.get(f"_{source}_finite_count") or 0)
        sample_count = int(raw.get(f"_{source}_sample_count") or 0)
        value_sum = float(raw.get(f"_{source}_value_sum") or 0.0)
        square_sum = float(raw.get(f"_{source}_square_sum") or 0.0)
        mean = value_sum / finite_count if finite_count else None
        variance = max(square_sum / finite_count - float(mean or 0.0) ** 2, 0.0) if finite_count else None
        output.update(
            {
                f"{source}_mean": mean,
                f"{source}_rms": math.sqrt(square_sum / finite_count) if finite_count else None,
                f"{source}_std": math.sqrt(variance) if variance is not None else None,
                f"{source}_abs_max": raw.get(f"_{source}_abs_max"),
                f"{source}_zero_fraction": (
                    float(raw.get(f"_{source}_zero_count") or 0.0) / finite_count
                    if finite_count
                    else None
                ),
                f"{source}_positive_fraction": (
                    float(raw.get(f"_{source}_positive_count") or 0.0) / finite_count
                    if finite_count
                    else None
                ),
                f"{source}_negative_fraction": (
                    float(raw.get(f"_{source}_negative_count") or 0.0) / finite_count
                    if finite_count
                    else None
                ),
                f"{source}_nonfinite_fraction": (
                    1.0 - finite_count / sample_count if sample_count else None
                ),
            }
        )
    input_rms = output.get("input_rms")
    output_rms = output.get("output_rms")
    input_std = output.get("input_std")
    output_std = output.get("output_std")
    output["output_input_rms_ratio"] = (
        float(output_rms) / max(float(input_rms), _EPS)
        if output_rms is not None and input_rms is not None
        else None
    )
    output["output_input_std_ratio"] = (
        float(output_std) / max(float(input_std), _EPS)
        if output_std is not None and input_std is not None
        else None
    )
    output["output_zero_fraction_delta"] = (
        float(output["output_zero_fraction"]) - float(output["input_zero_fraction"])
        if output.get("output_zero_fraction") is not None and output.get("input_zero_fraction") is not None
        else None
    )
    saturation_count = float(raw.get("_output_saturation_count") or 0.0)
    saturation_total = float(raw.get("_output_saturation_sample_count") or 0.0)
    output["activation_saturation_fraction"] = (
        saturation_count / saturation_total if saturation_total > 0.0 else None
    )
    output["activation_dead_fraction"] = (
        output.get("output_zero_fraction")
        if category == "activation" and module_type in {"ReLU", "ReLU6"}
        else None
    )
    return output


def _record(
    sample: _TensorSample,
    *,
    update: int,
    rank: int,
    objective_raw_sum: float | None,
    backward_objective_sum: float | None,
    objective_normalization_divisor: float | None,
    loss_cap_hit_fraction: float | None,
    objective_trace_status: str,
) -> dict[str, Any]:
    activation = sample.activation
    gradient = sample.gradient
    activation_stats = _sufficient_statistics(activation)
    gradient_stats = _sufficient_statistics(gradient)
    joint = torch.isfinite(activation)
    if gradient is not None:
        joint = joint & torch.isfinite(gradient)
    if gradient is None or not bool(joint.any().item()):
        joint_count = 0
        dot_sum = None
        activation_joint_square_sum = None
        gradient_joint_square_sum = None
        signed_product_sum = None
        abs_product_sum = None
    else:
        left = activation[joint].double()
        right = gradient[joint].double()
        product = left * right
        joint_count = int(left.numel())
        dot_sum = float(product.sum().item())
        activation_joint_square_sum = float(left.square().sum().item())
        gradient_joint_square_sum = float(right.square().sum().item())
        signed_product_sum = dot_sum
        abs_product_sum = float(product.abs().sum().item())
    cosine_denominator = math.sqrt(
        max(float(activation_joint_square_sum or 0.0), 0.0)
        * max(float(gradient_joint_square_sum or 0.0), 0.0)
    )
    cosine = (
        float(dot_sum) / cosine_denominator
        if dot_sum is not None and cosine_denominator > 0.0
        else None
    )
    normalized = (
        float(abs_product_sum) / abs(float(backward_objective_sum))
        if abs_product_sum is not None
        and backward_objective_sum is not None
        and abs(float(backward_objective_sum)) > _EPS
        else None
    )
    sample_numel = int(activation.numel())
    full_element_coverage = sample_numel == int(sample.source_numel)
    estimate_available = (
        abs_product_sum is not None
        and signed_product_sum is not None
        and joint_count == sample_numel
        and sample_numel > 0
    )
    expansion = (
        float(sample.source_numel) / float(sample_numel)
        if estimate_available
        else None
    )
    estimated_abs_sum = (
        float(abs_product_sum) * float(expansion)
        if expansion is not None
        else None
    )
    estimated_signed_sum = (
        float(signed_product_sum) * float(expansion)
        if expansion is not None
        else None
    )
    return {
        "update": update,
        "rank": rank,
        "output_path": sample.path,
        "shape": list(sample.shape),
        "num_elements": sample.source_numel,
        "sample_num_elements": sample_numel,
        "local_taylor_element_coverage": (
            sample_numel / int(sample.source_numel)
            if sample.source_numel > 0
            else None
        ),
        "activation_count": activation_stats["count"],
        "activation_finite_count": activation_stats["finite_count"],
        "activation_sum": activation_stats["sum"],
        "activation_square_sum": activation_stats["square_sum"],
        "activation_zero_count": activation_stats["zero_count"],
        "activation_nonfinite_count": activation_stats["nonfinite_count"],
        "activation_abs_max": activation_stats["abs_max"],
        "activation_grad_count": gradient_stats["count"],
        "activation_grad_finite_count": gradient_stats["finite_count"],
        "activation_grad_sum": gradient_stats["sum"],
        "activation_grad_square_sum": gradient_stats["square_sum"],
        "activation_grad_zero_count": gradient_stats["zero_count"],
        "activation_grad_nonfinite_count": gradient_stats["nonfinite_count"],
        "activation_grad_abs_max": gradient_stats["abs_max"],
        "joint_finite_count": joint_count,
        "dot_sum": dot_sum,
        "activation_joint_square_sum": activation_joint_square_sum,
        "gradient_joint_square_sum": gradient_joint_square_sum,
        "signed_product_sum": signed_product_sum,
        "abs_product_sum": abs_product_sum,
        "activation_rms": _rms(activation_stats),
        "activation_std": _std(activation_stats),
        "activation_zero_fraction": _fraction(
            activation_stats["zero_count"],
            activation_stats["finite_count"],
        ),
        "activation_nonfinite_fraction": _fraction(
            activation_stats["nonfinite_count"],
            activation_stats["count"],
        ),
        "activation_grad_rms": _rms(gradient_stats),
        "activation_grad_zero_fraction": _fraction(
            gradient_stats["zero_count"],
            gradient_stats["finite_count"],
        ),
        "activation_grad_nonfinite_fraction": _fraction(
            gradient_stats["nonfinite_count"],
            gradient_stats["count"],
        ),
        "activation_grad_cosine": cosine,
        "sampled_local_taylor_abs_sum": abs_product_sum,
        "sampled_local_taylor_signed_sum": signed_product_sum,
        "local_taylor_abs_sum": (
            abs_product_sum if full_element_coverage else None
        ),
        "local_taylor_abs_sum_estimate": estimated_abs_sum,
        "local_taylor_abs_mean": (
            float(abs_product_sum) / joint_count
            if abs_product_sum is not None and joint_count > 0
            else None
        ),
        "local_taylor_signed_sum": (
            signed_product_sum if full_element_coverage else None
        ),
        "local_taylor_signed_sum_estimate": estimated_signed_sum,
        "objective_normalized_local_taylor": (
            normalized if full_element_coverage else None
        ),
        "objective_normalized_local_taylor_estimate": (
            estimated_abs_sum / abs(float(backward_objective_sum))
            if estimated_abs_sum is not None
            and backward_objective_sum is not None
            and abs(float(backward_objective_sum)) > _EPS
            else None
        ),
        "local_taylor_sum_status": (
            "exact_full_coverage"
            if full_element_coverage and abs_product_sum is not None
            else "deterministic_expansion_estimate"
            if estimated_abs_sum is not None
            else "unavailable"
        ),
        "objective_raw_sum": objective_raw_sum,
        "backward_objective_sum": backward_objective_sum,
        "objective_normalization_divisor": objective_normalization_divisor,
        "loss_cap_hit_fraction": loss_cap_hit_fraction,
        "objective_trace_status": objective_trace_status,
    }


def _sufficient_statistics(values: torch.Tensor | None) -> dict[str, float | int | None]:
    """把一个有界样本归约成可跨 invocation/rank 精确相加的标量。"""

    if values is None or values.numel() == 0:
        return {
            "count": 0,
            "finite_count": 0,
            "sum": None,
            "square_sum": None,
            "abs_max": None,
            "zero_count": 0,
            "nonfinite_count": 0,
        }
    finite = torch.isfinite(values)
    finite_count = int(finite.sum().item())
    count = int(values.numel())
    if finite_count == 0:
        return {
            "count": count,
            "finite_count": 0,
            "sum": None,
            "square_sum": None,
            "abs_max": None,
            "zero_count": 0,
            "nonfinite_count": count,
        }
    selected = values[finite].double()
    return {
        "count": count,
        "finite_count": finite_count,
        "sum": float(selected.sum().item()),
        "square_sum": float(selected.square().sum().item()),
        "abs_max": float(selected.abs().max().item()),
        "zero_count": int((selected == 0).sum().item()),
        "nonfinite_count": count - finite_count,
    }


def _rms(stats: Mapping[str, float | int | None]) -> float | None:
    finite_count = int(stats.get("finite_count") or 0)
    square_sum = stats.get("square_sum")
    return (
        math.sqrt(max(float(square_sum), 0.0) / finite_count)
        if square_sum is not None and finite_count > 0
        else None
    )


def _std(stats: Mapping[str, float | int | None]) -> float | None:
    finite_count = int(stats.get("finite_count") or 0)
    value_sum = stats.get("sum")
    square_sum = stats.get("square_sum")
    if value_sum is None or square_sum is None or finite_count <= 0:
        return None
    variance = max(
        float(square_sum) / finite_count
        - (float(value_sum) / finite_count) ** 2,
        0.0,
    )
    return math.sqrt(variance)


def _fraction(numerator: float | int | None, denominator: float | int | None) -> float | None:
    denominator_value = int(denominator or 0)
    return (
        float(numerator or 0) / denominator_value
        if denominator_value > 0
        else None
    )


__all__ = [
    "DistributionObservation",
    "DistributionTap",
    "ModuleDiagnosticsProbe",
    "capture_distribution_observations",
    "capture_normalization_state",
    "discover_distribution_taps",
    "finalize_normalization_state",
    "merge_distribution_summaries",
    "summarize_distribution_observations",
]
