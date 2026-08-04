"""计算 checkpoint-conditioned 分支的梯度、输出与参数变化标量。

阶段: post-training diagnostics metrics。该文件只在内存中短暂保存 detached CPU
gradient/parameter snapshot，用于 baseline 与 intervention 的精确 dot/norm 比较；写盘层
只接收聚合标量。activation 与 local Taylor 数值口径遵循 package 共享 contract。
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Iterable, Mapping

import torch

from ..interventions import ModuleSite


_EPS = 1e-12


@dataclass(slots=True)
class GradientSnapshot:
    """当前分支完整但临时的 detached CPU 参数梯度。"""

    tensors: dict[str, torch.Tensor]

    @classmethod
    def capture(cls, model: torch.nn.Module) -> "GradientSnapshot":
        """在 accumulation 完成、unscale 后、clip 前捕获当前梯度。"""

        return cls(
            {
                name: parameter.grad.detach().to(device="cpu", dtype=torch.float32).clone()
                for name, parameter in model.named_parameters()
                if parameter.grad is not None
            }
        )


@dataclass(slots=True)
class ParameterSnapshot:
    """一个 probe step 开始前的 detached CPU 参数值。"""

    tensors: dict[str, torch.Tensor]

    @classmethod
    def capture(cls, model: torch.nn.Module) -> "ParameterSnapshot":
        """捕获 trainable parameters；registered buffers 由 branch base state 管理。"""

        return cls(
            {
                name: parameter.detach().to(device="cpu", dtype=torch.float32).clone()
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            }
        )


@dataclass(slots=True)
class GradientSketch:
    """用于多 probe step 流式比较的 bounded deterministic CountSketch。"""

    values: torch.Tensor
    source_numel: int


def hierarchy_nodes(
    model_name: str,
    module_sites: Iterable[ModuleSite],
) -> list[dict[str, str]]:
    """生成用于参数聚合的 root、model 与 module inclusive scopes。"""

    nodes = [
        {
            "node_id": "all",
            "parent_id": "",
            "hierarchy_level": "Root",
            "module_path": "",
            "model_name": model_name,
        },
        {
            "node_id": f"model:{model_name}",
            "parent_id": "all",
            "hierarchy_level": "Model",
            "module_path": "",
            "model_name": model_name,
        },
    ]
    seen = {node["node_id"] for node in nodes}
    for site in module_sites:
        node_id = site.node_id or site.site_id
        if node_id in seen:
            continue
        seen.add(node_id)
        nodes.append(
            {
                "node_id": node_id,
                "parent_id": site.parent_node_id or f"model:{model_name}",
                "hierarchy_level": str(
                    site.metadata.get("hierarchy_level", "Module")
                ),
                "module_path": site.module_path,
                "model_name": model_name,
            }
        )
    return nodes


def aggregate_gradients(
    model: torch.nn.Module,
    snapshot: GradientSnapshot,
    nodes: Iterable[Mapping[str, str]],
) -> list[dict[str, Any]]:
    """按 hierarchy node 计算 inclusive 参数与梯度健康统计。"""

    named_parameters = dict(model.named_parameters())
    all_grad_energy = sum(
        float(torch.square(values[torch.isfinite(values)]).sum().item())
        for values in snapshot.tensors.values()
    )
    records: list[dict[str, Any]] = []
    for node in nodes:
        names = [
            name
            for name in named_parameters
            if parameter_name_in_module_path(
                name,
                str(node.get("module_path", "")),
            )
        ]
        parameter_numel = sum(int(named_parameters[name].numel()) for name in names)
        parameter_tensor_count = len(names)
        grad_names = [name for name in names if name in snapshot.tensors]
        grad_numel = sum(int(snapshot.tensors[name].numel()) for name in grad_names)
        grad_energy = 0.0
        grad_abs_max = 0.0
        zero_count = 0
        nonfinite_count = 0
        for name in grad_names:
            values = snapshot.tensors[name]
            finite = torch.isfinite(values)
            finite_values = values[finite]
            nonfinite_count += int((~finite).sum().item())
            zero_count += int((finite_values == 0).sum().item())
            if int(finite_values.numel()) > 0:
                grad_energy += float(torch.square(finite_values).sum().item())
                grad_abs_max = max(grad_abs_max, float(finite_values.abs().max().item()))
        parameter_energy = sum(
            float(torch.square(named_parameters[name].detach().float()).sum().item()) for name in names
        )
        grad_l2 = math.sqrt(max(grad_energy, 0.0))
        parameter_rms = math.sqrt(parameter_energy / max(parameter_numel, 1))
        grad_rms = math.sqrt(grad_energy / max(grad_numel, 1)) if grad_numel else None
        records.append(
            {
                **dict(node),
                "parameter_tensor_count": parameter_tensor_count,
                "parameter_numel": parameter_numel,
                "gradient_tensor_count": len(grad_names),
                "no_grad_parameter_tensor_count": (
                    parameter_tensor_count - len(grad_names)
                ),
                "grad_l2": grad_l2,
                "grad_rms": grad_rms,
                "grad_abs_max": grad_abs_max if grad_numel else None,
                "grad_zero_fraction": zero_count / max(grad_numel, 1) if grad_numel else None,
                "grad_nonfinite_fraction": nonfinite_count / max(grad_numel, 1) if grad_numel else None,
                "parameter_rms": parameter_rms,
                "relative_grad_rms": (
                    grad_rms / parameter_rms
                    if grad_rms is not None and parameter_rms > _EPS
                    else None
                ),
                "gradient_energy_share": grad_energy / max(all_grad_energy, _EPS),
            }
        )
    return records


def compare_gradients(
    baseline: GradientSnapshot,
    intervention: GradientSnapshot,
    nodes: Iterable[Mapping[str, str]],
) -> list[dict[str, Any]]:
    """按参数元素汇总 dot/norm；不会平均各 tensor 的 cosine。"""

    records: list[dict[str, Any]] = []
    for node in nodes:
        module_path = str(node.get("module_path", ""))
        names = sorted(
            name
            for name in baseline.tensors.keys() | intervention.tensors.keys()
            if parameter_name_in_module_path(name, module_path)
        )
        dot = baseline_energy = intervention_energy = delta_energy = 0.0
        compared_numel = 0
        for name in names:
            left = baseline.tensors.get(name)
            right = intervention.tensors.get(name)
            if left is None and right is not None:
                left = torch.zeros_like(right)
            if right is None and left is not None:
                right = torch.zeros_like(left)
            if left is None or right is None or left.shape != right.shape:
                continue
            finite = torch.isfinite(left) & torch.isfinite(right)
            left_finite = left[finite]
            right_finite = right[finite]
            if int(left_finite.numel()) == 0:
                continue
            delta = right_finite - left_finite
            compared_numel += int(left_finite.numel())
            dot += float(torch.dot(left_finite.reshape(-1), right_finite.reshape(-1)).item())
            baseline_energy += float(torch.square(left_finite).sum().item())
            intervention_energy += float(torch.square(right_finite).sum().item())
            delta_energy += float(torch.square(delta).sum().item())
        baseline_l2 = math.sqrt(max(baseline_energy, 0.0))
        intervention_l2 = math.sqrt(max(intervention_energy, 0.0))
        denominator = baseline_l2 * intervention_l2
        records.append(
            {
                **dict(node),
                "compared_gradient_numel": compared_numel,
                "baseline_grad_l2": baseline_l2,
                "intervention_grad_l2": intervention_l2,
                "grad_delta_l2": math.sqrt(max(delta_energy, 0.0)),
                "grad_relative_delta": math.sqrt(max(delta_energy, 0.0)) / max(baseline_l2, _EPS),
                "gradient_dot": dot,
                "gradient_cosine": dot / denominator if denominator > _EPS else None,
            }
        )
    return records


def build_gradient_sketches(
    snapshot: GradientSnapshot,
    nodes: Iterable[Mapping[str, str]],
    *,
    sketch_size: int = 512,
) -> dict[str, GradientSketch]:
    """为每个 hierarchy node 构造 bounded sketch，不保留第二份完整梯度。"""

    size = max(int(sketch_size), 32)
    sketches: dict[str, GradientSketch] = {}
    for node in nodes:
        module_path = str(node.get("module_path", ""))
        values = torch.zeros(size, dtype=torch.float32)
        source_numel = 0
        for name, tensor in snapshot.tensors.items():
            if not parameter_name_in_module_path(name, module_path):
                continue
            flat = tensor.reshape(-1)
            finite = torch.where(torch.isfinite(flat), flat, torch.zeros_like(flat))
            positions = torch.arange(int(flat.numel()), dtype=torch.int64)
            digest = hashlib.sha256(name.encode("utf-8")).digest()
            offset = int.from_bytes(digest[:8], "little") % size
            stride = (int.from_bytes(digest[8:16], "little") | 1) % size
            if stride == 0:
                stride = 1
            buckets = (positions * stride + offset) % size
            signs = torch.where(
                ((positions + int(digest[16])) % 2) == 0,
                torch.ones_like(finite),
                -torch.ones_like(finite),
            )
            values.scatter_add_(0, buckets, finite * signs)
            source_numel += int(flat.numel())
        sketches[str(node.get("node_id"))] = GradientSketch(values=values, source_numel=source_numel)
    return sketches


def compare_gradient_sketches(
    baseline: Mapping[str, GradientSketch],
    intervention: Mapping[str, GradientSketch],
    nodes: Iterable[Mapping[str, str]],
) -> list[dict[str, Any]]:
    """用同一 deterministic projection 估计多步 gradient dot/norm。"""

    records: list[dict[str, Any]] = []
    for node in nodes:
        node_id = str(node.get("node_id"))
        left = baseline.get(node_id)
        right = intervention.get(node_id)
        if left is None or right is None:
            continue
        left_l2 = float(torch.linalg.vector_norm(left.values).item())
        right_l2 = float(torch.linalg.vector_norm(right.values).item())
        delta_l2 = float(torch.linalg.vector_norm(right.values - left.values).item())
        dot = float(torch.dot(left.values, right.values).item())
        denominator = left_l2 * right_l2
        records.append(
            {
                **dict(node),
                "compared_gradient_numel": min(left.source_numel, right.source_numel),
                "baseline_grad_l2": left_l2,
                "intervention_grad_l2": right_l2,
                "grad_delta_l2": delta_l2,
                "grad_relative_delta": delta_l2 / max(left_l2, _EPS),
                "gradient_dot": dot,
                "gradient_cosine": dot / denominator if denominator > _EPS else None,
                "gradient_comparison_mode": "deterministic_count_sketch",
                "gradient_comparison_approximate": True,
                "gradient_sketch_size": int(left.values.numel()),
            }
        )
    return records


def parameter_delta_metrics(
    model: torch.nn.Module,
    before: ParameterSnapshot,
    nodes: Iterable[Mapping[str, str]],
) -> list[dict[str, Any]]:
    """计算 optimizer step 对每个 inclusive hierarchy node 的参数变化。"""

    current = dict(model.named_parameters())
    records: list[dict[str, Any]] = []
    for node in nodes:
        module_path = str(node.get("module_path", ""))
        names = [
            name
            for name in before.tensors
            if parameter_name_in_module_path(name, module_path)
            and name in current
        ]
        delta_energy = parameter_energy = 0.0
        numel = 0
        for name in names:
            previous = before.tensors[name]
            value = current[name].detach().to(device="cpu", dtype=torch.float32)
            delta = value - previous
            delta_energy += float(torch.square(delta).sum().item())
            parameter_energy += float(torch.square(previous).sum().item())
            numel += int(previous.numel())
        delta_l2 = math.sqrt(max(delta_energy, 0.0))
        parameter_l2 = math.sqrt(max(parameter_energy, 0.0))
        records.append(
            {
                **dict(node),
                "parameter_delta_l2": delta_l2,
                "parameter_delta_rms": math.sqrt(delta_energy / max(numel, 1)) if numel else 0.0,
                "update_to_parameter_ratio": delta_l2 / max(parameter_l2, _EPS),
            }
        )
    return records


def compare_outputs(
    baseline: Mapping[str, torch.Tensor],
    intervention: Mapping[str, torch.Tensor],
) -> list[dict[str, Any]]:
    """按 adapter 声明的输出 role 计算 L2、相对变化与 cosine。"""

    records: list[dict[str, Any]] = []
    for role in sorted(baseline.keys() & intervention.keys()):
        left = baseline[role].detach().float().reshape(-1)
        right = intervention[role].detach().float().reshape(-1)
        if left.shape != right.shape:
            records.append({"output_role": role, "status": "skipped", "skip_reason": "shape_mismatch"})
            continue
        finite = torch.isfinite(left) & torch.isfinite(right)
        left = left[finite]
        right = right[finite]
        if int(left.numel()) == 0:
            records.append({"output_role": role, "status": "skipped", "skip_reason": "no_finite_values"})
            continue
        delta_l2 = float(torch.linalg.vector_norm(right - left).item())
        left_l2 = float(torch.linalg.vector_norm(left).item())
        right_l2 = float(torch.linalg.vector_norm(right).item())
        denominator = left_l2 * right_l2
        records.append(
            {
                "output_role": role,
                "status": "success",
                "output_delta_l2": delta_l2,
                "output_relative_delta": delta_l2 / max(left_l2, _EPS),
                "output_cosine": float(torch.dot(left, right).item()) / denominator if denominator > _EPS else None,
            }
        )
    return records


def scalar_terms(terms: Mapping[str, torch.Tensor]) -> dict[str, float | None]:
    """把 loss terms 转成 JSON-safe finite scalars。"""

    result: dict[str, float | None] = {}
    for name, value in terms.items():
        scalar = float(value.detach().float().cpu().item())
        result[str(name)] = scalar if math.isfinite(scalar) else None
    return result


def parameter_name_in_module_path(
    parameter_name: str,
    module_path: str,
) -> bool:
    """判断参数名是否落在 inclusive module parameter scope 内。"""

    path = str(module_path).strip(".")
    if not path:
        return True
    return parameter_name == path or parameter_name.startswith(f"{path}.")


__all__ = [
    "GradientSnapshot",
    "GradientSketch",
    "ParameterSnapshot",
    "aggregate_gradients",
    "compare_gradients",
    "build_gradient_sketches",
    "compare_gradient_sketches",
    "compare_outputs",
    "hierarchy_nodes",
    "parameter_name_in_module_path",
    "parameter_delta_metrics",
    "scalar_terms",
]
