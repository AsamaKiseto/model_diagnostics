"""验证最终 checkpoint 的 objective 分区并计算整体梯度几何。"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import torch

from model_diagnostics.base import (
    AnalyzerDefinition,
    EvidenceMetric,
    EvidenceRendererDefinition,
    compose_catalogs,
    compose_renderer_catalogs,
)
from model_diagnostics.base.measurements import (
    parameter_name_in_module_path,
)


_EPS = 1e-12

ANALYZER_DEFINITIONS = (
    AnalyzerDefinition(
        name="final_objective_conflict",
        evidence_kind="objective_gradient_geometry",
        execution_mode="aggregate",
        definition_version=1,
        claim_boundaries=(
            "gradient geometry is local optimization evidence",
            "gradient conflict does not identify physical causality",
        ),
    ),
)
ANALYZER_CATALOG = compose_catalogs(ANALYZER_DEFINITIONS)

REPORT_RENDERER_DEFINITIONS = (
    EvidenceRendererDefinition(
        evidence_kind="objective_gradient_geometry",
        title="输出目标之间的梯度关系",
        analyzer_definition_version=1,
        metrics=(
            EvidenceMetric(
                "gradient_cosine",
                "目标梯度 cosine",
                visualization="matrix",
                category="multi_objective",
                priority="P0",
                description="一个输出目标与其余目标的参数梯度方向关系。",
                reading="按输出目标查看；负值表示该 checkpoint 上局部方向相反。",
                reference="范围 [-1,1]，不得单独用于归因训练结果。",
                invalid_when="partition 不完整、任一梯度 norm 退化或 objective 与训练不一致。",
            ),
            EvidenceMetric(
                "norm_ratio",
                "目标梯度尺度比",
                visualization="bar",
                category="multi_objective",
                priority="P0",
                description="目标梯度 norm 与其余目标梯度 norm 的比值。",
                reading="查看输出目标之间是否存在明显尺度失衡。",
                reference="1 表示尺度相等，不表示优化最优。",
                invalid_when="分母梯度接近零或 supervision support 不完整。",
            ),
            EvidenceMetric(
                "partition_coverage",
                "目标分区覆盖率",
                role="control",
                visualization="bar",
                category="multi_objective",
                priority="P0",
                description="目标 A、其余目标和 shared terms 对训练 objective 的覆盖率。",
                reading="解释任何梯度图前先核对覆盖率与 residual。",
                reference="完整分区参考值为 1。",
                invalid_when="训练 objective identity 或 ledger 不可靠。",
            ),
            EvidenceMetric(
                "joint_active_parameter_fraction",
                "共同活跃参数比例",
                role="control",
                visualization="matrix",
                category="multi_objective",
                priority="P0",
                description="目标 A 和其余目标同时具有有效梯度的参数元素比例。",
                reading="判断梯度几何是否建立在足够大的共同参数支持上。",
                reference="越接近 1 表示两个目标共同作用的参数比例越高，不表示无冲突。",
                invalid_when="参数 owner、no-grad 处理或梯度有效性口径不一致。",
            ),
        ),
        status_fields=("status",),
    ),
)
REPORT_RENDERER_CATALOG = compose_renderer_catalogs(
    REPORT_RENDERER_DEFINITIONS
)


@dataclass(frozen=True, slots=True)
class ObjectivePartition:
    """保存 actual training objective 的 A / complement / shared 分区。"""

    training_total: torch.Tensor
    objective_a: torch.Tensor
    objective_b: torch.Tensor
    shared_terms: Mapping[str, torch.Tensor] = field(default_factory=dict)
    support_count: float = 0.0
    partition_coverage: float = 1.0
    normalization: str = "unspecified"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, value in (
            ("training_total", self.training_total),
            ("objective_a", self.objective_a),
            ("objective_b", self.objective_b),
        ):
            if not torch.is_tensor(value) or value.numel() != 1:
                raise TypeError(f"{name} must be a scalar tensor")
        shared = dict(self.shared_terms)
        if any(
            not torch.is_tensor(value) or value.numel() != 1
            for value in shared.values()
        ):
            raise TypeError("shared_terms values must be scalar tensors")
        if not math.isfinite(float(self.support_count)):
            raise ValueError("support_count must be finite")
        coverage = float(self.partition_coverage)
        if not math.isfinite(coverage) or not 0.0 <= coverage <= 1.0:
            raise ValueError("partition_coverage must be within [0,1]")
        object.__setattr__(self, "shared_terms", MappingProxyType(shared))
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )


def validate_partition(
    partition: ObjectivePartition,
) -> dict[str, Any]:
    """按固定 `1e-6` 相对容差验证 actual training objective 分区。"""

    shared = sum(
        partition.shared_terms.values(),
        start=partition.training_total.new_zeros(()),
    )
    reconstructed = (
        partition.objective_a + partition.objective_b + shared
    )
    residual = float(
        (partition.training_total - reconstructed)
        .detach()
        .abs()
        .cpu()
        .item()
    )
    total = float(
        partition.training_total.detach().abs().cpu().item()
    )
    relative_tolerance = 1e-6
    tolerance = relative_tolerance * (1.0 + total)
    finite = all(
        bool(torch.isfinite(value).all().item())
        for value in (
            partition.training_total,
            partition.objective_a,
            partition.objective_b,
            *partition.shared_terms.values(),
        )
    )
    status = (
        "success"
        if finite
        and residual <= tolerance
        and partition.partition_coverage >= 1.0 - relative_tolerance
        else "insufficient_evidence"
    )
    return {
        "status": status,
        "skip_reason": (
            None
            if status == "success"
            else "objective_partition_contract_failed"
        ),
        "partition_residual": residual,
        "partition_tolerance": tolerance,
        "partition_coverage": float(partition.partition_coverage),
        "support_count": float(partition.support_count),
        "normalization": partition.normalization,
        "shared_term_count": len(partition.shared_terms),
    }


def partition_scope_gradient_rows(
    partition: ObjectivePartition,
    model: torch.nn.Module,
    nodes: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    """按固定严格口径计算 A 与 complement 的整体/显式 scope 梯度。"""

    validation = validate_partition(partition)
    if validation["status"] != "success":
        return [
            {
                **validation,
                "scope_mode": "partition",
                "node_id": "training_objective",
                "hierarchy_level": None,
                "gradient_numel": 0,
            }
        ]
    named_parameters = tuple(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    if not named_parameters:
        return [
            {
                **validation,
                "status": "insufficient_evidence",
                "skip_reason": "no_trainable_parameters",
                "scope_mode": "model",
                "node_id": "model",
                "gradient_numel": 0,
            }
        ]
    parameters = tuple(parameter for _name, parameter in named_parameters)
    gradients_a = torch.autograd.grad(
        partition.objective_a,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    gradients_b = torch.autograd.grad(
        partition.objective_b,
        parameters,
        # final analyzer 会在同一次真实 objective graph 上遍历全部输出，避免
        # 为每个输出重复 forward；graph 在样本级 loop 结束后统一释放。
        retain_graph=True,
        allow_unused=True,
    )
    scopes = tuple(nodes) or (
        {
            "node_id": "model",
            "hierarchy_level": "Model",
            "module_path": "",
        },
    )
    rows: list[dict[str, Any]] = []
    for node in scopes:
        module_path = str(node.get("module_path", ""))
        dot = energy_a = energy_b = 0.0
        joint_numel = union_numel = 0
        for (
            (name, parameter),
            gradient_a,
            gradient_b,
        ) in zip(
            named_parameters,
            gradients_a,
            gradients_b,
            strict=True,
        ):
            if not parameter_name_in_module_path(name, module_path):
                continue
            union_numel += int(parameter.numel())
            if gradient_a is None or gradient_b is None:
                continue
            finite = torch.isfinite(gradient_a) & torch.isfinite(
                gradient_b
            )
            if not bool(finite.any().item()):
                continue
            left = gradient_a[finite].double()
            right = gradient_b[finite].double()
            joint_numel += int(left.numel())
            dot += float((left * right).sum().detach().cpu().item())
            energy_a += float(left.square().sum().detach().cpu().item())
            energy_b += float(right.square().sum().detach().cpu().item())
        norm_a = math.sqrt(max(energy_a, 0.0))
        norm_b = math.sqrt(max(energy_b, 0.0))
        denominator = norm_a * norm_b
        rows.append(
            {
                **dict(node),
                **validation,
                "scope_mode": "model",
                "gradient_numel": joint_numel,
                "gradient_dot": dot,
                "gradient_cosine": (
                    dot / denominator
                    if denominator > _EPS
                    else None
                ),
                "objective_a_gradient_norm": norm_a,
                "objective_b_gradient_norm": norm_b,
                "norm_ratio": (
                    norm_a / norm_b if norm_b > _EPS else None
                ),
                "negative_dot": dot < 0.0 if joint_numel else None,
                "joint_active_parameter_fraction": (
                    joint_numel / union_numel
                    if union_numel
                    else None
                ),
                "optimizer_transform_computed": False,
                "training_optimizer_writeback": False,
            }
        )
    return rows


__all__ = [
    "ANALYZER_CATALOG",
    "ANALYZER_DEFINITIONS",
    "ObjectivePartition",
    "REPORT_RENDERER_CATALOG",
    "REPORT_RENDERER_DEFINITIONS",
    "partition_scope_gradient_rows",
    "validate_partition",
]
