"""
验证 scalar/vector backward objective，并生成 local-Taylor 共用的标量口径。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class ObjectiveContraction:
    """保存 backward objective 的 detached 标量收缩及其可核验 provenance。"""

    value: torch.Tensor
    objective_shape: tuple[int, ...]
    objective_numel: int
    reduction: str
    cotangent_identity: str | None
    cotangent_norm: torch.Tensor | None


def contract_backward_objective(
    backward_objective: torch.Tensor,
    *,
    cotangent: torch.Tensor | None,
    cotangent_identity: str | None,
) -> ObjectiveContraction:
    """把实际 VJP 口径收缩成 denominator scalar，但不参与或替代训练 backward。

    非标量 objective 必须提供同 shape、同 device、同 dtype 的显式 cotangent 和稳定
    identity。严格拒绝 broadcasting，避免 artifact 声明的 objective 与调用方真实 VJP
    口径不一致。返回 Tensor 均已 detach，不保留训练计算图。
    """

    if not isinstance(backward_objective, torch.Tensor):
        raise TypeError("backward_objective must be a torch.Tensor")
    if backward_objective.is_complex():
        raise TypeError("complex backward_objective is not supported")
    shape = tuple(int(dimension) for dimension in backward_objective.shape)
    numel = int(backward_objective.numel())
    if numel == 1 and cotangent is None:
        return ObjectiveContraction(
            value=backward_objective.detach().reshape(()),
            objective_shape=shape,
            objective_numel=1,
            reduction="scalar_identity",
            cotangent_identity=None,
            cotangent_norm=None,
        )
    if cotangent is None:
        raise ValueError("vector backward_objective requires an explicit cotangent")
    if not isinstance(cotangent_identity, str) or not cotangent_identity.strip():
        raise ValueError(
            "cotangent_identity must be non-empty when cotangent is provided"
        )
    if cotangent.shape != backward_objective.shape:
        raise ValueError(
            "cotangent shape must exactly match backward_objective; broadcasting is forbidden"
        )
    if cotangent.device != backward_objective.device:
        raise ValueError("cotangent and backward_objective must use the same device")
    if cotangent.dtype != backward_objective.dtype:
        raise ValueError("cotangent and backward_objective must use the same dtype")
    if cotangent.is_complex():
        raise TypeError("complex cotangent is not supported")
    detached_cotangent = cotangent.detach()
    return ObjectiveContraction(
        value=torch.sum(
            backward_objective.detach() * detached_cotangent,
        ).reshape(()),
        objective_shape=shape,
        objective_numel=numel,
        reduction="explicit_vector_jacobian_product",
        cotangent_identity=cotangent_identity.strip(),
        cotangent_norm=torch.linalg.vector_norm(
            detached_cotangent.reshape(-1).to(dtype=torch.float64)
        ),
    )


__all__ = ["ObjectiveContraction", "contract_backward_objective"]
