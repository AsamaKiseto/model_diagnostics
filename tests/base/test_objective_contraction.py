"""
验证 scalar objective 与显式 vector-Jacobian cotangent 的严格收缩口径。
"""

from __future__ import annotations

import pytest
import torch

from model_diagnostics.base.runtime.objective import contract_backward_objective


def test_scalar_objective_preserves_identity() -> None:
    objective = torch.tensor(3.5, requires_grad=True)
    contraction = contract_backward_objective(
        objective,
        cotangent=None,
        cotangent_identity=None,
    )
    assert contraction.reduction == "scalar_identity"
    assert contraction.objective_shape == ()
    assert contraction.value.item() == pytest.approx(3.5)
    assert not contraction.value.requires_grad


def test_vector_objective_matches_explicit_vjp_scalar() -> None:
    features = torch.tensor([1.0, -2.0, 3.0], requires_grad=True)
    objective = features.square()
    cotangent = torch.tensor([0.5, -1.0, 2.0])
    contraction = contract_backward_objective(
        objective,
        cotangent=cotangent,
        cotangent_identity="signed_weights_v1",
    )
    expected = torch.sum(objective.detach() * cotangent)
    torch.testing.assert_close(contraction.value, expected)
    assert contraction.reduction == "explicit_vector_jacobian_product"
    assert contraction.cotangent_identity == "signed_weights_v1"
    assert contraction.cotangent_norm is not None

    objective.backward(cotangent)
    torch.testing.assert_close(features.grad, 2.0 * features.detach() * cotangent)


@pytest.mark.parametrize(
    ("cotangent", "identity", "message"),
    [
        (None, None, "requires an explicit cotangent"),
        (torch.ones(2), "weights", "shape must exactly match"),
        (torch.ones(3), None, "cotangent_identity"),
    ],
)
def test_vector_objective_rejects_ambiguous_cotangent(
    cotangent: torch.Tensor | None,
    identity: str | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        contract_backward_objective(
            torch.ones(3),
            cotangent=cotangent,
            cotangent_identity=identity,
        )
