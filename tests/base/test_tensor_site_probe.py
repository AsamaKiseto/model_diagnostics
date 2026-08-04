"""
验证 functional Tensor site 的显式探针、local Taylor 与任务无关聚合。
"""

from __future__ import annotations

import pytest
import torch

from model_diagnostics.base.interventions import (
    InterventionSpec,
    TensorSite,
    apply_tensor_intervention,
    transform_module_output,
)
from model_diagnostics.base.measurements.tensor_site import (
    TensorSiteProbe,
    aggregate_tensor_site_records,
)


def test_functional_tensor_site_matches_autograd_reference() -> None:
    source = torch.tensor([-1.0, 0.5, 2.0], requires_grad=True)
    hidden = torch.sin(source)
    storage_pointer = hidden.data_ptr()
    probe = TensorSiteProbe.from_tensor(
        site=TensorSite(path="encoder.functional_hidden"),
        invocation_index=0,
        microbatch_id="attempt-m0000",
        tensor=hidden,
        with_grad=True,
    )
    assert probe is not None
    assert hidden.data_ptr() == storage_pointer

    objective = hidden.sum()
    probe.bind_amp_scale(1.0)
    objective.backward()
    raw = probe.finalize(
        update=3,
        rank=0,
        objective_summary={
            "objective_raw_sum": float(objective.detach()),
            "backward_objective_sum": float(objective.detach()),
            "objective_normalization_divisor": 1.0,
            "loss_cap_hit_fraction": 0.0,
            "objective_trace_status": "complete",
        },
        objective_provenance={
            "objective_reduction": "scalar_identity",
            "objective_shape": [],
            "cotangent_identity": None,
            "cotangent_norm": None,
        },
    )
    rows = aggregate_tensor_site_records(
        raw,
        rank_objectives={
            0: {
                "objective_raw_sum": float(objective.detach()),
                "backward_objective_sum": float(objective.detach()),
                "objective_normalization_divisor": 1.0,
                "loss_cap_hit_fraction": 0.0,
                "objective_trace_status": "complete",
                "objective_identities": ["objective"],
            }
        },
        world_size=1,
        update=3,
    )
    assert len(rows) == 1
    row = rows[0]
    expected = float(hidden.detach().abs().sum())
    assert row["local_taylor_abs_sum"] == pytest.approx(expected)
    assert row["activation_source"] == "explicit_functional_tensor_site"
    assert row["conductance_computed"] is False
    assert row["integrated_gradients_computed"] is False
    assert row["activation_patching_computed"] is False


def test_unobserved_functional_local_is_not_claimed() -> None:
    source = torch.tensor([1.0, 2.0], requires_grad=True)
    hidden = torch.nn.functional.gelu(source)
    hidden.sum().backward()
    rows = aggregate_tensor_site_records(
        [],
        rank_objectives={},
        world_size=1,
        update=0,
    )
    assert rows == []


def test_explicit_tensor_site_supports_all_base_intervention_methods() -> None:
    """Tensor intervention 必须复用 Base 数值语义且不解析宿主容器。"""

    site = TensorSite(path="feature.hidden")
    original = torch.tensor([1.0, -2.0], requires_grad=True)
    identity = apply_tensor_intervention(
        original,
        InterventionSpec(method="identity", tensor_site=site),
    )
    assert identity is original

    scaled = apply_tensor_intervention(
        original,
        InterventionSpec(
            method="output_scale",
            tensor_site=site,
            scale=0.5,
        ),
    )
    scaled.sum().backward()
    torch.testing.assert_close(original.grad, torch.full_like(original, 0.5))

    for method, replacement, expected in (
        ("detach", None, original.detach()),
        ("constant_patch", 3.0, torch.full_like(original, 3.0)),
        ("mean_patch", torch.zeros_like(original), torch.zeros_like(original)),
        (
            "explicit_donor_patch",
            torch.tensor([-4.0, 5.0]),
            torch.tensor([-4.0, 5.0]),
        ),
    ):
        original.grad = None
        transformed = apply_tensor_intervention(
            original,
            InterventionSpec(
                method=method,
                tensor_site=site,
                replacement=replacement,
            ),
        )
        torch.testing.assert_close(transformed, expected)
        transformed.sum().backward()
        torch.testing.assert_close(original.grad, torch.zeros_like(original))


def test_constant_patch_broadcasts_across_structured_module_output() -> None:
    first = torch.tensor([1.0, 2.0], requires_grad=True)
    second = torch.tensor([3.0], requires_grad=True)

    transformed, tensor_count = transform_module_output(
        {"tokens": (first, {"second": second}), "metadata": "kept"},
        method="constant_patch",
        scale=1.0,
        replacement=0.0,
    )

    assert tensor_count == 2
    torch.testing.assert_close(transformed["tokens"][0], torch.zeros_like(first))
    torch.testing.assert_close(
        transformed["tokens"][1]["second"],
        torch.zeros_like(second),
    )
    assert transformed["metadata"] == "kept"
