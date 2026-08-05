"""验证三阶段方案保留的最小 extension 数学与 catalog。"""

from __future__ import annotations

import pytest
import torch

from model_diagnostics.base.reporting import build_metric_guide_catalog
from model_diagnostics.extensions import build_report_renderer_catalog
from model_diagnostics.extensions.input_dependence import (
    ANALYZER_CATALOG as INPUT_CATALOG,
    symmetric_relative_output_response,
)
from model_diagnostics.extensions.multi_objective import (
    ANALYZER_CATALOG as OBJECTIVE_CATALOG,
    ObjectivePartition,
    partition_scope_gradient_rows,
)
from model_diagnostics.extensions.rollout import (
    ANALYZER_CATALOG as ROLLOUT_CATALOG,
    RolloutTrace,
    ScenarioSpec,
    summarize_rollout_traces,
)


def test_retained_catalog_only_exposes_final_analyzers() -> None:
    assert tuple(INPUT_CATALOG.definitions) == (
        "final_channel_influence",
        "final_input_sensitivity",
    )
    assert tuple(OBJECTIVE_CATALOG.definitions) == (
        "final_objective_conflict",
    )
    assert tuple(ROLLOUT_CATALOG.definitions) == ("final_rollout",)
    kinds = build_report_renderer_catalog().definitions
    assert "predictive_dependence_intervention" in kinds
    assert "predictive_input_sensitivity" in kinds
    assert "objective_gradient_geometry" in kinds
    assert "rollout_stability_and_cohort_comparison" in kinds
    paths = {
        guide["path"]
        for guide in build_metric_guide_catalog(
            build_report_renderer_catalog()
        )
    }
    assert "joint_active_parameter_fraction" in paths
    assert "effective_supervision_coverage" not in paths
    assert "symmetric_relative_output_response" in paths


def test_symmetric_relative_output_response_uses_output_self_scale() -> None:
    assert symmetric_relative_output_response(
        baseline_output_square_sum=2.0,
        condition_output_square_sum=8.0,
        output_difference_square_sum=2.0,
        support_count=2,
    ) == pytest.approx(2.0 / 3.0)
    assert symmetric_relative_output_response(
        baseline_output_square_sum=0.0,
        condition_output_square_sum=0.0,
        output_difference_square_sum=0.0,
        support_count=2,
    ) == 0.0


def test_final_objective_conflict_reports_opposite_gradients() -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    weight = next(model.parameters())
    weight.data.zero_()
    objective_a = (weight.reshape(()) - 1.0).square()
    objective_b = (weight.reshape(()) + 1.0).square()
    rows = partition_scope_gradient_rows(
        ObjectivePartition(
            training_total=objective_a + objective_b,
            objective_a=objective_a,
            objective_b=objective_b,
            support_count=1.0,
            partition_coverage=1.0,
            normalization="unit",
        ),
        model,
    )
    assert rows[0]["status"] == "success"
    assert rows[0]["gradient_cosine"] == -1.0
    assert rows[0]["negative_dot"] is True
    assert "effective_supervision_coverage" not in rows[0]
    assert rows[0]["optimizer_transform_computed"] is False


def test_final_rollout_reuses_trace_prefix_with_core_stability_metrics() -> None:
    traces = (
        RolloutTrace(
            item_id="sample-1",
            condition_id="free",
            response_component_id="output-0",
            errors=torch.tensor([0.1, 0.2, 0.3, 0.4]),
        ),
        RolloutTrace(
            item_id="sample-2",
            condition_id="free",
            response_component_id="output-0",
            errors=torch.tensor([0.2, 0.3, 0.4, 0.5]),
        ),
    )
    rows = summarize_rollout_traces(
        traces,
        (
            ScenarioSpec("short", 2),
            ScenarioSpec("long", 4),
        ),
    )
    assert {
        (row["scenario_id"], row["cohort_policy"])
        for row in rows
    } == {
        ("short", "fixed_complete"),
        ("short", "available"),
        ("long", "fixed_complete"),
        ("long", "available"),
    }
    assert all(
        len(row["timestep_mean_absolute_error"]) == row["horizon"]
        and len(row["timestep_q90_absolute_error"]) == row["horizon"]
        and "available_item_count_by_condition" not in row
        for row in rows
    )
