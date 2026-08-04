"""提供显式 scenario、rollout trace、cohort 与局部稳定性度量。"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import statistics
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


_EPS = 1e-12

ANALYZER_DEFINITIONS = (
    AnalyzerDefinition(
        name="final_rollout",
        evidence_kind="rollout_stability_and_cohort_comparison",
        execution_mode="aggregate",
        option_keys=frozenset(),
        definition_version=1,
        claim_boundaries=(
            "scenario horizons must be supplied explicitly",
            "fixed-complete and available cohorts are reported separately",
            "rollout association does not establish physical causality",
        ),
    ),
)
ANALYZER_CATALOG = compose_catalogs(ANALYZER_DEFINITIONS)

REPORT_RENDERER_DEFINITIONS = (
    EvidenceRendererDefinition(
        evidence_kind="rollout_stability_and_cohort_comparison",
        title="最终模型自由滚动稳定性",
        analyzer_definition_version=1,
        metrics=(
            EvidenceMetric(
                "rmse",
                "RMSE",
                visualization="scatter",
                category="rollout",
                priority="P0",
                description="给定 scenario、condition、response component 和 cohort 上的 rollout RMSE。",
                reading="固定 cohort 与 horizon 比较条件、response component 或 checkpoint。",
                reference="越小只表示该定义下误差较小；跨 scale 不直接比较。",
                invalid_when="cohort、horizon、validity 或 response scale 不一致。",
            ),
            EvidenceMetric(
                "srmse",
                "sRMSE",
                visualization="scatter",
                category="rollout",
                priority="P0",
                description="使用显式训练尺度标准化的 rollout RMSE。",
                reading="在相同 normalization contract 下比较 response/condition/checkpoint。",
                reference="没有跨任务统一阈值；需与 free/null/control 条件对照。",
                invalid_when="normalization scale 无效或 fixed/available cohort 被混合。",
            ),
            EvidenceMetric(
                "rolling_local_slope_max",
                "max local slope",
                visualization="scatter",
                category="rollout",
                priority="P0",
                description="rolling window 局部误差斜率的最大值。",
                reading="定位局部最快误差增长条件，并回看对应 timestep 区间。",
                reference="没有通用阈值；需与 identity/free baseline 和误差尺度对照。",
                invalid_when="window 不一致、单点噪声主导或有效支持不足。",
            ),
            EvidenceMetric(
                "unstable_episode_count",
                "unstable episodes",
                role="context",
                visualization="bar",
                category="rollout",
                priority="P0",
                description="按显式 instability rule 合并后的不稳定 episode 数。",
                reading="结合 episode start/end/max growth 和 threshold 定义查看。",
                reference="0 只表示该阈值规则未触发，不证明闭环稳定。",
                invalid_when="threshold/window 未声明或 cohort/horizon 不一致。",
            ),
            EvidenceMetric(
                "time_to_threshold_median",
                "time to threshold",
                role="context",
                visualization="scatter",
                category="rollout",
                priority="P0",
                description="误差首次越过显式阈值的 timestep 中位数。",
                reading="对已达到阈值的相同 fixed cohort 比较条件。",
                reference="越晚只表示在该阈值下越晚越界；未越界需单独显示 censored。",
                invalid_when="阈值不同、censoring 未显示或 available cohort 混入。",
            ),
            EvidenceMetric(
                "timestep_mean_absolute_error",
                "mean |error|",
                value_kind="series",
                visualization="series",
                category="rollout",
                priority="P0",
                description="每个 rollout timestep 的 mean absolute error。",
                reading="按 scenario/condition/response component 分组画完整曲线，并显示逐时 support。",
                reference="只在 fixed cohort、validity 与 scale 对齐时比较。",
                invalid_when="不同 cohort 被逐时 survivor filtering 或 support 隐藏。",
            ),
            EvidenceMetric(
                "timestep_q90_absolute_error",
                "q90 |error|",
                role="control",
                value_kind="series",
                visualization="series",
                category="rollout",
                priority="P0",
                description="每个 timestep 的 absolute error 90% 分位数。",
                reading="与 mean 同图观察尾部误差增长、局部尖峰和条件间差异。",
                reference="没有通用阈值；需与 mean 和 support 联合查看。",
                invalid_when="样本数不足以稳定估计 q90 或 cohort 随时间变化。",
            ),
        ),
    ),
)
REPORT_RENDERER_CATALOG = compose_renderer_catalogs(
    REPORT_RENDERER_DEFINITIONS
)


@dataclass(frozen=True, slots=True)
class ScenarioSpec:
    """定义一个由宿主命名且显式给出 horizon 的 rollout scenario。"""

    scenario_id: str
    horizon: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        scenario_id = str(self.scenario_id).strip()
        if not scenario_id:
            raise ValueError("scenario_id must not be empty")
        if int(self.horizon) < 1:
            raise ValueError("scenario horizon must be positive")
        object.__setattr__(self, "scenario_id", scenario_id)
        object.__setattr__(self, "horizon", int(self.horizon))
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )


@dataclass(frozen=True, slots=True)
class RolloutTrace:
    """保存一个 item/condition/response component 的一维 residual trace。

    每个 output dimension 由宿主映射为独立 ``response_component_id``；extension
    不读取或复制任务数据类型。``valid_mask`` 与 nonfinite residual 都会影响
    cohort eligibility。已经停止计算的 group bootstrap 与 boundary jump 不再进入
    DTO，避免保存不会参与任何诊断结论的任务事实。
    """

    item_id: str
    condition_id: str
    response_component_id: str
    errors: torch.Tensor
    valid_mask: torch.Tensor | None = None
    normalization_scale: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        identifiers = (
            str(self.item_id).strip(),
            str(self.condition_id).strip(),
            str(self.response_component_id).strip(),
        )
        if any(not identifier for identifier in identifiers):
            raise ValueError(
                "item_id, condition_id, and response_component_id "
                "must not be empty"
            )
        errors = torch.as_tensor(self.errors).detach().to(
            dtype=torch.float64,
            device="cpu",
        )
        if errors.ndim != 1 or not errors.numel():
            raise ValueError("RolloutTrace errors must have shape [time]")
        if self.valid_mask is None:
            valid = torch.ones_like(errors, dtype=torch.bool)
        else:
            valid = torch.as_tensor(self.valid_mask).detach().to(
                dtype=torch.bool,
                device="cpu",
            )
            if valid.shape != errors.shape:
                raise ValueError("valid_mask must match errors")
        valid &= torch.isfinite(errors)
        scale = (
            None
            if self.normalization_scale is None
            else float(self.normalization_scale)
        )
        if scale is not None and (
            not math.isfinite(scale) or scale <= 0.0
        ):
            raise ValueError(
                "normalization_scale must be finite and positive"
            )
        object.__setattr__(self, "item_id", identifiers[0])
        object.__setattr__(self, "condition_id", identifiers[1])
        object.__setattr__(
            self,
            "response_component_id",
            identifiers[2],
        )
        object.__setattr__(self, "errors", errors.clone())
        object.__setattr__(self, "valid_mask", valid.clone())
        object.__setattr__(self, "normalization_scale", scale)
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )


def stability_metrics(
    errors: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    normalization_scales: torch.Tensor | Sequence[float] | float | None = None,
) -> dict[str, Any]:
    """计算跨 item 的核心误差曲线、最大局部 slope 与不稳定 episode。

    输入 shape 固定为 ``[items, time]``。完整 mean/q90 曲线承担主要趋势证据；
    不再重复计算不会进入报告的 AUC、中位 slope、边界 jump 或 bootstrap 派生量。
    """

    values = torch.as_tensor(errors).detach().to(
        dtype=torch.float64,
        device="cpu",
    )
    if values.ndim != 2 or not values.shape[0] or not values.shape[1]:
        raise ValueError("errors must have shape [items, time]")
    if valid_mask is None:
        valid = torch.ones_like(values, dtype=torch.bool)
    else:
        valid = torch.as_tensor(valid_mask).detach().to(
            dtype=torch.bool,
            device="cpu",
        )
        if valid.shape != values.shape:
            raise ValueError("valid_mask must match errors")
    valid &= torch.isfinite(values)
    window = max(2, min(20, int(values.shape[1])))
    threshold = 1.0
    if not bool(valid.any()):
        return {
            "status": "insufficient_evidence",
            "skip_reason": "no_finite_valid_error",
            "item_count": int(values.shape[0]),
            "observed_horizon": int(values.shape[1]),
        }
    absolute = values.abs()
    timestep_mean, timestep_q90 = _timestep_summaries(
        absolute,
        valid,
    )
    rmse = float(torch.sqrt(values[valid].square().mean()).item())
    normalized = _normalize_errors(
        values,
        valid,
        normalization_scales,
    )
    slopes = _rolling_slopes(absolute, valid, window)
    episodes, times_to_threshold = _unstable_episodes(
        absolute,
        valid,
        threshold,
    )
    finite_times = [
        time for time in times_to_threshold if time is not None
    ]
    return {
        "status": "success",
        "skip_reason": None,
        "item_count": int(values.shape[0]),
        "observed_horizon": int(values.shape[1]),
        "valid_element_count": int(valid.sum().item()),
        "rmse": rmse,
        "srmse": (
            float(
                torch.sqrt(
                    normalized[torch.isfinite(normalized)].square().mean()
                ).item()
            )
            if normalized is not None
            else None
        ),
        "timestep_mean_absolute_error": timestep_mean,
        "timestep_q90_absolute_error": timestep_q90,
        "local_slope_window": window,
        "rolling_local_slope_max": max(slopes) if slopes else None,
        "unstable_episode_count": len(episodes),
        "time_to_threshold_median": (
            statistics.median(finite_times) if finite_times else None
        ),
        "instability_threshold": threshold,
        "endpoint_slope_is_sole_stability_metric": False,
        "physical_causality_claimed": False,
    }


def cohort_membership(
    traces: Sequence[RolloutTrace],
    scenarios: Sequence[ScenarioSpec],
    *,
    condition_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """建立最大 horizon 上全 condition/response 共享的 fixed-complete cohort。

    每个 ``RolloutTrace`` 必须是调用方执行最大 horizon 后得到的完整 trace；较短
    scenario 只消费其 prefix。主 fixed cohort 在最大 horizon 上同时要求所有显式
    condition 与 response 可用，避免不同 horizon、condition 或 output 使用不同 survivor。
    available cohort 仍按 scenario/condition/response 单独报告，供 selection sensitivity
    审计，不能替代主 fixed cohort。
    """

    index = _trace_index(traces)
    scenario_list = _validate_scenarios(scenarios)
    conditions = (
        sorted({trace.condition_id for trace in traces})
        if condition_ids is None
        else list(dict.fromkeys(str(value) for value in condition_ids))
    )
    if not conditions or any(not condition.strip() for condition in conditions):
        raise ValueError(
            "condition_ids must select at least one non-empty condition"
        )
    response_components = sorted(
        {trace.response_component_id for trace in traces}
    )
    candidate_items = sorted({trace.item_id for trace in traces})
    fixed_horizon = max(scenario.horizon for scenario in scenario_list)
    fixed_items: list[str] = []
    fixed_exclusion_counts: dict[str, int] = {}
    for item_id in candidate_items:
        reasons = [
            reason
            for response_component in response_components
            for condition in conditions
            if (
                reason := _ineligibility_reason(
                    index.get(
                        (item_id, condition, response_component)
                    ),
                    fixed_horizon,
                )
            )
            is not None
        ]
        if not reasons:
            fixed_items.append(item_id)
            continue
        for reason in set(reasons):
            fixed_exclusion_counts[reason] = (
                fixed_exclusion_counts.get(reason, 0) + 1
            )

    rows: list[dict[str, Any]] = []
    for scenario in scenario_list:
        for response_component in response_components:
            available: dict[str, list[str]] = {}
            exclusion_counts: dict[str, dict[str, int]] = {}
            for condition in conditions:
                eligible: list[str] = []
                reasons: dict[str, int] = {}
                for item_id in candidate_items:
                    trace = index.get(
                        (item_id, condition, response_component)
                    )
                    reason = _ineligibility_reason(trace, scenario.horizon)
                    if reason is None:
                        eligible.append(item_id)
                    else:
                        reasons[reason] = reasons.get(reason, 0) + 1
                available[condition] = eligible
                exclusion_counts[condition] = reasons
            rows.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "horizon": scenario.horizon,
                    "response_component_id": response_component,
                    "condition_ids": conditions,
                    "candidate_item_count": len(candidate_items),
                    "fixed_complete_item_ids": fixed_items,
                    "fixed_complete_item_count": len(fixed_items),
                    "fixed_complete_horizon": fixed_horizon,
                    "fixed_complete_condition_ids": conditions,
                    "fixed_complete_response_component_ids": (
                        response_components
                    ),
                    "fixed_complete_exclusion_reason_counts": (
                        fixed_exclusion_counts
                    ),
                    "available_item_ids_by_condition": available,
                    "available_item_count_by_condition": {
                        condition: len(item_ids)
                        for condition, item_ids in available.items()
                    },
                    "exclusion_reason_count_by_condition": exclusion_counts,
                }
            )
    return rows


def summarize_rollout_traces(
    traces: Sequence[RolloutTrace],
    scenarios: Sequence[ScenarioSpec],
) -> list[dict[str, Any]]:
    """分别汇总 fixed-complete 与 available cohort，禁止支持集混用。"""

    index = _trace_index(traces)
    membership_rows = cohort_membership(
        traces,
        scenarios,
    )
    results: list[dict[str, Any]] = []
    for membership in membership_rows:
        scenario_id = str(membership["scenario_id"])
        horizon = int(membership["horizon"])
        response_component_id = str(
            membership["response_component_id"]
        )
        fixed_items = list(membership["fixed_complete_item_ids"])
        for condition in membership["condition_ids"]:
            available_items = list(
                membership["available_item_ids_by_condition"][condition]
            )
            for policy, item_ids in (
                ("fixed_complete", fixed_items),
                ("available", available_items),
            ):
                selected = [
                    index[
                        (item_id, condition, response_component_id)
                    ]
                    for item_id in item_ids
                ]
                if not selected:
                    results.append(
                        {
                            "scenario_id": scenario_id,
                            "horizon": horizon,
                            "condition_id": condition,
                            "response_component_id": response_component_id,
                            "cohort_policy": policy,
                            "status": "insufficient_evidence",
                            "skip_reason": "empty_eligible_cohort",
                            "item_count": 0,
                            "candidate_item_count": membership[
                                "candidate_item_count"
                            ],
                            "fixed_complete_item_count": membership[
                                "fixed_complete_item_count"
                            ],
                            "fixed_complete_horizon": membership[
                                "fixed_complete_horizon"
                            ],
                            "exclusion_reason_count_by_condition": membership[
                                "exclusion_reason_count_by_condition"
                            ],
                            "fixed_complete_exclusion_reason_counts": membership[
                                "fixed_complete_exclusion_reason_counts"
                            ],
                        }
                    )
                    continue
                metrics = _trace_subset_metrics(
                    selected,
                    horizon=horizon,
                )
                results.append(
                    {
                        "scenario_id": scenario_id,
                        "horizon": horizon,
                        "condition_id": condition,
                        "response_component_id": response_component_id,
                        "cohort_policy": policy,
                        "cohort_item_ids": item_ids,
                        "candidate_item_count": membership[
                            "candidate_item_count"
                        ],
                        "fixed_complete_item_count": membership[
                            "fixed_complete_item_count"
                        ],
                        "fixed_complete_horizon": membership[
                            "fixed_complete_horizon"
                        ],
                        "fixed_complete_condition_ids": membership[
                            "fixed_complete_condition_ids"
                        ],
                        "fixed_complete_response_component_ids": membership[
                            "fixed_complete_response_component_ids"
                        ],
                        "exclusion_reason_count_by_condition": membership[
                            "exclusion_reason_count_by_condition"
                        ],
                        "fixed_complete_exclusion_reason_counts": membership[
                            "fixed_complete_exclusion_reason_counts"
                        ],
                        **metrics,
                    }
                )
    return results


def _timestep_summaries(
    values: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[list[float | None], list[float | None]]:
    means: list[float | None] = []
    q90s: list[float | None] = []
    for time_index in range(values.shape[1]):
        observed = values[:, time_index][valid[:, time_index]]
        if observed.numel():
            means.append(float(observed.mean().item()))
            q90s.append(float(torch.quantile(observed, 0.9).item()))
        else:
            means.append(None)
            q90s.append(None)
    return means, q90s


def _normalize_errors(
    errors: torch.Tensor,
    valid: torch.Tensor,
    scales: torch.Tensor | Sequence[float] | float | None,
) -> torch.Tensor | None:
    if scales is None:
        return None
    normalized_scales = torch.as_tensor(scales).detach().to(
        dtype=torch.float64,
        device="cpu",
    )
    if normalized_scales.ndim == 0:
        normalized_scales = normalized_scales.repeat(errors.shape[0])
    if normalized_scales.shape != (errors.shape[0],):
        raise ValueError(
            "normalization_scales must be scalar or match the item axis"
        )
    if (
        not bool(torch.isfinite(normalized_scales).all())
        or bool((normalized_scales <= 0.0).any())
    ):
        raise ValueError(
            "normalization_scales must be finite and positive"
        )
    normalized = errors / normalized_scales[:, None]
    return torch.where(
        valid,
        normalized,
        torch.full_like(normalized, math.nan),
    )


def _rolling_slopes(
    absolute: torch.Tensor,
    valid: torch.Tensor,
    window: int,
) -> list[float]:
    x = torch.arange(window, dtype=torch.float64)
    centered_x = x - x.mean()
    denominator = float(centered_x.square().sum().item())
    slopes: list[float] = []
    for item_index in range(absolute.shape[0]):
        for start in range(0, absolute.shape[1] - window + 1):
            selected = slice(start, start + window)
            if bool(valid[item_index, selected].all()):
                y = absolute[item_index, selected]
                slopes.append(
                    float(
                        torch.dot(centered_x, y - y.mean()).item()
                        / max(denominator, _EPS)
                    )
                )
    return slopes


def _unstable_episodes(
    absolute: torch.Tensor,
    valid: torch.Tensor,
    threshold: float,
) -> tuple[list[dict[str, Any]], list[int | None]]:
    episodes: list[dict[str, Any]] = []
    first_times: list[int | None] = []
    for item_index in range(absolute.shape[0]):
        active_start: int | None = None
        first_time: int | None = None
        for time_index in range(absolute.shape[1] + 1):
            active = (
                time_index < absolute.shape[1]
                and bool(valid[item_index, time_index])
                and float(absolute[item_index, time_index].item()) >= threshold
            )
            if active and active_start is None:
                active_start = time_index
                if first_time is None:
                    first_time = time_index
            if not active and active_start is not None:
                end = time_index - 1
                segment = absolute[item_index, active_start : end + 1]
                episodes.append(
                    {
                        "item_index": item_index,
                        "start_time_index": active_start,
                        "end_time_index": end,
                        "duration": end - active_start + 1,
                        "max_absolute_error": float(segment.max().item()),
                        "end_minus_start_growth": float(
                            segment[-1].item() - segment[0].item()
                        ),
                    }
                )
                active_start = None
        first_times.append(first_time)
    return episodes, first_times


def _trace_index(
    traces: Sequence[RolloutTrace],
) -> dict[tuple[str, str, str], RolloutTrace]:
    index: dict[tuple[str, str, str], RolloutTrace] = {}
    for trace in traces:
        if not isinstance(trace, RolloutTrace):
            raise TypeError("traces must contain RolloutTrace values")
        key = (
            trace.item_id,
            trace.condition_id,
            trace.response_component_id,
        )
        if key in index:
            raise ValueError(f"duplicate RolloutTrace identity: {key}")
        index[key] = trace
    if not index:
        raise ValueError("traces must not be empty")
    return index


def _validate_scenarios(
    scenarios: Sequence[ScenarioSpec],
) -> list[ScenarioSpec]:
    result: list[ScenarioSpec] = []
    seen: set[str] = set()
    for scenario in scenarios:
        if not isinstance(scenario, ScenarioSpec):
            raise TypeError("scenarios must contain ScenarioSpec values")
        if scenario.scenario_id in seen:
            raise ValueError(
                f"duplicate scenario_id: {scenario.scenario_id}"
            )
        seen.add(scenario.scenario_id)
        result.append(scenario)
    if not result:
        raise ValueError("scenarios must not be empty")
    return result


def _ineligibility_reason(
    trace: RolloutTrace | None,
    horizon: int,
) -> str | None:
    if trace is None:
        return "missing_trace"
    if trace.errors.numel() < horizon:
        return "insufficient_horizon"
    if not bool(trace.valid_mask[:horizon].all()):
        return "invalid_or_nonfinite_prefix"
    return None


def _trace_subset_metrics(
    traces: Sequence[RolloutTrace],
    *,
    horizon: int,
) -> dict[str, Any]:
    """按共同 horizon 堆叠同一 response 的 traces 并计算核心稳定性指标。"""

    errors = torch.stack([trace.errors[:horizon] for trace in traces])
    valid = torch.stack([trace.valid_mask[:horizon] for trace in traces])
    scales = (
        [trace.normalization_scale for trace in traces]
        if all(trace.normalization_scale is not None for trace in traces)
        else None
    )
    return stability_metrics(
        errors,
        valid_mask=valid,
        normalization_scales=scales,
    )


__all__ = [
    "ANALYZER_CATALOG",
    "ANALYZER_DEFINITIONS",
    "REPORT_RENDERER_CATALOG",
    "REPORT_RENDERER_DEFINITIONS",
    "RolloutTrace",
    "ScenarioSpec",
    "cohort_membership",
    "stability_metrics",
    "summarize_rollout_traces",
]
