"""收口 portable 逐 component 证据的 inventory、配对与标量化契约。

本模块只处理 Host Runtime 已物化的任务中立 DTO。它不执行 forward、不解释任务
payload，也不定义 loss；各 analyzer 因而共享同一套 output inventory 校验和
``baseline → condition`` effect 口径，而不会各自复制一套成对分支数学。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math
from typing import Any

import torch

from model_diagnostics.base.artifacts import stable_json_hash
from model_diagnostics.host_runtime.contracts import (
    RuntimeOutputObjective,
    RuntimeOutputRef,
    RuntimeOutputResponse,
)


@dataclass(frozen=True, slots=True)
class PairedResponseEffect:
    """保存同一 output response 在 baseline 与 condition 间的规范化差值。"""

    response_component_id: str
    response_id: str
    metric_id: str
    response_normalization: str
    higher_is_better: bool
    baseline_value: float
    condition_value: float
    raw_delta: float
    effect_value: float | None
    normalized_effect: float | None
    baseline_support_count: int
    condition_support_count: int
    support_count: int | None
    status: str
    skip_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        """返回 analyzer row 共用的规范 effect block。"""

        return {
            "response_component_id": self.response_component_id,
            "response_id": self.response_id,
            "metric_id": self.metric_id,
            "response_normalization": self.response_normalization,
            "higher_is_better": self.higher_is_better,
            "baseline_value": self.baseline_value,
            "condition_value": self.condition_value,
            "raw_delta": self.raw_delta,
            "effect_value": self.effect_value,
            "normalized_effect": self.normalized_effect,
            "effect_direction": "positive_means_degradation",
            "normalized_effect_denominator": (
                "max(abs(baseline_value),1e-12)"
            ),
            "baseline_support_count": self.baseline_support_count,
            "condition_support_count": self.condition_support_count,
            "support_count": self.support_count,
            "status": self.status,
            "skip_reason": self.skip_reason,
        }


def output_response_map(
    evaluator: Any,
    *,
    batch: Any,
    unit: Any,
    outputs: Iterable[RuntimeOutputRef],
) -> tuple[
    dict[str, RuntimeOutputResponse],
    dict[str, str],
]:
    """逐 output 评价同一次 objective unit，并隔离无有效 support 的输出。"""

    expected_outputs = tuple(outputs)
    expected_ids = tuple(
        output.component.component_id for output in expected_outputs
    )
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError("output component catalog contains duplicate ids")
    evaluate_available = getattr(
        evaluator,
        "evaluate_available_outputs",
        None,
    )
    if callable(evaluate_available):
        result, unavailable = evaluate_available(
            batch=batch,
            unit=unit,
            outputs=expected_outputs,
        )
        responses = tuple(result)
        by_id = {
            response.output.component.component_id: response
            for response in responses
        }
        if (
            any(
                not isinstance(item, RuntimeOutputResponse)
                for item in responses
            )
            or len(by_id) != len(responses)
            or not set(by_id).issubset(expected_ids)
            or set(by_id) & set(unavailable)
            or set(by_id) | set(unavailable) != set(expected_ids)
        ):
            raise ValueError(
                "available output response inventory does not match catalog"
            )
        for response in responses:
            scalar_tensor_value(response.value)
        return by_id, {
            str(output_id): str(reason)
            for output_id, reason in unavailable.items()
        }
    responses: dict[str, RuntimeOutputResponse] = {}
    unavailable: dict[str, str] = {}
    for output, output_id in zip(expected_outputs, expected_ids):
        try:
            result = tuple(
                evaluator.evaluate_outputs(
                    batch=batch,
                    unit=unit,
                    outputs=(output,),
                )
            )
        except RuntimeError as error:
            code = getattr(error, "code", None)
            if code is None:
                raise
            unavailable[output_id] = str(code)
            continue
        if (
            len(result) != 1
            or not isinstance(result[0], RuntimeOutputResponse)
            or result[0].output.component.component_id != output_id
        ):
            raise ValueError(
                "output evaluation response does not match requested component"
            )
        scalar_tensor_value(result[0].value)
        responses[output_id] = result[0]
    return responses, unavailable


def output_objective_map(
    capability: Any,
    *,
    unit: Any,
    outputs: Iterable[RuntimeOutputRef],
) -> dict[str, RuntimeOutputObjective]:
    """从 actual ledger 读取完整且有序的逐输出 objective inventory。"""

    expected_outputs = tuple(outputs)
    expected_ids = tuple(
        output.component.component_id for output in expected_outputs
    )
    objectives = tuple(
        capability.objective_slices(
            unit=unit,
            outputs=expected_outputs,
        )
    )
    if any(
        not isinstance(item, RuntimeOutputObjective)
        for item in objectives
    ):
        raise TypeError(
            "multi_objective must return RuntimeOutputObjective values"
        )
    by_id = {
        objective.output.component.component_id: objective
        for objective in objectives
    }
    if len(by_id) != len(objectives) or tuple(by_id) != expected_ids:
        raise ValueError(
            "output objective inventory does not match component catalog"
        )
    for objective in objectives:
        scalar_tensor_value(objective.value)
    return by_id


def pair_output_responses(
    baseline: RuntimeOutputResponse,
    condition: RuntimeOutputResponse,
) -> PairedResponseEffect:
    """验证成对 response contract，并统一 effect 符号与 support 语义。"""

    baseline_component_id = baseline.output.component.component_id
    condition_component_id = condition.output.component.component_id
    if baseline_component_id != condition_component_id:
        raise ValueError(
            "paired output responses refer to different components"
        )
    if (
        baseline.response_id != condition.response_id
        or baseline.metric_id != condition.metric_id
        or baseline.normalization != condition.normalization
        or baseline.higher_is_better != condition.higher_is_better
    ):
        raise ValueError(
            "output response contract changed between paired branches"
        )
    baseline_value = scalar_tensor_value(baseline.value)
    condition_value = scalar_tensor_value(condition.value)
    raw_delta = condition_value - baseline_value
    effect_value = (
        raw_delta if not condition.higher_is_better else -raw_delta
    )
    support_matches = (
        baseline.support_count == condition.support_count
    )
    return PairedResponseEffect(
        response_component_id=baseline_component_id,
        response_id=condition.response_id,
        metric_id=condition.metric_id,
        response_normalization=condition.normalization,
        higher_is_better=condition.higher_is_better,
        baseline_value=baseline_value,
        condition_value=condition_value,
        raw_delta=raw_delta,
        effect_value=effect_value if support_matches else None,
        normalized_effect=(
            effect_value / max(abs(baseline_value), 1e-12)
            if support_matches
            else None
        ),
        baseline_support_count=baseline.support_count,
        condition_support_count=condition.support_count,
        support_count=(
            condition.support_count if support_matches else None
        ),
        status="success" if support_matches else "insufficient_evidence",
        skip_reason=(
            None
            if support_matches
            else "output_response_support_mismatch"
        ),
    )


def scalar_tensor_value(value: torch.Tensor) -> float:
    """读取 finite scalar；非有限值不能进入 component evidence。"""

    if not torch.is_tensor(value) or value.numel() != 1:
        raise TypeError("component response value must be a scalar tensor")
    scalar = float(value.detach().to(dtype=torch.float64).cpu().item())
    if not math.isfinite(scalar):
        raise ValueError("component response value must be finite")
    return scalar


def pair_evidence_key(
    evidence_kind: str,
    *,
    sample_id: str,
    condition_id: str,
    intervened_component_id: str,
    response_component_id: str,
) -> str:
    """构造不含测量值与显示名称的稳定 pair evidence identity。"""

    return stable_json_hash(
        {
            "evidence_kind": str(evidence_kind),
            "sample_id": str(sample_id),
            "condition_id": str(condition_id),
            "intervened_component_id": str(intervened_component_id),
            "response_component_id": str(response_component_id),
        }
    )


__all__ = [
    "PairedResponseEffect",
    "output_objective_map",
    "output_response_map",
    "pair_evidence_key",
    "pair_output_responses",
    "scalar_tensor_value",
]
