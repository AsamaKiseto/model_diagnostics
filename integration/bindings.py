"""
组合 portable 三阶段 checkpoint analyzer 与 Host Runtime capability。

批量 checkpoint 只运行训练健康 sweep；输入、模块、目标冲突和 rollout 影响只在
final-selected 阶段运行。这里不复制宿主 objective、rollout 或数据语义。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass
import math
import time
from types import MappingProxyType
from typing import Any

import torch

from model_diagnostics.base.artifacts import stable_json_hash
from model_diagnostics.base.checkpoint import (
    capture_canonical_model_state,
    paired_branch_rng,
    restore_canonical_model_state,
)
from model_diagnostics.base.interventions import preserve_module_modes
from model_diagnostics.base.measurements import (
    GradientSnapshot,
    aggregate_gradients,
    capture_distribution_observations,
    capture_normalization_state,
    discover_distribution_taps,
    finalize_normalization_state,
    summarize_distribution_observations,
)
from model_diagnostics.base.registry import (
    AnalyzerBinding,
    AnalyzerBindingCatalog,
    AnalyzerCatalog,
    AnalyzerDefinition,
    AnalyzerExecutionResult,
    AnalyzerRunContext,
    BASE_ANALYZER_CATALOG,
    compose_bindings,
    compose_catalogs,
)
from model_diagnostics.integration.component_evidence import (
    output_response_map,
    pair_output_responses,
)
from model_diagnostics.host_runtime.composition import HostTaskRuntime


DEFAULT_ANALYZER_CATALOG = BASE_ANALYZER_CATALOG


@dataclass(frozen=True, slots=True)
class CapabilityAnalyzerBinding:
    """把一个 analyzer runner 绑定到显式 runtime capabilities。"""

    binding: AnalyzerBinding
    required_capability_ids: frozenset[str]

    def __post_init__(self) -> None:
        if not isinstance(self.binding, AnalyzerBinding):
            raise TypeError("binding must be an AnalyzerBinding")
        requirements = frozenset(
            str(item).strip()
            for item in self.required_capability_ids
            if str(item).strip()
        )
        if not requirements:
            raise ValueError("required_capability_ids must not be empty")
        object.__setattr__(self, "required_capability_ids", requirements)


@dataclass(frozen=True, slots=True)
class CheckpointBindingPlan:
    """保存一次 composition 的 analyzer definitions 与 runners。"""

    analyzer_catalog: AnalyzerCatalog = DEFAULT_ANALYZER_CATALOG
    capability_bindings: tuple[CapabilityAnalyzerBinding, ...] = ()
    explicit_bindings: tuple[AnalyzerBinding, ...] = ()
    extra_definitions: tuple[AnalyzerDefinition, ...] = ()

    def __post_init__(self) -> None:
        capability_bindings = tuple(self.capability_bindings)
        explicit_bindings = tuple(self.explicit_bindings)
        extra_definitions = tuple(self.extra_definitions)
        names = [
            item.binding.name for item in capability_bindings
        ] + [item.name for item in explicit_bindings]
        if len(set(names)) != len(names):
            raise ValueError("analyzer runner bindings must be unique")
        object.__setattr__(self, "capability_bindings", capability_bindings)
        object.__setattr__(self, "explicit_bindings", explicit_bindings)
        object.__setattr__(self, "extra_definitions", extra_definitions)

    def compose_catalog(self) -> AnalyzerCatalog:
        if not self.extra_definitions:
            return self.analyzer_catalog
        return compose_catalogs(
            self.analyzer_catalog,
            self.extra_definitions,
        )

    def compose_bindings(
        self,
        runtime: HostTaskRuntime,
    ) -> AnalyzerBindingCatalog:
        catalog = self.compose_catalog()
        bindings = list(self.explicit_bindings)
        for item in self.capability_bindings:
            missing = sorted(
                capability_id
                for capability_id in item.required_capability_ids
                if runtime.capabilities.get(capability_id) is None
            )
            if missing:
                raise ValueError(
                    "analyzer requires unavailable runtime capabilities: "
                    f"{missing}"
                )
            if catalog.definition(item.binding.name) != item.binding.definition:
                raise ValueError(
                    "analyzer binding definition does not match catalog"
                )
            bindings.append(item.binding)
        result = compose_bindings(bindings)
        result.validate_against(catalog)
        return result

    @property
    def capability_requirements_by_analyzer(
        self,
    ) -> Mapping[str, tuple[str, ...]]:
        return MappingProxyType(
            {
                item.binding.name: tuple(
                    sorted(item.required_capability_ids)
                )
                for item in self.capability_bindings
            }
        )


def compose_checkpoint_binding_plan(
    runtime: HostTaskRuntime,
    *,
    capability_bindings: Iterable[CapabilityAnalyzerBinding] | None = None,
    explicit_bindings: Iterable[AnalyzerBinding] = (),
    extra_definitions: Iterable[AnalyzerDefinition] = (),
) -> CheckpointBindingPlan:
    """只组合当前三阶段仍保留的 analyzer。"""

    if not isinstance(runtime, HostTaskRuntime):
        raise TypeError("runtime must be a HostTaskRuntime")
    selected = tuple(
        _default_capability_bindings(runtime)
        if capability_bindings is None
        else (
            item
            for item in capability_bindings
            if all(
                runtime.capabilities.get(capability_id) is not None
                for capability_id in item.required_capability_ids
            )
        )
    )
    sweep_definition = BASE_ANALYZER_CATALOG.definition(
        "checkpoint_sweep"
    )
    built_in = AnalyzerBinding(
        definition=sweep_definition,
        runner=_run_checkpoint_sweep,
    )
    return CheckpointBindingPlan(
        analyzer_catalog=compose_catalogs(
            BASE_ANALYZER_CATALOG,
            (item.binding.definition for item in selected),
        ),
        capability_bindings=selected,
        explicit_bindings=(built_in, *tuple(explicit_bindings)),
        extra_definitions=tuple(extra_definitions),
    )


def _default_capability_bindings(
    runtime: HostTaskRuntime,
) -> Iterable[CapabilityAnalyzerBinding]:
    specifications = (
        (
            "model_diagnostics.extensions.input_dependence",
            "final_channel_influence",
            _run_final_channel_influence,
            frozenset(
                {
                    "component_catalog",
                    "input_dependence",
                    "output_evaluation",
                }
            ),
        ),
        (
            "model_diagnostics.extensions.input_dependence",
            "final_input_sensitivity",
            _run_final_input_sensitivity,
            frozenset({"component_catalog", "input_sensitivity"}),
        ),
        (
            "model_diagnostics.extensions.multi_objective",
            "final_objective_conflict",
            _run_final_objective_conflict,
            frozenset({"component_catalog", "multi_objective"}),
        ),
        (
            "model_diagnostics.extensions.rollout",
            "final_rollout",
            _run_final_rollout,
            frozenset({"component_catalog", "rollout"}),
        ),
    )
    for module_name, analyzer_name, runner, requirements in specifications:
        if not all(
            runtime.capabilities.get(name) is not None
            for name in requirements
        ):
            continue
        module = __import__(module_name, fromlist=["ANALYZER_CATALOG"])
        definition = module.ANALYZER_CATALOG.definition(analyzer_name)
        yield CapabilityAnalyzerBinding(
            binding=AnalyzerBinding(
                definition=definition,
                runner=runner,
            ),
            required_capability_ids=requirements,
        )


@dataclass(frozen=True, slots=True)
class _CheckpointNode:
    node_id: str
    model_name: str
    module_path: str
    hierarchy_level: str


def _run_checkpoint_sweep(
    context: AnalyzerRunContext,
) -> AnalyzerExecutionResult:
    """每个样本一次 baseline forward/backward，输出训练健康证据。"""

    if not context.samples:
        return _execution_result(
            "checkpoint_sweep",
            (_insufficient("empty_cohort"),),
        )
    started_at = time.perf_counter()
    adapter = context.adapter
    model = context.loaded.model
    nodes = _checkpoint_parameter_nodes(
        model,
        model_name=context.loaded.model_name,
    )
    taps = discover_distribution_taps(
        {context.loaded.model_name: model},
        nodes,
    )
    observations: dict[str, list[Any]] = {
        tap.tap_id: [] for tap in taps
    }
    rows: list[dict[str, Any]] = []
    model_state = capture_canonical_model_state(model)
    parameter = next(model.parameters(), None)
    if parameter is not None:
        device = parameter.device
    else:
        buffer = next(model.buffers(), None)
        device = buffer.device if buffer is not None else torch.device("cpu")
    sweep_seed = int(
        stable_json_hash(
            {
                "checkpoint_identity": context.checkpoint.identity,
                "analyzer": "checkpoint_sweep",
            }
        )[:16],
        16,
    )
    try:
        model.zero_grad(set_to_none=True)
        with (
            preserve_module_modes(model),
            paired_branch_rng(sweep_seed, device),
            ExitStack() as stack,
        ):
            # 训练健康扫描需要保留 RNN/dropout 等训练分支，否则 cuDNN RNN
            # 无法反向；带 running statistics 的归一化层单独冻结，避免 cohort
            # 顺序改变 checkpoint 状态。canonical state 会在分支后严格恢复。
            model.train()
            for module in model.modules():
                if any(
                    name in module._buffers
                    for name in (
                        "running_mean",
                        "running_var",
                        "num_batches_tracked",
                    )
                ):
                    module.eval()
            for tap in taps:
                input_handle = tap.module.register_forward_pre_hook(
                    _distribution_input_hook(tap, observations)
                )
                stack.callback(input_handle.remove)
                output_handle = tap.module.register_forward_hook(
                    _distribution_output_hook(tap, observations)
                )
                stack.callback(output_handle.remove)
            for sample in context.samples:
                batch = adapter.materialize_sample(
                    context.loaded,
                    sample,
                )
                result = adapter.forward_objective(
                    context.loaded,
                    batch,
                    objective={},
                    precision=adapter.execution_precision,
                    return_aux=True,
                )
                retained_gradient_tensors: set[int] = set()
                for response in result.component_responses:
                    tensor = response.gradient_tensor
                    if (
                        tensor is not None
                        and tensor.requires_grad
                        and id(tensor) not in retained_gradient_tensors
                    ):
                        tensor.retain_grad()
                        retained_gradient_tensors.add(id(tensor))
                backward = result.effective_backward_objective()
                if backward.requires_grad:
                    backward.backward()
                raw = (
                    result.raw_total
                    if result.raw_total is not None
                    else result.total
                )
                raw_value, raw_finite = _observed_scalar(raw)
                backward_value, backward_finite = _observed_scalar(
                    backward
                )
                rows.append(
                    {
                        "status": (
                            "success"
                            if raw_finite and backward_finite
                            else "nonfinite_observed"
                        ),
                        "record_kind": "checkpoint_objective_health",
                        "sample_id": sample.sample_id,
                        "group_id": sample.group_id,
                        "raw_objective": raw_value,
                        "raw_objective_finite": raw_finite,
                        "backward_objective": backward_value,
                        "backward_objective_finite": backward_finite,
                        "objective_identity": result.objective_identity,
                        "loss_cap_applied": bool(
                            result.objective_metadata.get(
                                "loss_cap_applied",
                                False,
                            )
                        ),
                        "normalization_divisor": (
                            result.objective_metadata.get(
                                "normalization_divisor"
                            )
                        ),
                    }
                )
                for response in result.component_responses:
                    gradient_statistics = (
                        _component_output_gradient_statistics(response)
                    )
                    response_value, response_finite = _observed_scalar(
                        response.value
                    )
                    rows.append(
                        {
                            "status": (
                                "success"
                                if response_finite
                                else "nonfinite_observed"
                            ),
                            "record_kind": "checkpoint_output_health",
                            "sample_id": sample.sample_id,
                            "group_id": sample.group_id,
                            "response_component_id": (
                                response.response_component_id
                            ),
                            "metric_id": response.metric_id,
                            "response_value": response_value,
                            "response_value_finite": response_finite,
                            "support_count": response.support_count,
                            "response_normalization": (
                                response.normalization_id
                            ),
                            "higher_is_better": (
                                response.higher_is_better
                            ),
                            **gradient_statistics,
                        }
                    )
            gradient = GradientSnapshot.capture(model)
            for gradient_row in aggregate_gradients(
                model,
                gradient,
                (_node_mapping(node) for node in nodes),
            ):
                rows.append(
                    {
                        "status": "success",
                        "record_kind": "checkpoint_gradient_health",
                        **gradient_row,
                        "gradient_capture_provenance": (
                            "post_backward_pre_clip_checkpoint_baseline"
                        ),
                    }
                )
            for tap in taps:
                row = {
                    "status": "success",
                    "record_kind": "checkpoint_distribution_health",
                    **tap.manifest_record(),
                    **summarize_distribution_observations(
                        observations[tap.tap_id],
                        category=tap.category,
                        module_type=tap.module_type,
                    ),
                }
                if tap.category == "normalization":
                    row.update(
                        finalize_normalization_state(
                            capture_normalization_state(tap.module)
                        )
                    )
                rows.append(row)
    finally:
        restore_canonical_model_state(model, model_state)
        model.zero_grad(set_to_none=True)
    analyzer_seconds = time.perf_counter() - started_at
    load_seconds = float(
        context.execution_metadata.get("checkpoint_load_seconds", 0.0)
    )
    rows.append(
        {
            "status": "success",
            "record_kind": "checkpoint_sweep_runtime",
            "elapsed_seconds": load_seconds + analyzer_seconds,
            "checkpoint_load_seconds": load_seconds,
            "analyzer_seconds": analyzer_seconds,
            "sample_count": len(context.samples),
            "forward_count": len(context.samples),
            "backward_count": len(context.samples),
            "cohort_execution_policy": "fixed_group_stratified_cohort",
            "module_mode_policy": (
                "training_with_running_statistics_frozen"
            ),
        }
    )
    return _execution_result("checkpoint_sweep", rows)


def _component_output_gradient_statistics(
    response: Any,
) -> dict[str, Any]:
    """从同一次 total-objective backward 读取逐输出梯度，不增加反向次数。"""

    tensor = response.gradient_tensor
    gradient = None if tensor is None else tensor.grad
    if (
        gradient is None
        or response.gradient_axis is None
        or response.gradient_index is None
    ):
        return {
            "output_gradient_status": "not_available",
            "output_gradient_rms": None,
            "output_gradient_zero_fraction": None,
            "output_gradient_nonfinite_fraction": None,
        }
    axis = int(response.gradient_axis) % gradient.ndim
    index = int(response.gradient_index)
    if index < 0 or index >= int(gradient.shape[axis]):
        return {
            "output_gradient_status": "invalid_component_coordinate",
            "output_gradient_rms": None,
            "output_gradient_zero_fraction": None,
            "output_gradient_nonfinite_fraction": None,
        }
    selected = gradient.select(axis, index).detach()
    numel = int(selected.numel())
    if numel <= 0:
        return {
            "output_gradient_status": "empty",
            "output_gradient_rms": None,
            "output_gradient_zero_fraction": None,
            "output_gradient_nonfinite_fraction": None,
        }
    finite = torch.isfinite(selected)
    finite_count = int(finite.sum().item())
    finite_values = selected[finite].to(dtype=torch.float64)
    rms = (
        float(finite_values.square().mean().sqrt().cpu().item())
        if finite_count
        else None
    )
    zero_fraction = (
        float((finite_values == 0).sum().item()) / float(finite_count)
        if finite_count
        else None
    )
    return {
        "output_gradient_status": (
            "success" if finite_count == numel else "nonfinite_observed"
        ),
        "output_gradient_rms": rms,
        "output_gradient_zero_fraction": zero_fraction,
        "output_gradient_nonfinite_fraction": (
            float(numel - finite_count) / float(numel)
        ),
    }


def _checkpoint_parameter_nodes(
    model: torch.nn.Module,
    *,
    model_name: str,
) -> tuple[_CheckpointNode, ...]:
    """只统计模型整体和参数叶子层，避免容器重复计数。"""

    nodes = [
        _CheckpointNode(
            node_id=f"model:{model_name}",
            model_name=model_name,
            module_path="",
            hierarchy_level="Model",
        )
    ]
    for path, module in model.named_modules():
        if not path or tuple(module.children()):
            continue
        if not any(
            parameter.requires_grad
            for parameter in module.parameters(recurse=False)
        ):
            continue
        nodes.append(
            _CheckpointNode(
                node_id=f"parameter_leaf:{path}",
                model_name=model_name,
                module_path=path,
                hierarchy_level="ParameterLeaf",
            )
        )
    return tuple(nodes)


def _node_mapping(node: _CheckpointNode) -> dict[str, str]:
    return {
        "node_id": node.node_id,
        "parent_id": "",
        "hierarchy_level": node.hierarchy_level,
        "module_path": node.module_path,
        "model_name": node.model_name,
    }


def _distribution_input_hook(
    tap: Any,
    observations: dict[str, list[Any]],
) -> Any:
    def hook(module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
        observations[tap.tap_id].extend(
            capture_distribution_observations(
                inputs,
                source="input",
                module=module,
            )
        )

    return hook


def _distribution_output_hook(
    tap: Any,
    observations: dict[str, list[Any]],
) -> Any:
    def hook(
        module: torch.nn.Module,
        _inputs: tuple[Any, ...],
        output: Any,
    ) -> None:
        observations[tap.tap_id].extend(
            capture_distribution_observations(
                output,
                source="output",
                module=module,
            )
        )

    return hook


def _run_final_channel_influence(
    context: AnalyzerRunContext,
) -> AnalyzerExecutionResult:
    """以 identity/mean replacement 计算完整 input × output 响应矩阵。

    最终报告只需要 cohort 层面的方向证据。这里在内存中累计充分统计，每个
    input × output × method 只持久化一行，避免把逐样本笛卡尔积写成数百万行。
    """

    capability = _runtime_capability(context, "input_dependence")
    evaluator = _runtime_capability(context, "output_evaluation")
    adapter = context.adapter
    session = adapter.runtime_session(context.loaded)
    summaries: dict[tuple[str, str, str], dict[str, Any]] = {}
    availability: dict[tuple[str, str], dict[str, Any]] = {}
    condition_failures: dict[tuple[str, str, str], dict[str, Any]] = {}
    for sample in context.samples:
        batch = adapter.materialize_sample(context.loaded, sample)
        inputs, outputs, _digest = adapter.component_catalog(
            context.loaded,
            batch,
        )
        input_ids = {
            item.component.component_id for item in inputs
        }
        baseline_unit = adapter.execute_objective_unit(
            context.loaded,
            batch,
            objective={"response_only": True},
            return_aux=False,
        )
        baseline, baseline_unavailable = output_response_map(
            evaluator,
            batch=batch,
            unit=baseline_unit,
            outputs=outputs,
        )
        for output_id, reason in baseline_unavailable.items():
            state = availability.setdefault(
                (output_id, reason),
                {
                    "sample_count": 0,
                    "group_ids": set(),
                },
            )
            state["sample_count"] += 1
            if sample.group_id is not None:
                state["group_ids"].add(sample.group_id)
        for target in capability.list_targets(
            session=session,
            batch=batch,
        ):
            input_id = target.input.component.component_id
            if input_id not in input_ids:
                raise ValueError(
                    "input target is absent from component catalog"
                )
            for method in ("identity", "mean"):
                condition_id = f"input:{input_id}:{method}"
                try:
                    replacement = capability.replace(
                        batch=batch,
                        target=target,
                        method=method,
                    )
                    unit = adapter.execute_objective_unit(
                        context.loaded,
                        replacement.batch,
                        objective={"response_only": True},
                        return_aux=False,
                    )
                    changed, changed_unavailable = output_response_map(
                        evaluator,
                        batch=replacement.batch,
                        unit=unit,
                        outputs=outputs,
                    )
                except RuntimeError as error:
                    if not hasattr(error, "code"):
                        raise
                    failure = condition_failures.setdefault(
                        (input_id, method, str(error)),
                        {
                            "sample_count": 0,
                            "group_ids": set(),
                        },
                    )
                    failure["sample_count"] += 1
                    if sample.group_id is not None:
                        failure["group_ids"].add(sample.group_id)
                    continue
                for output_id, reason in changed_unavailable.items():
                    if output_id not in baseline:
                        continue
                    summary = summaries.setdefault(
                        (input_id, method, output_id),
                        {
                            "sample_count": 0,
                            "group_ids": set(),
                            "unavailable_sample_count": 0,
                            "skip_reasons": set(),
                            "effect_count": 0,
                        },
                    )
                    summary["unavailable_sample_count"] += 1
                    summary["skip_reasons"].add(reason)
                for output_id, baseline_response in baseline.items():
                    if output_id not in changed:
                        continue
                    paired = pair_output_responses(
                        baseline_response,
                        changed[output_id],
                    )
                    paired_row = paired.to_dict()
                    summary = summaries.setdefault(
                        (input_id, method, output_id),
                        {
                            "sample_count": 0,
                            "group_ids": set(),
                            "unavailable_sample_count": 0,
                            "skip_reasons": set(),
                            "effect_count": 0,
                        },
                    )
                    summary["sample_count"] += 1
                    if sample.group_id is not None:
                        summary["group_ids"].add(sample.group_id)
                    summary.setdefault(
                        "template",
                        {
                            **paired_row,
                            "record_kind": "input_output_influence",
                            "intervened_component_id": input_id,
                            "method": method,
                            "input_condition_id": condition_id,
                            "modified_paths": list(
                                replacement.modified_paths
                            ),
                            "affected_value_element_count": (
                                replacement.affected_value_element_count
                            ),
                            "affected_validity_element_count": (
                                replacement.affected_validity_element_count
                            ),
                            "changed_element_count": (
                                replacement.changed_element_count
                            ),
                            "preserved_paths_verified": (
                                replacement.preserved_paths_verified
                            ),
                            "replacement_provenance": dict(
                                replacement.provenance
                            ),
                            "control_status": (
                                "identity_control"
                                if method == "identity"
                                else "intervention"
                            ),
                            "predictive_dependence_only": True,
                            "physical_causality_claimed": False,
                        },
                    )
                    for field in (
                        "baseline_value",
                        "condition_value",
                        "raw_delta",
                        "effect_value",
                        "normalized_effect",
                    ):
                        value = paired_row.get(field)
                        if value is None:
                            continue
                        numeric = float(value)
                        if not math.isfinite(numeric):
                            continue
                        summary[f"{field}_sum"] = (
                            float(summary.get(f"{field}_sum", 0.0))
                            + numeric
                        )
                        summary[f"{field}_count"] = (
                            int(summary.get(f"{field}_count", 0)) + 1
                        )
                        summary[f"{field}_min"] = min(
                            numeric,
                            float(summary.get(f"{field}_min", numeric)),
                        )
                        summary[f"{field}_max"] = max(
                            numeric,
                            float(summary.get(f"{field}_max", numeric)),
                        )
                    summary["effect_count"] += 1
                    for field in (
                        "baseline_support_count",
                        "condition_support_count",
                        "support_count",
                    ):
                        value = paired_row.get(field)
                        if value is not None:
                            summary[field] = int(
                                summary.get(field, 0)
                            ) + int(value)

    rows: list[dict[str, Any]] = []
    for (output_id, reason), state in sorted(availability.items()):
        rows.append(
            {
                **_insufficient(reason),
                "record_kind": "output_response_availability",
                "response_component_id": output_id,
                "evaluation_branch": "baseline",
                "sample_count": state["sample_count"],
                "group_count": len(state["group_ids"]),
            }
        )
    for (input_id, method, reason), state in sorted(
        condition_failures.items()
    ):
        rows.append(
            {
                **_insufficient(reason),
                "record_kind": "input_output_influence",
                "intervened_component_id": input_id,
                "method": method,
                "input_condition_id": f"input:{input_id}:{method}",
                "sample_count": state["sample_count"],
                "group_count": len(state["group_ids"]),
            }
        )
    for (input_id, method, output_id), state in sorted(summaries.items()):
        template = state.get("template")
        if template is None:
            reason = "、".join(sorted(state["skip_reasons"]))
            rows.append(
                {
                    **_insufficient(
                        reason or "output_response_unavailable"
                    ),
                    "record_kind": "input_output_influence",
                    "response_component_id": output_id,
                    "intervened_component_id": input_id,
                    "method": method,
                    "input_condition_id": f"input:{input_id}:{method}",
                    "sample_count": 0,
                    "unavailable_sample_count": (
                        state["unavailable_sample_count"]
                    ),
                }
            )
            continue
        row = dict(template)
        for field in (
            "baseline_value",
            "condition_value",
            "raw_delta",
            "effect_value",
            "normalized_effect",
        ):
            count = int(state.get(f"{field}_count", 0))
            row[field] = (
                float(state[f"{field}_sum"]) / count
                if count
                else None
            )
            row[f"{field}_min"] = state.get(f"{field}_min")
            row[f"{field}_max"] = state.get(f"{field}_max")
        row.update(
            {
                "baseline_support_count": state.get(
                    "baseline_support_count", 0
                ),
                "condition_support_count": state.get(
                    "condition_support_count", 0
                ),
                "support_count": state.get("support_count", 0),
                "sample_count": state["sample_count"],
                "group_count": len(state["group_ids"]),
                "unavailable_sample_count": (
                    state["unavailable_sample_count"]
                ),
                "skip_reasons": sorted(state["skip_reasons"]),
                "aggregation": "cohort_mean_with_min_max",
            }
        )
        rows.append(row)
    return _execution_result(
        "final_channel_influence",
        rows or (_insufficient("empty_cohort"),),
    )


def _run_final_input_sensitivity(
    context: AnalyzerRunContext,
) -> AnalyzerExecutionResult:
    """计算多尺度、正负配对的完整 input × output 相对输出响应。

    每个样本只执行一次未扰动基线。某个 ``input × scale`` 只有在相同确定性
    Rademacher 方向的正负两个分支都成功时才进入 cohort 统计，避免单侧扰动把
    非对称响应误写成尺度敏感度。每个方向先相对于该样本输出自身 RMS 归一化，
    再对方向和样本做等权 RMS；输入差分能量只验证干预执行，不进入指标分母。
    """

    from model_diagnostics.extensions.input_dependence import (
        symmetric_relative_output_response,
    )

    capability = _runtime_capability(context, "input_sensitivity")
    adapter = context.adapter
    summaries: dict[tuple[str, str, float], dict[str, Any]] = {}
    failures: dict[tuple[str, str, float, str], dict[str, Any]] = {}
    scales = tuple(sorted({float(value) for value in capability.scales}))
    if not scales or any(not math.isfinite(value) or value <= 0 for value in scales):
        return _execution_result(
            "final_input_sensitivity",
            (_insufficient("input_sensitivity_scales_are_invalid"),),
        )

    for sample in context.samples:
        batch = adapter.materialize_sample(context.loaded, sample)
        inputs, outputs, _digest = adapter.component_catalog(
            context.loaded,
            batch,
        )
        baseline_unit = adapter.execute_objective_unit(
            context.loaded,
            batch,
            objective={"response_only": True},
            return_aux=False,
        )
        for input_ref in inputs:
            input_id = input_ref.component.component_id
            seed = int(
                stable_json_hash(
                    {
                        "analyzer": "final_input_sensitivity",
                        "sample_id": sample.sample_id,
                        "input_component_id": input_id,
                    }
                )[:16],
                16,
            ) % (2**63 - 1)
            for scale in scales:
                branches: dict[int, tuple[Any, dict[str, Any]]] = {}
                failure_reason: str | None = None
                for direction in (-1, 1):
                    try:
                        perturbation = capability.perturb(
                            batch=batch,
                            input=input_ref,
                            scale=scale,
                            direction=direction,
                            direction_seed=seed,
                        )
                        condition_unit = adapter.execute_objective_unit(
                            context.loaded,
                            perturbation.batch,
                            objective={"response_only": True},
                            return_aux=False,
                        )
                        output_responses = capability.compare_outputs(
                            batch=perturbation.batch,
                            baseline_unit=baseline_unit,
                            condition_unit=condition_unit,
                            outputs=outputs,
                        )
                        branches[direction] = (
                            perturbation,
                            {
                                item.output.component.component_id: item
                                for item in output_responses
                            },
                        )
                    except RuntimeError as error:
                        if not hasattr(error, "code"):
                            raise
                        failure_reason = str(error)
                        break
                if set(branches) != {-1, 1}:
                    reason = failure_reason or "paired_perturbation_incomplete"
                    for output in outputs:
                        state = failures.setdefault(
                            (
                                input_id,
                                output.component.component_id,
                                scale,
                                reason,
                            ),
                            {"sample_count": 0, "group_ids": set()},
                        )
                        state["sample_count"] += 1
                        if sample.group_id is not None:
                            state["group_ids"].add(sample.group_id)
                    continue

                negative, negative_outputs = branches[-1]
                positive, positive_outputs = branches[1]
                paired_output_ids = set(negative_outputs) & set(positive_outputs)
                for output in outputs:
                    output_id = output.component.component_id
                    if output_id in paired_output_ids:
                        continue
                    state = failures.setdefault(
                        (
                            input_id,
                            output_id,
                            scale,
                            "output_response_unavailable",
                        ),
                        {"sample_count": 0, "group_ids": set()},
                    )
                    state["sample_count"] += 1
                    if sample.group_id is not None:
                        state["group_ids"].add(sample.group_id)
                for output_id in paired_output_ids:
                    negative_output = negative_outputs[output_id]
                    positive_output = positive_outputs[output_id]
                    negative_response = symmetric_relative_output_response(
                        baseline_output_square_sum=(
                            negative_output.baseline_output_square_sum
                        ),
                        condition_output_square_sum=(
                            negative_output.condition_output_square_sum
                        ),
                        output_difference_square_sum=(
                            negative_output.output_difference_square_sum
                        ),
                        support_count=negative_output.support_count,
                    )
                    positive_response = symmetric_relative_output_response(
                        baseline_output_square_sum=(
                            positive_output.baseline_output_square_sum
                        ),
                        condition_output_square_sum=(
                            positive_output.condition_output_square_sum
                        ),
                        output_difference_square_sum=(
                            positive_output.output_difference_square_sum
                        ),
                        support_count=positive_output.support_count,
                    )
                    key = (input_id, output_id, scale)
                    state = summaries.setdefault(
                        key,
                        {
                            "paired_response_square_sum": 0.0,
                            "output_support_count": 0,
                            "sample_count": 0,
                            "group_ids": set(),
                            "affected_value_element_count": 0,
                            "modified_paths": set(),
                            "normalizations": set(),
                            "provenance": None,
                        },
                    )
                    state["paired_response_square_sum"] += 0.5 * (
                        negative_response * negative_response
                        + positive_response * positive_response
                    )
                    state["output_support_count"] += (
                        negative_output.support_count
                        + positive_output.support_count
                    )
                    state["sample_count"] += 1
                    if sample.group_id is not None:
                        state["group_ids"].add(sample.group_id)
                    state["affected_value_element_count"] += (
                        negative.affected_value_element_count
                        + positive.affected_value_element_count
                    )
                    state["modified_paths"].update(negative.modified_paths)
                    state["modified_paths"].update(positive.modified_paths)
                    state["normalizations"].update(
                        {
                            negative.normalization,
                            positive.normalization,
                            negative_output.normalization,
                            positive_output.normalization,
                        }
                    )
                    state["provenance"] = dict(negative.provenance)

    rows: list[dict[str, Any]] = []
    for (input_id, output_id, scale, reason), state in sorted(
        failures.items(),
        key=lambda item: (
            item[0][0],
            item[0][1],
            item[0][2],
            item[0][3],
        ),
    ):
        rows.append(
            {
                **_insufficient(reason),
                "record_kind": "input_output_sensitivity",
                "intervened_component_id": input_id,
                "response_component_id": output_id,
                "scale": scale,
                "sample_count": state["sample_count"],
                "group_count": len(state["group_ids"]),
            }
        )
    for (input_id, output_id, scale), state in sorted(summaries.items()):
        response = math.sqrt(
            state["paired_response_square_sum"] / state["sample_count"]
        )
        rows.append(
            {
                "status": "success",
                "record_kind": "input_output_sensitivity",
                "intervened_component_id": input_id,
                "response_component_id": output_id,
                "scale": scale,
                "symmetric_relative_output_response": response,
                "sample_count": state["sample_count"],
                "group_count": len(state["group_ids"]),
                "support_count": state["output_support_count"],
                "direction_count": 2,
                "affected_value_element_count": (
                    state["affected_value_element_count"]
                ),
                "modified_paths": sorted(state["modified_paths"]),
                "normalization_contracts": sorted(
                    state["normalizations"]
                ),
                "perturbation_distribution": "deterministic_rademacher",
                "perturbation_provenance": state["provenance"],
                "aggregation": "sample_equal_paired_direction_rms",
                "predictive_sensitivity_only": True,
                "physical_causality_claimed": False,
            }
        )
    return _execution_result(
        "final_input_sensitivity",
        rows or (_insufficient("empty_cohort"),),
    )


def _run_final_objective_conflict(
    context: AnalyzerRunContext,
) -> AnalyzerExecutionResult:
    """比较每个输出 objective 与其余输出，只计算模型整体梯度几何。"""

    from model_diagnostics.extensions.multi_objective import (
        ObjectivePartition,
        partition_scope_gradient_rows,
    )

    capability = _runtime_capability(context, "multi_objective")
    adapter = context.adapter
    rows: list[dict[str, Any]] = []
    model_node = {
        "node_id": f"model:{context.loaded.model_name}",
        "parent_id": "all",
        "hierarchy_level": "Model",
        "module_path": "",
        "model_name": context.loaded.model_name,
    }
    for sample in context.samples:
        batch = adapter.materialize_sample(context.loaded, sample)
        _inputs, outputs, _digest = adapter.component_catalog(
            context.loaded,
            batch,
        )
        unit = adapter.execute_objective_unit(
            context.loaded,
            batch,
            objective={"collect_objective_ledger": True},
        )
        for output in outputs:
            try:
                facts = capability.partition(
                    unit=unit,
                    output=output,
                )
            except RuntimeError as error:
                if not hasattr(error, "code"):
                    raise
                rows.append(
                    {
                        **_insufficient(str(error)),
                        "record_kind": "objective_gradient_geometry",
                        "sample_id": sample.sample_id,
                        "group_id": sample.group_id,
                        "response_component_id": (
                            output.component.component_id
                        ),
                    }
                )
                continue
            partition = ObjectivePartition(
                training_total=facts.training_total,
                objective_a=facts.objective_a,
                objective_b=facts.objective_b,
                shared_terms=dict(facts.shared_terms),
                support_count=float(facts.support_count),
                partition_coverage=float(facts.partition_coverage),
                normalization=str(facts.normalization),
                metadata=dict(facts.metadata),
            )
            for geometry in partition_scope_gradient_rows(
                partition,
                context.loaded.model,
                (model_node,),
            ):
                rows.append(
                    {
                        **_json_safe(geometry),
                        "record_kind": "objective_gradient_geometry",
                        "sample_id": sample.sample_id,
                        "group_id": sample.group_id,
                        "response_component_id": (
                            output.component.component_id
                        ),
                        "optimizer_writeback": False,
                        "physical_causality_claimed": False,
                    }
                )
    return _execution_result(
        "final_objective_conflict",
        rows or (_insufficient("empty_cohort"),),
    )


def _run_final_rollout(
    context: AnalyzerRunContext,
) -> AnalyzerExecutionResult:
    """最终 checkpoint 只执行最大 horizon free rollout，并复用 prefix。"""

    from model_diagnostics.extensions.rollout import (
        RolloutTrace,
        ScenarioSpec,
        summarize_rollout_traces,
    )

    rollout = _runtime_capability(context, "rollout")
    scenarios = tuple(rollout.configured_scenarios)
    if not scenarios:
        return _execution_result(
            "final_rollout",
            (_insufficient("rollout_scenarios_not_declared"),),
        )
    max_scenario = max(
        scenarios,
        key=lambda item: int(item.horizon),
    )
    adapter = context.adapter
    session = adapter.runtime_session(context.loaded)
    traces: list[RolloutTrace] = []
    rows: list[dict[str, Any]] = []
    for sample in context.samples:
        batch = adapter.materialize_sample(context.loaded, sample)
        execution = rollout.execute(
            session=session,
            batch=batch,
            scenario=max_scenario,
            condition_id="free",
            precision=adapter.execution_precision,
            feedback_policy=None,
            provenance={"condition": "free_feedback"},
        )
        for series in execution.series:
            residual = (
                series.prediction.detach().to(
                    dtype=torch.float64,
                    device="cpu",
                )
                - series.truth.detach().to(
                    dtype=torch.float64,
                    device="cpu",
                )
            )
            valid = series.validity.detach().to(
                dtype=torch.bool,
                device="cpu",
            )
            if residual.ndim < 2:
                residual = residual.reshape(1, -1)
                valid = valid.reshape(1, -1)
            for item_index in range(int(residual.shape[0])):
                item_residual = residual[item_index]
                item_valid = valid[item_index]
                steps = int(item_residual.shape[0])
                flattened = item_residual.reshape(steps, -1)
                flattened_valid = item_valid.reshape(steps, -1)
                counts = flattened_valid.sum(dim=1)
                rms = torch.sqrt(
                    torch.where(
                        flattened_valid,
                        flattened.square(),
                        torch.zeros_like(flattened),
                    ).sum(dim=1)
                    / counts.clamp_min(1)
                )
                traces.append(
                    RolloutTrace(
                        item_id=f"{sample.sample_id}:{item_index}",
                        condition_id="free",
                        response_component_id=(
                            series.output.component.component_id
                        ),
                        errors=rms,
                        valid_mask=counts > 0,
                        normalization_scale=(
                            series.normalization_scale
                        ),
                    )
                )
    scenario_specs = tuple(
        ScenarioSpec(
            scenario_id=item.scenario_id,
            horizon=int(item.horizon),
            metadata=dict(item.metadata),
        )
        for item in scenarios
    )
    if traces:
        summaries = summarize_rollout_traces(
            traces,
            scenario_specs,
        )
        retained_fields = {
            "status",
            "skip_reason",
            "scenario_id",
            "horizon",
            "response_component_id",
            "cohort_policy",
            "cohort_item_ids",
            "candidate_item_count",
            "fixed_complete_item_count",
            "fixed_complete_horizon",
            "exclusion_reason_count_by_condition",
            "fixed_complete_exclusion_reason_counts",
            "item_count",
            "valid_element_count",
            "rmse",
            "srmse",
            "timestep_mean_absolute_error",
            "timestep_q90_absolute_error",
            "local_slope_window",
            "rolling_local_slope_max",
            "unstable_episode_count",
            "instability_threshold",
            "time_to_threshold_median",
        }
        rows.extend(
            {
                "record_kind": "rollout_stability",
                **{
                    key: _json_safe(value)
                    for key, value in summary.items()
                    if key in retained_fields
                },
                "rollout_condition_id": summary.get("condition_id"),
                "physical_causality_claimed": False,
            }
            for summary in summaries
        )
    return _execution_result(
        "final_rollout",
        rows or (_insufficient("rollout_trace_unavailable"),),
    )


def _runtime_capability(
    context: AnalyzerRunContext,
    name: str,
) -> Any:
    capability = context.adapter.runtime.capabilities.get(name)
    if capability is None:
        raise ValueError(f"runtime capability unavailable: {name}")
    return capability


def _insufficient(reason: str) -> dict[str, Any]:
    return {
        "status": "insufficient_evidence",
        "skip_reason": str(reason),
    }


def _json_safe(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
            if not torch.is_tensor(item) or item.numel() <= 1024
        }
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _observed_scalar(value: torch.Tensor) -> tuple[float | None, bool]:
    """异常 sweep 保留非有限状态，不让 NaN/Inf 中断整个 checkpoint。"""

    if not torch.is_tensor(value) or value.numel() != 1:
        raise TypeError("checkpoint health value must be a scalar tensor")
    scalar = float(value.detach().to(dtype=torch.float64).cpu().item())
    return (scalar, True) if math.isfinite(scalar) else (None, False)


def _execution_result(
    analyzer: str,
    rows: Iterable[Mapping[str, Any]],
) -> AnalyzerExecutionResult:
    normalized: list[dict[str, Any]] = []
    for ordinal, raw in enumerate(rows):
        row = dict(raw)
        row["evidence_key"] = str(
            row.get("evidence_key")
            or stable_json_hash(
                {
                    "analyzer": analyzer,
                    "record_kind": row.get("record_kind"),
                    "sample_id": row.get("sample_id"),
                    "group_id": row.get("group_id"),
                    "intervened_component_id": row.get(
                        "intervened_component_id"
                    ),
                    "response_component_id": row.get(
                        "response_component_id"
                    ),
                    "node_id": row.get("node_id"),
                    "tap_id": row.get("tap_id"),
                    "method": row.get("method"),
                    "scale": row.get("scale"),
                    "scenario_id": row.get("scenario_id"),
                    "cohort_policy": row.get("cohort_policy"),
                    "ordinal": ordinal,
                }
            )
        )
        normalized.append(row)
    if not normalized:
        normalized.append(
            {
                **_insufficient("analyzer_returned_no_rows"),
                "evidence_key": stable_json_hash(
                    {"analyzer": analyzer, "empty": True}
                ),
            }
        )
    return AnalyzerExecutionResult(
        rows=tuple(normalized),
        expected_evidence_keys=tuple(
            row["evidence_key"] for row in normalized
        ),
    )


__all__ = [
    "CapabilityAnalyzerBinding",
    "CheckpointBindingPlan",
    "DEFAULT_ANALYZER_CATALOG",
    "compose_checkpoint_binding_plan",
]
