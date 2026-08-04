"""实现通用训练 attempt、同步边界与可组合 instrumentation 生命周期。"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass, field, replace
import math
from types import MappingProxyType
from typing import Any, ContextManager, Protocol, runtime_checkable

import torch
from torch import nn

from .composition import ExecutionSession
from .contracts import (
    BatchEnvelope,
    ModelHandle,
    ObjectiveContext,
    ObjectiveExecution,
    ObjectiveUnit,
    RuntimeCheckpointRef,
)


def _frozen_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(value))


@dataclass(frozen=True)
class AttemptContext:
    """instrumentation 可见的 optimizer-update attempt 坐标。"""

    attempt_id: str
    runtime_id: str
    update_index: int
    accumulation_steps: int
    apply_optimizer: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("attempt_id", "runtime_id"):
            value = str(getattr(self, field_name)).strip()
            if not value:
                raise ValueError(f"{field_name} must be non-empty")
            object.__setattr__(self, field_name, value)
        if int(self.update_index) < 0:
            raise ValueError("update_index must be non-negative")
        if int(self.accumulation_steps) <= 0:
            raise ValueError("accumulation_steps must be positive")
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))


@dataclass(frozen=True)
class MicrobatchContext:
    """instrumentation 可见的单个 accumulation microbatch 坐标。"""

    attempt_id: str
    microbatch_index: int
    batch: BatchEnvelope
    objective_context: ObjectiveContext


@dataclass(frozen=True)
class BackwardObservation:
    """记录一次真实 backward unit 及其 execution identity。"""

    attempt_id: str
    microbatch_index: int
    execution_id: str
    requested_objective_id: str
    unit: ObjectiveUnit


@dataclass(frozen=True)
class GradientSummary:
    """宿主同步服务返回的全局 post-unscale、pre-clip 梯度统计。"""

    grad_norm: float
    gradient_numel: int
    gradients_finite: bool
    contributing_rank_count: int
    metrics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", _frozen_mapping(self.metrics))


@dataclass(frozen=True)
class GradientObservation:
    """描述 post-unscale、pre-clip 梯度边界。"""

    handle: ModelHandle
    summary: GradientSummary
    amp_unscaled: bool
    clipping_requested: bool


@dataclass(frozen=True)
class OptimizerStepObservation:
    """记录 optimizer step 尝试、AMP skip 与 scale 变化。"""

    attempted: bool
    applied: bool | str
    scaler_scale_before: float | None = None
    scaler_scale_after: float | None = None


@dataclass(frozen=True)
class SchedulerStepObservation:
    """记录 scheduler 是否配置以及是否成功完成。"""

    configured: bool
    completed: bool


@dataclass(frozen=True)
class AuxiliaryStep:
    """声明必须在 attempt commit 前执行的 task-neutral 辅助更新。"""

    step_id: str
    callback: Callable[[], Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        step_id = str(self.step_id).strip()
        if not step_id:
            raise ValueError("step_id must be non-empty")
        if not callable(self.callback):
            raise TypeError("callback must be callable")
        object.__setattr__(self, "step_id", step_id)
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))


@dataclass(frozen=True)
class AuxiliaryStepObservation:
    """记录一个辅助更新的终态，不解释其任务语义。"""

    step_id: str
    completed: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))


@dataclass(frozen=True)
class CheckpointMilestone:
    """把已完成 checkpoint 事实通知 instrumentation。"""

    milestone_id: str
    checkpoint: RuntimeCheckpointRef
    update_index: int
    weight_boundary: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("milestone_id", "weight_boundary"):
            value = str(getattr(self, field_name)).strip()
            if not value:
                raise ValueError(f"{field_name} must be non-empty")
            object.__setattr__(self, field_name, value)
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))


@dataclass(frozen=True)
class TrainingAttemptRequest:
    """一次完整 optimizer update 的 batch、objective 与辅助步骤输入。"""

    attempt_id: str
    update_index: int
    batches: Iterable[BatchEnvelope]
    objective_contexts: Iterable[ObjectiveContext]
    accumulation_steps: int
    apply_optimizer: bool = True
    grad_clip_max_norm: float | None = None
    auxiliary_steps: Sequence[AuxiliaryStep] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        auxiliary_steps = tuple(self.auxiliary_steps)
        if int(self.accumulation_steps) <= 0:
            raise ValueError("accumulation_steps must be positive")
        if self.grad_clip_max_norm is not None and (
            not math.isfinite(float(self.grad_clip_max_norm))
            or float(self.grad_clip_max_norm) <= 0
        ):
            raise ValueError(
                "grad_clip_max_norm must be finite and positive when provided"
            )
        if len({step.step_id for step in auxiliary_steps}) != len(auxiliary_steps):
            raise ValueError("auxiliary step_id values must be unique")
        object.__setattr__(self, "auxiliary_steps", auxiliary_steps)
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))


@dataclass(frozen=True)
class TrainingAttemptResult:
    """训练 attempt 终态；失败事件保留已发生 weight boundary。"""

    attempt_id: str
    status: str
    optimizer_step_attempted: bool
    optimizer_step_applied: bool | str
    scheduler_step_completed: bool
    completed_auxiliary_steps: tuple[str, ...]
    processed_microbatches: int
    finite_microbatch_count: int
    rejected_microbatch_count: int
    objective_unit_count: int
    raw_objective_sum: float
    backward_objective_sum: float
    objective_terms: Mapping[str, float]
    loss_cap_hit_count: int
    gradient_summary: GradientSummary | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"completed", "failed"}:
            raise ValueError("status must be 'completed' or 'failed'")
        object.__setattr__(
            self,
            "objective_terms",
            MappingProxyType(dict(self.objective_terms)),
        )


TrainingAttemptOutcome = TrainingAttemptResult


@runtime_checkable
class AttemptInstrumentation(Protocol):
    """训练 runtime 可注入的任务中立观察接口。"""

    def begin_attempt(self, context: AttemptContext) -> Any: ...

    def begin_microbatch(
        self,
        attempt_token: Any,
        context: MicrobatchContext,
    ) -> Any: ...

    def observe_backward(
        self,
        microbatch_token: Any,
        observation: BackwardObservation,
    ) -> ContextManager[None]: ...

    def commit_microbatch(self, microbatch_token: Any) -> None: ...

    def rollback_microbatch(self, microbatch_token: Any, reason: str) -> None: ...

    def gradients_ready(
        self,
        attempt_token: Any,
        observation: GradientObservation,
    ) -> None: ...

    def optimizer_finished(
        self,
        attempt_token: Any,
        observation: OptimizerStepObservation,
    ) -> None: ...

    def scheduler_finished(
        self,
        attempt_token: Any,
        observation: SchedulerStepObservation,
    ) -> None: ...

    def auxiliary_step_finished(
        self,
        attempt_token: Any,
        observation: AuxiliaryStepObservation,
    ) -> None: ...

    def commit_attempt(
        self,
        attempt_token: Any,
        result: TrainingAttemptResult,
    ) -> None: ...

    def attempt_failed(
        self,
        attempt_token: Any,
        result: TrainingAttemptResult,
    ) -> None: ...

    def checkpoint_milestone(self, milestone: CheckpointMilestone) -> None: ...


@runtime_checkable
class AttemptSynchronization(Protocol):
    """隔离 DDP/no-sync/collective 细节的宿主中立同步服务。"""

    def objective_unit_context(
        self,
        attempt: AttemptContext,
        microbatch_index: int,
        unit_index: int,
        execution: ObjectiveExecution,
        objective_context: ObjectiveContext,
    ) -> ContextManager[None]: ...

    def objective_is_finite(self, value: torch.Tensor) -> bool: ...

    def synchronize_gradients(
        self,
        parameters: Sequence[nn.Parameter],
    ) -> None: ...

    def gradient_summary(
        self,
        parameters: Sequence[nn.Parameter],
        *,
        max_norm: float | None,
    ) -> GradientSummary: ...


class LocalAttemptSynchronization:
    """单进程默认实现；分布式宿主必须显式替换本服务。"""

    def objective_unit_context(
        self,
        attempt: AttemptContext,
        microbatch_index: int,
        unit_index: int,
        execution: ObjectiveExecution,
        objective_context: ObjectiveContext,
    ) -> ContextManager[None]:
        return nullcontext()

    def objective_is_finite(self, value: torch.Tensor) -> bool:
        return bool(torch.isfinite(value.detach()).all().item())

    def synchronize_gradients(
        self,
        parameters: Sequence[nn.Parameter],
    ) -> None:
        return None

    def gradient_summary(
        self,
        parameters: Sequence[nn.Parameter],
        *,
        max_norm: float | None,
    ) -> GradientSummary:
        squared_norm = 0.0
        gradient_numel = 0
        gradients_finite = True
        for parameter in parameters:
            gradient = parameter.grad
            if gradient is None:
                continue
            detached = gradient.detach()
            gradient_numel += detached.numel()
            finite = bool(torch.isfinite(detached).all().item())
            gradients_finite = gradients_finite and finite
            if finite:
                squared_norm += float(
                    torch.sum(detached.to(dtype=torch.float64) ** 2).item()
                )
        grad_norm = math.sqrt(squared_norm)
        clip_scale = (
            min(1.0, float(max_norm) / (grad_norm + 1e-6))
            if max_norm is not None
            else 1.0
        )
        return GradientSummary(
            grad_norm=grad_norm,
            gradient_numel=gradient_numel,
            gradients_finite=gradients_finite,
            contributing_rank_count=1,
            metrics={
                "grad_norm_pre_clip_local": grad_norm,
                "grad_norm_pre_clip_mean": grad_norm,
                "grad_norm_pre_clip_min": grad_norm,
                "grad_norm_pre_clip_max": grad_norm,
                "grad_clip_scale": clip_scale,
                "grad_clip_applied": float(clip_scale < 1.0),
            },
        )


class NoOpAttemptInstrumentation:
    """保持默认执行路径无额外观察副作用。"""

    def begin_attempt(self, context: AttemptContext) -> None:
        return None

    def begin_microbatch(
        self,
        attempt_token: Any,
        context: MicrobatchContext,
    ) -> None:
        return None

    def observe_backward(
        self,
        microbatch_token: Any,
        observation: BackwardObservation,
    ) -> ContextManager[None]:
        return nullcontext()

    def commit_microbatch(self, microbatch_token: Any) -> None:
        return None

    def rollback_microbatch(self, microbatch_token: Any, reason: str) -> None:
        return None

    def gradients_ready(
        self,
        attempt_token: Any,
        observation: GradientObservation,
    ) -> None:
        return None

    def optimizer_finished(
        self,
        attempt_token: Any,
        observation: OptimizerStepObservation,
    ) -> None:
        return None

    def scheduler_finished(
        self,
        attempt_token: Any,
        observation: SchedulerStepObservation,
    ) -> None:
        return None

    def auxiliary_step_finished(
        self,
        attempt_token: Any,
        observation: AuxiliaryStepObservation,
    ) -> None:
        return None

    def commit_attempt(
        self,
        attempt_token: Any,
        result: TrainingAttemptResult,
    ) -> None:
        return None

    def attempt_failed(
        self,
        attempt_token: Any,
        result: TrainingAttemptResult,
    ) -> None:
        return None

    def checkpoint_milestone(self, milestone: CheckpointMilestone) -> None:
        return None


@dataclass(frozen=True)
class _CompositeAttemptToken:
    values: tuple[Any, ...]


@dataclass(frozen=True)
class _CompositeMicrobatchToken:
    values: tuple[Any, ...]


class CompositeAttemptInstrumentation:
    """按声明顺序组合 observer，并保持各自 token 与清理次序隔离。"""

    def __init__(self, observers: Sequence[AttemptInstrumentation]) -> None:
        self._observers = tuple(observers)
        for observer in self._observers:
            if not isinstance(observer, AttemptInstrumentation):
                raise TypeError(
                    "observers must implement AttemptInstrumentation"
                )

    def begin_attempt(self, context: AttemptContext) -> _CompositeAttemptToken:
        return _CompositeAttemptToken(
            tuple(observer.begin_attempt(context) for observer in self._observers)
        )

    def objective_execution_options(self) -> Mapping[str, Any]:
        """合并 observer 对同一次真实 objective 的只读证据请求。"""

        merged: dict[str, Any] = {}
        for observer in self._observers:
            provider = getattr(
                observer,
                "objective_execution_options",
                None,
            )
            if not callable(provider):
                continue
            for key, value in dict(provider()).items():
                if key in merged and merged[key] != value:
                    raise ValueError(
                        "instrumentation objective option conflict: "
                        f"{key}"
                    )
                merged[str(key)] = value
        return MappingProxyType(merged)

    def begin_microbatch(
        self,
        attempt_token: _CompositeAttemptToken,
        context: MicrobatchContext,
    ) -> _CompositeMicrobatchToken:
        self._validate_attempt_token(attempt_token)
        return _CompositeMicrobatchToken(
            tuple(
                observer.begin_microbatch(token, context)
                for observer, token in zip(
                    self._observers,
                    attempt_token.values,
                    strict=True,
                )
            )
        )

    @contextmanager
    def observe_backward(
        self,
        microbatch_token: _CompositeMicrobatchToken,
        observation: BackwardObservation,
    ):
        self._validate_microbatch_token(microbatch_token)
        with ExitStack() as stack:
            for observer, token in zip(
                self._observers,
                microbatch_token.values,
                strict=True,
            ):
                stack.enter_context(observer.observe_backward(token, observation))
            yield

    def commit_microbatch(
        self,
        microbatch_token: _CompositeMicrobatchToken,
    ) -> None:
        self._validate_microbatch_token(microbatch_token)
        for observer, token in self._microbatch_pairs(microbatch_token):
            observer.commit_microbatch(token)

    def rollback_microbatch(
        self,
        microbatch_token: _CompositeMicrobatchToken,
        reason: str,
    ) -> None:
        self._validate_microbatch_token(microbatch_token)
        for observer, token in reversed(
            tuple(self._microbatch_pairs(microbatch_token))
        ):
            observer.rollback_microbatch(token, reason)

    def gradients_ready(
        self,
        attempt_token: _CompositeAttemptToken,
        observation: GradientObservation,
    ) -> None:
        for observer, token in self._attempt_pairs(attempt_token):
            observer.gradients_ready(token, observation)

    def optimizer_finished(
        self,
        attempt_token: _CompositeAttemptToken,
        observation: OptimizerStepObservation,
    ) -> None:
        for observer, token in self._attempt_pairs(attempt_token):
            observer.optimizer_finished(token, observation)

    def scheduler_finished(
        self,
        attempt_token: _CompositeAttemptToken,
        observation: SchedulerStepObservation,
    ) -> None:
        for observer, token in self._attempt_pairs(attempt_token):
            observer.scheduler_finished(token, observation)

    def auxiliary_step_finished(
        self,
        attempt_token: _CompositeAttemptToken,
        observation: AuxiliaryStepObservation,
    ) -> None:
        for observer, token in self._attempt_pairs(attempt_token):
            observer.auxiliary_step_finished(token, observation)

    def commit_attempt(
        self,
        attempt_token: _CompositeAttemptToken,
        result: TrainingAttemptResult,
    ) -> None:
        for observer, token in self._attempt_pairs(attempt_token):
            observer.commit_attempt(token, result)

    def attempt_failed(
        self,
        attempt_token: _CompositeAttemptToken,
        result: TrainingAttemptResult,
    ) -> None:
        for observer, token in reversed(tuple(self._attempt_pairs(attempt_token))):
            observer.attempt_failed(token, result)

    def checkpoint_milestone(self, milestone: CheckpointMilestone) -> None:
        for observer in self._observers:
            observer.checkpoint_milestone(milestone)

    def _attempt_pairs(
        self,
        token: _CompositeAttemptToken,
    ) -> tuple[tuple[AttemptInstrumentation, Any], ...]:
        self._validate_attempt_token(token)
        return tuple(zip(self._observers, token.values, strict=True))

    def _microbatch_pairs(
        self,
        token: _CompositeMicrobatchToken,
    ) -> tuple[tuple[AttemptInstrumentation, Any], ...]:
        self._validate_microbatch_token(token)
        return tuple(zip(self._observers, token.values, strict=True))

    def _validate_attempt_token(self, token: _CompositeAttemptToken) -> None:
        if not isinstance(token, _CompositeAttemptToken):
            raise TypeError("invalid composite attempt token")
        if len(token.values) != len(self._observers):
            raise ValueError("composite attempt token length mismatch")

    def _validate_microbatch_token(
        self,
        token: _CompositeMicrobatchToken,
    ) -> None:
        if not isinstance(token, _CompositeMicrobatchToken):
            raise TypeError("invalid composite microbatch token")
        if len(token.values) != len(self._observers):
            raise ValueError("composite microbatch token length mismatch")


class TrainingAttemptExecutor:
    """流式执行 objective units，并拥有 optimizer 到 auxiliary 的终态边界。"""

    def __init__(
        self,
        instrumentation: AttemptInstrumentation | None = None,
        synchronization: AttemptSynchronization | None = None,
    ) -> None:
        self._instrumentation = instrumentation or NoOpAttemptInstrumentation()
        self._synchronization = (
            synchronization or LocalAttemptSynchronization()
        )
        if not isinstance(self._instrumentation, AttemptInstrumentation):
            raise TypeError(
                "instrumentation must implement AttemptInstrumentation"
            )
        if not isinstance(self._synchronization, AttemptSynchronization):
            raise TypeError(
                "synchronization must implement AttemptSynchronization"
            )

    def _with_instrumentation_options(
        self,
        context: ObjectiveContext,
    ) -> ObjectiveContext:
        """把 observer 的证据请求合并到真实 objective，不建立第二套执行。"""

        provider = getattr(
            self._instrumentation,
            "objective_execution_options",
            None,
        )
        if not callable(provider):
            return context
        requested = {
            str(key): value for key, value in dict(provider()).items()
        }
        if not requested:
            return context
        return replace(
            context,
            options={**dict(context.options), **requested},
        )

    def execute(
        self,
        session: ExecutionSession,
        request: TrainingAttemptRequest,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any = None,
        scaler: Any = None,
    ) -> TrainingAttemptResult:
        if session.closed:
            raise RuntimeError("cannot execute against a closed ExecutionSession")
        if request.apply_optimizer and optimizer is None:
            raise ValueError("optimizer is required when apply_optimizer=True")
        if scheduler is not None and not request.apply_optimizer:
            raise ValueError("scheduler requires apply_optimizer=True")
        scaler_enabled = bool(
            scaler is not None
            and callable(getattr(scaler, "is_enabled", None))
            and scaler.is_enabled()
        )
        if scaler_enabled and optimizer is None:
            raise ValueError("enabled scaler requires an optimizer")
        attempt_context = AttemptContext(
            attempt_id=request.attempt_id,
            runtime_id=session.runtime.descriptor.runtime_id,
            update_index=request.update_index,
            accumulation_steps=request.accumulation_steps,
            apply_optimizer=request.apply_optimizer,
            metadata=request.metadata,
        )
        attempt_token = self._instrumentation.begin_attempt(attempt_context)
        self._zero_gradients(session.model, optimizer)
        processed_microbatches = 0
        rejected_microbatch_count = 0
        objective_unit_count = 0
        raw_objective_sum = 0.0
        backward_objective_sum = 0.0
        objective_terms: dict[str, float] = {}
        loss_cap_hit_count = 0
        gradient_summary: GradientSummary | None = None
        optimizer_step_attempted = False
        optimizer_step_applied: bool | str = False
        scheduler_step_completed = False
        completed_auxiliary_steps: list[str] = []
        active_microbatch_token: Any = None
        active_microbatch_finalized = True

        try:
            for microbatch_index, (batch, objective_context) in enumerate(
                zip(request.batches, request.objective_contexts, strict=True)
            ):
                if not isinstance(batch, BatchEnvelope):
                    raise TypeError("batches must yield BatchEnvelope values")
                if not isinstance(objective_context, ObjectiveContext):
                    raise TypeError(
                        "objective_contexts must yield ObjectiveContext values"
                    )
                objective_context = self._with_instrumentation_options(
                    objective_context
                )
                if not math.isclose(
                    float(objective_context.normalization_divisor),
                    float(request.accumulation_steps),
                    rel_tol=0.0,
                    abs_tol=0.0,
                ):
                    raise ValueError(
                        "objective normalization_divisor must equal "
                        "accumulation_steps"
                    )
                microbatch_context = MicrobatchContext(
                    attempt_id=request.attempt_id,
                    microbatch_index=microbatch_index,
                    batch=batch,
                    objective_context=objective_context,
                )
                active_microbatch_token = (
                    self._instrumentation.begin_microbatch(
                        attempt_token,
                        microbatch_context,
                    )
                )
                active_microbatch_finalized = False
                execution = session.runtime.objectives.execute(
                    session.handle,
                    batch,
                    objective_context,
                )
                unit_iterator = iter(execution.units)
                microbatch_unit_count = 0
                microbatch_backward_count = 0
                microbatch_raw_sum = 0.0
                microbatch_backward_sum = 0.0
                microbatch_terms: dict[str, float] = {}
                microbatch_cap_hits = 0
                rejection_reason: str | None = None
                fatal_reason: str | None = None
                try:
                    while True:
                        with self._synchronization.objective_unit_context(
                            attempt_context,
                            microbatch_index,
                            microbatch_unit_count,
                            execution,
                            objective_context,
                        ):
                            try:
                                unit = next(unit_iterator)
                            except StopIteration:
                                break
                            if not isinstance(unit, ObjectiveUnit):
                                raise TypeError(
                                    "objective execution must yield ObjectiveUnit"
                                )
                            expected_amp_scale = (
                                float(scaler.get_scale())
                                if scaler_enabled
                                else 1.0
                            )
                            if not math.isclose(
                                float(unit.amp_scale),
                                expected_amp_scale,
                                rel_tol=1e-12,
                                abs_tol=0.0,
                            ):
                                raise ValueError(
                                    "ObjectiveUnit.amp_scale does not match "
                                    "the active GradScaler"
                                )
                            if not math.isclose(
                                float(unit.normalization_divisor),
                                float(objective_context.normalization_divisor),
                                rel_tol=0.0,
                                abs_tol=0.0,
                            ):
                                raise ValueError(
                                    "ObjectiveUnit.normalization_divisor does "
                                    "not match ObjectiveContext"
                                )
                            cap_matches = (
                                unit.loss_cap is None
                                and objective_context.loss_cap is None
                            ) or (
                                unit.loss_cap is not None
                                and objective_context.loss_cap is not None
                                and math.isclose(
                                    float(unit.loss_cap),
                                    float(objective_context.loss_cap),
                                    rel_tol=0.0,
                                    abs_tol=0.0,
                                )
                            )
                            if not cap_matches:
                                raise ValueError(
                                    "ObjectiveUnit.loss_cap does not match "
                                    "ObjectiveContext"
                                )
                            # raw 与实际 backward objective 合并为一次 finite
                            # consensus，避免 DDP 每个 unit 新增重复 collective。
                            effective_backward = (
                                unit.effective_backward_objective()
                            )
                            objective_values = torch.stack(
                                (
                                    unit.raw_total.detach().reshape(()),
                                    effective_backward.detach(),
                                )
                            )
                            if not self._synchronization.objective_is_finite(
                                objective_values
                            ):
                                reason = (
                                    f"nonfinite_objective:{unit.unit_id}"
                                )
                                if microbatch_backward_count == 0:
                                    rejection_reason = reason
                                else:
                                    fatal_reason = (
                                        "partial_microbatch_"
                                        f"{reason}"
                                    )
                                break

                            observation = BackwardObservation(
                                attempt_id=request.attempt_id,
                                microbatch_index=microbatch_index,
                                execution_id=execution.execution_id,
                                requested_objective_id=(
                                    objective_context.objective_id
                                ),
                                unit=unit,
                            )
                            if effective_backward.requires_grad:
                                with self._instrumentation.observe_backward(
                                    active_microbatch_token,
                                    observation,
                                ):
                                    if scaler_enabled:
                                        scaler.scale(
                                            effective_backward
                                        ).backward(
                                            retain_graph=unit.retain_graph
                                        )
                                    else:
                                        effective_backward.backward(
                                            retain_graph=unit.retain_graph
                                        )
                                microbatch_backward_count += 1
                            microbatch_unit_count += 1
                            microbatch_raw_sum += float(
                                unit.raw_total.detach().item()
                            )
                            microbatch_backward_sum += float(
                                effective_backward.detach().item()
                            )
                            microbatch_cap_hits += int(
                                unit.loss_cap_applied
                            )
                            for name, term in unit.terms.items():
                                term_value = (
                                    float(term.detach().item())
                                    if isinstance(term, torch.Tensor)
                                    else float(term)
                                )
                                microbatch_terms[name] = (
                                    microbatch_terms.get(name, 0.0)
                                    + term_value
                                )
                finally:
                    close_units = getattr(unit_iterator, "close", None)
                    if callable(close_units):
                        close_units()

                if rejection_reason is not None:
                    self._instrumentation.rollback_microbatch(
                        active_microbatch_token,
                        rejection_reason,
                    )
                    active_microbatch_finalized = True
                    rejected_microbatch_count += 1
                    continue
                if fatal_reason is not None:
                    self._instrumentation.rollback_microbatch(
                        active_microbatch_token,
                        fatal_reason,
                    )
                    active_microbatch_finalized = True
                    result = self._failed_result(
                        request=request,
                        processed_microbatches=processed_microbatches,
                        rejected_microbatch_count=rejected_microbatch_count,
                        objective_unit_count=objective_unit_count,
                        raw_objective_sum=raw_objective_sum,
                        backward_objective_sum=backward_objective_sum,
                        objective_terms=objective_terms,
                        loss_cap_hit_count=loss_cap_hit_count,
                        gradient_summary=gradient_summary,
                        optimizer_step_attempted=optimizer_step_attempted,
                        optimizer_step_applied=optimizer_step_applied,
                        scheduler_step_completed=scheduler_step_completed,
                        completed_auxiliary_steps=completed_auxiliary_steps,
                        reason=fatal_reason,
                    )
                    self._zero_gradients(session.model, optimizer)
                    self._instrumentation.attempt_failed(
                        attempt_token,
                        result,
                    )
                    return result
                if microbatch_unit_count == 0:
                    reason = "empty_objective_units"
                    self._instrumentation.rollback_microbatch(
                        active_microbatch_token,
                        reason,
                    )
                    active_microbatch_finalized = True
                    result = self._failed_result(
                        request=request,
                        processed_microbatches=processed_microbatches,
                        rejected_microbatch_count=rejected_microbatch_count,
                        objective_unit_count=objective_unit_count,
                        raw_objective_sum=raw_objective_sum,
                        backward_objective_sum=backward_objective_sum,
                        objective_terms=objective_terms,
                        loss_cap_hit_count=loss_cap_hit_count,
                        gradient_summary=gradient_summary,
                        optimizer_step_attempted=optimizer_step_attempted,
                        optimizer_step_applied=optimizer_step_applied,
                        scheduler_step_completed=scheduler_step_completed,
                        completed_auxiliary_steps=completed_auxiliary_steps,
                        reason=reason,
                    )
                    self._zero_gradients(session.model, optimizer)
                    self._instrumentation.attempt_failed(
                        attempt_token,
                        result,
                    )
                    return result
                self._instrumentation.commit_microbatch(
                    active_microbatch_token
                )
                active_microbatch_finalized = True
                processed_microbatches += 1
                objective_unit_count += microbatch_unit_count
                raw_objective_sum += microbatch_raw_sum
                backward_objective_sum += microbatch_backward_sum
                loss_cap_hit_count += microbatch_cap_hits
                for name, value in microbatch_terms.items():
                    objective_terms[name] = (
                        objective_terms.get(name, 0.0) + value
                    )

            if (
                processed_microbatches + rejected_microbatch_count
                != request.accumulation_steps
            ):
                raise ValueError(
                    "batch/context streams ended before accumulation_steps"
                )
            if processed_microbatches == 0:
                result = self._failed_result(
                    request=request,
                    processed_microbatches=0,
                    rejected_microbatch_count=rejected_microbatch_count,
                    objective_unit_count=0,
                    raw_objective_sum=0.0,
                    backward_objective_sum=0.0,
                    objective_terms={},
                    loss_cap_hit_count=0,
                    gradient_summary=None,
                    optimizer_step_attempted=False,
                    optimizer_step_applied=False,
                    scheduler_step_completed=False,
                    completed_auxiliary_steps=(),
                    reason="all_microbatches_rejected",
                )
                self._zero_gradients(session.model, optimizer)
                self._instrumentation.attempt_failed(attempt_token, result)
                return result
            parameters = self._parameters_for_gradient_boundary(
                session.model,
                optimizer,
            )
            if scaler_enabled:
                scaler.unscale_(optimizer)
            self._synchronization.synchronize_gradients(parameters)
            gradient_summary = self._synchronization.gradient_summary(
                parameters,
                max_norm=request.grad_clip_max_norm,
            )
            self._instrumentation.gradients_ready(
                attempt_token,
                GradientObservation(
                    handle=session.handle,
                    summary=gradient_summary,
                    amp_unscaled=scaler_enabled,
                    clipping_requested=(
                        request.grad_clip_max_norm is not None
                    ),
                ),
            )
            if not gradient_summary.gradients_finite:
                result = self._failed_result(
                    request=request,
                    processed_microbatches=processed_microbatches,
                    rejected_microbatch_count=rejected_microbatch_count,
                    objective_unit_count=objective_unit_count,
                    raw_objective_sum=raw_objective_sum,
                    backward_objective_sum=backward_objective_sum,
                    objective_terms=objective_terms,
                    loss_cap_hit_count=loss_cap_hit_count,
                    gradient_summary=gradient_summary,
                    optimizer_step_attempted=False,
                    optimizer_step_applied=False,
                    scheduler_step_completed=False,
                    completed_auxiliary_steps=completed_auxiliary_steps,
                    reason="nonfinite_gradient",
                )
                self._zero_gradients(session.model, optimizer)
                if scaler_enabled:
                    scaler.update()
                self._instrumentation.attempt_failed(attempt_token, result)
                return result

            if request.grad_clip_max_norm is not None:
                clip_scale = float(
                    gradient_summary.metrics.get(
                        "grad_clip_scale",
                        min(
                            1.0,
                            float(request.grad_clip_max_norm)
                            / (float(gradient_summary.grad_norm) + 1e-6),
                        ),
                    )
                )
                if clip_scale < 1.0:
                    for parameter in parameters:
                        if parameter.grad is not None:
                            parameter.grad.mul_(clip_scale)

            scale_before: float | None = None
            scale_after: float | None = None
            if request.apply_optimizer:
                optimizer_step_attempted = True
                optimizer_step_applied = "unknown_partial"
                try:
                    if scaler_enabled:
                        scale_before = float(scaler.get_scale())
                        scaler.step(optimizer)
                        scaler.update()
                        scale_after = float(scaler.get_scale())
                        optimizer_step_applied = scale_after >= scale_before
                    else:
                        optimizer.step()
                        optimizer_step_applied = True
                except BaseException:
                    self._instrumentation.optimizer_finished(
                        attempt_token,
                        OptimizerStepObservation(
                            attempted=True,
                            applied="unknown_partial",
                            scaler_scale_before=scale_before,
                            scaler_scale_after=scale_after,
                        ),
                    )
                    raise
            self._instrumentation.optimizer_finished(
                attempt_token,
                OptimizerStepObservation(
                    attempted=optimizer_step_attempted,
                    applied=optimizer_step_applied,
                    scaler_scale_before=scale_before,
                    scaler_scale_after=scale_after,
                ),
            )
            if optimizer_step_attempted and optimizer_step_applied is False:
                result = self._failed_result(
                    request=request,
                    processed_microbatches=processed_microbatches,
                    rejected_microbatch_count=rejected_microbatch_count,
                    objective_unit_count=objective_unit_count,
                    raw_objective_sum=raw_objective_sum,
                    backward_objective_sum=backward_objective_sum,
                    objective_terms=objective_terms,
                    loss_cap_hit_count=loss_cap_hit_count,
                    gradient_summary=gradient_summary,
                    optimizer_step_attempted=True,
                    optimizer_step_applied=False,
                    scheduler_step_completed=False,
                    completed_auxiliary_steps=completed_auxiliary_steps,
                    reason="amp_optimizer_step_skipped",
                )
                self._zero_gradients(session.model, optimizer)
                self._instrumentation.attempt_failed(attempt_token, result)
                return result

            scheduler_configured = scheduler is not None
            if scheduler_configured:
                try:
                    scheduler.step()
                    scheduler_step_completed = True
                except BaseException:
                    self._instrumentation.scheduler_finished(
                        attempt_token,
                        SchedulerStepObservation(
                            configured=True,
                            completed=False,
                        ),
                    )
                    raise
            self._instrumentation.scheduler_finished(
                attempt_token,
                SchedulerStepObservation(
                    configured=scheduler_configured,
                    completed=scheduler_step_completed,
                ),
            )

            for auxiliary_step in request.auxiliary_steps:
                try:
                    auxiliary_step.callback()
                except BaseException:
                    self._instrumentation.auxiliary_step_finished(
                        attempt_token,
                        AuxiliaryStepObservation(
                            step_id=auxiliary_step.step_id,
                            completed=False,
                            metadata=auxiliary_step.metadata,
                        ),
                    )
                    raise
                completed_auxiliary_steps.append(auxiliary_step.step_id)
                self._instrumentation.auxiliary_step_finished(
                    attempt_token,
                    AuxiliaryStepObservation(
                        step_id=auxiliary_step.step_id,
                        completed=True,
                        metadata=auxiliary_step.metadata,
                    ),
                )

            result = TrainingAttemptResult(
                attempt_id=request.attempt_id,
                status="completed",
                optimizer_step_attempted=optimizer_step_attempted,
                optimizer_step_applied=optimizer_step_applied,
                scheduler_step_completed=scheduler_step_completed,
                completed_auxiliary_steps=tuple(
                    completed_auxiliary_steps
                ),
                processed_microbatches=(
                    processed_microbatches + rejected_microbatch_count
                ),
                finite_microbatch_count=processed_microbatches,
                rejected_microbatch_count=rejected_microbatch_count,
                objective_unit_count=objective_unit_count,
                raw_objective_sum=raw_objective_sum,
                backward_objective_sum=backward_objective_sum,
                objective_terms=objective_terms,
                loss_cap_hit_count=loss_cap_hit_count,
                gradient_summary=gradient_summary,
            )
            self._instrumentation.commit_attempt(attempt_token, result)
            return result
        except BaseException as exc:
            if (
                active_microbatch_token is not None
                and not active_microbatch_finalized
            ):
                self._instrumentation.rollback_microbatch(
                    active_microbatch_token,
                    f"{type(exc).__name__}:{exc}",
                )
            self._zero_gradients(session.model, optimizer)
            result = self._failed_result(
                request=request,
                processed_microbatches=processed_microbatches,
                rejected_microbatch_count=rejected_microbatch_count,
                objective_unit_count=objective_unit_count,
                raw_objective_sum=raw_objective_sum,
                backward_objective_sum=backward_objective_sum,
                objective_terms=objective_terms,
                loss_cap_hit_count=loss_cap_hit_count,
                gradient_summary=gradient_summary,
                optimizer_step_attempted=optimizer_step_attempted,
                optimizer_step_applied=optimizer_step_applied,
                scheduler_step_completed=scheduler_step_completed,
                completed_auxiliary_steps=completed_auxiliary_steps,
                reason=f"{type(exc).__name__}:{exc}",
            )
            self._instrumentation.attempt_failed(attempt_token, result)
            raise

    def checkpoint_milestone(self, milestone: CheckpointMilestone) -> None:
        """通知 observer 一个已由宿主完成的 checkpoint milestone。"""

        self._instrumentation.checkpoint_milestone(milestone)

    @staticmethod
    def _zero_gradients(
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None,
    ) -> None:
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        else:
            model.zero_grad(set_to_none=True)

    @staticmethod
    def _parameters_for_gradient_boundary(
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None,
    ) -> tuple[nn.Parameter, ...]:
        if optimizer is None:
            return tuple(model.parameters())
        parameters: list[nn.Parameter] = []
        seen: set[int] = set()
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if id(parameter) in seen:
                    continue
                seen.add(id(parameter))
                parameters.append(parameter)
        return tuple(parameters)

    @staticmethod
    def _failed_result(
        *,
        request: TrainingAttemptRequest,
        processed_microbatches: int,
        rejected_microbatch_count: int,
        objective_unit_count: int,
        raw_objective_sum: float,
        backward_objective_sum: float,
        objective_terms: Mapping[str, float],
        loss_cap_hit_count: int,
        gradient_summary: GradientSummary | None,
        optimizer_step_attempted: bool,
        optimizer_step_applied: bool | str,
        scheduler_step_completed: bool,
        completed_auxiliary_steps: Sequence[str],
        reason: str,
    ) -> TrainingAttemptResult:
        return TrainingAttemptResult(
            attempt_id=request.attempt_id,
            status="failed",
            optimizer_step_attempted=optimizer_step_attempted,
            optimizer_step_applied=optimizer_step_applied,
            scheduler_step_completed=scheduler_step_completed,
            completed_auxiliary_steps=tuple(completed_auxiliary_steps),
            processed_microbatches=(
                processed_microbatches + rejected_microbatch_count
            ),
            finite_microbatch_count=processed_microbatches,
            rejected_microbatch_count=rejected_microbatch_count,
            objective_unit_count=objective_unit_count,
            raw_objective_sum=raw_objective_sum,
            backward_objective_sum=backward_objective_sum,
            objective_terms=objective_terms,
            loss_cap_hit_count=loss_cap_hit_count,
            gradient_summary=gradient_summary,
            reason=reason,
        )


__all__ = [
    "AttemptContext",
    "AttemptInstrumentation",
    "AttemptSynchronization",
    "AuxiliaryStep",
    "AuxiliaryStepObservation",
    "BackwardObservation",
    "CheckpointMilestone",
    "CompositeAttemptInstrumentation",
    "GradientObservation",
    "GradientSummary",
    "LocalAttemptSynchronization",
    "MicrobatchContext",
    "NoOpAttemptInstrumentation",
    "OptimizerStepObservation",
    "SchedulerStepObservation",
    "TrainingAttemptExecutor",
    "TrainingAttemptOutcome",
    "TrainingAttemptRequest",
    "TrainingAttemptResult",
]
