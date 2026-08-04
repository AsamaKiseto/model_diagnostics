"""
验证统一宿主运行时的公共 contract、流式 attempt 与资源边界。
"""

from __future__ import annotations

import ast
from contextlib import contextmanager
from pathlib import Path
import sys
from typing import Any

import pytest
import torch
from torch import nn

from model_diagnostics.host_runtime import (
    BatchEnvelope,
    CapabilityCatalog,
    CheckpointObjectiveSpec,
    CheckpointState,
    CompositeAttemptInstrumentation,
    ExecutionPreparationCapability,
    HostTaskRuntime,
    LocalAttemptSynchronization,
    ModelHandle,
    ModelSpec,
    NoOpAttemptInstrumentation,
    ObjectiveContext,
    ObjectiveExecution,
    ObjectiveUnit,
    RunRef,
    RuntimeCheckpointRef,
    RuntimeDescriptor,
    TaskDefinition,
    TaskRegistry,
    TrainingAttemptExecutor,
    TrainingAttemptRequest,
)


class _ModelProvider:
    def __init__(self) -> None:
        self.closed: list[str] = []

    def create(
        self,
        model_spec: ModelSpec,
        build_payload: Any,
        *,
        device: torch.device,
    ) -> ModelHandle:
        model = nn.Linear(1, 1, bias=False, device=device)
        return ModelHandle(model_id=model_spec.model_id, model=model)

    def close(self, handle: ModelHandle) -> None:
        self.closed.append(handle.model_id)


class _CheckpointProvider:
    def __init__(self, state_dict: dict[str, torch.Tensor]) -> None:
        self._state_dict = state_dict

    def resolve(self, run: RunRef):
        return (RuntimeCheckpointRef("only", f"{run.run_id}/opaque"),)

    def load(
        self,
        checkpoint: RuntimeCheckpointRef,
        *,
        map_location: torch.device,
    ) -> CheckpointState:
        return CheckpointState(
            checkpoint=checkpoint,
            model_spec=ModelSpec("linear"),
            state_payload={
                name: tensor.to(map_location)
                for name, tensor in self._state_dict.items()
            },
        )

    def restore(self, handle: ModelHandle, state: CheckpointState) -> None:
        handle.model.load_state_dict(state.state_payload)


class _BatchProvider:
    def select(self, run, selector):
        return ()

    def materialize(self, samples, *, device):
        raise NotImplementedError

    def clone(self, batch):
        return batch


class _ObjectiveExecutor:
    definition_version = 1

    def __init__(self, events: list[str]) -> None:
        self.events = events

    def execute(
        self,
        handle: ModelHandle,
        batch: BatchEnvelope,
        context: ObjectiveContext,
    ) -> ObjectiveExecution:
        self.events.append(f"prepare:{batch.batch_id}")

        def units():
            self.events.append(f"forward:{batch.batch_id}")
            features, target = batch.payload
            prediction = handle.model(features)
            raw_total = torch.mean((prediction - target) ** 2)
            bounded = (
                torch.clamp(raw_total, max=context.loss_cap)
                if context.loss_cap is not None
                else raw_total
            )
            yield ObjectiveUnit(
                unit_id=f"{batch.batch_id}:0",
                objective_identity=context.objective_id,
                raw_total=raw_total,
                backward_total=bounded / context.normalization_divisor,
                terms={"loss": raw_total},
                schedule_coordinate=context.schedule_coordinate,
                normalization_divisor=context.normalization_divisor,
                loss_cap=context.loss_cap,
                loss_cap_applied=bool(
                    context.loss_cap is not None
                    and raw_total.detach().item() > context.loss_cap
                ),
                amp_scale=context.amp_scale,
                chunk_identity="single",
                chunk_index=0,
                chunk_count=1,
            )
            self.events.append(f"released:{batch.batch_id}")

        return ObjectiveExecution(
            execution_id=f"execution:{batch.batch_id}",
            units=units(),
            metadata={"gradient_sync_mode": "manual"},
        )


class _RecordingSynchronization(LocalAttemptSynchronization):
    def __init__(self, events: list[str]) -> None:
        self.events = events

    @contextmanager
    def objective_unit_context(
        self,
        attempt,
        microbatch_index: int,
        unit_index: int,
        execution,
        objective_context,
    ):
        self.events.append(
            "sync-enter:"
            f"{microbatch_index}:{unit_index}:"
            f"{execution.metadata['gradient_sync_mode']}"
        )
        try:
            yield
        finally:
            self.events.append(f"sync-exit:{microbatch_index}:{unit_index}")

    def synchronize_gradients(self, parameters) -> None:
        self.events.append("gradients-synchronized")

    def gradient_summary(self, parameters, *, max_norm):
        self.events.append("gradient-summary")
        return super().gradient_summary(parameters, max_norm=max_norm)


class _RecordingInstrumentation(NoOpAttemptInstrumentation):
    def __init__(self, name: str, events: list[str]) -> None:
        self.name = name
        self.events = events

    def begin_attempt(self, context):
        self.events.append(f"{self.name}:begin-attempt")
        return self.name

    def begin_microbatch(self, attempt_token, context):
        token = f"{attempt_token}:{context.microbatch_index}"
        self.events.append(f"{self.name}:begin:{context.microbatch_index}")
        return token

    @contextmanager
    def observe_backward(self, microbatch_token, observation):
        self.events.append(f"{self.name}:backward-enter")
        try:
            yield
        finally:
            self.events.append(f"{self.name}:backward-exit")

    def commit_microbatch(self, microbatch_token):
        self.events.append(f"{self.name}:commit-microbatch")

    def rollback_microbatch(self, microbatch_token, reason):
        self.events.append(f"{self.name}:rollback:{reason}")

    def gradients_ready(self, attempt_token, observation):
        self.events.append(f"{self.name}:gradients-ready")

    def optimizer_finished(self, attempt_token, observation):
        self.events.append(
            f"{self.name}:optimizer:{observation.applied}"
        )

    def scheduler_finished(self, attempt_token, observation):
        self.events.append(f"{self.name}:scheduler")

    def auxiliary_step_finished(self, attempt_token, observation):
        self.events.append(f"{self.name}:aux:{observation.step_id}")

    def commit_attempt(self, attempt_token, result):
        self.events.append(f"{self.name}:commit-attempt")

    def attempt_failed(self, attempt_token, result):
        self.events.append(f"{self.name}:failed:{result.reason}")


def _runtime(
    model: nn.Module,
    objective_executor: _ObjectiveExecutor,
    attempt_executor: TrainingAttemptExecutor,
) -> tuple[HostTaskRuntime, _ModelProvider]:
    models = _ModelProvider()
    runtime = HostTaskRuntime(
        descriptor=RuntimeDescriptor(
            runtime_id="runtime",
            task_definition_id="generic_linear",
            task_definition_version=1,
            run=RunRef("run"),
            objective_executor_version=objective_executor.definition_version,
            checkpoint_objective=CheckpointObjectiveSpec(
                objective_id="generic_linear_objective",
            ),
        ),
        models=models,
        checkpoints=_CheckpointProvider(model.state_dict()),
        batches=_BatchProvider(),
        objectives=objective_executor,
        attempts=attempt_executor,
        capabilities=CapabilityCatalog(),
    )
    return runtime, models


def test_training_attempt_streams_batches_and_matches_reference() -> None:
    events: list[str] = []
    instrumentation = _RecordingInstrumentation("observer", events)
    executor = TrainingAttemptExecutor(
        instrumentation=instrumentation,
        synchronization=_RecordingSynchronization(events),
    )
    model = nn.Linear(1, 1, bias=False)
    nn.init.constant_(model.weight, 0.0)
    runtime, _ = _runtime(model, _ObjectiveExecutor(events), executor)
    session = runtime.bind_live_model(ModelHandle("linear", model))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    def batches():
        events.append("batch-stream-start")
        yield BatchEnvelope(
            "a",
            (torch.tensor([[1.0]]), torch.tensor([[1.0]])),
        )
        events.append("batch-stream-middle")
        yield BatchEnvelope(
            "b",
            (torch.tensor([[2.0]]), torch.tensor([[0.0]])),
        )

    def contexts():
        for index in range(2):
            yield ObjectiveContext(
                objective_id="training_total",
                training=True,
                update_index=0,
                microbatch_index=index,
                normalization_divisor=2.0,
            )

    request = TrainingAttemptRequest(
        attempt_id="attempt-0",
        update_index=0,
        batches=batches(),
        objective_contexts=contexts(),
        accumulation_steps=2,
    )
    assert events == []
    result = runtime.attempts.execute(
        session,
        request,
        optimizer=optimizer,
    )

    reference = nn.Linear(1, 1, bias=False)
    nn.init.constant_(reference.weight, 0.0)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    reference_loss = (
        torch.mean((reference(torch.tensor([[1.0]])) - 1.0) ** 2)
        + torch.mean((reference(torch.tensor([[2.0]])) - 0.0) ** 2)
    ) / 2.0
    reference_loss.backward()
    reference_optimizer.step()

    assert result.status == "completed"
    assert result.processed_microbatches == 2
    assert result.objective_unit_count == 2
    torch.testing.assert_close(model.weight, reference.weight)
    assert events.index("sync-enter:0:0:manual") < events.index("forward:a")
    assert events.index("forward:a") < events.index("observer:backward-enter")
    assert events.index("observer:backward-exit") < events.index("sync-exit:0:0")
    assert events.index("sync-exit:0:0") < events.index("released:a")
    assert events.index("released:a") < events.index("batch-stream-middle")
    assert events.count("gradient-summary") == 1
    assert result.gradient_summary is not None
    assert result.gradient_summary.metrics["grad_norm_pre_clip_max"] >= 0.0
    assert result.gradient_summary.metrics["grad_clip_scale"] == 1.0


def test_attempt_instrumentation_extends_the_actual_objective_context() -> None:
    """observer 只能向唯一 objective 请求证据，不能触发第二次执行。"""

    events: list[str] = []
    observed_options: list[dict[str, Any]] = []

    class EvidenceInstrumentation(_RecordingInstrumentation):
        def objective_execution_options(self):
            return {"collect_objective_ledger": True}

    class CapturingObjective(_ObjectiveExecutor):
        def execute(self, handle, batch, context):
            observed_options.append(dict(context.options))
            return super().execute(handle, batch, context)

    instrumentation = EvidenceInstrumentation("observer", events)
    executor = TrainingAttemptExecutor(instrumentation=instrumentation)
    model = nn.Linear(1, 1, bias=False)
    objective = CapturingObjective(events)
    runtime, _ = _runtime(model, objective, executor)
    session = runtime.bind_live_model(ModelHandle("linear", model))
    result = executor.execute(
        session,
        TrainingAttemptRequest(
            attempt_id="objective-evidence",
            update_index=0,
            batches=(
                BatchEnvelope(
                    "only",
                    (torch.tensor([[1.0]]), torch.tensor([[0.0]])),
                ),
            ),
            objective_contexts=(
                ObjectiveContext(
                    objective_id="training_total",
                    training=True,
                    normalization_divisor=1.0,
                    options={"host_option": "kept"},
                ),
            ),
            accumulation_steps=1,
        ),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
    )

    assert result.status == "completed"
    assert observed_options == [
        {
            "host_option": "kept",
            "collect_objective_ledger": True,
        }
    ]
    assert events.count("prepare:only") == 1
    assert events.count("forward:only") == 1


def test_nonfinite_objective_rolls_back_without_weight_update() -> None:
    events: list[str] = []
    executor = TrainingAttemptExecutor(
        instrumentation=_RecordingInstrumentation("observer", events)
    )
    model = nn.Linear(1, 1, bias=False)
    initial = model.weight.detach().clone()
    runtime, _ = _runtime(model, _ObjectiveExecutor(events), executor)
    session = runtime.bind_live_model(ModelHandle("linear", model))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    request = TrainingAttemptRequest(
        attempt_id="attempt-nan",
        update_index=0,
        batches=(
            BatchEnvelope(
                "nan",
                (torch.tensor([[float("nan")]]), torch.tensor([[0.0]])),
            ),
        ),
        objective_contexts=(
            ObjectiveContext(
                objective_id="training_total",
                training=True,
                normalization_divisor=1.0,
            ),
        ),
        accumulation_steps=1,
    )

    result = executor.execute(session, request, optimizer=optimizer)

    assert result.status == "failed"
    assert result.reason == "all_microbatches_rejected"
    assert result.finite_microbatch_count == 0
    assert result.rejected_microbatch_count == 1
    assert not result.optimizer_step_applied
    torch.testing.assert_close(model.weight, initial)
    assert any(":rollback:nonfinite_objective:nan:0" in event for event in events)


def test_nonfinite_microbatch_is_rejected_between_finite_microbatches() -> None:
    events: list[str] = []
    executor = TrainingAttemptExecutor(
        instrumentation=_RecordingInstrumentation("observer", events)
    )
    model = nn.Linear(1, 1, bias=False)
    nn.init.constant_(model.weight, 0.0)
    runtime, _ = _runtime(model, _ObjectiveExecutor(events), executor)
    session = runtime.bind_live_model(ModelHandle("linear", model))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    batches = (
        BatchEnvelope(
            "finite-before",
            (torch.tensor([[1.0]]), torch.tensor([[1.0]])),
        ),
        BatchEnvelope(
            "rejected",
            (torch.tensor([[float("nan")]]), torch.tensor([[0.0]])),
        ),
        BatchEnvelope(
            "finite-after",
            (torch.tensor([[2.0]]), torch.tensor([[0.0]])),
        ),
    )
    contexts = tuple(
        ObjectiveContext(
            objective_id="training_total",
            training=True,
            microbatch_index=index,
            normalization_divisor=3.0,
        )
        for index in range(3)
    )

    result = executor.execute(
        session,
        TrainingAttemptRequest(
            attempt_id="finite-nan-finite",
            update_index=0,
            batches=batches,
            objective_contexts=contexts,
            accumulation_steps=3,
        ),
        optimizer=optimizer,
    )

    assert result.status == "completed"
    assert result.processed_microbatches == 3
    assert result.finite_microbatch_count == 2
    assert result.rejected_microbatch_count == 1
    torch.testing.assert_close(
        model.weight,
        torch.tensor([[2.0 / 30.0]]),
    )
    assert events.count("observer:commit-microbatch") == 2
    assert any(":rollback:nonfinite_objective:rejected:0" in event for event in events)


def test_nonfinite_after_partial_chunk_backward_fails_whole_attempt() -> None:
    class _PartialChunkExecutor:
        definition_version = 1

        def execute(self, handle, batch, context):
            def units():
                finite = torch.mean(handle.model(batch.payload) ** 2)
                yield ObjectiveUnit(
                    unit_id="finite",
                    objective_identity="actual",
                    raw_total=finite,
                    backward_total=finite,
                )
                nonfinite = finite.detach() * float("nan")
                yield ObjectiveUnit(
                    unit_id="nonfinite",
                    objective_identity="actual",
                    raw_total=nonfinite,
                    backward_total=nonfinite,
                )

            return ObjectiveExecution("partial", units())

    model = nn.Linear(1, 1, bias=False)
    initial = model.weight.detach().clone()
    executor = TrainingAttemptExecutor()
    runtime, _ = _runtime(model, _PartialChunkExecutor(), executor)
    request = TrainingAttemptRequest(
        attempt_id="partial",
        update_index=0,
        batches=(BatchEnvelope("batch", torch.ones(1, 1)),),
        objective_contexts=(
            ObjectiveContext(
                objective_id="requested",
                training=True,
                normalization_divisor=1.0,
            ),
        ),
        accumulation_steps=1,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    result = executor.execute(
        runtime.bind_live_model(ModelHandle("linear", model)),
        request,
        optimizer=optimizer,
    )

    assert result.status == "failed"
    assert result.reason == (
        "partial_microbatch_nonfinite_objective:nonfinite"
    )
    assert result.finite_microbatch_count == 0
    torch.testing.assert_close(model.weight, initial)


@pytest.mark.parametrize(
    ("divisor", "loss_cap", "amp_scale", "message"),
    (
        (2.0, None, 1.0, "normalization_divisor"),
        (1.0, 3.0, 1.0, "loss_cap"),
        (1.0, None, 2.0, "amp_scale"),
    ),
)
def test_objective_unit_provenance_must_match_context(
    divisor: float,
    loss_cap: float | None,
    amp_scale: float,
    message: str,
) -> None:
    class _MismatchedExecutor:
        definition_version = 1

        def execute(self, handle, batch, context):
            raw = torch.mean(handle.model(batch.payload) ** 2)
            return ObjectiveExecution(
                execution_id="mismatch",
                units=(
                    ObjectiveUnit(
                        unit_id="unit",
                        objective_identity="host_actual_objective",
                        raw_total=raw,
                        backward_total=raw,
                        normalization_divisor=divisor,
                        loss_cap=loss_cap,
                        amp_scale=amp_scale,
                    ),
                ),
            )

    events: list[str] = []
    executor = TrainingAttemptExecutor()
    model = nn.Linear(1, 1, bias=False)
    runtime, _ = _runtime(model, _MismatchedExecutor(), executor)
    session = runtime.bind_live_model(ModelHandle("linear", model))
    request = TrainingAttemptRequest(
        attempt_id="mismatch",
        update_index=0,
        batches=(BatchEnvelope("batch", torch.ones(1, 1)),),
        objective_contexts=(
            ObjectiveContext(
                objective_id="training_total",
                training=True,
                normalization_divisor=1.0,
            ),
        ),
        accumulation_steps=1,
        apply_optimizer=False,
    )

    with pytest.raises(ValueError, match=message):
        executor.execute(session, request)


def test_nonfinite_post_unscale_gradient_updates_scaler_state() -> None:
    class _FakeScaler:
        def __init__(self) -> None:
            self.scale_value = 4.0
            self.update_calls = 0
            self.step_calls = 0

        def is_enabled(self):
            return True

        def get_scale(self):
            return self.scale_value

        def scale(self, value):
            return value * self.scale_value

        def unscale_(self, optimizer):
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    if parameter.grad is not None:
                        parameter.grad.div_(self.scale_value)

        def step(self, optimizer):
            self.step_calls += 1
            optimizer.step()

        def update(self):
            self.update_calls += 1

    model = nn.Linear(1, 1, bias=False)
    model.weight.register_hook(
        lambda gradient: torch.full_like(gradient, float("inf"))
    )
    scaler = _FakeScaler()
    executor = TrainingAttemptExecutor()
    runtime, _ = _runtime(model, _ObjectiveExecutor([]), executor)
    context = ObjectiveContext(
        objective_id="training_total",
        training=True,
        normalization_divisor=1.0,
        amp_scale=4.0,
    )
    request = TrainingAttemptRequest(
        attempt_id="nonfinite-gradient",
        update_index=0,
        batches=(
            BatchEnvelope(
                "batch",
                (torch.ones(1, 1), torch.zeros(1, 1)),
            ),
        ),
        objective_contexts=(context,),
        accumulation_steps=1,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    result = executor.execute(
        runtime.bind_live_model(ModelHandle("linear", model)),
        request,
        optimizer=optimizer,
        scaler=scaler,
    )

    assert result.status == "failed"
    assert result.reason == "nonfinite_gradient"
    assert scaler.update_calls == 1
    assert scaler.step_calls == 0


def test_composite_instrumentation_preserves_order() -> None:
    events: list[str] = []
    composite = CompositeAttemptInstrumentation(
        (
            _RecordingInstrumentation("first", events),
            _RecordingInstrumentation("second", events),
        )
    )
    context = type(
        "Context",
        (),
        {
            "attempt_id": "a",
            "runtime_id": "r",
            "update_index": 0,
            "accumulation_steps": 1,
            "apply_optimizer": False,
            "replay": None,
            "metadata": {},
        },
    )()
    token = composite.begin_attempt(context)
    assert events == ["first:begin-attempt", "second:begin-attempt"]
    assert len(token.values) == 2


def test_live_and_checkpoint_sessions_have_distinct_ownership() -> None:
    events: list[str] = []
    model = nn.Linear(1, 1, bias=False)
    executor = TrainingAttemptExecutor()
    runtime, models = _runtime(model, _ObjectiveExecutor(events), executor)

    live = runtime.bind_live_model(ModelHandle("live", model))
    live.close()
    assert models.closed == []

    checkpoint = RuntimeCheckpointRef("checkpoint", "opaque")
    loaded = runtime.open_session(checkpoint, device="cpu")
    assert loaded.checkpoint == checkpoint
    loaded.close()
    loaded.close()
    assert models.closed == ["linear"]


def test_registry_constructs_runtime_without_global_discovery() -> None:
    events: list[str] = []
    model = nn.Linear(1, 1, bias=False)
    executor = TrainingAttemptExecutor()
    runtime, models = _runtime(model, _ObjectiveExecutor(events), executor)
    definition = TaskDefinition(
        task_definition_id="generic_linear",
        definition_version=3,
        models=models,
        checkpoints=runtime.checkpoints,
        batches=runtime.batches,
        objectives=runtime.objectives,
        attempts=executor,
        checkpoint_objective=CheckpointObjectiveSpec(
            objective_id="generic_linear_objective",
        ),
    )
    registry = TaskRegistry((definition,))

    created = registry.create_runtime("generic_linear", RunRef("new-run"))

    assert created.descriptor.task_definition_version == 3
    assert created.descriptor.run.run_id == "new-run"
    with pytest.raises(ValueError, match="duplicate task definition"):
        TaskRegistry((definition, definition))


def test_execution_preparation_capability_is_structural() -> None:
    class Preparation:
        def prepare_execution(self) -> None:
            return None

    assert isinstance(Preparation(), ExecutionPreparationCapability)


def test_runtime_rejects_missing_or_mismatched_objective_version() -> None:
    """v4 identity 不得把未版本化或错绑 objective 退化为占位字符串。"""

    class UnversionedObjective:
        def execute(self, handle, batch, context):
            raise AssertionError("not executed")

    models = _ModelProvider()
    checkpoints = _CheckpointProvider(nn.Linear(1, 1).state_dict())
    batches = _BatchProvider()
    attempts = TrainingAttemptExecutor()
    specification = CheckpointObjectiveSpec(
        objective_id="generic_linear_objective",
        precision="BF16",
    )
    assert specification.precision == "bf16"
    with pytest.raises(
        ValueError,
        match="objectives.definition_version",
    ):
        TaskDefinition(
            task_definition_id="unversioned",
            models=models,
            checkpoints=checkpoints,
            batches=batches,
            objectives=UnversionedObjective(),
            attempts=attempts,
            checkpoint_objective=specification,
        )

    versioned = _ObjectiveExecutor([])
    with pytest.raises(
        ValueError,
        match="objective_executor_version does not match",
    ):
        HostTaskRuntime(
            descriptor=RuntimeDescriptor(
                runtime_id="mismatched",
                task_definition_id="generic_linear",
                task_definition_version=1,
                run=RunRef("run"),
                objective_executor_version=versioned.definition_version + 1,
                checkpoint_objective=specification,
            ),
            models=models,
            checkpoints=checkpoints,
            batches=batches,
            objectives=versioned,
            attempts=attempts,
        )


def test_package_has_only_standard_library_pytorch_and_relative_imports() -> None:
    package_root = Path(__file__).resolve().parents[2] / "host_runtime"
    allowed_roots = set(sys.stdlib_module_names) | {"torch"}
    for source_path in package_root.glob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = {alias.name.split(".", 1)[0] for alias in node.names}
                assert roots <= allowed_roots
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                root = (node.module or "").split(".", 1)[0]
                assert root in allowed_roots
