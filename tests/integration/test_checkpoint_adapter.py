"""验证 Generic checkpoint adapter 的显式 session 复用与资源边界。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from model_diagnostics.base.checkpoint import DiagnosticsRecipe
from model_diagnostics.base.registry import (
    BASE_ANALYZER_CATALOG,
    AnalyzerBindingCatalog,
)
from model_diagnostics.host_runtime import (
    CapabilityCatalog,
    CheckpointObjectiveSpec,
    CheckpointState,
    HostTaskRuntime,
    ModelHandle,
    ModelSpec,
    RunRef,
    RuntimeCheckpointRef,
    RuntimeDescriptor,
)
from model_diagnostics.integration import GenericCheckpointAdapter


class _ReusableModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.register_buffer("persistent", torch.zeros(1))
        self.register_buffer(
            "ephemeral",
            torch.tensor([7.0]),
            persistent=False,
        )


class _ModelProvider:
    def __init__(self) -> None:
        self.created: list[str] = []
        self.closed: list[str] = []

    def create(
        self,
        model_spec: ModelSpec,
        build_payload: Any,
        *,
        device: torch.device,
    ) -> ModelHandle:
        self.created.append(model_spec.model_id)
        model = _ReusableModel().to(device)
        return ModelHandle(
            model_id=model_spec.model_id,
            model=model,
            metadata={"checkpoint_sha256": build_payload["checkpoint_id"]},
        )

    def close(self, handle: ModelHandle) -> None:
        self.closed.append(handle.model_id)


class _CheckpointProvider:
    def __init__(
        self,
        refs: tuple[RuntimeCheckpointRef, ...],
        specs: tuple[ModelSpec, ...],
    ) -> None:
        self.refs = refs
        self.specs = dict(zip((ref.checkpoint_id for ref in refs), specs))
        self.load_count = 0
        self.restore_count = 0

    def resolve(self, run: RunRef):
        del run
        return self.refs

    def load(
        self,
        checkpoint: RuntimeCheckpointRef,
        *,
        map_location: torch.device,
    ) -> CheckpointState:
        self.load_count += 1
        value = float(self.refs.index(checkpoint) + 1)
        return CheckpointState(
            checkpoint=checkpoint,
            model_spec=self.specs[checkpoint.checkpoint_id],
            state_payload={
                "weight": torch.tensor([value], device=map_location),
                "persistent": torch.tensor(
                    [value * 10.0],
                    device=map_location,
                ),
            },
            build_payload={"checkpoint_id": checkpoint.checkpoint_id},
            metadata={"checkpoint_sha256": checkpoint.checkpoint_id},
        )

    def restore(self, handle: ModelHandle, state: CheckpointState) -> None:
        self.restore_count += 1
        if handle.model_id != state.model_spec.model_id:
            raise ValueError("model identity mismatch")
        result = handle.model.load_state_dict(state.state_payload, strict=True)
        assert result.missing_keys == []
        assert result.unexpected_keys == []
        handle.model.train()


class _ObjectiveExecutor:
    definition_version = 1

    def execute(self, handle, batch, context):
        raise NotImplementedError


class _UnusedBatchProvider:
    pass


class _UnusedAttemptExecutor:
    pass


def _build_adapter(
    tmp_path: Path,
    *,
    model_ids: tuple[str, str] = ("same", "same"),
) -> tuple[
    GenericCheckpointAdapter,
    _ModelProvider,
    _CheckpointProvider,
]:
    refs = (
        RuntimeCheckpointRef("first", str(tmp_path / "first.pt")),
        RuntimeCheckpointRef("second", str(tmp_path / "second.pt")),
    )
    models = _ModelProvider()
    checkpoints = _CheckpointProvider(
        refs,
        tuple(ModelSpec(model_id, config={"width": 1}) for model_id in model_ids),
    )
    runtime = HostTaskRuntime(
        descriptor=RuntimeDescriptor(
            runtime_id="adapter-reuse-test",
            task_definition_id="adapter-reuse-test",
            task_definition_version=1,
            run=RunRef("run", locator=str(tmp_path)),
            objective_executor_version=1,
            checkpoint_objective=CheckpointObjectiveSpec(
                objective_id="unused",
            ),
            device="cpu",
        ),
        models=models,
        checkpoints=checkpoints,
        batches=_UnusedBatchProvider(),
        objectives=_ObjectiveExecutor(),
        attempts=_UnusedAttemptExecutor(),
        capabilities=CapabilityCatalog(),
    )
    adapter = GenericCheckpointAdapter(
        runtime=runtime,
        run_dir=tmp_path,
        analyzer_catalog=BASE_ANALYZER_CATALOG,
        analyzer_bindings=AnalyzerBindingCatalog({}),
        reuse_compatible_checkpoint_session=True,
    )
    adapter.configure(
        DiagnosticsRecipe.checkpoint_sweep(
            analyzer_catalog=BASE_ANALYZER_CATALOG,
        )
    )
    adapter.resolve_checkpoints(run_dir=tmp_path)
    return adapter, models, checkpoints


def test_compatible_sweep_reuses_model_and_restores_complete_runtime_state(
    tmp_path: Path,
) -> None:
    adapter, models, checkpoints = _build_adapter(tmp_path)

    first = adapter.load_model(
        tmp_path / "first.pt",
        precision="fp32",
    )
    first.model.eval()
    first.model.weight.grad = torch.ones_like(first.model.weight)
    first.model.ephemeral.fill_(99.0)

    second = adapter.load_model(
        tmp_path / "second.pt",
        precision="fp32",
    )

    assert second.model is first.model
    assert models.created == ["same"]
    assert checkpoints.restore_count == 2
    assert second.model.training is True
    assert second.model.weight.grad is None
    assert torch.equal(second.model.weight, torch.tensor([2.0]))
    assert torch.equal(second.model.persistent, torch.tensor([20.0]))
    assert torch.equal(second.model.ephemeral, torch.tensor([7.0]))
    assert second.structure_metadata["checkpoint_sha256"] == "second"
    assert second.structure_metadata["checkpoint_state_metadata"] == {
        "checkpoint_sha256": "second"
    }
    with pytest.raises(RuntimeError, match="already released"):
        adapter.runtime_session(first)

    adapter.close_loaded_model(second)
    assert models.closed == []
    third = adapter.load_model(
        tmp_path / "second.pt",
        precision="fp32",
    )
    assert third.model is second.model
    assert checkpoints.restore_count == 3
    adapter.close_loaded_model(third)
    adapter.close_loaded_model(first)
    adapter.close()
    assert models.closed == ["same"]


def test_incompatible_model_spec_rebuilds_instead_of_reusing(
    tmp_path: Path,
) -> None:
    adapter, models, checkpoints = _build_adapter(
        tmp_path,
        model_ids=("first-model", "second-model"),
    )
    first = adapter.load_model(
        tmp_path / "first.pt",
        precision="fp32",
    )
    second = adapter.load_model(
        tmp_path / "second.pt",
        precision="fp32",
    )

    assert second.model is not first.model
    assert models.created == ["first-model", "second-model"]
    assert models.closed == ["first-model"]
    assert checkpoints.restore_count == 2

    adapter.close_loaded_model(second)
    adapter.close_loaded_model(first)
    adapter.close()
    assert models.closed == ["first-model", "second-model"]


def test_leaked_module_hook_forces_fresh_session(
    tmp_path: Path,
) -> None:
    adapter, models, checkpoints = _build_adapter(tmp_path)
    first = adapter.load_model(
        tmp_path / "first.pt",
        precision="fp32",
    )
    first.model.register_forward_hook(lambda _module, _inputs, output: output)

    second = adapter.load_model(
        tmp_path / "second.pt",
        precision="fp32",
    )

    assert second.model is not first.model
    assert models.created == ["same", "same"]
    assert models.closed == ["same"]
    assert checkpoints.restore_count == 2

    adapter.close_loaded_model(second)
    adapter.close_loaded_model(first)
    adapter.close()
    assert models.closed == ["same", "same"]
