"""提供无全局副作用的显式 task runtime registry。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from .composition import CapabilityCatalog, HostTaskRuntime
from .contracts import (
    AttemptExecutor,
    BatchProvider,
    CheckpointObjectiveSpec,
    CheckpointProvider,
    ModelProvider,
    ObjectiveExecutor,
    RunRef,
    RuntimeDescriptor,
)


@dataclass(frozen=True)
class TaskDefinition:
    """声明一个任务的 provider/capability 组合，不在 import 时自动注册。"""

    task_definition_id: str
    definition_version: int = 1
    models: ModelProvider = field(default=None)
    checkpoints: CheckpointProvider = field(default=None)
    batches: BatchProvider = field(default=None)
    objectives: ObjectiveExecutor = field(default=None)
    attempts: AttemptExecutor = field(default=None)
    checkpoint_objective: CheckpointObjectiveSpec | None = field(default=None)
    device: str = "cpu"
    capabilities: CapabilityCatalog = field(default_factory=CapabilityCatalog)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        task_definition_id = str(self.task_definition_id).strip()
        if not task_definition_id:
            raise ValueError("task_definition_id must be non-empty")
        if int(self.definition_version) <= 0:
            raise ValueError("definition_version must be positive")
        for field_name in (
            "models",
            "checkpoints",
            "batches",
            "objectives",
            "attempts",
        ):
            if getattr(self, field_name) is None:
                raise ValueError(f"{field_name} provider must not be None")
        objective_executor_version = getattr(
            self.objectives,
            "definition_version",
            None,
        )
        if (
            isinstance(objective_executor_version, bool)
            or not isinstance(objective_executor_version, int)
            or objective_executor_version <= 0
        ):
            raise ValueError(
                "objectives.definition_version must be a positive integer"
            )
        if not isinstance(self.checkpoint_objective, CheckpointObjectiveSpec):
            raise TypeError(
                "checkpoint_objective must be CheckpointObjectiveSpec"
            )
        device = str(self.device).strip()
        if not device:
            raise ValueError("device must be non-empty")
        object.__setattr__(self, "task_definition_id", task_definition_id)
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, init=False)
class TaskRegistry:
    """按稳定名称解析任务定义；重复名称在构造期 fail-fast。"""

    definitions: Mapping[str, TaskDefinition]

    def __init__(self, definitions: Iterable[TaskDefinition] = ()) -> None:
        indexed: dict[str, TaskDefinition] = {}
        for definition in definitions:
            if not isinstance(definition, TaskDefinition):
                raise TypeError("definitions must contain TaskDefinition values")
            if definition.task_definition_id in indexed:
                raise ValueError(
                    "duplicate task definition: "
                    f"{definition.task_definition_id!r}"
                )
            indexed[definition.task_definition_id] = definition
        object.__setattr__(self, "definitions", MappingProxyType(indexed))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self.definitions))

    def resolve(self, name: str) -> TaskDefinition:
        normalized = str(name).strip()
        try:
            return self.definitions[normalized]
        except KeyError as exc:
            available = ", ".join(self.names) or "<none>"
            raise KeyError(
                f"Unknown task definition {normalized!r}; available: {available}"
            ) from exc

    def create_runtime(self, name: str, run: RunRef) -> HostTaskRuntime:
        definition = self.resolve(name)
        return HostTaskRuntime(
            descriptor=RuntimeDescriptor(
                runtime_id=f"{definition.task_definition_id}:{run.run_id}",
                task_definition_id=definition.task_definition_id,
                task_definition_version=definition.definition_version,
                run=run,
                objective_executor_version=(
                    definition.objectives.definition_version
                ),
                checkpoint_objective=definition.checkpoint_objective,
                device=definition.device,
                metadata=definition.metadata,
            ),
            models=definition.models,
            checkpoints=definition.checkpoints,
            batches=definition.batches,
            objectives=definition.objectives,
            attempts=definition.attempts,
            capabilities=definition.capabilities,
        )


__all__ = ["TaskDefinition", "TaskRegistry"]
