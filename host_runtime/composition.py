"""组合宿主 providers、可选 capability 与已加载任务 session。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import torch
from torch import nn

from .contracts import (
    AttemptExecutor,
    BatchProvider,
    CapabilityDescriptor,
    CheckpointProvider,
    CheckpointState,
    ModelHandle,
    ModelProvider,
    ObjectiveExecutor,
    RuntimeCheckpointRef,
    RuntimeDescriptor,
)


@dataclass(frozen=True)
class CapabilityBinding:
    """把 capability descriptor 与宿主实现显式绑定。"""

    descriptor: CapabilityDescriptor
    implementation: Any

    def __post_init__(self) -> None:
        if self.implementation is None:
            raise ValueError("capability implementation must not be None")


@dataclass(frozen=True)
class CapabilityCatalog:
    """保存显式 capability，不执行 entry-point 或 import-time discovery。"""

    entries: Mapping[str, CapabilityBinding] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized: dict[str, Any] = {}
        for raw_name, binding in self.entries.items():
            name = str(raw_name).strip()
            if not name:
                raise ValueError("capability name must be non-empty")
            if not isinstance(binding, CapabilityBinding):
                raise TypeError("capability entries must be CapabilityBinding values")
            if binding.descriptor.capability_id != name:
                raise ValueError(
                    f"capability key {name!r} does not match descriptor "
                    f"{binding.descriptor.capability_id!r}"
                )
            if name in normalized:
                raise ValueError(f"duplicate capability name: {name!r}")
            normalized[name] = binding
        object.__setattr__(self, "entries", MappingProxyType(normalized))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self.entries))

    def get(self, name: str) -> Any | None:
        binding = self.entries.get(str(name))
        return None if binding is None else binding.implementation

    def require(self, name: str) -> Any:
        normalized = str(name).strip()
        try:
            return self.entries[normalized].implementation
        except KeyError as exc:
            available = ", ".join(self.names) or "<none>"
            raise KeyError(
                f"Runtime capability {normalized!r} is unavailable; available: {available}"
            ) from exc


@dataclass
class ExecutionSession:
    """持有 live 或 checkpoint-restored 模型的执行上下文。

    `owns_handle=False` 用于 live training，关闭 session 不释放训练模型；
    checkpoint session 则由 runtime 统一释放，避免 analyzer 各自猜测生命周期。
    """

    runtime: "HostTaskRuntime"
    handle: ModelHandle
    checkpoint_state: CheckpointState | None = None
    owns_handle: bool = False
    _closed: bool = field(default=False, init=False, repr=False)

    @property
    def model(self) -> nn.Module:
        return self.handle.model

    @property
    def checkpoint(self) -> RuntimeCheckpointRef | None:
        if self.checkpoint_state is None:
            return None
        return self.checkpoint_state.checkpoint

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        if self._closed:
            return
        if self.owns_handle:
            self.runtime.models.close(self.handle)
        self._closed = True

    def __enter__(self) -> "ExecutionSession":
        if self._closed:
            raise RuntimeError("cannot re-enter a closed ExecutionSession")
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


class LoadedTaskSession(ExecutionSession):
    """明确表示模型来自 checkpoint restore 的 owned session。"""


@dataclass(frozen=True)
class HostTaskRuntime:
    """一个任务在统一框架中的显式 provider/capability 组合。"""

    descriptor: RuntimeDescriptor
    models: ModelProvider
    checkpoints: CheckpointProvider
    batches: BatchProvider
    objectives: ObjectiveExecutor
    attempts: AttemptExecutor
    capabilities: CapabilityCatalog = field(default_factory=CapabilityCatalog)

    def __post_init__(self) -> None:
        objective_executor_version = getattr(
            self.objectives,
            "definition_version",
            None,
        )
        if objective_executor_version != (
            self.descriptor.objective_executor_version
        ):
            raise ValueError(
                "RuntimeDescriptor objective_executor_version does not match "
                "the bound ObjectiveExecutor"
            )

    def bind_live_model(self, handle: ModelHandle) -> ExecutionSession:
        """为 live training 模型创建不接管资源所有权的 session。"""

        return ExecutionSession(
            runtime=self,
            handle=handle,
            checkpoint_state=None,
            owns_handle=False,
        )

    def open_session(
        self,
        checkpoint: RuntimeCheckpointRef,
        *,
        device: torch.device | str,
    ) -> LoadedTaskSession:
        """加载 checkpoint、构造模型并恢复状态，失败时释放已构造模型。"""

        resolved_device = torch.device(device)
        state = self.checkpoints.load(checkpoint, map_location=resolved_device)
        if state.checkpoint != checkpoint:
            raise ValueError(
                "checkpoint provider returned state for a different CheckpointRef"
            )
        handle = self.models.create(
            state.model_spec,
            state.build_payload,
            device=resolved_device,
        )
        try:
            self.checkpoints.restore(handle, state)
        except BaseException:
            self.models.close(handle)
            raise
        return LoadedTaskSession(
            runtime=self,
            checkpoint_state=state,
            handle=handle,
            owns_handle=True,
        )


__all__ = [
    "CapabilityBinding",
    "CapabilityCatalog",
    "ExecutionSession",
    "HostTaskRuntime",
    "LoadedTaskSession",
]
