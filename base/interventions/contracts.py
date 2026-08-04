"""定义任务无关的 module/tensor intervention site。

阶段: diagnostics Base contract。该模块只描述可观测或可干预位置，不决定输入、
donor、序列或宿主任务语义；checkpoint runner 与在线 probe 都可依赖这些 DTO。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping


def canonical_activation_output_path(value: str | None) -> str | None:
    """把 module output path 规范化为以 ``$`` 为根的稳定表示。"""

    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    if raw in {"$", "output"}:
        return "$"
    if raw.startswith("output."):
        parts = raw[len("output.") :].split(".")
        path = "$"
        for part in parts:
            path += f"[{int(part)}]" if part.isdigit() else f".{part}"
        return path
    if raw.startswith("$."):
        parts = raw[2:].split(".")
        path = "$"
        for part in parts:
            path += f"[{int(part)}]" if part.isdigit() else f".{part}"
        return path
    return raw


@dataclass(frozen=True, slots=True)
class ModuleSite:
    """标识可观测或可干预的 ``nn.Module`` output site。"""

    site_id: str
    module_path: str
    module_type: str
    node_id: str | None = None
    parent_node_id: str | None = None
    stage_id: str | None = None
    invocation_index: int | None = None
    output_path: str | None = None
    alias_node_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.site_id).strip():
            raise ValueError("ModuleSite.site_id must not be empty")
        if self.invocation_index is not None and int(self.invocation_index) < 0:
            raise ValueError("ModuleSite.invocation_index must be non-negative")
        object.__setattr__(
            self,
            "output_path",
            canonical_activation_output_path(self.output_path),
        )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class TensorSite:
    """标识任意 tensor 容器中的一个 task-neutral site。"""

    path: str
    axis: int | None = None
    index: int | tuple[int, ...] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.path).strip():
            raise ValueError("TensorSite.path must not be empty")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


__all__ = [
    "ModuleSite",
    "TensorSite",
    "canonical_activation_output_path",
]
