"""提供与任务和模型源码解耦的 module output 干预机制。

``ModuleOutputIntervention`` 通过 forward hook 对 Tensor 或嵌套容器应用 identity、
scale、detach、constant、mean 或显式 donor patch，并在 condition 退出时移除 hook。
"""

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass, replace
import math
from types import MappingProxyType
from typing import Any, Iterator

import torch

from .contracts import (
    ModuleSite,
    TensorSite,
    canonical_activation_output_path,
)


class UnsupportedModuleOutput(TypeError):
    """模块输出不含可安全转换 Tensor 时的 structured-skip 原因。"""


@dataclass(frozen=True, slots=True)
class InterventionSpec:
    """描述一个 task-neutral module 或 tensor intervention。"""

    method: str
    module_site: ModuleSite | None = None
    tensor_site: TensorSite | None = None
    scale: float = 1.0
    replacement: Any = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        method = str(self.method).strip().lower()
        if method not in {
            "identity",
            "output_scale",
            "detach",
            "constant_patch",
            "mean_patch",
            "explicit_donor_patch",
        }:
            raise ValueError(f"Unsupported intervention method: {method}")
        if self.module_site is None and self.tensor_site is None:
            raise ValueError("InterventionSpec requires module_site or tensor_site")
        if self.module_site is not None and self.tensor_site is not None:
            raise ValueError("InterventionSpec cannot target module and tensor together")
        if not math.isfinite(float(self.scale)):
            raise ValueError("InterventionSpec.scale must be finite")
        if method.endswith("_patch") and self.replacement is None:
            raise ValueError(f"{method} requires an explicit replacement")
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))


def capture_module_modes(model: torch.nn.Module) -> dict[str, bool]:
    """捕获完整 module tree 的 mixed train/eval mode。"""

    return {name: bool(module.training) for name, module in model.named_modules()}


def restore_module_modes(
    model: torch.nn.Module,
    modes: Mapping[str, bool],
) -> None:
    """在 module tree identity 未改变时精确恢复 mixed mode。"""

    modules = dict(model.named_modules())
    if set(modules) != set(modes):
        raise RuntimeError("module tree changed while restoring diagnostic branch modes")
    for name, mode in modes.items():
        modules[name].training = bool(mode)


@contextmanager
def preserve_module_modes(model: torch.nn.Module) -> Iterator[None]:
    """在诊断分支退出时恢复完整 mixed train/eval mode，包括异常路径。"""

    modes = capture_module_modes(model)
    try:
        yield
    finally:
        restore_module_modes(model, modes)


@dataclass(slots=True)
class ModuleOutputIntervention:
    """在一个 condition 生命周期内管理单个模块的输出 hook。"""

    module: torch.nn.Module
    spec: InterventionSpec
    call_count: int = 0
    transformed_call_count: int = 0
    tensor_count: int = 0
    forward_call_count: int | None = None
    _handle: Any = None

    def __enter__(self) -> "ModuleOutputIntervention":
        """注册 hook；同一模块每次调用默认都执行相同干预。"""

        if self.spec.module_site is None:
            raise ValueError("ModuleOutputIntervention requires spec.module_site")
        self._handle = self.module.register_forward_hook(self._hook)
        return self

    def _hook(self, _module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
        invocation = self.call_count
        self.call_count += 1
        logical_invocation = (
            invocation
            if self.forward_call_count is None or self.forward_call_count <= 0
            else invocation % self.forward_call_count
        )
        site = self.spec.module_site
        assert site is not None
        if (
            site.invocation_index is not None
            and logical_invocation != int(site.invocation_index)
        ):
            return output
        transformed, tensor_count = transform_module_output(
            output,
            method=self.spec.method,
            scale=self.spec.scale,
            replacement=self.spec.replacement,
            output_path=site.output_path,
        )
        if tensor_count == 0:
            raise UnsupportedModuleOutput(
                "module output contains no Tensor for intervention "
                f"method={self.spec.method}"
            )
        self.transformed_call_count += 1
        self.tensor_count += int(tensor_count)
        return transformed

    def mark_forward_complete(self) -> None:
        """记录原始 forward 调用数，使单调用模块的 checkpoint 重算复用同一选择。"""

        self.forward_call_count = int(self.call_count)

    def close(self) -> None:
        """幂等移除 hook，防止 condition 间状态泄漏。"""

        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


@dataclass(slots=True)
class ModuleOutputCapture:
    """捕获指定 invocation/output path 的 detached 输出。

    ``output_path=None`` 保留完整结构，供 patch reference 使用；调用方也可以传
    ``$`` 或具体 leaf path，使 capture 与 intervention 使用同一选择语义。
    """

    module: torch.nn.Module
    invocation_index: int | None = None
    output_path: str | None = None
    call_count: int = 0
    output: Any = None
    _handle: Any = None

    def __enter__(self) -> "ModuleOutputCapture":
        if self.output_path is not None:
            self.output_path = canonical_activation_output_path(self.output_path)
        self._handle = self.module.register_forward_hook(self._hook)
        return self

    def _hook(self, _module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        invocation = self.call_count
        self.call_count += 1
        if self.invocation_index is not None and invocation != int(self.invocation_index):
            return
        if self.output is None:
            selected = (
                output
                if self.output_path is None
                else select_module_output_path(output, self.output_path)
            )
            self.output = clone_tensor_structure(selected)

    def close(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def apply_tensor_intervention(
    tensor: torch.Tensor,
    spec: InterventionSpec,
) -> torch.Tensor:
    """对调用方显式提供的完整 Tensor site 应用任务无关 intervention。

    Base 不按 ``TensorSite.path/axis/index`` 进入任意宿主容器，也不猜测 donor；
    调用方必须先解析并传入真正参与后续计算的目标 Tensor。identity 返回原对象，其它
    方法复用 module-output 的数值与 autograd 语义。
    """

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor must be a torch.Tensor")
    if not isinstance(spec, InterventionSpec):
        raise TypeError("spec must be an InterventionSpec")
    if spec.tensor_site is None:
        raise ValueError("apply_tensor_intervention requires spec.tensor_site")
    transformed, tensor_count = transform_module_output(
        tensor,
        method=spec.method,
        scale=spec.scale,
        replacement=spec.replacement,
        output_path="$",
    )
    if tensor_count != 1 or not isinstance(transformed, torch.Tensor):
        raise UnsupportedModuleOutput(
            "explicit Tensor site did not produce exactly one Tensor"
        )
    return transformed


def transform_module_output(
    output: Any,
    *,
    method: str,
    scale: float,
    replacement: Any = None,
    output_path: str | None = None,
    _path: str = "$",
) -> tuple[Any, int]:
    """递归转换 Tensor 与常见结构容器，同时保留类型和非 Tensor metadata。

    dataclass 使用 ``dataclasses.replace`` 重建原类型；含 Tensor 的 ``init=False``
    字段无法由该公共协议安全替换，因此返回零转换并让 condition structured skip。
    """

    selected = output_path is None or str(output_path) in {"", "$", _path}
    if torch.is_tensor(output):
        if not selected:
            return output, 0
        if method == "identity":
            return output, 1
        if method == "output_scale":
            return output * float(scale), 1
        if method == "detach":
            # 零系数 graph edge 让 terminal module 仍可 backward，同时对该输出上游的
            # 导数严格为 0；数值等于 detach 前输出。
            return output.detach() + output * 0.0, 1
        if method == "constant_patch":
            if torch.is_tensor(replacement):
                if replacement.numel() != 1 and replacement.shape != output.shape:
                    raise UnsupportedModuleOutput(
                        f"constant replacement shape mismatch at {_path}"
                    )
                patched = replacement.to(device=output.device, dtype=output.dtype)
                if patched.numel() == 1:
                    patched = torch.full_like(output, float(patched.item()))
            elif isinstance(replacement, (int, float, bool)):
                patched = torch.full_like(output, float(replacement))
            else:
                raise UnsupportedModuleOutput(
                    f"constant replacement is not scalar/tensor at {_path}"
                )
            return patched.detach() + output * 0.0, 1
        if method == "mean_patch":
            if not torch.is_tensor(replacement):
                raise UnsupportedModuleOutput(
                    f"replacement shape/type mismatch at {_path}: "
                    f"output={tuple(output.shape)}, replacement={getattr(replacement, 'shape', None)}"
                )
            patched = replacement.to(device=output.device, dtype=output.dtype)
            try:
                patched = torch.broadcast_to(patched, output.shape)
            except RuntimeError as error:
                raise UnsupportedModuleOutput(
                    f"mean replacement is not broadcastable at {_path}: "
                    f"output={tuple(output.shape)}, replacement={tuple(patched.shape)}"
                ) from error
            return patched.detach() + output * 0.0, 1
        if method == "explicit_donor_patch":
            if (
                not torch.is_tensor(replacement)
                or replacement.shape != output.shape
            ):
                raise UnsupportedModuleOutput(
                    f"donor replacement shape/type mismatch at {_path}: "
                    f"output={tuple(output.shape)}, replacement={getattr(replacement, 'shape', None)}"
                )
            patched = replacement.to(device=output.device, dtype=output.dtype)
            return patched.detach() + output * 0.0, 1
        raise ValueError(f"Unsupported module intervention: {method}")
    if isinstance(output, tuple):
        replacement_items = (
            replacement
            if isinstance(replacement, tuple)
            else (replacement,) * len(output)
            if method == "constant_patch"
            else (None,) * len(output)
        )
        items = [
            transform_module_output(
                item,
                method=method,
                scale=scale,
                replacement=replacement_items[index] if index < len(replacement_items) else None,
                output_path=output_path,
                _path=f"{_path}[{index}]",
            )
            for index, item in enumerate(output)
        ]
        values = tuple(item[0] for item in items)
        if hasattr(output, "_fields"):
            values = type(output)(*values)
        return values, sum(item[1] for item in items)
    if isinstance(output, list):
        replacement_items = (
            replacement
            if isinstance(replacement, list)
            else [replacement] * len(output)
            if method == "constant_patch"
            else [None] * len(output)
        )
        items = [
            transform_module_output(
                item,
                method=method,
                scale=scale,
                replacement=replacement_items[index] if index < len(replacement_items) else None,
                output_path=output_path,
                _path=f"{_path}[{index}]",
            )
            for index, item in enumerate(output)
        ]
        return [item[0] for item in items], sum(item[1] for item in items)
    if isinstance(output, Mapping):
        replacement_mapping = (
            replacement
            if isinstance(replacement, Mapping)
            else {key: replacement for key in output}
            if method == "constant_patch"
            else {}
        )
        items = {
            key: transform_module_output(
                value,
                method=method,
                scale=scale,
                replacement=replacement_mapping.get(key),
                output_path=output_path,
                _path=f"{_path}.{key}",
            )
            for key, value in output.items()
        }
        return {key: item[0] for key, item in items.items()}, sum(item[1] for item in items.values())
    if is_dataclass(output) and not isinstance(output, type):
        replacement_object = replacement if is_dataclass(replacement) else None
        replacements: dict[str, Any] = {}
        tensor_count = 0
        for field in fields(output):
            transformed, field_tensor_count = transform_module_output(
                getattr(output, field.name),
                method=method,
                scale=scale,
                replacement=(
                    getattr(replacement_object, field.name)
                    if replacement_object is not None
                    else replacement
                    if method == "constant_patch"
                    else None
                ),
                output_path=output_path,
                _path=f"{_path}.{field.name}",
            )
            if field_tensor_count and not field.init:
                return output, 0
            if field_tensor_count:
                replacements[field.name] = transformed
                tensor_count += int(field_tensor_count)
        return (
            replace(output, **replacements) if tensor_count else output,
            tensor_count,
        )
    return output, 0


def iter_module_output_tensors(
    output: Any,
    *,
    _path: str = "$",
) -> list[tuple[str, torch.Tensor]]:
    """按 intervention 的 ``$`` path 语法枚举 structured output Tensor。

    返回顺序对 Mapping key 做稳定排序；baseline 与 intervention branch 因而可以用
    ``path + shape + deterministic indices`` 对齐有限 Taylor comparison，而无需依赖
    模型私有输出类型。
    """

    if torch.is_tensor(output):
        return [(_path, output)]
    if isinstance(output, (tuple, list)):
        return [
            pair
            for index, item in enumerate(output)
            for pair in iter_module_output_tensors(
                item,
                _path=f"{_path}[{index}]",
            )
        ]
    if isinstance(output, Mapping):
        return [
            pair
            for key in sorted(output, key=lambda value: str(value))
            for pair in iter_module_output_tensors(
                output[key],
                _path=f"{_path}.{key}",
            )
        ]
    if is_dataclass(output) and not isinstance(output, type):
        return [
            pair
            for field in fields(output)
            for pair in iter_module_output_tensors(
                getattr(output, field.name),
                _path=f"{_path}.{field.name}",
            )
        ]
    return []


def clone_tensor_structure(output: Any) -> Any:
    """递归复制 Tensor storage，同时保留常见输出容器和 metadata。"""

    if torch.is_tensor(output):
        return output.detach().clone()
    if isinstance(output, tuple):
        values = tuple(clone_tensor_structure(item) for item in output)
        return type(output)(*values) if hasattr(output, "_fields") else values
    if isinstance(output, list):
        return [clone_tensor_structure(item) for item in output]
    if isinstance(output, Mapping):
        return {key: clone_tensor_structure(value) for key, value in output.items()}
    if is_dataclass(output) and not isinstance(output, type):
        replacements: dict[str, Any] = {}
        for field in fields(output):
            value = clone_tensor_structure(getattr(output, field.name))
            if field.init:
                replacements[field.name] = value
            elif _contains_tensor(value):
                raise UnsupportedModuleOutput(
                    f"cannot capture Tensor in init=False dataclass field {field.name}"
                )
        return replace(output, **replacements)
    return output


def select_module_output_path(output: Any, output_path: str) -> Any:
    """按 intervention 的 ``$`` path 语法选择 structured output 子树。"""

    path = canonical_activation_output_path(output_path)
    if path is None:
        raise UnsupportedModuleOutput("module output path is missing")
    if path == "$":
        return output
    if not path.startswith("$"):
        raise UnsupportedModuleOutput(f"unsupported module output path: {output_path!r}")
    current = output
    cursor = 1
    while cursor < len(path):
        if path[cursor] == ".":
            end = cursor + 1
            while end < len(path) and path[end] not in ".[":
                end += 1
            key = path[cursor + 1 : end]
            if not key:
                raise UnsupportedModuleOutput(f"invalid module output path: {path!r}")
            if isinstance(current, Mapping):
                if key not in current:
                    raise UnsupportedModuleOutput(
                        f"module output path key is unavailable: {path!r}"
                    )
                current = current[key]
            elif is_dataclass(current) and not isinstance(current, type):
                if not hasattr(current, key):
                    raise UnsupportedModuleOutput(
                        f"module output path field is unavailable: {path!r}"
                    )
                current = getattr(current, key)
            elif isinstance(current, tuple) and hasattr(current, "_fields"):
                if key not in current._fields:
                    raise UnsupportedModuleOutput(
                        f"module output path field is unavailable: {path!r}"
                    )
                current = getattr(current, key)
            else:
                raise UnsupportedModuleOutput(
                    f"module output path cannot select field {key!r}: {path!r}"
                )
            cursor = end
            continue
        if path[cursor] == "[":
            end = path.find("]", cursor + 1)
            if end < 0:
                raise UnsupportedModuleOutput(f"invalid module output path: {path!r}")
            try:
                index = int(path[cursor + 1 : end])
            except ValueError as error:
                raise UnsupportedModuleOutput(
                    f"module output index is not an integer: {path!r}"
                ) from error
            if not isinstance(current, (tuple, list)) or not (
                -len(current) <= index < len(current)
            ):
                raise UnsupportedModuleOutput(
                    f"module output index is unavailable: {path!r}"
                )
            current = current[index]
            cursor = end + 1
            continue
        raise UnsupportedModuleOutput(f"invalid module output path: {path!r}")
    return current


def _contains_tensor(value: Any) -> bool:
    if torch.is_tensor(value):
        return True
    if isinstance(value, (tuple, list)):
        return any(_contains_tensor(item) for item in value)
    if isinstance(value, Mapping):
        return any(_contains_tensor(item) for item in value.values())
    if is_dataclass(value) and not isinstance(value, type):
        return any(_contains_tensor(getattr(value, field.name)) for field in fields(value))
    return False


def resolve_module(model: torch.nn.Module, module_path: str) -> torch.nn.Module:
    """按稳定 dotted path 解析模块，并透明剥离常见 wrapper 根。"""

    root = model
    while isinstance(getattr(root, "module", None), torch.nn.Module) and root.__class__.__name__ in {
        "DistributedDataParallel",
        "DataParallel",
    }:
        root = root.module
    if not str(module_path).strip():
        return root
    current = root
    for segment in str(module_path).split("."):
        if segment.isdigit() and isinstance(current, (torch.nn.ModuleList, torch.nn.Sequential)):
            current = current[int(segment)]
            continue
        candidate = getattr(current, segment, None)
        if not isinstance(candidate, torch.nn.Module):
            raise KeyError(f"Unknown module path {module_path!r} at segment {segment!r}")
        current = candidate
    return current


__all__ = [
    "InterventionSpec",
    "ModuleOutputIntervention",
    "ModuleOutputCapture",
    "UnsupportedModuleOutput",
    "apply_tensor_intervention",
    "capture_module_modes",
    "clone_tensor_structure",
    "iter_module_output_tensors",
    "preserve_module_modes",
    "resolve_module",
    "restore_module_modes",
    "select_module_output_path",
    "transform_module_output",
]
