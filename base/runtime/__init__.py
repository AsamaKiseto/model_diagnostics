"""公开三阶段诊断仍使用的任务无关 runtime contract。"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_PUBLIC_EXPORTS = {
    "ObjectiveTrace": (".contracts", "ObjectiveTrace"),
    "ObjectiveContraction": (".objective", "ObjectiveContraction"),
    "contract_backward_objective": (
        ".objective",
        "contract_backward_objective",
    ),
}

__all__ = sorted(_PUBLIC_EXPORTS)


def __getattr__(name: str) -> Any:
    """按实际 owner 延迟解析公共符号，避免仅读 DTO 时加载 controller。"""

    try:
        module_name, attribute_name = _PUBLIC_EXPORTS[name]
    except KeyError as error:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from error
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
