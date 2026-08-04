"""
隔离 portable diagnostics pytest，使测试收集无法加载宿主 ``src`` package。
"""

from __future__ import annotations

from importlib.abc import MetaPathFinder
import sys

import pytest


class _BlockHostImports(MetaPathFinder):
    """在 portable suite 中拒绝任何直接或传递的 ``src`` import。"""

    def find_spec(self, fullname, path=None, target=None):
        del path, target
        if fullname == "src" or fullname.startswith("src."):
            raise ImportError(
                f"portable model_diagnostics tests cannot import host module {fullname!r}"
            )
        return None


_HOST_IMPORT_BLOCKER = _BlockHostImports()
sys.meta_path.insert(0, _HOST_IMPORT_BLOCKER)


def pytest_collection_modifyitems(items) -> None:
    """统一标记 package-owned portable contract tests。"""

    for item in items:
        item.add_marker(pytest.mark.portable)


def pytest_unconfigure(config) -> None:
    """测试会话结束后移除 import guard，避免污染嵌入式 pytest 调用方。"""

    del config
    if _HOST_IMPORT_BLOCKER in sys.meta_path:
        sys.meta_path.remove(_HOST_IMPORT_BLOCKER)

