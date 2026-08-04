"""公开通用 PyTorch Stage/Block 层级发现。"""

from .discovery import (
    HierarchyDiscoveryPolicy,
    HierarchyDiscoveryResult,
    discover_model_hierarchy,
)

__all__ = [
    "HierarchyDiscoveryPolicy",
    "HierarchyDiscoveryResult",
    "discover_model_hierarchy",
]
