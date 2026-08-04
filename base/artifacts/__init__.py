"""延迟公开 Base artifact 原语、在线事务和 checkpoint validator。"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_SYMBOL_OWNERS = {
    "ArtifactTransaction": ".common",
    "V2ArtifactInventory": ".common",
    "V2_ROOT_FORMAT_VERSION": ".common",
    "ValidationIssue": ".common",
    "discover_v2_artifacts": ".common",
    "file_sha256": ".common",
    "stable_json_hash": ".common",
    "update_v2_index": ".common",
    "write_json_atomic": ".common",
    "FAILURE_BUNDLE_COMMIT_FORMAT_VERSION": ".failure",
    "FAILURE_BUNDLE_FORMAT_VERSION": ".failure",
    "FAILURE_RANK_COMMIT_FORMAT_VERSION": ".failure",
    "FailureAttachment": ".failure",
    "FailureBundleReference": ".failure",
    "FailureBundleValidation": ".failure",
    "FailureBundleWriter": ".failure",
    "FailureEvidenceScope": ".failure",
    "FailureRankValidation": ".failure",
    "finalize_failure_bundle": ".failure",
    "validate_failure_bundle": ".failure",
    "validate_failure_rank": ".failure",
    "CommittedTransaction": ".online",
    "ONLINE_ARTIFACT_FORMAT_VERSION": ".online",
    "ONLINE_DIAGNOSTICS_FORMAT_VERSION": ".online",
    "OnlineArtifactWriter": ".online",
    "TRANSACTION_FORMAT_VERSION": ".online",
    "ValidationResult": ".online",
    "iter_committed_transactions": ".online",
    "latest_valid_online_hierarchy": ".online",
    "validate_online_session": ".online",
    "DiagnosticsArtifactStore": ".checkpoint",
    "CHECKPOINT_FORMAT_VERSION": ".checkpoint_validation",
    "CHECKPOINT_TRANSACTION_CONTRACT": ".checkpoint_validation",
    "CheckpointTransactionTerminalValidator": ".checkpoint_validation",
    "CheckpointValidation": ".checkpoint_validation",
    "validate_checkpoint_analysis": ".checkpoint_validation",
}

__all__ = sorted(_SYMBOL_OWNERS)


def __getattr__(name: str) -> Any:
    """按职责 owner 延迟加载，避免 writer 与 checkpoint contract 循环。"""

    owner = _SYMBOL_OWNERS.get(name)
    if owner is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(owner, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
