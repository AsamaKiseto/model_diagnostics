"""定义 checkpoint artifact 的版本与任务无关运行时身份规范。"""

from __future__ import annotations

import json
import math
from typing import Any, Mapping, Sequence

from .common import stable_json_hash


CHECKPOINT_FORMAT_VERSION = "checkpoint_diagnostics_v4"
RUNTIME_DESCRIPTOR_FIELDS = (
    "runtime_descriptor_digest",
    "task_definition_id",
    "task_definition_version",
    "objective_executor_version",
    "capability_descriptors",
)


def _json_safe_copy(value: Any, *, path: str) -> Any:
    """复制严格 JSON 值，拒绝非有限数和会被 ``default=str`` 隐式改写的对象。"""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must not contain non-finite numbers")
        return value
    if isinstance(value, Mapping):
        copied: dict[str, Any] = {}
        for raw_key, item in value.items():
            if not isinstance(raw_key, str) or not raw_key:
                raise ValueError(f"{path} keys must be non-empty strings")
            copied[raw_key] = _json_safe_copy(
                item,
                path=f"{path}.{raw_key}",
            )
        return copied
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [
            _json_safe_copy(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(f"{path} contains a non-JSON value: {type(value).__name__}")


def canonical_capability_descriptors(
    descriptors: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """规范化 capability 描述符集合，使语义相同但顺序不同的集合身份一致。"""

    canonical: list[tuple[str, dict[str, Any]]] = []
    for index, descriptor in enumerate(descriptors):
        if not isinstance(descriptor, Mapping) or not descriptor:
            raise ValueError(
                f"capability_descriptors[{index}] must be a non-empty object"
            )
        copied = _json_safe_copy(
            descriptor,
            path=f"capability_descriptors[{index}]",
        )
        encoded = json.dumps(
            copied,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        canonical.append((encoded, copied))
    canonical.sort(key=lambda item: item[0])
    if len({encoded for encoded, _descriptor in canonical}) != len(canonical):
        raise ValueError("capability_descriptors must not contain duplicates")
    return [descriptor for _encoded, descriptor in canonical]


def runtime_descriptor_payload(
    *,
    task_definition_id: str,
    task_definition_version: str,
    objective_executor_version: str,
    capability_descriptors: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """返回 runtime digest 的唯一 canonical payload，不包含 digest 自身。"""

    normalized_task_id = str(task_definition_id).strip()
    normalized_task_version = str(task_definition_version).strip()
    normalized_objective_version = str(objective_executor_version).strip()
    if not normalized_task_id:
        raise ValueError("task_definition_id must not be empty")
    if not normalized_task_version:
        raise ValueError("task_definition_version must not be empty")
    if not normalized_objective_version:
        raise ValueError("objective_executor_version must not be empty")
    return {
        "task_definition_id": normalized_task_id,
        "task_definition_version": normalized_task_version,
        "objective_executor_version": normalized_objective_version,
        "capability_descriptors": canonical_capability_descriptors(
            capability_descriptors
        ),
    }


def normalize_runtime_descriptor(
    descriptor: Mapping[str, Any],
) -> dict[str, Any]:
    """校验调用方 runtime identity，并返回可直接持久化的 canonical mapping。"""

    missing = sorted(set(RUNTIME_DESCRIPTOR_FIELDS) - set(descriptor))
    unknown = sorted(set(descriptor) - set(RUNTIME_DESCRIPTOR_FIELDS))
    if missing or unknown:
        raise ValueError(
            "runtime_descriptor fields mismatch: "
            f"missing={missing}, unknown={unknown}"
        )
    raw_capabilities = descriptor.get("capability_descriptors")
    if not isinstance(raw_capabilities, Sequence) or isinstance(
        raw_capabilities,
        (str, bytes),
    ):
        raise TypeError("capability_descriptors must be a sequence of objects")
    payload = runtime_descriptor_payload(
        task_definition_id=str(descriptor.get("task_definition_id", "")),
        task_definition_version=str(
            descriptor.get("task_definition_version", "")
        ),
        objective_executor_version=str(
            descriptor.get("objective_executor_version", "")
        ),
        capability_descriptors=raw_capabilities,
    )
    digest = str(descriptor.get("runtime_descriptor_digest", "")).strip().lower()
    expected_digest = stable_json_hash(payload)
    if digest != expected_digest:
        raise ValueError(
            "runtime_descriptor_digest does not match the canonical runtime "
            "descriptor payload"
        )
    return {
        "runtime_descriptor_digest": digest,
        **payload,
    }


def build_checkpoint_analysis_identity_payload(
    *,
    recipe: Mapping[str, Any],
    analyzer_definitions: Sequence[Mapping[str, Any]],
    checkpoint_identities: Sequence[Any],
    cohort_status: Any,
    cohort_identity: Any,
    cohort_samples: Sequence[Mapping[str, Any]],
    runtime_descriptor: Mapping[str, Any],
) -> dict[str, Any]:
    """按当前 artifact contract 构造 analysis identity。"""

    identity = {
        "recipe": dict(recipe),
        "analyzer_definitions": [dict(item) for item in analyzer_definitions],
        "checkpoints": list(checkpoint_identities),
        "cohort": {
            "status": cohort_status,
            "identity": cohort_identity,
            "samples": [dict(sample) for sample in cohort_samples],
        },
    }
    identity.update(normalize_runtime_descriptor(runtime_descriptor))
    return identity


__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "RUNTIME_DESCRIPTOR_FIELDS",
    "build_checkpoint_analysis_identity_payload",
    "canonical_capability_descriptors",
    "normalize_runtime_descriptor",
    "runtime_descriptor_payload",
]
