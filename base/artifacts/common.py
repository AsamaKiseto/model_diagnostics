"""提供在线、checkpoint 与 reporting 共用的 diagnostics v2 artifact 原语。

本模块只依赖 Python 标准库，拥有 canonical hash、原子 JSON 写入、v2 source
inventory 和顶层 index。各执行路径保留自己的 writer、transaction 与 source
validator，不通过其它执行子包取得这些跨路径原语。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping
import uuid


V2_ROOT_FORMAT_VERSION = "diagnostics_v2"


@dataclass(frozen=True)
class ValidationIssue:
    """一个 artifact 完整性问题。"""

    code: str
    severity: str
    message: str
    path: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "path": self.path,
        }


@dataclass(frozen=True)
class V2ArtifactInventory:
    """调用方 run directory 中可发现的 diagnostics v2 sources。"""

    root: Path
    online_sessions: tuple[Path, ...]
    checkpoint_analyses: tuple[Path, ...]
    failure_bundles: tuple[Path, ...] = ()


@dataclass(frozen=True, slots=True)
class ArtifactTransaction:
    """描述一个不可拆分写入的 task-neutral artifact transaction。"""

    transaction_id: str
    transaction_kind: str
    payload: Mapping[str, Any]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not str(self.transaction_id).strip():
            raise ValueError("ArtifactTransaction.transaction_id must not be empty")
        if not str(self.transaction_kind).strip():
            raise ValueError("ArtifactTransaction.transaction_kind must not be empty")
        if int(self.schema_version) < 1:
            raise ValueError("ArtifactTransaction.schema_version must be positive")
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))


def stable_json_hash(value: Any) -> str:
    """对 JSON-compatible identity payload 生成稳定 SHA-256。"""

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    """流式计算文件 SHA-256，避免 validator 整体物化大文件。"""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: str | Path, value: Mapping[str, Any]) -> None:
    """在目标目录内完成 fsync + replace，供所有 v2 manifest 复用。"""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        try:
            directory_fd = os.open(destination.parent, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def update_v2_index(
    run_dir: str | Path,
    section: str,
    identity: str,
    record: Mapping[str, Any],
) -> Path:
    """保留未知字段地更新调用方 ``diagnostics/v2/index.json`` 的一个条目。"""

    root = Path(run_dir).resolve() / "diagnostics" / "v2"
    path = root / "index.json"
    try:
        current = (
            json.loads(path.read_text(encoding="utf-8"))
            if path.is_file()
            else {}
        )
    except (OSError, json.JSONDecodeError):
        current = {}
    if not isinstance(current, dict):
        current = {}
    entries = current.get(section)
    if not isinstance(entries, dict):
        entries = {}
    entries[str(identity)] = dict(record)
    current.update(
        format_version=V2_ROOT_FORMAT_VERSION,
        updated_at=datetime.now(timezone.utc).isoformat(
            timespec="milliseconds"
        ),
    )
    if section == "checkpoint_analyses":
        runtime_digest = record.get("runtime_descriptor_digest")
        if isinstance(runtime_digest, str) and runtime_digest.strip():
            current["active_checkpoint_runtime_descriptor_digest"] = (
                runtime_digest.strip()
            )
    current[section] = entries
    write_json_atomic(path, current)
    return path


def discover_v2_artifacts(run_dir: str | Path) -> V2ArtifactInventory:
    """只发现调用方 ``diagnostics/v2``，不探测或转换其它 artifact 目录。"""

    root = Path(run_dir).resolve() / "diagnostics" / "v2"
    sessions_root = root / "online" / "sessions"
    analyses_root = root / "checkpoint" / "analyses"
    online_sessions = tuple(
        sorted(
            path.parent
            for path in sessions_root.glob("*/manifest.json")
            if path.is_file()
        )
    )
    checkpoint_analyses = (
        tuple(
            sorted(
                path
                for path in analyses_root.iterdir()
                if path.is_dir()
                and (
                    (path / "manifest.json").is_file()
                    or (path / "run_state.json").is_file()
                )
            )
        )
        if analyses_root.is_dir()
        else ()
    )
    failure_bundles = tuple(
        sorted(
            bundle
            for session in online_sessions
            for failure_root in (session / "failure_bundles",)
            if failure_root.is_dir()
            for bundle in failure_root.iterdir()
            if bundle.is_dir() and (bundle / "ranks").is_dir()
        )
    )
    return V2ArtifactInventory(
        root,
        online_sessions,
        checkpoint_analyses,
        failure_bundles,
    )


__all__ = [
    "ArtifactTransaction",
    "V2ArtifactInventory",
    "V2_ROOT_FORMAT_VERSION",
    "ValidationIssue",
    "discover_v2_artifacts",
    "file_sha256",
    "stable_json_hash",
    "update_v2_index",
    "write_json_atomic",
]
