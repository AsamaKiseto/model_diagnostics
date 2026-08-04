"""
提供任务无关 failure bundle v1 的 rank-local 写入、根级提交与完整性校验。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Iterable, Mapping
import uuid

from .common import (
    V2_ROOT_FORMAT_VERSION,
    ValidationIssue,
    file_sha256,
    write_json_atomic,
)


FAILURE_BUNDLE_FORMAT_VERSION = "diagnostics_failure_bundle_v1"
FAILURE_RANK_COMMIT_FORMAT_VERSION = "diagnostics_failure_rank_commit_v1"
FAILURE_BUNDLE_COMMIT_FORMAT_VERSION = "diagnostics_failure_bundle_commit_v1"


class FailureEvidenceScope(str, Enum):
    """声明 failure bundle 只保存异常事实，不承诺训练重放。"""

    METADATA_ONLY = "metadata_only"


@dataclass(frozen=True, slots=True)
class FailureAttachment:
    """绑定一个 rank-local attachment 的相对路径、大小与内容哈希。"""

    relative_path: str
    size_bytes: int
    sha256: str
    media_type: str = "application/octet-stream"

    def __post_init__(self) -> None:
        _validated_relative_path(self.relative_path)
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes < 0
        ):
            raise ValueError(
                "FailureAttachment.size_bytes must be a non-negative integer"
            )
        digest = str(self.sha256).strip().lower()
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError("FailureAttachment.sha256 must be a SHA-256 hex digest")
        if not str(self.media_type).strip():
            raise ValueError("FailureAttachment.media_type must not be empty")
        object.__setattr__(self, "relative_path", str(self.relative_path))
        object.__setattr__(self, "size_bytes", int(self.size_bytes))
        object.__setattr__(self, "sha256", digest)
        object.__setattr__(self, "media_type", str(self.media_type).strip())

    def to_dict(self) -> dict[str, Any]:
        """返回 manifest 与 commit 共用的 canonical descriptor。"""

        return {
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "media_type": self.media_type,
        }


@dataclass(frozen=True, slots=True)
class FailureBundleReference:
    """供 failed-attempt transaction 引用 sidecar bundle 的稳定 DTO。"""

    bundle_id: str
    relative_path: str
    format_version: str = FAILURE_BUNDLE_FORMAT_VERSION

    def __post_init__(self) -> None:
        if not str(self.bundle_id).strip():
            raise ValueError("FailureBundleReference.bundle_id must not be empty")
        _validated_relative_path(self.relative_path)
        if self.format_version != FAILURE_BUNDLE_FORMAT_VERSION:
            raise ValueError(
                "FailureBundleReference.format_version is not supported"
            )

    def to_dict(self) -> dict[str, str]:
        """返回 JSON-safe transaction reference。"""

        return {
            "bundle_id": str(self.bundle_id),
            "relative_path": str(self.relative_path),
            "format_version": self.format_version,
        }


@dataclass(frozen=True, slots=True)
class FailureRankValidation:
    """描述一个 rank-local failure transaction 的验证结果。"""

    rank_dir: Path
    status: str
    valid: bool
    bundle_id: str | None
    rank: int | None
    world_size: int | None
    evidence_scope: str | None
    attachments: tuple[FailureAttachment, ...]
    manifest: Mapping[str, Any]
    issues: tuple[ValidationIssue, ...]


@dataclass(frozen=True, slots=True)
class FailureBundleValidation:
    """聚合全部已发现 rank-local commit，不把缺 rank 提升为完整证据。"""

    bundle_dir: Path
    status: str
    valid: bool
    bundle_id: str | None
    expected_rank_count: int | None
    captured_rank_count: int
    rank_validations: tuple[FailureRankValidation, ...]
    issues: tuple[ValidationIssue, ...]


class FailureBundleWriter:
    """同步写一个 rank-local failure bundle。

    writer 不使用在线 bounded queue。构造时立即原子写 ``running`` header，
    attachment 在目标 rank 目录内逐个 replace，最后才写 commit。各 rank 只拥有
    自己的目录，因而失败路径不需要 collective 或共享 manifest 写锁。
    """

    def __init__(
        self,
        bundle_dir: str | Path,
        *,
        bundle_id: str,
        session_id: str,
        attempt_id: str,
        rank: int,
        world_size: int,
        failure_phase: str,
        failure_reason: str,
        evidence_scope: FailureEvidenceScope | str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not str(bundle_id).strip():
            raise ValueError("bundle_id must not be empty")
        if not str(session_id).strip():
            raise ValueError("session_id must not be empty")
        if not str(attempt_id).strip():
            raise ValueError("attempt_id must not be empty")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
            raise ValueError("rank must be a non-negative integer")
        if (
            isinstance(world_size, bool)
            or not isinstance(world_size, int)
            or world_size <= 0
        ):
            raise ValueError("world_size must be a positive integer")
        if rank >= world_size:
            raise ValueError("rank must be smaller than world_size")
        if not str(failure_phase).strip():
            raise ValueError("failure_phase must not be empty")
        if not str(failure_reason).strip():
            raise ValueError("failure_reason must not be empty")

        self.bundle_dir = Path(bundle_dir).resolve()
        self.bundle_id = str(bundle_id).strip()
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.rank_dir = (
            self.bundle_dir / "ranks" / f"rank_{self.rank:05d}"
        )
        if self.rank_dir.exists():
            raise FileExistsError(f"failure rank directory exists: {self.rank_dir}")
        self.rank_dir.parent.mkdir(parents=True, exist_ok=True)
        self.rank_dir.mkdir()
        self.attachments_dir = self.rank_dir / "attachments"
        self.attachments_dir.mkdir()
        self.manifest_path = self.rank_dir / "manifest.json"
        self.commit_path = self.rank_dir / "commit.json"
        self._attachments: dict[str, FailureAttachment] = {}
        self._committed = False
        self._manifest: dict[str, Any] = {
            "format_version": FAILURE_BUNDLE_FORMAT_VERSION,
            "root_format_version": V2_ROOT_FORMAT_VERSION,
            "bundle_id": self.bundle_id,
            "session_id": str(session_id).strip(),
            "attempt_id": str(attempt_id).strip(),
            "rank": self.rank,
            "world_size": self.world_size,
            "failure_phase": str(failure_phase).strip(),
            "failure_reason": str(failure_reason).strip(),
            "evidence_scope": FailureEvidenceScope(evidence_scope).value,
            "status": "running",
            "created_at": _utc_now(),
            "metadata": dict(metadata or {}),
            "attachments": [],
        }
        # 先落最小 header；后续序列化失败时仍保留可审计的 failure identity。
        write_json_atomic(self.manifest_path, self._manifest)

    def write_bytes(
        self,
        relative_path: str,
        payload: bytes | bytearray | memoryview,
        *,
        media_type: str = "application/octet-stream",
    ) -> FailureAttachment:
        """原子写内存 payload，并更新 running manifest 的 attachment inventory。"""

        data = bytes(payload)
        return self._write_chunks(
            relative_path,
            (data,),
            media_type=media_type,
        )

    def write_json(
        self,
        relative_path: str,
        payload: Mapping[str, Any],
    ) -> FailureAttachment:
        """按 canonical JSON 编码原子写小型结构化 attachment。"""

        encoded = (
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
        return self._write_chunks(
            relative_path,
            (encoded,),
            media_type="application/json",
        )

    def copy_file(
        self,
        relative_path: str,
        source: str | Path,
        *,
        media_type: str = "application/octet-stream",
    ) -> FailureAttachment:
        """流式复制宿主文件，不把大型 Tensor attachment 全部载入内存。"""

        source_path = Path(source)
        with source_path.open("rb") as handle:
            return self._write_chunks(
                relative_path,
                iter(lambda: handle.read(1024 * 1024), b""),
                media_type=media_type,
            )

    def commit(
        self,
        *,
        status: str = "complete",
        note: str | None = None,
    ) -> Path:
        """以 final manifest + commit 完成 rank-local durability boundary。

        ``degraded`` 表示 writer 已完整提交现有证据，但调用方明确知道 capture
        不完整；它不会被 validator 提升为 ``success``。
        """

        if self._committed:
            raise RuntimeError("failure bundle rank has already been committed")
        normalized_status = str(status).strip().lower()
        if normalized_status not in {"complete", "degraded"}:
            raise ValueError("failure bundle status must be complete or degraded")
        final_manifest = {
            **self._manifest,
            "status": normalized_status,
            "completed_at": _utc_now(),
            "attachments": [
                attachment.to_dict()
                for attachment in self._attachments.values()
            ],
        }
        if note is not None:
            final_manifest["note"] = str(note)
        write_json_atomic(self.manifest_path, final_manifest)
        commit = {
            "format_version": FAILURE_RANK_COMMIT_FORMAT_VERSION,
            "bundle_format_version": FAILURE_BUNDLE_FORMAT_VERSION,
            "bundle_id": self.bundle_id,
            "rank": self.rank,
            "world_size": self.world_size,
            "status": normalized_status,
            "manifest_file": "manifest.json",
            "manifest_size_bytes": self.manifest_path.stat().st_size,
            "manifest_sha256": file_sha256(self.manifest_path),
            "attachments": final_manifest["attachments"],
            "committed_at": _utc_now(),
        }
        # commit 永远最后写；缺 commit 的 final manifest 仍只能视为 partial。
        write_json_atomic(self.commit_path, commit)
        self._manifest = final_manifest
        self._committed = True
        return self.commit_path

    def reference(self, *, relative_to: str | Path) -> FailureBundleReference:
        """构造 transaction reference，并拒绝逃逸调用方 artifact root 的路径。"""

        relative = self.bundle_dir.relative_to(Path(relative_to).resolve())
        return FailureBundleReference(
            bundle_id=self.bundle_id,
            relative_path=relative.as_posix(),
        )

    def _write_chunks(
        self,
        relative_path: str,
        chunks: Iterable[bytes],
        *,
        media_type: str,
    ) -> FailureAttachment:
        """在 rank-local attachment root 内流式写入、fsync 并原子 replace。"""

        if self._committed:
            raise RuntimeError("cannot add attachments after commit")
        normalized = _validated_relative_path(relative_path)
        if normalized in self._attachments:
            raise ValueError(f"duplicate failure attachment: {normalized}")
        destination = self.attachments_dir.joinpath(*PurePosixPath(normalized).parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(
            f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        digest = hashlib.sha256()
        size_bytes = 0
        try:
            with temporary.open("wb") as handle:
                for chunk in chunks:
                    if not isinstance(chunk, bytes):
                        raise TypeError("failure attachment chunks must be bytes")
                    handle.write(chunk)
                    digest.update(chunk)
                    size_bytes += len(chunk)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
            _fsync_directory(destination.parent)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        attachment = FailureAttachment(
            relative_path=normalized,
            size_bytes=size_bytes,
            sha256=digest.hexdigest(),
            media_type=media_type,
        )
        self._attachments[normalized] = attachment
        self._manifest["attachments"] = [
            item.to_dict() for item in self._attachments.values()
        ]
        write_json_atomic(self.manifest_path, self._manifest)
        return attachment


def validate_failure_rank(rank_dir: str | Path) -> FailureRankValidation:
    """校验一个 rank-local manifest、attachment inventory 和 terminal commit。"""

    directory = Path(rank_dir).resolve()
    issues: list[ValidationIssue] = []
    manifest = _read_json_mapping(
        directory / "manifest.json",
        issues,
        code="invalid_failure_manifest",
    )
    bundle_id = _optional_string(manifest.get("bundle_id"))
    rank = _optional_nonnegative_integer(manifest.get("rank"))
    world_size = _optional_positive_integer(manifest.get("world_size"))
    evidence_scope = _optional_string(manifest.get("evidence_scope"))

    if manifest.get("format_version") != FAILURE_BUNDLE_FORMAT_VERSION:
        issues.append(
            ValidationIssue(
                "failure_bundle_format_version_mismatch",
                "error",
                f"expected {FAILURE_BUNDLE_FORMAT_VERSION}",
                str(directory / "manifest.json"),
            )
        )
    for field_name, field_value in (
        ("bundle_id", bundle_id),
        ("session_id", _optional_string(manifest.get("session_id"))),
        ("attempt_id", _optional_string(manifest.get("attempt_id"))),
        ("failure_phase", _optional_string(manifest.get("failure_phase"))),
        ("failure_reason", _optional_string(manifest.get("failure_reason"))),
        ("rank", rank),
        ("world_size", world_size),
    ):
        if field_value is None:
            issues.append(
                ValidationIssue(
                    "invalid_failure_manifest_identity",
                    "error",
                    f"manifest field {field_name} is missing or invalid",
                    str(directory / "manifest.json"),
                )
            )
    if rank is not None and world_size is not None and rank >= world_size:
        issues.append(
            ValidationIssue(
                "invalid_failure_rank_range",
                "error",
                "manifest rank must be smaller than world_size",
                str(directory / "manifest.json"),
            )
        )
    expected_rank_dir_name = (
        f"rank_{rank:05d}" if rank is not None else None
    )
    if expected_rank_dir_name is not None and directory.name != expected_rank_dir_name:
        issues.append(
            ValidationIssue(
                "failure_rank_directory_mismatch",
                "error",
                "rank directory name does not match manifest rank",
                str(directory),
            )
        )
    if manifest.get("root_format_version") != V2_ROOT_FORMAT_VERSION:
        issues.append(
            ValidationIssue(
                "failure_root_format_version_mismatch",
                "error",
                f"expected {V2_ROOT_FORMAT_VERSION}",
                str(directory / "manifest.json"),
            )
        )
    try:
        if evidence_scope is not None:
            FailureEvidenceScope(evidence_scope)
    except ValueError:
        issues.append(
            ValidationIssue(
                "invalid_failure_evidence_scope",
                "error",
                f"unsupported failure evidence scope: {evidence_scope}",
                str(directory / "manifest.json"),
            )
        )

    attachments = _validate_attachments(directory, manifest, issues)
    commit_path = directory / "commit.json"
    if not commit_path.is_file():
        issues.append(
            ValidationIssue(
                "missing_failure_rank_commit",
                "warning",
                "rank-local failure evidence has no terminal commit",
                str(commit_path),
            )
        )
        status = (
            "corrupt"
            if any(issue.severity == "error" for issue in issues)
            else "partial"
        )
        return FailureRankValidation(
            directory,
            status,
            False,
            bundle_id,
            rank,
            world_size,
            evidence_scope,
            attachments,
            MappingProxyType(dict(manifest)),
            tuple(issues),
        )

    commit = _read_json_mapping(
        commit_path,
        issues,
        code="invalid_failure_rank_commit",
    )
    if commit.get("format_version") != FAILURE_RANK_COMMIT_FORMAT_VERSION:
        issues.append(
            ValidationIssue(
                "failure_rank_commit_format_version_mismatch",
                "error",
                f"expected {FAILURE_RANK_COMMIT_FORMAT_VERSION}",
                str(commit_path),
            )
        )
    if commit.get("bundle_format_version") != FAILURE_BUNDLE_FORMAT_VERSION:
        issues.append(
            ValidationIssue(
                "failure_commit_bundle_format_version_mismatch",
                "error",
                f"expected {FAILURE_BUNDLE_FORMAT_VERSION}",
                str(commit_path),
            )
        )
    expected_identity = (bundle_id, rank, world_size)
    committed_identity = (
        _optional_string(commit.get("bundle_id")),
        _optional_nonnegative_integer(commit.get("rank")),
        _optional_positive_integer(commit.get("world_size")),
    )
    if committed_identity != expected_identity:
        issues.append(
            ValidationIssue(
                "failure_rank_commit_identity_mismatch",
                "error",
                "commit identity does not match manifest",
                str(commit_path),
            )
        )
    if commit.get("manifest_file") != "manifest.json":
        issues.append(
            ValidationIssue(
                "invalid_failure_manifest_reference",
                "error",
                "commit must bind manifest.json",
                str(commit_path),
            )
        )
    manifest_path = directory / "manifest.json"
    if (
        not manifest_path.is_file()
        or commit.get("manifest_size_bytes") != manifest_path.stat().st_size
        or commit.get("manifest_sha256") != file_sha256(manifest_path)
    ):
        issues.append(
            ValidationIssue(
                "failure_manifest_digest_mismatch",
                "error",
                "committed manifest size or SHA-256 does not match",
                str(directory / "manifest.json"),
            )
        )
    canonical_attachments = [attachment.to_dict() for attachment in attachments]
    if commit.get("attachments") != canonical_attachments:
        issues.append(
            ValidationIssue(
                "failure_attachment_inventory_mismatch",
                "error",
                "commit attachments do not match validated manifest inventory",
                str(commit_path),
            )
        )

    manifest_status = str(manifest.get("status") or "").strip().lower()
    commit_status = str(commit.get("status") or "").strip().lower()
    if (
        manifest_status not in {"complete", "degraded"}
        or commit_status != manifest_status
    ):
        issues.append(
            ValidationIssue(
                "invalid_failure_terminal_status",
                "error",
                "manifest and commit must agree on complete or degraded",
                str(commit_path),
            )
        )
    has_error = any(issue.severity == "error" for issue in issues)
    status = (
        "corrupt"
        if has_error
        else "degraded"
        if manifest_status == "degraded"
        else "success"
    )
    return FailureRankValidation(
        directory,
        status,
        not has_error,
        bundle_id,
        rank,
        world_size,
        evidence_scope,
        attachments,
        MappingProxyType(dict(manifest)),
        tuple(issues),
    )


def finalize_failure_bundle(bundle_dir: str | Path) -> Path | None:
    """在全部预期 rank commit 可验证后幂等写根级 terminal commit。

    任一 rank 都可在完成自己的 rank-local commit 后调用本函数。调用不使用
    collective、queue 或共享进程锁；并发调用者从同一组不可变 rank commit 构造完全
    相同的 root manifest，再通过同目录原子 replace 写入。若 rank 尚未齐全、identity
    分歧或任一 rank 未通过校验，函数不写根级 artifact 并返回 ``None``。

    该协议依赖共享文件系统对同目录 ``os.replace`` 的原子性和已关闭文件的可见性。
    已存在且合法的根 commit 直接幂等返回；已存在但损坏的根 commit 不会被覆盖。
    """

    directory = Path(bundle_dir).resolve()
    root_commit_path = directory / "commit.json"
    if root_commit_path.is_file():
        validation = validate_failure_bundle(directory)
        if validation.valid and validation.status in {"success", "degraded"}:
            return root_commit_path
        raise RuntimeError(
            "existing failure bundle root commit did not pass validation"
        )

    rank_dirs = _failure_rank_directories(directory)
    validations = tuple(validate_failure_rank(path) for path in rank_dirs)
    if not validations:
        return None
    bundle_ids = {item.bundle_id for item in validations}
    session_ids = {
        _optional_string(item.manifest.get("session_id"))
        for item in validations
    }
    attempt_ids = {
        _optional_string(item.manifest.get("attempt_id"))
        for item in validations
    }
    world_sizes = {item.world_size for item in validations}
    ranks = {item.rank for item in validations}
    if (
        len(bundle_ids) != 1
        or None in bundle_ids
        or len(session_ids) != 1
        or None in session_ids
        or len(attempt_ids) != 1
        or None in attempt_ids
        or len(world_sizes) != 1
        or None in world_sizes
    ):
        return None
    expected_rank_count = next(iter(world_sizes))
    assert expected_rank_count is not None
    if (
        ranks != set(range(expected_rank_count))
        or len(validations) != expected_rank_count
        or any(
            not item.valid or item.status not in {"success", "degraded"}
            for item in validations
        )
    ):
        return None

    rank_commits = _rank_commit_inventory(validations)
    terminal_status = (
        "degraded"
        if any(item.status == "degraded" for item in validations)
        else "complete"
    )
    bundle_id = next(iter(bundle_ids))
    root_manifest = {
        "format_version": FAILURE_BUNDLE_FORMAT_VERSION,
        "root_format_version": V2_ROOT_FORMAT_VERSION,
        "record_kind": "failure_bundle_manifest",
        "bundle_id": bundle_id,
        "session_id": next(iter(session_ids)),
        "attempt_id": next(iter(attempt_ids)),
        "status": terminal_status,
        "expected_rank_count": expected_rank_count,
        "captured_rank_count": len(validations),
        "rank_commits": rank_commits,
    }
    root_manifest_path = directory / "manifest.json"
    # root manifest 不含 wall-clock 字段，因此所有并发 finalizer 生成相同字节。
    write_json_atomic(root_manifest_path, root_manifest)
    root_commit = {
        "format_version": FAILURE_BUNDLE_COMMIT_FORMAT_VERSION,
        "bundle_format_version": FAILURE_BUNDLE_FORMAT_VERSION,
        "bundle_id": bundle_id,
        "status": terminal_status,
        "manifest_file": "manifest.json",
        "manifest_size_bytes": root_manifest_path.stat().st_size,
        "manifest_sha256": file_sha256(root_manifest_path),
        "expected_rank_count": expected_rank_count,
        "rank_commit_count": len(rank_commits),
        "committed_at": _utc_now(),
    }
    # commit 始终最后写；并发 replace 的 payload 虽有不同时间戳，但绑定同一 manifest。
    write_json_atomic(root_commit_path, root_commit)
    validation = validate_failure_bundle(directory)
    expected_status = (
        "degraded" if terminal_status == "degraded" else "success"
    )
    if not validation.valid or validation.status != expected_status:
        raise RuntimeError("failure bundle root commit did not pass validation")
    return root_commit_path


def validate_failure_bundle(bundle_dir: str | Path) -> FailureBundleValidation:
    """验证 rank-local evidence 与根级 terminal commit。

    rank commit 只证明单个 rank 的 durability。整个 bundle 只有在根 manifest/commit
    绑定全部预期 rank commit 后才能成为 ``success``；根 commit 缺失时，即使每个 rank
    均已完成，也只能是 ``partial`` 或 ``degraded``，不能冒充完整异常证据。
    """

    directory = Path(bundle_dir).resolve()
    issues: list[ValidationIssue] = []
    ranks_root = directory / "ranks"
    rank_dirs = _failure_rank_directories(directory)
    validations = tuple(validate_failure_rank(path) for path in rank_dirs)
    for validation in validations:
        issues.extend(validation.issues)
    if not validations:
        issues.append(
            ValidationIssue(
                "missing_failure_rank_evidence",
                "error",
                "failure bundle contains no rank-local evidence",
                str(ranks_root),
            )
        )
        return FailureBundleValidation(
            directory,
            "corrupt",
            False,
            None,
            None,
            0,
            (),
            tuple(issues),
        )

    bundle_ids = {item.bundle_id for item in validations}
    session_ids = {
        _optional_string(item.manifest.get("session_id"))
        for item in validations
    }
    attempt_ids = {
        _optional_string(item.manifest.get("attempt_id"))
        for item in validations
    }
    world_sizes = {item.world_size for item in validations}
    ranks = [item.rank for item in validations]
    if len(bundle_ids) != 1 or None in bundle_ids:
        issues.append(
            ValidationIssue(
                "failure_bundle_identity_divergence",
                "error",
                "rank-local bundle identities do not agree",
                str(directory),
            )
        )
    if len(session_ids) != 1 or None in session_ids:
        issues.append(
            ValidationIssue(
                "failure_session_identity_divergence",
                "error",
                "rank-local session identities do not agree",
                str(directory),
            )
        )
    if len(attempt_ids) != 1 or None in attempt_ids:
        issues.append(
            ValidationIssue(
                "failure_attempt_identity_divergence",
                "error",
                "rank-local attempt identities do not agree",
                str(directory),
            )
        )
    if len(world_sizes) != 1 or None in world_sizes:
        issues.append(
            ValidationIssue(
                "failure_world_size_divergence",
                "error",
                "rank-local world_size values do not agree",
                str(directory),
            )
        )
    if None in ranks or len(set(ranks)) != len(ranks):
        issues.append(
            ValidationIssue(
                "failure_rank_identity_divergence",
                "error",
                "rank-local identities are missing or duplicated",
                str(directory),
            )
        )

    expected_rank_count = (
        next(iter(world_sizes))
        if len(world_sizes) == 1 and None not in world_sizes
        else None
    )
    captured_rank_count = len(validations)
    if (
        expected_rank_count is not None
        and captured_rank_count != expected_rank_count
    ):
        issues.append(
            ValidationIssue(
                "missing_failure_ranks",
                "warning",
                f"captured {captured_rank_count} of {expected_rank_count} ranks",
                str(directory),
            )
        )
    if (
        expected_rank_count is not None
        and None not in ranks
        and (
            any(
                rank is not None
                and (rank < 0 or rank >= expected_rank_count)
                for rank in ranks
            )
            or (
                len(ranks) == expected_rank_count
                and set(ranks) != set(range(expected_rank_count))
            )
        )
    ):
        issues.append(
            ValidationIssue(
                "failure_rank_set_mismatch",
                "error",
                "captured rank identities do not match the expected rank set",
                str(directory),
            )
        )

    root_commit_path = directory / "commit.json"
    root_commit_present = root_commit_path.is_file()
    root_terminal_status: str | None = None
    if root_commit_present:
        root_terminal_status = _validate_failure_bundle_root(
            directory=directory,
            validations=validations,
            bundle_ids=bundle_ids,
            session_ids=session_ids,
            attempt_ids=attempt_ids,
            expected_rank_count=expected_rank_count,
            issues=issues,
        )
    else:
        issues.append(
            ValidationIssue(
                "missing_failure_bundle_commit",
                "warning",
                "failure bundle has no root-level terminal commit",
                str(root_commit_path),
            )
        )

    has_error = any(issue.severity == "error" for issue in issues)
    has_partial = any(item.status == "partial" for item in validations)
    has_degraded = (
        any(item.status == "degraded" for item in validations)
        or (
            expected_rank_count is not None
            and captured_rank_count != expected_rank_count
        )
    )
    status = (
        "corrupt"
        if has_error
        else "partial"
        if has_partial
        or (
            not root_commit_present
            and not has_degraded
        )
        else "degraded"
        if has_degraded or root_terminal_status == "degraded"
        else "success"
    )
    return FailureBundleValidation(
        directory,
        status,
        status in {"success", "degraded"},
        next(iter(bundle_ids)) if len(bundle_ids) == 1 else None,
        expected_rank_count,
        captured_rank_count,
        validations,
        tuple(issues),
    )


def _validate_failure_bundle_root(
    *,
    directory: Path,
    validations: tuple[FailureRankValidation, ...],
    bundle_ids: set[str | None],
    session_ids: set[str | None],
    attempt_ids: set[str | None],
    expected_rank_count: int | None,
    issues: list[ValidationIssue],
) -> str | None:
    """把根 manifest/commit 与当前合法 rank commit inventory 逐字段绑定。"""

    manifest_path = directory / "manifest.json"
    commit_path = directory / "commit.json"
    manifest = _read_json_mapping(
        manifest_path,
        issues,
        code="invalid_failure_bundle_manifest",
    )
    commit = _read_json_mapping(
        commit_path,
        issues,
        code="invalid_failure_bundle_commit",
    )
    if manifest.get("format_version") != FAILURE_BUNDLE_FORMAT_VERSION:
        issues.append(
            ValidationIssue(
                "failure_bundle_root_format_version_mismatch",
                "error",
                f"expected {FAILURE_BUNDLE_FORMAT_VERSION}",
                str(manifest_path),
            )
        )
    if manifest.get("root_format_version") != V2_ROOT_FORMAT_VERSION:
        issues.append(
            ValidationIssue(
                "failure_bundle_v2_root_format_version_mismatch",
                "error",
                f"expected {V2_ROOT_FORMAT_VERSION}",
                str(manifest_path),
            )
        )
    if manifest.get("record_kind") != "failure_bundle_manifest":
        issues.append(
            ValidationIssue(
                "invalid_failure_bundle_manifest_kind",
                "error",
                "root manifest must identify failure_bundle_manifest",
                str(manifest_path),
            )
        )

    expected_bundle_id = (
        next(iter(bundle_ids)) if len(bundle_ids) == 1 else None
    )
    expected_session_id = (
        next(iter(session_ids)) if len(session_ids) == 1 else None
    )
    expected_attempt_id = (
        next(iter(attempt_ids)) if len(attempt_ids) == 1 else None
    )
    captured_rank_count = len(validations)
    rank_commits = _rank_commit_inventory(validations)
    all_expected_ranks_valid = (
        expected_rank_count is not None
        and captured_rank_count == expected_rank_count
        and {item.rank for item in validations}
        == set(range(expected_rank_count))
        and all(
            item.valid and item.status in {"success", "degraded"}
            for item in validations
        )
    )
    if not all_expected_ranks_valid:
        issues.append(
            ValidationIssue(
                "premature_failure_bundle_commit",
                "error",
                "root commit exists without every expected valid rank commit",
                str(commit_path),
            )
        )
    expected_terminal_status = (
        "degraded"
        if any(item.status == "degraded" for item in validations)
        else "complete"
    )
    expected_manifest_fields = {
        "bundle_id": expected_bundle_id,
        "session_id": expected_session_id,
        "attempt_id": expected_attempt_id,
        "status": expected_terminal_status,
        "expected_rank_count": expected_rank_count,
        "captured_rank_count": captured_rank_count,
        "rank_commits": rank_commits,
    }
    for field_name, expected_value in expected_manifest_fields.items():
        if manifest.get(field_name) != expected_value:
            issues.append(
                ValidationIssue(
                    "failure_bundle_manifest_mismatch",
                    "error",
                    f"root manifest field {field_name} does not match rank evidence",
                    str(manifest_path),
                )
            )

    if commit.get("format_version") != FAILURE_BUNDLE_COMMIT_FORMAT_VERSION:
        issues.append(
            ValidationIssue(
                "failure_bundle_commit_format_version_mismatch",
                "error",
                f"expected {FAILURE_BUNDLE_COMMIT_FORMAT_VERSION}",
                str(commit_path),
            )
        )
    if commit.get("bundle_format_version") != FAILURE_BUNDLE_FORMAT_VERSION:
        issues.append(
            ValidationIssue(
                "failure_bundle_commit_bundle_format_mismatch",
                "error",
                f"expected {FAILURE_BUNDLE_FORMAT_VERSION}",
                str(commit_path),
            )
        )
    if (
        commit.get("bundle_id") != expected_bundle_id
        or commit.get("status") != expected_terminal_status
        or commit.get("expected_rank_count") != expected_rank_count
        or commit.get("rank_commit_count") != len(rank_commits)
    ):
        issues.append(
            ValidationIssue(
                "failure_bundle_commit_identity_mismatch",
                "error",
                "root commit identity or coverage does not match rank evidence",
                str(commit_path),
            )
        )
    if commit.get("manifest_file") != "manifest.json":
        issues.append(
            ValidationIssue(
                "invalid_failure_bundle_manifest_reference",
                "error",
                "root commit must bind manifest.json",
                str(commit_path),
            )
        )
    if (
        not manifest_path.is_file()
        or commit.get("manifest_size_bytes") != manifest_path.stat().st_size
        or commit.get("manifest_sha256") != file_sha256(manifest_path)
    ):
        issues.append(
            ValidationIssue(
                "failure_bundle_manifest_digest_mismatch",
                "error",
                "root manifest size or SHA-256 does not match root commit",
                str(manifest_path),
            )
        )
    return (
        "degraded"
        if expected_terminal_status == "degraded"
        else "complete"
    )


def _failure_rank_directories(bundle_dir: Path) -> tuple[Path, ...]:
    """只枚举规范 rank 目录；其它 bundle root 文件不参与 rank evidence。"""

    ranks_root = bundle_dir / "ranks"
    return (
        tuple(
            sorted(
                path
                for path in ranks_root.iterdir()
                if path.is_dir() and path.name.startswith("rank_")
            )
        )
        if ranks_root.is_dir()
        else ()
    )


def _rank_commit_inventory(
    validations: Iterable[FailureRankValidation],
) -> list[dict[str, Any]]:
    """按 rank 生成根 manifest 绑定的 commit size/hash inventory。"""

    inventory: list[dict[str, Any]] = []
    for validation in sorted(
        validations,
        key=lambda item: -1 if item.rank is None else item.rank,
    ):
        commit_path = validation.rank_dir / "commit.json"
        if validation.rank is None or not commit_path.is_file():
            continue
        inventory.append(
            {
                "rank": validation.rank,
                "status": (
                    str(validation.manifest.get("status") or "")
                    .strip()
                    .lower()
                ),
                "commit_file": (
                    f"ranks/rank_{validation.rank:05d}/commit.json"
                ),
                "commit_size_bytes": commit_path.stat().st_size,
                "commit_sha256": file_sha256(commit_path),
            }
        )
    return inventory


def _validated_relative_path(value: str) -> str:
    """规范 attachment/reference 路径，拒绝绝对路径、反斜杠和父级逃逸。"""

    raw = str(value)
    path = PurePosixPath(raw)
    canonical = path.as_posix()
    if (
        not raw
        or "\\" in raw
        or path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
        or raw != canonical
    ):
        raise ValueError("failure artifact path must be a safe POSIX relative path")
    return canonical


def _validate_attachments(
    rank_dir: Path,
    manifest: Mapping[str, Any],
    issues: list[ValidationIssue],
) -> tuple[FailureAttachment, ...]:
    """流式核对 manifest 声明的 attachment，不消费未声明的 orphan 文件。"""

    raw_records = manifest.get("attachments")
    if not isinstance(raw_records, list):
        issues.append(
            ValidationIssue(
                "invalid_failure_attachment_inventory",
                "error",
                "manifest attachments must be a list",
                str(rank_dir / "manifest.json"),
            )
        )
        return ()
    attachments: list[FailureAttachment] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_records):
        try:
            if not isinstance(raw, Mapping):
                raise TypeError("attachment descriptor must be an object")
            attachment = FailureAttachment(
                relative_path=str(raw["relative_path"]),
                size_bytes=raw["size_bytes"],
                sha256=str(raw["sha256"]),
                media_type=str(
                    raw.get("media_type", "application/octet-stream")
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            issues.append(
                ValidationIssue(
                    "invalid_failure_attachment_descriptor",
                    "error",
                    f"attachment {index}: {error}",
                    str(rank_dir / "manifest.json"),
                )
            )
            continue
        if attachment.relative_path in seen:
            issues.append(
                ValidationIssue(
                    "duplicate_failure_attachment",
                    "error",
                    f"duplicate attachment: {attachment.relative_path}",
                    str(rank_dir / "manifest.json"),
                )
            )
            continue
        seen.add(attachment.relative_path)
        attachments_root = (rank_dir / "attachments").resolve()
        path = (attachments_root / attachment.relative_path).resolve()
        if not path.is_relative_to(attachments_root):
            issues.append(
                ValidationIssue(
                    "failure_attachment_path_escape",
                    "error",
                    f"attachment escapes rank root: {attachment.relative_path}",
                    str(path),
                )
            )
        elif not path.is_file():
            issues.append(
                ValidationIssue(
                    "missing_failure_attachment",
                    "error",
                    f"missing attachment: {attachment.relative_path}",
                    str(path),
                )
            )
        elif (
            path.stat().st_size != attachment.size_bytes
            or file_sha256(path) != attachment.sha256
        ):
            issues.append(
                ValidationIssue(
                    "failure_attachment_digest_mismatch",
                    "error",
                    f"size or SHA-256 mismatch: {attachment.relative_path}",
                    str(path),
                )
            )
        attachments.append(attachment)
    return tuple(attachments)


def _read_json_mapping(
    path: Path,
    issues: list[ValidationIssue],
    *,
    code: str,
) -> dict[str, Any]:
    """读取单个小型 metadata 文件，失败时返回空 mapping 供 validator 收口。"""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        issues.append(ValidationIssue(code, "error", str(error), str(path)))
        return {}
    if not isinstance(payload, dict):
        issues.append(
            ValidationIssue(code, "error", "payload must be a JSON object", str(path))
        )
        return {}
    return payload


def _optional_string(value: Any) -> str | None:
    return str(value) if isinstance(value, str) and value else None


def _optional_nonnegative_integer(value: Any) -> int | None:
    return (
        int(value)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0
        else None
    )


def _optional_positive_integer(value: Any) -> int | None:
    return (
        int(value)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0
        else None
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _fsync_directory(path: Path) -> None:
    """把 attachment replace 的目录项纳入 failure-path durability boundary。"""

    try:
        directory_fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


__all__ = [
    "FAILURE_BUNDLE_COMMIT_FORMAT_VERSION",
    "FAILURE_BUNDLE_FORMAT_VERSION",
    "FAILURE_RANK_COMMIT_FORMAT_VERSION",
    "FailureAttachment",
    "FailureBundleReference",
    "FailureBundleValidation",
    "FailureBundleWriter",
    "FailureRankValidation",
    "FailureEvidenceScope",
    "finalize_failure_bundle",
    "validate_failure_bundle",
    "validate_failure_rank",
]
