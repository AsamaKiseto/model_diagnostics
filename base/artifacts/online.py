"""Diagnostics v2 的事务式在线 artifact 写入、发现与校验。

该模块只依赖标准库。训练热路径把一个 logical update 或 failed attempt 作为不可拆分的
transaction 提交；后台线程按 segment 追加 payload rows，最后写入包含行数和 SHA-256
的 commit。reader 只暴露通过 commit 校验的 transaction。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import queue
import threading
import time
from typing import Any, Iterable, Iterator, Mapping
import uuid

from .common import (
    V2ArtifactInventory,
    V2_ROOT_FORMAT_VERSION,
    ValidationIssue,
    discover_v2_artifacts,
    file_sha256,
    stable_json_hash,
    update_v2_index,
    write_json_atomic,
)

ONLINE_ARTIFACT_FORMAT_VERSION = "training_diagnostics_v2"
TRANSACTION_FORMAT_VERSION = "diagnostics_transaction_v1"
ONLINE_DIAGNOSTICS_FORMAT_VERSION = 10

DEFAULT_SEGMENT_MAX_TRANSACTIONS = 256
DEFAULT_SEGMENT_MAX_UNCOMPRESSED_BYTES = 32 * 1024 * 1024
DEFAULT_WRITER_QUEUE_CAPACITY = 64
DEFAULT_WRITER_CLOSE_TIMEOUT_SECONDS = 30.0
# running manifest 只提供有界进度快照，不是 transaction durability boundary。
# 与 segment transaction 上限对齐，避免后台线程每 32 个训练 update 原子替换
# manifest 而周期性争用热路径；close/rotation 仍会强制刷新完整终态。
_MANIFEST_CHECKPOINT_INTERVAL_TRANSACTIONS = (
    DEFAULT_SEGMENT_MAX_TRANSACTIONS
)

_FINAL_STATUSES = frozenset({"complete", "degraded", "corrupt"})
_STATUS_PRIORITY = {"running": 0, "complete": 1, "partial": 2, "degraded": 3, "corrupt": 4}
_SENTINEL = object()
_ANALYZER_DEFINITION_FIELDS = frozenset(
    {
        "name",
        "evidence_kind",
        "execution_mode",
        "option_keys",
        "definition_version",
        "claim_boundaries",
    }
)


@dataclass(frozen=True)
class ValidationResult:
    """单个 online session 的完整性校验结果。"""

    session_id: str | None
    status: str
    valid: bool
    manifest_status: str | None
    committed_transaction_count: int
    committed_row_count: int
    issues: tuple[ValidationIssue, ...]
    manifest: Mapping[str, Any]
    content_summary: Mapping[str, Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
        kw_only=True,
    )

    def as_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "status": self.status,
            "valid": self.valid,
            "manifest_status": self.manifest_status,
            "committed_transaction_count": self.committed_transaction_count,
            "committed_row_count": self.committed_row_count,
            "issues": [issue.as_dict() for issue in self.issues],
        }


@dataclass(frozen=True)
class CommittedTransaction:
    """通过 payload hash 与 row-count 校验的 transaction。"""

    transaction_id: str
    transaction_kind: str
    identity: Any
    rows: tuple[Mapping[str, Any], ...]
    commit: Mapping[str, Any]
    segment_path: Path


@dataclass(frozen=True)
class _QueuedTransaction:
    sequence: int
    transaction_kind: str
    identity: Any
    rows: tuple[Mapping[str, Any], ...]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _canonical_json_bytes(value: Any) -> bytes:
    """按 online transaction wire contract 编码无换行 JSON bytes。"""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_analyzer_definitions(
    definitions: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """校验并规范化 online stream analyzer descriptors。"""

    materialized = list(definitions)
    if not materialized:
        raise ValueError(
            "online sessions must declare at least one analyzer definition"
        )
    normalized: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, raw in enumerate(materialized):
        if not isinstance(raw, Mapping):
            raise TypeError(f"analyzer descriptor {index} must be an object")
        if set(raw) != _ANALYZER_DEFINITION_FIELDS:
            raise ValueError(
                f"analyzer descriptor {index} fields do not match contract"
            )
        name = str(raw.get("name") or "").strip().lower()
        evidence_kind = str(raw.get("evidence_kind") or "").strip()
        execution_mode = str(raw.get("execution_mode") or "").strip().lower()
        version = raw.get("definition_version")
        option_keys = raw.get("option_keys")
        claim_boundaries = raw.get("claim_boundaries")
        if (
            not name
            or name in names
            or not evidence_kind
            or execution_mode != "stream"
            or not isinstance(version, int)
            or isinstance(version, bool)
            or version < 1
            or not isinstance(option_keys, list)
            or not isinstance(claim_boundaries, list)
            or any(not isinstance(key, str) or not key.strip() for key in option_keys)
            or len(set(option_keys)) != len(option_keys)
            or any(
                not isinstance(boundary, str) or not boundary.strip()
                for boundary in claim_boundaries
            )
        ):
            raise ValueError(
                f"analyzer descriptor {index} contains invalid values"
            )
        names.add(name)
        normalized.append(
            {
                "name": name,
                "evidence_kind": evidence_kind,
                "execution_mode": execution_mode,
                "option_keys": sorted(option_keys),
                "definition_version": int(version),
                "claim_boundaries": list(claim_boundaries),
            }
        )
    return normalized


class OnlineArtifactWriter:
    """有界、fail-stop 的 diagnostics v2 在线 transaction writer。

    ``submit_transaction`` 从不拆分 transaction。queue 满时返回 ``False``，记录缺口并把
    session 降级为 ``degraded``；首次不可恢复写错误后拒绝所有新 transaction。
    """

    def __init__(
        self,
        run_dir: str | Path,
        *,
        analyzer_definitions: Iterable[Mapping[str, Any]],
        session_id: str | None = None,
    ) -> None:
        self.run_dir = Path(run_dir).resolve()
        self._analyzer_definitions = _canonical_analyzer_definitions(
            analyzer_definitions
        )
        self.session_id = session_id or (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:12]
        )
        if not self.session_id or "/" in self.session_id or "\\" in self.session_id:
            raise ValueError("session_id must be a non-empty path component")
        self._transaction_format_json = _canonical_json_bytes(
            TRANSACTION_FORMAT_VERSION
        )
        self._session_id_json = _canonical_json_bytes(self.session_id)
        self._transaction_kind_json: dict[str, bytes] = {}
        self.session_dir = (
            self.run_dir
            / "diagnostics"
            / "v2"
            / "online"
            / "sessions"
            / self.session_id
        )
        if self.session_dir.exists():
            raise FileExistsError(f"diagnostics session already exists: {self.session_dir}")
        self.segments_dir = self.session_dir / "segments"
        self.segments_dir.mkdir(parents=True)
        self.manifest_path = self.session_dir / "manifest.json"
        self.hierarchy_path = self.session_dir / "hierarchy.json"

        self._queue: queue.Queue[_QueuedTransaction | object] = queue.Queue(
            maxsize=DEFAULT_WRITER_QUEUE_CAPACITY
        )
        self._close_timeout_seconds = DEFAULT_WRITER_CLOSE_TIMEOUT_SECONDS
        self._segment_max_transactions = DEFAULT_SEGMENT_MAX_TRANSACTIONS
        self._segment_max_bytes = DEFAULT_SEGMENT_MAX_UNCOMPRESSED_BYTES
        self._lock = threading.RLock()
        self._manifest_io_lock = threading.Lock()
        self._closed = False
        self._accepting = True
        self._sequence = 0
        self._status = "running"
        self._writer_error: str | None = None
        self._dropped_count = 0
        self._dropped_ranges: list[dict[str, int]] = []
        self._dropped_examples: list[dict[str, Any]] = []
        self._committed_transactions = 0
        self._committed_rows = 0
        self._last_committed_identity: Any = None
        self._hierarchy_sha256: str | None = None
        self._segments: list[dict[str, Any]] = []
        self._segment_handle: Any = None
        self._segment_digest: Any = None
        self._last_manifest_checkpoint_transaction_count = 0
        self._created_at = _utc_now()
        self._updated_at = self._created_at

        self._write_manifest()
        self._update_index()
        self._thread = threading.Thread(
            target=self._writer_loop,
            name=f"diagnostics-writer-{self.session_id}",
            daemon=True,
        )
        self._thread.start()

    @property
    def status(self) -> str:
        with self._lock:
            return self._status

    @property
    def writer_error(self) -> str | None:
        with self._lock:
            return self._writer_error

    def write_hierarchy(self, hierarchy: Mapping[str, Any]) -> None:
        """原子写入该 session 的唯一 hierarchy，并把 digest 绑定到 manifest。"""

        hierarchy_payload = dict(hierarchy)
        persisted_definitions = hierarchy_payload.get(
            "analyzer_definitions"
        )
        if (
            persisted_definitions is not None
            and persisted_definitions != self._analyzer_definitions
        ):
            raise ValueError(
                "hierarchy analyzer definitions conflict with session manifest"
            )
        hierarchy_payload["analyzer_definitions"] = [
            dict(descriptor)
            for descriptor in self._analyzer_definitions
        ]
        payload = _canonical_json_bytes(hierarchy_payload) + b"\n"
        temporary = self.hierarchy_path.with_name(
            f".{self.hierarchy_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        try:
            with temporary.open("wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.hierarchy_path)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        with self._lock:
            self._hierarchy_sha256 = hashlib.sha256(payload).hexdigest()
            self._updated_at = _utc_now()
        self._write_manifest()

    def submit_transaction(
        self,
        transaction_kind: str,
        identity: Any,
        rows: Iterable[Mapping[str, Any]],
    ) -> bool:
        """非阻塞提交完整 transaction；queue 满或 writer 失败时返回 ``False``。"""

        if not transaction_kind:
            raise ValueError("transaction_kind must be non-empty")
        materialized = tuple(dict(row) for row in rows)
        if not materialized:
            raise ValueError("transaction must contain at least one row")
        with self._lock:
            self._sequence += 1
            sequence = self._sequence
            if not self._accepting or self._writer_error is not None:
                self._record_drop_locked(
                    sequence,
                    transaction_kind,
                    identity,
                    "writer_unavailable",
                )
                return False
        item = _QueuedTransaction(sequence, transaction_kind, identity, materialized)
        try:
            self._queue.put_nowait(item)
            return True
        except queue.Full:
            with self._lock:
                self._record_drop_locked(sequence, transaction_kind, identity, "queue_full")
            return False

    def close(self, status: str | None = None) -> None:
        """幂等、noexcept 地收口 writer；超时或写错误产生 ``corrupt`` manifest。"""

        try:
            deadline = time.monotonic() + self._close_timeout_seconds
            metadata_reserve = min(
                max(self._close_timeout_seconds * 0.5, 0.001),
                self._close_timeout_seconds,
            )
            writer_deadline = deadline - metadata_reserve
            with self._lock:
                if self._closed:
                    return
                self._closed = True
                self._accepting = False
            try:
                self._queue.put(
                    _SENTINEL,
                    timeout=max(writer_deadline - time.monotonic(), 0.0),
                )
            except queue.Full:
                with self._lock:
                    self._mark_status_locked("corrupt")
                    self._writer_error = "writer_close_queue_timeout"
            self._thread.join(timeout=max(writer_deadline - time.monotonic(), 0.0))
            with self._lock:
                if self._thread.is_alive():
                    self._writer_error = self._writer_error or "writer_close_thread_timeout"
                    self._mark_status_locked("corrupt")
                elif self._writer_error is not None:
                    self._mark_status_locked("corrupt")
                elif status is not None:
                    if status not in _FINAL_STATUSES:
                        self._writer_error = f"invalid_close_status:{status}"
                        self._mark_status_locked("corrupt")
                    else:
                        self._mark_status_locked(status)
                elif self._dropped_count:
                    self._mark_status_locked("degraded")
                else:
                    self._mark_status_locked("complete")
                self._updated_at = _utc_now()
            self._flush_terminal_metadata(deadline)
        except Exception as error:  # pragma: no cover - close 必须保护训练退出路径
            try:
                with self._lock:
                    self._writer_error = self._writer_error or f"{type(error).__name__}: {error}"
                    self._status = "corrupt"
                self._flush_terminal_metadata(
                    time.monotonic() + self._close_timeout_seconds
                )
            except Exception:
                pass

    def __enter__(self) -> "OnlineArtifactWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close(status="degraded" if exc_type is not None else None)

    def _writer_loop(self) -> None:
        """串行持久化 queue；payload、sync 或 metadata 错误都令 writer fail-stop。"""

        while True:
            item = self._queue.get()
            try:
                if item is _SENTINEL:
                    try:
                        self._close_segment()
                    except Exception as error:
                        with self._lock:
                            self._writer_error = (
                                f"{type(error).__name__}: {error}"
                            )
                            self._mark_status_locked("corrupt")
                        try:
                            self._write_manifest()
                        except Exception:
                            pass
                    return
                assert isinstance(item, _QueuedTransaction)
                with self._lock:
                    failed = self._writer_error is not None
                if failed:
                    with self._lock:
                        self._record_drop_locked(
                            item.sequence,
                            item.transaction_kind,
                            item.identity,
                            "writer_failed",
                        )
                    continue
                try:
                    self._write_transaction(item)
                except Exception as error:
                    with self._lock:
                        self._writer_error = f"{type(error).__name__}: {error}"
                        self._mark_status_locked("corrupt")
                        self._record_drop_locked(
                            item.sequence,
                            item.transaction_kind,
                            item.identity,
                            "writer_failed",
                        )
                    try:
                        self._write_manifest()
                    except Exception:
                        pass
            finally:
                self._queue.task_done()

    def _write_transaction(self, item: _QueuedTransaction) -> None:
        """按 rows→commit 顺序追加完整 transaction，并有界 checkpoint manifest。

        commit 始终是 transaction 的最后一行。普通 transaction 只进入进程与内核
        缓冲；segment rotation/close 才建立 fsync durability boundary。running
        manifest 每 256 个 transaction checkpoint 一次，避免每个训练 update 都执行
        原子替换与目录 fsync。
        """

        payload_lines: list[bytes] = []
        row_counts: dict[str, int] = {}
        identity_json = _canonical_json_bytes(item.identity)
        transaction_kind_json = self._transaction_kind_json.get(
            item.transaction_kind
        )
        if transaction_kind_json is None:
            transaction_kind_json = _canonical_json_bytes(
                item.transaction_kind
            )
            self._transaction_kind_json[item.transaction_kind] = (
                transaction_kind_json
            )
        identity_digest = hashlib.sha256(
            b"["
            + transaction_kind_json
            + b","
            + identity_json
            + b"]"
        ).hexdigest()[:16]
        transaction_id = f"tx-{item.sequence:012d}-{identity_digest}"
        envelope_lines: list[bytes] = []
        # payload 既参与 transaction hash，又原样嵌在 row envelope 中。先编码一次
        # 再拼接固定 envelope 字段，避免对 heartbeat 的 40+ provenance fields 递归
        # JSON 编码两次。字段顺序与 sort_keys=True 的 canonical JSON 完全一致。
        envelope_prefix = (
            b'{"format_version":'
            + self._transaction_format_json
            + b',"identity":'
            + identity_json
            + b',"payload":'
        )
        envelope_suffix_prefix = (
            b',"record_type":"transaction_row","row_index":'
        )
        envelope_suffix = (
            b',"session_id":'
            + self._session_id_json
            + b',"transaction_id":'
            + b'"'
            + transaction_id.encode("ascii")
            + b'"'
            + b',"transaction_kind":'
            + transaction_kind_json
            + b"}\n"
        )
        for index, row in enumerate(item.rows):
            payload = _canonical_json_bytes(row)
            payload_lines.append(payload + b"\n")
            row_kind = str(row.get("record_kind", "unspecified"))
            row_counts[row_kind] = row_counts.get(row_kind, 0) + 1
            envelope_lines.append(
                envelope_prefix
                + payload
                + envelope_suffix_prefix
                + str(index).encode("ascii")
                + envelope_suffix
            )
        payload_sha256 = hashlib.sha256(b"".join(payload_lines)).hexdigest()
        committed_at = _utc_now()
        commit = {
            "format_version": TRANSACTION_FORMAT_VERSION,
            "record_type": "transaction_commit",
            "session_id": self.session_id,
            "transaction_id": transaction_id,
            "transaction_kind": item.transaction_kind,
            "identity": item.identity,
            "row_count": len(item.rows),
            "row_counts": row_counts,
            "payload_sha256": payload_sha256,
            "committed_at": committed_at,
        }
        envelope_lines.append(_canonical_json_bytes(commit) + b"\n")
        encoded = b"".join(envelope_lines)
        self._ensure_segment(len(encoded))
        assert self._segment_handle is not None and self._segment_digest is not None
        self._segment_handle.write(encoded)
        self._segment_digest.update(encoded)

        with self._lock:
            segment = self._segments[-1]
            segment["transaction_count"] += 1
            segment["row_count"] += len(item.rows)
            segment["source_bytes"] += len(encoded)
            segment["sha256"] = self._segment_digest.copy().hexdigest()
            segment["last_transaction_id"] = transaction_id
            if segment["first_transaction_id"] is None:
                segment["first_transaction_id"] = transaction_id
            self._committed_transactions += 1
            self._committed_rows += len(item.rows)
            self._last_committed_identity = item.identity
            self._updated_at = committed_at
        self._checkpoint_manifest_if_due()

    def _ensure_segment(self, incoming_bytes: int) -> None:
        """在 count/byte 上限前轮转，并先持久化已关闭 segment 的索引。"""

        current = self._segments[-1] if self._segments else None
        should_rotate = bool(
            current
            and current["transaction_count"] > 0
            and (
                current["transaction_count"] >= self._segment_max_transactions
                or current["source_bytes"] + incoming_bytes > self._segment_max_bytes
            )
        )
        if should_rotate:
            self._close_segment()
            self._checkpoint_manifest_if_due(force=True)
        if self._segment_handle is not None:
            return
        index = len(self._segments)
        relative = f"segments/segment-{index:06d}.jsonl"
        path = self.session_dir / relative
        self._segment_handle = path.open("ab")
        self._segment_digest = hashlib.sha256()
        self._segments.append(
            {
                "index": index,
                "file": relative,
                "transaction_count": 0,
                "row_count": 0,
                "source_bytes": 0,
                "sha256": hashlib.sha256(b"").hexdigest(),
                "first_transaction_id": None,
                "last_transaction_id": None,
            }
        )

    def _close_segment(self) -> None:
        """把当前 segment flush+fsync 后关闭，建立明确的 durability boundary。"""

        if self._segment_handle is None:
            return
        handle = self._segment_handle
        self._segment_handle = None
        self._segment_digest = None
        try:
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            handle.close()

    def _checkpoint_manifest_if_due(self, *, force: bool = False) -> None:
        """有界刷新 running manifest，但不把 transaction cadence 变成 fsync cadence。

        checkpoint 前只把 Python buffer flush 到内核，使 live validator 能看到与
        snapshot 一致的字节。若进程或主机在 segment fsync 前崩溃，running/stale
        manifest 最多被 validator 判为 ``partial`` 或 ``corrupt``；终态 ``success``
        只能由 close 后的 durable segment 与 terminal manifest 共同产生。
        """

        with self._lock:
            committed = self._committed_transactions
            pending = (
                committed
                - self._last_manifest_checkpoint_transaction_count
            )
            due = pending > 0 and (
                force
                or pending
                >= _MANIFEST_CHECKPOINT_INTERVAL_TRANSACTIONS
            )
        if not due:
            return
        if self._segment_handle is not None:
            self._segment_handle.flush()
        self._write_manifest()
        with self._lock:
            self._last_manifest_checkpoint_transaction_count = max(
                self._last_manifest_checkpoint_transaction_count,
                committed,
            )

    def _record_drop_locked(
        self,
        sequence: int,
        transaction_kind: str,
        identity: Any,
        reason: str,
    ) -> None:
        self._dropped_count += 1
        if self._dropped_ranges and self._dropped_ranges[-1]["end_sequence"] + 1 == sequence:
            self._dropped_ranges[-1]["end_sequence"] = sequence
        else:
            self._dropped_ranges.append(
                {"start_sequence": sequence, "end_sequence": sequence}
            )
        if len(self._dropped_examples) < 32:
            self._dropped_examples.append(
                {
                    "sequence": sequence,
                    "transaction_kind": transaction_kind,
                    "identity": identity,
                    "reason": reason,
                }
            )
        if self._status != "corrupt":
            self._status = "degraded"
        self._updated_at = _utc_now()

    def _mark_status_locked(self, status: str) -> None:
        if _STATUS_PRIORITY.get(status, -1) >= _STATUS_PRIORITY.get(self._status, -1):
            self._status = status

    def _manifest_snapshot_locked(self) -> dict[str, Any]:
        return {
            "format_version": ONLINE_ARTIFACT_FORMAT_VERSION,
            "root_format_version": V2_ROOT_FORMAT_VERSION,
            "transaction_format_version": TRANSACTION_FORMAT_VERSION,
            "artifact_kind": "online_session",
            "session_id": self.session_id,
            "analyzer_definitions": [
                dict(descriptor)
                for descriptor in self._analyzer_definitions
            ],
            "status": self._status,
            "created_at": self._created_at,
            "updated_at": self._updated_at,
            "hierarchy_file": "hierarchy.json" if self._hierarchy_sha256 else None,
            "hierarchy_sha256": self._hierarchy_sha256,
            "committed_transaction_count": self._committed_transactions,
            "committed_row_count": self._committed_rows,
            "last_committed_identity": self._last_committed_identity,
            "dropped_transaction_count": self._dropped_count,
            "dropped_transaction_ranges": list(self._dropped_ranges),
            "dropped_transaction_examples": list(self._dropped_examples),
            "writer_error": self._writer_error,
            "writer_queue_capacity": self._queue.maxsize,
            "segment_max_transactions": self._segment_max_transactions,
            "segment_max_uncompressed_bytes": self._segment_max_bytes,
            "segments": [dict(segment) for segment in self._segments],
        }

    def _write_manifest(self) -> None:
        # I/O ownership 必须先于 snapshot；否则较早的 running snapshot 可能在终态
        # manifest 之后完成 replace，令状态倒退。
        with self._manifest_io_lock:
            with self._lock:
                snapshot = self._manifest_snapshot_locked()
            write_json_atomic(self.manifest_path, snapshot)

    def _update_index(self) -> None:
        with self._lock:
            record = {
                "path": str(
                    self.session_dir.relative_to(self.run_dir / "diagnostics" / "v2")
                ),
                "status": self._status,
                "updated_at": self._updated_at,
                "committed_transaction_count": self._committed_transactions,
                "dropped_transaction_count": self._dropped_count,
            }
        update_v2_index(
            self.run_dir,
            "online_sessions",
            self.session_id,
            record,
        )

    def _flush_terminal_metadata(self, deadline: float) -> None:
        """在 close deadline 内尽力写终态；底层 I/O 卡死时不阻塞训练退出。"""

        finished = threading.Event()

        def flush() -> None:
            try:
                self._write_manifest()
                self._update_index()
            except Exception as error:
                with self._lock:
                    self._writer_error = (
                        self._writer_error or f"{type(error).__name__}: {error}"
                    )
                    self._status = "corrupt"
            finally:
                finished.set()

        thread = threading.Thread(
            target=flush,
            name=f"diagnostics-close-{self.session_id}",
            daemon=True,
        )
        thread.start()
        finished.wait(timeout=max(deadline - time.monotonic(), 0.0))


def _load_json_object(path: Path) -> tuple[dict[str, Any], ValidationIssue | None]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}, ValidationIssue("missing_file", "error", "required file is missing", str(path))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        return {}, ValidationIssue(
            "invalid_json",
            "error",
            f"{type(error).__name__}: {error}",
            str(path),
        )
    if not isinstance(value, dict):
        return {}, ValidationIssue("invalid_document", "error", "expected JSON object", str(path))
    return value, None


def _segment_transactions(
    path: Path,
    *,
    issues: list[ValidationIssue] | None = None,
    expected_session_id: str | None = None,
    raw_digest: Any | None = None,
    raw_digest_bytes_read: list[int] | None = None,
) -> Iterator[CommittedTransaction]:
    active_id: str | None = None
    active_kind = ""
    active_identity: Any = None
    active_rows: list[Mapping[str, Any]] = []
    active_payload_lines: list[bytes] = []
    try:
        handle = path.open("rb")
    except OSError as error:
        if issues is not None:
            issues.append(
                ValidationIssue("segment_unreadable", "error", str(error), str(path))
            )
        return
    with handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if raw_digest is not None:
                raw_digest.update(raw_line)
            if raw_digest_bytes_read is not None:
                raw_digest_bytes_read[0] += len(raw_line)
            if not raw_line.strip():
                continue
            try:
                envelope = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                if issues is not None:
                    issues.append(
                        ValidationIssue(
                            "truncated_or_invalid_segment_row",
                            "error",
                            f"line {line_number}: {error}",
                            str(path),
                        )
                    )
                return
            if not isinstance(envelope, dict):
                if issues is not None:
                    issues.append(
                        ValidationIssue(
                            "invalid_segment_envelope",
                            "error",
                            f"line {line_number}: expected object",
                            str(path),
                        )
                    )
                return
            if envelope.get("format_version") != TRANSACTION_FORMAT_VERSION:
                if issues is not None:
                    issues.append(
                        ValidationIssue(
                            "transaction_format_version_mismatch",
                            "error",
                            f"line {line_number}: expected {TRANSACTION_FORMAT_VERSION}",
                            str(path),
                        )
                    )
                return
            if (
                expected_session_id is not None
                and str(envelope.get("session_id", "")) != expected_session_id
            ):
                if issues is not None:
                    issues.append(
                        ValidationIssue(
                            "transaction_session_mismatch",
                            "error",
                            f"line {line_number}: envelope belongs to another session",
                            str(path),
                        )
                    )
                return
            record_type = envelope.get("record_type")
            transaction_id = str(envelope.get("transaction_id", ""))
            if not transaction_id:
                if issues is not None:
                    issues.append(
                        ValidationIssue(
                            "missing_transaction_id",
                            "error",
                            f"line {line_number}",
                            str(path),
                        )
                    )
                return
            if record_type == "transaction_row":
                if active_id is None:
                    active_id = transaction_id
                    active_kind = str(envelope.get("transaction_kind", ""))
                    active_identity = envelope.get("identity")
                if transaction_id != active_id:
                    if issues is not None:
                        issues.append(
                            ValidationIssue(
                                "interleaved_transaction",
                                "error",
                                (
                                    f"line {line_number}: {transaction_id} before "
                                    f"commit of {active_id}"
                                ),
                                str(path),
                            )
                        )
                    return
                if (
                    str(envelope.get("transaction_kind", "")) != active_kind
                    or envelope.get("identity") != active_identity
                ):
                    if issues is not None:
                        issues.append(
                            ValidationIssue(
                                "transaction_envelope_identity_mismatch",
                                "error",
                                f"line {line_number}: row identity differs within transaction",
                                str(path),
                            )
                        )
                    return
                payload = envelope.get("payload")
                if not isinstance(payload, dict):
                    if issues is not None:
                        issues.append(
                            ValidationIssue(
                                "invalid_transaction_payload",
                                "error",
                                f"line {line_number}: payload must be an object",
                                str(path),
                            )
                        )
                    return
                if envelope.get("row_index") != len(active_rows):
                    if issues is not None:
                        issues.append(
                            ValidationIssue(
                                "row_index_mismatch",
                                "error",
                                f"line {line_number}: expected row_index {len(active_rows)}",
                                str(path),
                            )
                        )
                    return
                active_rows.append(payload)
                try:
                    active_payload_lines.append(_canonical_json_bytes(payload) + b"\n")
                except (TypeError, ValueError) as error:
                    if issues is not None:
                        issues.append(
                            ValidationIssue(
                                "noncanonical_payload",
                                "error",
                                f"line {line_number}: {error}",
                                str(path),
                            )
                        )
                    return
                continue
            if record_type != "transaction_commit" or active_id != transaction_id:
                if issues is not None:
                    issues.append(
                        ValidationIssue(
                            "unexpected_commit",
                            "error",
                            f"line {line_number}: commit has no matching payload rows",
                            str(path),
                        )
                    )
                return
            if (
                str(envelope.get("transaction_kind", "")) != active_kind
                or envelope.get("identity") != active_identity
            ):
                if issues is not None:
                    issues.append(
                        ValidationIssue(
                            "transaction_commit_identity_mismatch",
                            "error",
                            f"line {line_number}: commit identity differs from payload rows",
                            str(path),
                        )
                    )
                return
            expected_counts: dict[str, int] = {}
            for row in active_rows:
                key = str(row.get("record_kind", "unspecified"))
                expected_counts[key] = expected_counts.get(key, 0) + 1
            expected_sha = hashlib.sha256(b"".join(active_payload_lines)).hexdigest()
            if envelope.get("row_count") != len(active_rows):
                if issues is not None:
                    issues.append(
                        ValidationIssue(
                            "transaction_row_count_mismatch",
                            "error",
                            f"{transaction_id}: expected {len(active_rows)} rows",
                            str(path),
                        )
                    )
                return
            if envelope.get("row_counts") != expected_counts:
                if issues is not None:
                    issues.append(
                        ValidationIssue(
                            "transaction_row_counts_mismatch",
                            "error",
                            f"{transaction_id}: record-kind counts do not match",
                            str(path),
                        )
                    )
                return
            if envelope.get("payload_sha256") != expected_sha:
                if issues is not None:
                    issues.append(
                        ValidationIssue(
                            "transaction_hash_mismatch",
                            "error",
                            f"{transaction_id}: payload hash does not match",
                            str(path),
                        )
                    )
                return
            yield CommittedTransaction(
                transaction_id=transaction_id,
                transaction_kind=active_kind,
                identity=active_identity,
                rows=tuple(active_rows),
                commit=envelope,
                segment_path=path,
            )
            active_id = None
            active_kind = ""
            active_identity = None
            active_rows = []
            active_payload_lines = []
        if active_id is not None and issues is not None:
            issues.append(
                ValidationIssue(
                    "uncommitted_transaction_tail",
                    "error",
                    f"{active_id} has no terminal commit",
                    str(path),
                )
            )


def iter_committed_transactions(session_dir: str | Path) -> Iterator[CommittedTransaction]:
    """按 manifest segment 顺序流式产出通过校验的 transaction。"""

    directory = Path(session_dir).resolve()
    manifest, issue = _load_json_object(directory / "manifest.json")
    if issue is not None:
        return
    for descriptor in manifest.get("segments", []):
        if not isinstance(descriptor, Mapping) or not descriptor.get("file"):
            continue
        path = (directory / str(descriptor["file"])).resolve()
        try:
            path.relative_to(directory)
        except ValueError:
            continue
        yield from _segment_transactions(
            path,
            expected_session_id=str(manifest.get("session_id") or ""),
        )


def validate_online_session(session_dir: str | Path) -> ValidationResult:
    """验证 manifest、hierarchy、segment digest、transaction commit 与重复 ID。"""

    directory = Path(session_dir).resolve()
    issues: list[ValidationIssue] = []
    manifest, manifest_issue = _load_json_object(directory / "manifest.json")
    if manifest_issue is not None:
        issues.append(manifest_issue)
        return ValidationResult(
            None,
            "corrupt",
            False,
            None,
            0,
            0,
            tuple(issues),
            manifest,
        )
    session_id = manifest.get("session_id")
    if manifest.get("format_version") != ONLINE_ARTIFACT_FORMAT_VERSION:
        issues.append(
            ValidationIssue(
                "format_version_mismatch",
                "error",
                f"expected {ONLINE_ARTIFACT_FORMAT_VERSION}",
                str(directory / "manifest.json"),
            )
        )
    if str(session_id) != directory.name:
        issues.append(
            ValidationIssue(
                "session_identity_mismatch",
                "error",
                "manifest session_id does not match directory",
                str(directory),
            )
        )
    raw_analyzer_definitions = manifest.get("analyzer_definitions")
    try:
        analyzer_definitions = _canonical_analyzer_definitions(
            raw_analyzer_definitions
            if isinstance(raw_analyzer_definitions, list)
            else ()
        )
    except (TypeError, ValueError) as error:
        analyzer_definitions = []
        issues.append(
            ValidationIssue(
                "invalid_analyzer_definitions",
                "error",
                str(error),
                str(directory / "manifest.json"),
            )
        )
    else:
        if raw_analyzer_definitions != analyzer_definitions:
            issues.append(
                ValidationIssue(
                    "noncanonical_analyzer_definitions",
                    "error",
                    "manifest analyzer definitions are not canonical",
                    str(directory / "manifest.json"),
                )
            )
    hierarchy_file = manifest.get("hierarchy_file")
    hierarchy_sha = manifest.get("hierarchy_sha256")
    hierarchy_complete = bool(hierarchy_file and hierarchy_sha)
    if hierarchy_file or hierarchy_sha:
        hierarchy_path = directory / str(hierarchy_file or "hierarchy.json")
        try:
            hierarchy_bytes = hierarchy_path.read_bytes()
            actual = hashlib.sha256(hierarchy_bytes).hexdigest()
            hierarchy = json.loads(hierarchy_bytes)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            issues.append(
                ValidationIssue("hierarchy_unreadable", "error", str(error), str(hierarchy_path))
            )
        else:
            if actual != hierarchy_sha:
                issues.append(
                    ValidationIssue(
                        "hierarchy_hash_mismatch",
                        "error",
                        "hierarchy digest does not match manifest",
                        str(hierarchy_path),
                    )
                )
            if (
                not isinstance(hierarchy, Mapping)
                or hierarchy.get("kind") != "online_diagnostics_v2_hierarchy"
                or str(hierarchy.get("session_id", "")) != str(session_id)
                or hierarchy.get("format_version")
                != ONLINE_DIAGNOSTICS_FORMAT_VERSION
                or not isinstance(hierarchy.get("nodes"), list)
            ):
                issues.append(
                    ValidationIssue(
                        "hierarchy_schema_or_identity_mismatch",
                        "error",
                        "hierarchy kind/session/format/nodes do not match the session contract",
                        str(hierarchy_path),
                    )
                )
            if (
                isinstance(hierarchy, Mapping)
                and hierarchy.get("analyzer_definitions")
                != analyzer_definitions
            ):
                issues.append(
                    ValidationIssue(
                        "hierarchy_analyzer_definitions_mismatch",
                        "error",
                        "hierarchy analyzer definitions do not match manifest",
                        str(hierarchy_path),
                    )
                )

    descriptors = manifest.get("segments")
    if not isinstance(descriptors, list):
        descriptors = []
        issues.append(
            ValidationIssue(
                "invalid_segment_index",
                "error",
                "manifest segments must be a list",
                str(directory / "manifest.json"),
            )
        )
    indexed_paths: set[Path] = set()
    transaction_ids: set[str] = set()
    record_ids: set[Any] = set()
    transaction_kinds: Counter[str] = Counter()
    record_kinds: Counter[str] = Counter()
    anomaly_count = 0
    update_run_count = 0
    last_update: str | None = None
    committed_transactions = 0
    committed_rows = 0
    for descriptor in descriptors:
        if not isinstance(descriptor, Mapping) or not descriptor.get("file"):
            issues.append(
                ValidationIssue(
                    "invalid_segment_descriptor",
                    "error",
                    "segment descriptor is missing file",
                    str(directory / "manifest.json"),
                )
            )
            continue
        path = (directory / str(descriptor["file"])).resolve()
        try:
            path.relative_to(directory)
        except ValueError:
            issues.append(
                ValidationIssue(
                    "segment_path_escape",
                    "error",
                    "segment path leaves the session directory",
                    str(path),
                )
            )
            continue
        indexed_paths.add(path)
        segment_issues: list[ValidationIssue] = []
        segment_transactions = 0
        segment_rows = 0
        raw_digest = hashlib.sha256()
        raw_digest_bytes_read = [0]
        for transaction in _segment_transactions(
            path,
            issues=segment_issues,
            expected_session_id=str(session_id),
            raw_digest=raw_digest,
            raw_digest_bytes_read=raw_digest_bytes_read,
        ):
            segment_transactions += 1
            segment_rows += len(transaction.rows)
            transaction_kinds[transaction.transaction_kind] += 1
            if transaction.transaction_id in transaction_ids:
                issues.append(
                    ValidationIssue(
                        "duplicate_transaction_id",
                        "error",
                        transaction.transaction_id,
                        str(path),
                    )
                )
            transaction_ids.add(transaction.transaction_id)
            for row in transaction.rows:
                record_kind = str(row.get("record_kind", "unspecified"))
                record_kinds[record_kind] += 1
                if record_kind in {"anomaly", "anomaly_episode"}:
                    anomaly_count += 1
                if row.get("update") is not None:
                    update = str(row["update"])
                    if update != last_update:
                        update_run_count += 1
                        last_update = update
                record_id = row.get("record_id")
                if record_id is None:
                    update = row.get("update")
                    node_id = row.get("node_id") or row.get("tap_id")
                    if (
                        update is not None
                        and node_id is not None
                        and record_kind in {"hierarchy_node", "distribution_tap"}
                    ):
                        record_id = (
                            str(session_id),
                            transaction.transaction_id,
                            str(update),
                            str(record_kind),
                            str(node_id),
                        )
                if record_id is not None:
                    try:
                        duplicate = record_id in record_ids
                        record_ids.add(record_id)
                    except TypeError:
                        duplicate = False
                    if duplicate:
                        issues.append(
                            ValidationIssue(
                                "duplicate_record_id",
                                "error",
                                str(record_id),
                                str(path),
                            )
                        )
        digest_complete = True
        try:
            segment_size = path.stat().st_size
            if raw_digest_bytes_read[0] < segment_size:
                with path.open("rb") as handle:
                    handle.seek(raw_digest_bytes_read[0])
                    for chunk in iter(
                        lambda: handle.read(1024 * 1024),
                        b"",
                    ):
                        raw_digest.update(chunk)
        except OSError as error:
            digest_complete = False
            issues.append(
                ValidationIssue(
                    "segment_unreadable",
                    "error",
                    str(error),
                    str(path),
                )
            )
        if (
            digest_complete
            and raw_digest.hexdigest() != descriptor.get("sha256")
        ):
            issues.append(
                ValidationIssue(
                    "segment_hash_mismatch",
                    "error",
                    "segment digest does not match manifest",
                    str(path),
                )
            )
        issues.extend(segment_issues)
        committed_transactions += segment_transactions
        committed_rows += segment_rows
        if descriptor.get("transaction_count") != segment_transactions:
            issues.append(
                ValidationIssue(
                    "segment_transaction_count_mismatch",
                    "error",
                    f"manifest={descriptor.get('transaction_count')} actual={segment_transactions}",
                    str(path),
                )
            )
        if descriptor.get("row_count") != segment_rows:
            issues.append(
                ValidationIssue(
                    "segment_row_count_mismatch",
                    "error",
                    f"manifest={descriptor.get('row_count')} actual={segment_rows}",
                    str(path),
                )
            )
    for path in sorted((directory / "segments").glob("segment-*.jsonl")):
        if path.resolve() not in indexed_paths:
            issues.append(
                ValidationIssue(
                    "unindexed_segment",
                    "error",
                    "segment is not referenced by manifest",
                    str(path),
                )
            )
    if manifest.get("committed_transaction_count") != committed_transactions:
        issues.append(
            ValidationIssue(
                "manifest_transaction_count_mismatch",
                "error",
                (
                    f"manifest={manifest.get('committed_transaction_count')} "
                    f"actual={committed_transactions}"
                ),
                str(directory / "manifest.json"),
            )
        )
    if manifest.get("committed_row_count") != committed_rows:
        issues.append(
            ValidationIssue(
                "manifest_row_count_mismatch",
                "error",
                f"manifest={manifest.get('committed_row_count')} actual={committed_rows}",
                str(directory / "manifest.json"),
            )
        )

    valid = not any(issue.severity == "error" for issue in issues)
    manifest_status = str(manifest.get("status")) if manifest.get("status") is not None else None
    semantically_complete = hierarchy_complete and committed_transactions > 0
    if manifest_status == "complete" and not semantically_complete:
        issues.append(
            ValidationIssue(
                "incomplete_online_source",
                "warning",
                "complete manifest requires a committed transaction and hierarchy",
                str(directory / "manifest.json"),
            )
        )
    if not valid or manifest_status == "corrupt":
        status = "corrupt"
    elif manifest_status == "degraded":
        status = "degraded"
    elif manifest_status == "complete" and semantically_complete:
        status = "success"
    else:
        status = "partial"
    return ValidationResult(
        str(session_id) if session_id is not None else None,
        status,
        valid,
        manifest_status,
        committed_transactions,
        committed_rows,
        tuple(issues),
        manifest,
        content_summary={
            "transaction_kinds": dict(transaction_kinds),
            "record_kinds": dict(record_kinds),
            "anomaly_count": anomaly_count,
            "ordered_update_run_count": update_run_count,
        },
    )


def latest_valid_online_hierarchy(run_dir: str | Path) -> Path | None:
    """选择最近一个完整且通过 validator 的 canonical v2 hierarchy。

    session 的可用性只由 ``validate_online_session()`` 决定；调用方不得仅凭 manifest
    的终态字段消费 hierarchy。多个 session 并存时按 manifest ``created_at`` 和
    ``session_id`` 稳定选择最近的合法来源。存在合法旧 session 时忽略较新的损坏
    session；只有完整 session 全部校验失败时才 fail-fast，避免静默回退结构发现。
    """

    valid_candidates: list[tuple[str, str, Path]] = []
    rejected_complete: list[tuple[str, tuple[str, ...]]] = []
    for session_dir in discover_v2_artifacts(run_dir).online_sessions:
        validation = validate_online_session(session_dir)
        hierarchy_file = validation.manifest.get("hierarchy_file")
        canonical_hierarchy = hierarchy_file == "hierarchy.json"
        session_id = validation.session_id or session_dir.name
        if (
            validation.valid
            and validation.status == "success"
            and canonical_hierarchy
        ):
            valid_candidates.append(
                (
                    str(validation.manifest.get("created_at") or ""),
                    str(session_id),
                    session_dir / "hierarchy.json",
                )
            )
            continue
        if validation.manifest_status == "complete":
            issue_codes = tuple(
                sorted({issue.code for issue in validation.issues})
            )
            if not canonical_hierarchy:
                issue_codes = tuple(
                    sorted({*issue_codes, "noncanonical_hierarchy_file"})
                )
            rejected_complete.append((str(session_id), issue_codes))

    if valid_candidates:
        return max(valid_candidates)[2]
    if rejected_complete:
        details = "; ".join(
            f"{session_id}[{','.join(issue_codes) or 'invalid'}]"
            for session_id, issue_codes in rejected_complete
        )
        raise ValueError(
            "No complete online diagnostics session passed validation: "
            f"{details}"
        )
    return None


__all__ = [
    "CommittedTransaction",
    "DEFAULT_SEGMENT_MAX_TRANSACTIONS",
    "DEFAULT_SEGMENT_MAX_UNCOMPRESSED_BYTES",
    "DEFAULT_WRITER_CLOSE_TIMEOUT_SECONDS",
    "DEFAULT_WRITER_QUEUE_CAPACITY",
    "ONLINE_ARTIFACT_FORMAT_VERSION",
    "ONLINE_DIAGNOSTICS_FORMAT_VERSION",
    "OnlineArtifactWriter",
    "TRANSACTION_FORMAT_VERSION",
    "V2ArtifactInventory",
    "V2_ROOT_FORMAT_VERSION",
    "ValidationIssue",
    "ValidationResult",
    "discover_v2_artifacts",
    "iter_committed_transactions",
    "latest_valid_online_hierarchy",
    "update_v2_index",
    "validate_online_session",
    "write_json_atomic",
]
