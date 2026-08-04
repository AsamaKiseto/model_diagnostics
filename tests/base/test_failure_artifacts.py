"""
验证 portable failure bundle 的 rank-local durability 与 fail-closed 校验。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import pytest

from model_diagnostics.base.artifacts import (
    FAILURE_BUNDLE_COMMIT_FORMAT_VERSION,
    FAILURE_BUNDLE_FORMAT_VERSION,
    FailureBundleWriter,
    FailureEvidenceScope,
    finalize_failure_bundle,
    validate_failure_bundle,
    validate_failure_rank,
)


def _writer(
    bundle_dir: Path,
    *,
    rank: int = 0,
    world_size: int = 1,
) -> FailureBundleWriter:
    return FailureBundleWriter(
        bundle_dir,
        bundle_id="failure-a",
        session_id="session-a",
        attempt_id="attempt-a",
        rank=rank,
        world_size=world_size,
        failure_phase="backward",
        failure_reason="nonfinite_gradient",
        evidence_scope=FailureEvidenceScope.METADATA_ONLY,
        metadata={"logical_update_candidate": 17},
    )


def test_failure_bundle_commits_rank_local_attachments_and_validates(
    tmp_path: Path,
) -> None:
    bundle_dir = tmp_path / "failure_bundles" / "failure-a"
    rank_zero = _writer(bundle_dir, rank=0, world_size=2)
    rank_zero.write_json("metadata/objective.json", {"identity": "loss"})
    rank_zero.write_bytes("state/gradient.bin", b"gradient")
    rank_zero.commit()
    rank_one = _writer(bundle_dir, rank=1, world_size=2)
    rank_one.write_bytes("state/gradient.bin", b"peer-gradient")
    rank_one.commit()
    root_commit = finalize_failure_bundle(bundle_dir)

    validation = validate_failure_bundle(bundle_dir)
    rank_validation = validate_failure_rank(rank_zero.rank_dir)
    reference = rank_zero.reference(relative_to=tmp_path)

    assert validation.status == "success"
    assert validation.valid is True
    assert validation.expected_rank_count == 2
    assert validation.captured_rank_count == 2
    assert rank_validation.status == "success"
    assert rank_validation.valid is True
    assert [item.relative_path for item in rank_validation.attachments] == [
        "metadata/objective.json",
        "state/gradient.bin",
    ]
    assert reference.bundle_id == "failure-a"
    assert reference.relative_path == "failure_bundles/failure-a"
    assert reference.format_version == FAILURE_BUNDLE_FORMAT_VERSION
    manifest = json.loads(
        rank_zero.manifest_path.read_text(encoding="utf-8")
    )
    assert manifest["format_version"] == FAILURE_BUNDLE_FORMAT_VERSION
    assert manifest["status"] == "complete"
    assert root_commit == bundle_dir / "commit.json"
    root_manifest = json.loads(
        (bundle_dir / "manifest.json").read_text(encoding="utf-8")
    )
    root_commit_payload = json.loads(
        (bundle_dir / "commit.json").read_text(encoding="utf-8")
    )
    assert root_manifest["expected_rank_count"] == 2
    assert [item["rank"] for item in root_manifest["rank_commits"]] == [0, 1]
    assert (
        root_commit_payload["format_version"]
        == FAILURE_BUNDLE_COMMIT_FORMAT_VERSION
    )


def test_failure_bundle_missing_rank_is_structurally_valid_but_degraded(
    tmp_path: Path,
) -> None:
    bundle_dir = tmp_path / "failure-a"
    writer = _writer(bundle_dir, rank=0, world_size=2)
    writer.commit(status="degraded", note="peer rank unavailable")

    assert finalize_failure_bundle(bundle_dir) is None
    validation = validate_failure_bundle(bundle_dir)

    assert validation.status == "degraded"
    assert validation.valid is True
    assert validation.expected_rank_count == 2
    assert validation.captured_rank_count == 1
    assert {issue.code for issue in validation.issues} >= {
        "missing_failure_ranks",
        "missing_failure_bundle_commit",
    }


def test_complete_rank_set_without_root_commit_remains_partial(
    tmp_path: Path,
) -> None:
    bundle_dir = tmp_path / "failure-a"
    writer = _writer(bundle_dir)
    writer.commit()

    validation = validate_failure_bundle(bundle_dir)

    assert validation.status == "partial"
    assert validation.valid is False
    assert {issue.code for issue in validation.issues} >= {
        "missing_failure_bundle_commit"
    }


def test_failure_bundle_without_commit_remains_partial(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path / "failure-a")
    writer.write_json("metadata/objective.json", {"identity": "loss"})

    rank_validation = validate_failure_rank(writer.rank_dir)
    bundle_validation = validate_failure_bundle(writer.bundle_dir)

    assert rank_validation.status == "partial"
    assert rank_validation.valid is False
    assert bundle_validation.status == "partial"
    assert bundle_validation.valid is False
    assert {issue.code for issue in rank_validation.issues} >= {
        "missing_failure_rank_commit"
    }


def test_failure_bundle_tampering_fails_closed(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path / "failure-a")
    writer.write_bytes("state/gradient.bin", b"gradient")
    writer.commit()
    assert finalize_failure_bundle(writer.bundle_dir) is not None
    attachment = writer.attachments_dir / "state" / "gradient.bin"
    attachment.write_bytes(b"tampered")

    validation = validate_failure_bundle(writer.bundle_dir)

    assert validation.status == "corrupt"
    assert validation.valid is False
    assert {issue.code for issue in validation.issues} >= {
        "failure_attachment_digest_mismatch"
    }


def test_failure_bundle_finalize_is_race_safe_and_idempotent(
    tmp_path: Path,
) -> None:
    """多个 rank 观察到完整 inventory 时可并发完成同一根 commit。"""

    bundle_dir = tmp_path / "failure-a"
    for rank in range(2):
        writer = _writer(bundle_dir, rank=rank, world_size=2)
        writer.write_json("metadata.json", {"rank": rank})
        writer.commit()

    with ThreadPoolExecutor(max_workers=8) as executor:
        commits = tuple(
            executor.map(
                lambda _: finalize_failure_bundle(bundle_dir),
                range(16),
            )
        )

    assert set(commits) == {bundle_dir / "commit.json"}
    validation = validate_failure_bundle(bundle_dir)
    assert validation.status == "success"
    assert validation.valid is True


def test_failure_bundle_root_manifest_tampering_fails_closed(
    tmp_path: Path,
) -> None:
    bundle_dir = tmp_path / "failure-a"
    writer = _writer(bundle_dir)
    writer.commit()
    assert finalize_failure_bundle(bundle_dir) is not None
    root_manifest_path = bundle_dir / "manifest.json"
    root_manifest = json.loads(root_manifest_path.read_text(encoding="utf-8"))
    root_manifest["captured_rank_count"] = 0
    root_manifest_path.write_text(json.dumps(root_manifest), encoding="utf-8")

    validation = validate_failure_bundle(bundle_dir)

    assert validation.status == "corrupt"
    assert validation.valid is False
    assert {issue.code for issue in validation.issues} >= {
        "failure_bundle_manifest_mismatch",
        "failure_bundle_manifest_digest_mismatch",
    }


@pytest.mark.parametrize(
    "relative_path",
    ("../escape.bin", "/absolute.bin", "nested\\windows.bin", "./alias.bin"),
)
def test_failure_bundle_rejects_unsafe_attachment_paths(
    tmp_path: Path,
    relative_path: str,
) -> None:
    writer = _writer(tmp_path / "failure-a")

    with pytest.raises(ValueError, match="safe POSIX relative path"):
        writer.write_bytes(relative_path, b"payload")
