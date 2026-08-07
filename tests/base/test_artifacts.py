"""验证 portable diagnostics transaction、完整性校验与统一报告契约。"""

from __future__ import annotations

import errno
import json
from pathlib import Path
import threading
from types import MethodType

import model_diagnostics.base.reporting.report as report_module
import pytest

from model_diagnostics.base.artifacts import (
    FailureBundleWriter,
    FailureEvidenceScope,
    ONLINE_ARTIFACT_FORMAT_VERSION,
    ONLINE_DIAGNOSTICS_FORMAT_VERSION,
    OnlineArtifactWriter,
    TRANSACTION_FORMAT_VERSION,
    V2_ROOT_FORMAT_VERSION,
    finalize_failure_bundle,
    file_sha256,
    iter_committed_transactions,
    latest_valid_online_hierarchy,
    validate_online_session,
)
from model_diagnostics.base.registry import (
    FLIGHT_RECORDER_ANALYZER_DEFINITION,
)
from model_diagnostics.base.reporting import (
    REPORT_FORMAT_VERSION,
    generate_diagnostics_report,
)
from model_diagnostics.base.reporting import (
    BASE_EVIDENCE_RENDERER_CATALOG,
    EvidenceMetric,
    EvidenceRendererCatalog,
    EvidenceRendererDefinition,
    render_evidence_figure,
)


def _hierarchy(session_id: str) -> dict[str, object]:
    return {
        "format_version": ONLINE_DIAGNOSTICS_FORMAT_VERSION,
        "kind": "online_diagnostics_v2_hierarchy",
        "session_id": session_id,
        "nodes": [
            {
                "node_id": "all",
                "parent_id": None,
                "hierarchy_level": "All",
                "model_name": None,
                "module_path": "",
                "module_type": "AllModels",
            }
        ],
    }


def _writer(
    run_dir: Path,
    *,
    session_id: str,
) -> OnlineArtifactWriter:
    return OnlineArtifactWriter(
        run_dir,
        analyzer_definitions=(
            FLIGHT_RECORDER_ANALYZER_DEFINITION.to_manifest(),
        ),
        session_id=session_id,
    )


def test_report_selects_only_the_active_checkpoint_runtime(
    tmp_path: Path,
) -> None:
    """当前报告不得把不同 runtime generation 的证据混在同一图中。"""

    analyses = tmp_path / "diagnostics" / "v2" / "checkpoint" / "analyses"
    old = analyses / "old"
    current = analyses / "current"
    old.mkdir(parents=True)
    current.mkdir()
    (old / "manifest.json").write_text(
        json.dumps({"runtime_descriptor_digest": "old-runtime"}),
        encoding="utf-8",
    )
    (current / "manifest.json").write_text(
        json.dumps({"runtime_descriptor_digest": "current-runtime"}),
        encoding="utf-8",
    )
    (tmp_path / "diagnostics" / "v2" / "index.json").write_text(
        json.dumps(
            {
                "active_checkpoint_runtime_descriptor_digest": (
                    "current-runtime"
                )
            }
        ),
        encoding="utf-8",
    )

    assert report_module._active_checkpoint_analyses(
        tmp_path,
        (old, current),
    ) == (current,)


def test_online_schema_v10_preserves_root_and_transaction_identity() -> None:
    assert ONLINE_DIAGNOSTICS_FORMAT_VERSION == 10
    assert ONLINE_ARTIFACT_FORMAT_VERSION == "training_diagnostics_v2"
    assert V2_ROOT_FORMAT_VERSION == "diagnostics_v2"
    assert TRANSACTION_FORMAT_VERSION == "diagnostics_transaction_v1"


def test_noncurrent_online_schema_is_rejected_by_validator_and_report(
    tmp_path: Path,
) -> None:
    """validator/report 只接受当前 hierarchy schema。"""

    writer = _writer(tmp_path, session_id="session-unsupported")
    writer.write_hierarchy(_hierarchy(writer.session_id))
    assert writer.submit_transaction(
        "logical_update",
        "attempt-1",
        [{"record_kind": "update_heartbeat", "update": 1}],
    )
    writer.close()

    hierarchy = json.loads(
        writer.hierarchy_path.read_text(encoding="utf-8")
    )
    hierarchy["format_version"] = -1
    writer.hierarchy_path.write_text(
        json.dumps(
            hierarchy,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = json.loads(writer.manifest_path.read_text(encoding="utf-8"))
    manifest["hierarchy_sha256"] = file_sha256(writer.hierarchy_path)
    writer.manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )

    validation = validate_online_session(writer.session_dir)
    report = generate_diagnostics_report(tmp_path)
    summary = json.loads(
        (report / "summary.json").read_text(encoding="utf-8")
    )
    assert ONLINE_DIAGNOSTICS_FORMAT_VERSION == 10
    assert validation.status == "corrupt"
    assert validation.valid is False
    assert {
        issue.code for issue in validation.issues
    } >= {"hierarchy_schema_or_identity_mismatch"}
    assert summary["status"] == "corrupt"


def test_report_summarizes_failure_without_embedding_attachments(
    tmp_path: Path,
) -> None:
    """failure source 只展示完整性与安全状态，不把敏感 attachment 嵌入报告。"""

    writer = _writer(tmp_path, session_id="session-failure")
    writer.write_hierarchy(_hierarchy(writer.session_id))
    assert writer.submit_transaction(
        "failed_attempt",
        "attempt-failure",
        [{"record_kind": "failed_attempt", "status": "failed"}],
    )
    writer.close(status="degraded")
    bundle = writer.session_dir / "failure_bundles" / "failure-a"
    failure_writer = FailureBundleWriter(
        bundle,
        bundle_id="failure-a",
        session_id=writer.session_id,
        attempt_id="attempt-failure",
        rank=0,
        world_size=1,
        failure_phase="backward",
        failure_reason="nonfinite_gradient",
        evidence_scope=FailureEvidenceScope.METADATA_ONLY,
    )
    secret = b"attachment-payload-must-not-enter-report"
    failure_writer.write_bytes("failure_context.bin", secret)
    failure_writer.commit()
    assert finalize_failure_bundle(bundle) is not None

    report = generate_diagnostics_report(tmp_path)
    summary = json.loads(
        (report / "summary.json").read_text(encoding="utf-8")
    )
    standalone = (
        tmp_path / "diagnostics" / "v2" / "diagnostics-report.html"
    )
    document = standalone.read_text(encoding="utf-8")

    assert REPORT_FORMAT_VERSION == 18
    assert summary["format_version"] == 18
    assert summary["status"] == "degraded"
    assert summary["source_count"] == 2
    assert summary["failure_bundles"] == [
        {
            "bundle_id": "failure-a",
            "path": str(bundle),
            "status": "success",
            "valid": True,
            "expected_rank_count": 1,
            "captured_rank_count": 1,
            "evidence_scope": ["metadata_only"],
            "issue_codes": [],
            "attachments_exposed": False,
        }
    ]
    assert secret.decode("ascii") not in document
    assert "诊断栏目" in document
    assert "当前诊断产物没有有限观测；未生成替代图" in document
    assert "证据与原始数据" in document
    assert "字段可视化审计" in document
    assert "可视化覆盖" in document
    assert "完整字段可视化" in document
    assert 'id="chart-dialog"' in document
    assert "训练过程监测" in document
    assert "逐检查点异常监测" in document
    assert "最终模型分析" in document
    assert "Input-to-Output Influence" in document
    assert "Stage/Block Influence" in document
    assert 'categories:["input"]' in document
    assert 'categories:["module"]' in document
    assert "选择输出通道" in document
    assert "其它已采集诊断量可在下方“补充证据”中按需绘图" in document
    assert "补充证据" in document
    assert "这个目的只保留下钻或审计字段" not in document
    assert "validity gate" not in document
    assert "个方向图" not in document
    assert "项 validity" not in document
    assert "放大图表" not in document
    assert 'id="secondary-evidence"' not in document
    assert "categorical_paths" in document
    assert "未运行" in document
    assert "不输出根因或下一步建议" not in document
    assert 'id="run"' not in document
    assert 'id="zoom-y-scale"' in document
    assert "online-session-failure-segment-000000" in document
    assert standalone.is_file()
    assert "self_contained_gzip_embedded" in standalone.read_text(
        encoding="utf-8"
    )
    assert summary["metric_guides"]
    guides = {
        guide["path"]: guide for guide in summary["metric_guides"]
    }
    assert guides["raw_objective_sum"]["category"] == "training"
    assert guides["gradient_norm"]["role"] == "primary"
    assert "activation_rms" not in guides
    assert "forward_call_count" not in guides
    assert not any(
        path.name == "failure_context.bin"
        for path in (report / "data").glob("**/*")
    )


def test_online_writer_commits_whole_transactions_with_fixed_policy(
    tmp_path: Path,
) -> None:
    writer = _writer(
        tmp_path,
        session_id="session-a",
    )
    writer.write_hierarchy(_hierarchy(writer.session_id))
    assert writer.submit_transaction(
        "logical_update",
        "attempt-1",
        [
            {"record_kind": "update_heartbeat", "update": 1},
            {"record_kind": "hierarchy_node", "update": 1, "node_id": "all"},
        ],
    )
    assert writer.submit_transaction(
        "failed_attempt",
        "attempt-2",
        [{"record_kind": "failed_attempt", "update": 2}],
    )
    writer.close()

    result = validate_online_session(writer.session_dir)
    transactions = list(iter_committed_transactions(writer.session_dir))
    manifest = json.loads(writer.manifest_path.read_text(encoding="utf-8"))
    assert result.status == "success"
    assert result.valid is True
    assert result.committed_transaction_count == 2
    assert result.committed_row_count == 3
    assert [item.transaction_kind for item in transactions] == [
        "logical_update",
        "failed_attempt",
    ]
    assert [len(item.rows) for item in transactions] == [2, 1]
    assert manifest["format_version"] == ONLINE_ARTIFACT_FORMAT_VERSION
    assert manifest["status"] == "complete"
    assert manifest["analyzer_definitions"] == [
        FLIGHT_RECORDER_ANALYZER_DEFINITION.to_manifest()
    ]
    hierarchy = json.loads(
        writer.hierarchy_path.read_text(encoding="utf-8")
    )
    assert hierarchy["format_version"] == ONLINE_DIAGNOSTICS_FORMAT_VERSION
    assert hierarchy["analyzer_definitions"] == manifest[
        "analyzer_definitions"
    ]
    assert len(manifest["segments"]) == 1
    assert all(segment["sha256"] for segment in manifest["segments"])
    for segment in manifest["segments"]:
        segment_path = writer.session_dir / str(segment["file"])
        for line in segment_path.read_bytes().splitlines():
            parsed = json.loads(line)
            canonical = json.dumps(
                parsed,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            assert line == canonical


def test_queue_overflow_drops_one_complete_transaction(
    tmp_path: Path,
) -> None:
    writer = _writer(
        tmp_path,
        session_id="session-overflow",
    )
    started = threading.Event()
    release = threading.Event()
    original = writer._write_transaction

    def blocked(self: OnlineArtifactWriter, item: object) -> None:
        started.set()
        assert release.wait(timeout=5)
        original(item)  # type: ignore[arg-type]

    writer._write_transaction = MethodType(  # type: ignore[method-assign]
        blocked,
        writer,
    )
    assert writer.submit_transaction(
        "logical_update",
        "attempt-1",
        [{"record_kind": "update_heartbeat", "update": 1}],
    )
    assert started.wait(timeout=5)
    for update in range(2, 66):
        assert writer.submit_transaction(
            "logical_update",
            f"attempt-{update}",
            [{"record_kind": "update_heartbeat", "update": update}],
        )
    assert not writer.submit_transaction(
        "logical_update",
        "attempt-66",
        [
            {"record_kind": "update_heartbeat", "update": 66},
            {"record_kind": "hierarchy_node", "update": 66},
        ],
    )
    release.set()
    writer.close()

    manifest = json.loads(writer.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "degraded"
    assert manifest["dropped_transaction_count"] == 1
    assert [
        transaction.identity
        for transaction in iter_committed_transactions(writer.session_dir)
    ] == [f"attempt-{update}" for update in range(1, 66)]
    assert validate_online_session(writer.session_dir).status == "degraded"


def test_online_validator_rejects_truncated_segment_and_report_excludes_rows(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path, session_id="session-corrupt")
    writer.write_hierarchy(_hierarchy(writer.session_id))
    assert writer.submit_transaction(
        "logical_update",
        "attempt-1",
        [{"record_kind": "update_heartbeat", "update": 1}],
    )
    writer.close()
    segment = next((writer.session_dir / "segments").glob("*.jsonl"))
    with segment.open("ab") as handle:
        handle.write(b'{"record_type":"transaction_row"')

    result = validate_online_session(writer.session_dir)
    report = generate_diagnostics_report(tmp_path)
    summary = json.loads((report / "summary.json").read_text(encoding="utf-8"))
    document = (
        tmp_path / "diagnostics" / "v2" / "diagnostics-report.html"
    ).read_text(encoding="utf-8")
    assert result.status == "corrupt"
    assert result.valid is False
    assert {
        "segment_hash_mismatch",
        "truncated_or_invalid_segment_row",
    }.issubset({issue.code for issue in result.issues})
    assert summary["status"] == "corrupt"
    assert f"online {writer.session_id} · {segment.name}" not in document


def test_writer_disk_failure_is_fail_stop_and_never_complete(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path, session_id="session-enospc")
    attempted = threading.Event()

    def fail_with_enospc(self: OnlineArtifactWriter, item: object) -> None:
        del self, item
        attempted.set()
        raise OSError(errno.ENOSPC, "injected diagnostics disk exhaustion")

    writer._write_transaction = MethodType(  # type: ignore[method-assign]
        fail_with_enospc,
        writer,
    )
    assert writer.submit_transaction(
        "logical_update",
        "attempt-1",
        [{"record_kind": "update_heartbeat", "update": 1}],
    )
    assert attempted.wait(timeout=5)
    writer.close()

    manifest = json.loads(writer.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "corrupt"
    assert "injected diagnostics disk exhaustion" in manifest["writer_error"]
    assert manifest["committed_transaction_count"] == 0
    assert list(iter_committed_transactions(writer.session_dir)) == []


def test_latest_hierarchy_uses_only_complete_validator_valid_session(
    tmp_path: Path,
) -> None:
    valid = _writer(tmp_path, session_id="session-valid")
    valid.write_hierarchy(_hierarchy(valid.session_id))
    assert valid.submit_transaction(
        "logical_update",
        "attempt-valid",
        [{"record_kind": "update_heartbeat", "update": 1}],
    )
    valid.close()

    invalid = _writer(tmp_path, session_id="session-invalid")
    invalid.write_hierarchy(_hierarchy(invalid.session_id))
    assert invalid.submit_transaction(
        "logical_update",
        "attempt-invalid",
        [{"record_kind": "update_heartbeat", "update": 2}],
    )
    invalid.close()
    damaged = json.loads(invalid.hierarchy_path.read_text(encoding="utf-8"))
    damaged["session_id"] = "wrong-session"
    invalid.hierarchy_path.write_text(json.dumps(damaged), encoding="utf-8")

    assert latest_valid_online_hierarchy(tmp_path) == valid.hierarchy_path


def test_report_is_always_one_self_contained_html(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path, session_id="session-linked")
    writer.write_hierarchy(_hierarchy(writer.session_id))
    assert writer.submit_transaction(
        "logical_update",
        "attempt-1",
        [
            {
                "record_kind": "update_heartbeat",
                "update": 1,
                "payload": "x" * 4096,
            }
        ],
    )
    writer.close()

    report = generate_diagnostics_report(tmp_path)
    summary = json.loads((report / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((report / "manifest.json").read_text(encoding="utf-8"))
    standalone_path = (
        tmp_path / "diagnostics" / "v2" / "diagnostics-report.html"
    )
    standalone = standalone_path.read_text(encoding="utf-8")

    assert summary["status"] == "success"
    assert (
        summary["interactive_report"]["delivery"]
        == "self_contained_gzip_embedded"
    )
    assert manifest["files"]["html"]["path"] == "../diagnostics-report.html"
    assert not (report / "data").exists()
    assert not (report / "report.html").exists()
    assert standalone_path.is_file()
    assert '"delivery":"self_contained_gzip_embedded"' in standalone
    assert '"embedded_key":' in standalone
    assert 'type="application/octet-stream"' in standalone
    assert 'id="payloads"' not in standalone
    assert "DATA_WORKER_SOURCE" in standalone
    assert 'mode==="observations"' in standalone
    assert ".chart-controls[hidden]" in standalone
    assert 'view.kind==="rollout_overview"?' in standalone
    assert "function updateAxisTicks" in standalone
    assert 'data-role="horizon"' not in standalone
    assert 'data-role="summary-metric"' in standalone
    assert "function drawObjectiveOverview" in standalone
    assert "function drawRolloutOverview" in standalone
    assert "function drawRolloutHeatmap" in standalone
    assert "function drawRolloutCoverage" in standalone
    assert "完整样本判定长度" in standalone
    assert "rolloutExclusionSummary" in standalone
    assert "grid-template-columns:repeat(2,minmax(0,1fr))" in standalone
    assert "tall-chart" not in standalone
    assert "查看当前筛选的色块影响矩阵" in standalone
    assert "purpose-tabs" in standalone
    assert "drawInfluenceList" in standalone


def test_online_validator_rejects_hierarchy_descriptor_drift(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path, session_id="session-descriptor-drift")
    writer.write_hierarchy(_hierarchy(writer.session_id))
    assert writer.submit_transaction(
        "logical_update",
        "attempt-1",
        [{"record_kind": "update_heartbeat", "update": 1}],
    )
    writer.close()
    hierarchy = json.loads(
        writer.hierarchy_path.read_text(encoding="utf-8")
    )
    hierarchy["analyzer_definitions"][0]["definition_version"] = 999
    writer.hierarchy_path.write_text(
        json.dumps(hierarchy, sort_keys=True),
        encoding="utf-8",
    )

    result = validate_online_session(writer.session_dir)

    assert result.valid is False
    assert {
        "hierarchy_hash_mismatch",
        "hierarchy_analyzer_definitions_mismatch",
    }.issubset({issue.code for issue in result.issues})


def test_report_keeps_complete_categorical_statistics() -> None:
    definition = EvidenceRendererDefinition(
        evidence_kind="bounded_test",
        title="Bounded test",
        metrics=(EvidenceMetric("value", "value"),),
        null_control_fields=("identity_exact",),
        status_fields=("trace_status",),
        analyzer_definition_version=1,
    )
    rows = [
        {
            "status": "success" if index % 2 == 0 else "skipped",
            "skip_reason": f"reason-{index}-" + "x" * 256,
            "identity_exact": index % 2 == 0,
            "trace_status": f"trace-{index}",
            "value": float(index),
        }
        for index in range(256)
    ]

    figure = render_evidence_figure(
        source_id="bounded",
        analyzer_name="bounded",
        descriptor={
            "name": "bounded",
            "evidence_kind": "bounded_test",
            "execution_mode": "stream",
            "option_keys": [],
            "definition_version": 1,
            "claim_boundaries": [],
        },
        rows=rows,
        renderer_catalog=EvidenceRendererCatalog(
            {"bounded_test": definition}
        ),
        figure_id="bounded",
    )

    assert figure["status"] == "partial"
    assert figure["null_control_status"] == "mixed"
    assert len(figure["skip_reasons"]) == 256
    assert len(figure["secondary_statuses"]["trace_status"]) == 256
    assert figure["skip_reasons"][0]["reason"].endswith("x" * 256)
    assert figure["aggregation_policy"] == {
        "categorical_values": "complete",
        "categorical_labels": "complete",
        "series_points": "complete",
        "scalar_fields": "complete",
    }
