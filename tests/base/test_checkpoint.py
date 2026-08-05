"""Base checkpoint 执行、显式 analyzer catalog 与 portable adapter 回归。

测试以任意文件名 checkpoint 和任务无关 toy adapter 保护 Base contract，不依赖
具体宿主数据、checkpoint 命名规则或 host integration 实现。
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import pytest
import torch

from model_diagnostics.base import (
    AnalyzerBindingCatalog,
    AnalyzerDefinition,
    AnalyzerExecutionResult,
    BASE_ANALYZER_CATALOG,
    DiagnosticComponent,
    ModuleSite,
    bind_adapter_capability,
    compose_catalogs,
)
from model_diagnostics.base.artifacts import (
    CHECKPOINT_FORMAT_VERSION,
    DiagnosticsArtifactStore,
    file_sha256,
    stable_json_hash,
    validate_checkpoint_analysis,
)
from model_diagnostics.base.checkpoint import (
    CheckpointRef,
    CheckpointRuntimeDescriptor,
    ComponentResponse,
    CohortSelection,
    ConditionUnavailable,
    DiagnosticsRecipe,
    LoadedModel,
    ModuleActivationCapture,
    ModulePatchReference,
    ObjectiveResult,
    SampleRef,
    run_checkpoint_diagnostics,
)
from model_diagnostics.base.reporting import (
    BASE_EVIDENCE_RENDERER_CATALOG,
    build_metric_guide_catalog,
    generate_diagnostics_report,
)


class _ToyModel(torch.nn.Module):
    """带 registered buffer 的通用 MLP，用于验证 branch state isolation。"""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Linear(3, 5)
        self.normalization = torch.nn.BatchNorm1d(5)
        self.head = torch.nn.Linear(5, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(self.normalization(self.encoder(value)))
        return self.head(hidden)


_EXAMPLE_ANALYZER = AnalyzerDefinition(
    name="example_analyzer",
    evidence_kind="example_scalar_evidence",
    execution_mode="aggregate",
    option_keys=frozenset(),
    definition_version=3,
    claim_boundaries=("example evidence is descriptive only",),
)


class _ToyAdapter:
    """实现 Base protocol，但不携带时序、channel 或宿主目录知识。"""

    execution_precision = "fp32"
    analyzer_catalog = compose_catalogs(
        BASE_ANALYZER_CATALOG,
        (_EXAMPLE_ANALYZER,),
    )
    analyzer_bindings = AnalyzerBindingCatalog(
        {
            _EXAMPLE_ANALYZER.name: bind_adapter_capability(
                _EXAMPLE_ANALYZER,
                "analyze_example",
            )
        }
    )

    def __init__(self, checkpoint_path: Path) -> None:
        self.device = torch.device("cpu")
        self.rank = 0
        self.world_size = 1
        self.checkpoint_path = checkpoint_path.resolve()
        self.closed_states: list[dict[str, torch.Tensor]] = []
        self.analyzer_calls: list[dict[str, Any]] = []
        self.module_site_call_count = 0
        self.objective_call_count = 0
        self.alternate_objective_identity = False
        self.vector_objective = False
        self.vector_cotangent = False
        self.inventory_mode = "complete"
        self.samples = {
            0: {
                "features": torch.tensor(
                    [[0.2, -0.5, 1.0], [1.2, 0.3, -0.7]],
                    dtype=torch.float32,
                ),
                "label": torch.tensor([[0.1], [-0.2]], dtype=torch.float32),
            },
            1: {
                "features": torch.tensor(
                    [[-0.4, 0.8, 0.1], [0.7, -1.1, 0.2]],
                    dtype=torch.float32,
                ),
                "label": torch.tensor([[0.4], [0.0]], dtype=torch.float32),
            },
        }
        self.recipe: DiagnosticsRecipe | None = None

    def configure(self, recipe: DiagnosticsRecipe) -> None:
        """保存 Base 已解析 recipe；重复配置只允许相同事实源。"""

        if self.recipe is not None and self.recipe.to_dict() != recipe.to_dict():
            raise RuntimeError("toy adapter cannot be reconfigured")
        self.recipe = recipe

    def resolve_checkpoints(
        self,
        *,
        run_dir: Path,
    ) -> Sequence[CheckpointRef]:
        del run_dir
        assert self.recipe is not None
        paths = (
            (
                self.recipe.final_checkpoint
                or str(self.checkpoint_path)
            ),
        )
        return tuple(
            CheckpointRef(
                path=str(Path(path).resolve()),
                identity=stable_json_hash(
                    {
                        "path": str(Path(path).resolve()),
                        "sha256": file_sha256(path),
                    }
                ),
                update=None,
                kind="caller_selected",
            )
            for path in paths
        )

    def load_model(self, checkpoint_path: str, *, precision: str) -> LoadedModel:
        assert precision == "fp32"
        model = _ToyModel()
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        model.train()
        return LoadedModel(
            model=model,
            checkpoint_update=None,
            model_name="generic_mlp",
            structure_metadata={"source": "toy"},
        )

    def build_cohort(
        self,
        loaded: LoadedModel,
        *,
        checkpoints: Sequence[CheckpointRef],
    ) -> CohortSelection:
        del loaded, checkpoints
        count = len(self.samples)
        refs = tuple(
            SampleRef(
                sample_id=f"sample-{index}",
                partition="validation",
                sample_index=index,
                group_id=f"subject-{index}",
                position=index,
            )
            for index in range(count)
        )
        return CohortSelection(
            status="available",
            samples=refs,
            identity=stable_json_hash([ref.to_dict() for ref in refs]),
        )

    def materialize_sample(
        self,
        loaded: LoadedModel,
        sample: SampleRef,
    ) -> dict[str, torch.Tensor]:
        del loaded
        assert sample.sample_index is not None
        return self.clone_sample_payload(self.samples[sample.sample_index])

    def clone_sample_payload(
        self,
        payload: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        return {name: value.detach().clone() for name, value in payload.items()}

    def list_module_sites(self, loaded: LoadedModel) -> Sequence[ModuleSite]:
        del loaded
        self.module_site_call_count += 1
        return (
            ModuleSite(
                site_id="normalization-output",
                module_path="normalization",
                module_type="BatchNorm1d",
                node_id="normalization-output",
                stage_id="feature_extractor",
                output_path="$",
            ),
        )

    def forward_objective(
        self,
        loaded: LoadedModel,
        payload: Mapping[str, torch.Tensor],
        *,
        objective: Mapping[str, Any],
        precision: str,
        return_aux: bool,
    ) -> ObjectiveResult:
        del objective, precision, return_aux
        prediction = loaded.model(payload["features"])
        squared_error = torch.square(prediction - payload["label"])
        loss = squared_error if self.vector_objective else squared_error.mean()
        self.objective_call_count += 1
        objective_identity = (
            "alternate_generic_mse"
            if self.alternate_objective_identity
            and self.objective_call_count % 2 == 0
            else "generic_mse"
        )
        return ObjectiveResult(
            total=(
                squared_error.mean()
                if self.vector_objective
                else loss
            ),
            raw_total=squared_error.mean(),
            backward_total=loss,
            backward_cotangent=(
                torch.full_like(loss, 1.0 / loss.numel())
                if self.vector_cotangent
                else None
            ),
            cotangent_identity=(
                "explicit_mean_vjp"
                if self.vector_cotangent
                else None
            ),
            terms={"mse": squared_error.mean()},
            outputs={"prediction": prediction},
            component_responses=(
                ComponentResponse(
                    response_component_id="prediction",
                    metric_id="mean_squared_error",
                    value=squared_error.mean(),
                    support_count=squared_error.numel(),
                    normalization_id="target-scale",
                    higher_is_better=False,
                    metadata={"response_id": "prediction:mse"},
                ),
            ),
            objective_identity=objective_identity,
            objective_metadata={"reduction": "mean"},
        )

    def module_patch_reference(
        self,
        loaded: LoadedModel,
        *,
        site: ModuleSite,
        method: str,
        sample: SampleRef,
        payload: Any,
    ) -> ModulePatchReference:
        del loaded, site, sample, payload
        if method != "mean_patch":
            raise ConditionUnavailable("toy adapter only provides a mean patch")
        return ModulePatchReference(
            replacement=torch.zeros(5),
            provenance={"source": "toy_training_mean"},
        )

    def analyze_example(
        self,
        *,
        loaded: LoadedModel,
        samples: Sequence[SampleRef],
        options: Mapping[str, Any],
    ) -> AnalyzerExecutionResult:
        """验证 public capability binding 传递完整执行上下文。"""

        self.analyzer_calls.append(
            {
                "model_name": loaded.model_name,
                "sample_ids": tuple(sample.sample_id for sample in samples),
                "options": dict(options),
            }
        )
        evidence_key = "feature-0:prediction"
        expected_keys = (
            evidence_key,
            *(
                ("feature-1:prediction",)
                if self.inventory_mode in {"missing", "skipped"}
                else ()
            ),
        )
        rows: list[Mapping[str, Any]] = [
            {
                "evidence_key": evidence_key,
                "record_kind": "example_metric",
                "status": "success",
                "intervened_component_id": "feature-0",
                "response_component_id": "prediction",
                "value": float(options.get("gain", 1.0)) * len(samples),
                "physical_causality_claimed": False,
            }
        ]
        if self.inventory_mode == "skipped":
            rows.append(
                {
                    "evidence_key": "feature-1:prediction",
                    "record_kind": "example_metric",
                    "status": "skipped",
                    "skip_reason": "explicit_test_skip",
                    "intervened_component_id": "feature-1",
                    "response_component_id": "prediction",
                }
            )
        return AnalyzerExecutionResult(
            rows=tuple(rows),
            expected_evidence_keys=expected_keys,
        )

    def list_diagnostic_components(
        self,
        loaded: LoadedModel,
        samples: Sequence[SampleRef],
    ) -> Sequence[DiagnosticComponent]:
        """声明普通 MLP 的输入 feature 与输出 prediction 身份。"""

        del loaded, samples
        return (
            DiagnosticComponent(
                component_id="feature-0",
                semantic_id="feature-0",
                component_kind="input",
                display_label="输入特征 0",
                tensor_path="features",
                tensor_axis=1,
                tensor_index=0,
            ),
            DiagnosticComponent(
                component_id="feature-1",
                semantic_id="feature-1",
                component_kind="input",
                display_label="输入特征 1",
                tensor_path="features",
                tensor_axis=1,
                tensor_index=1,
            ),
            DiagnosticComponent(
                component_id="prediction",
                semantic_id="prediction",
                component_kind="output",
                display_label="预测值",
                tensor_path="prediction",
                tensor_axis=1,
                tensor_index=0,
                normalization_id="target-scale",
            ),
        )

    def close_loaded_model(self, loaded: LoadedModel) -> None:
        self.closed_states.append(
            {
                name: value.detach().cpu().clone()
                for name, value in loaded.model.state_dict().items()
            }
        )

    def barrier(self) -> None:
        return None


def _write_arbitrary_checkpoint(run_dir: Path) -> tuple[Path, dict[str, torch.Tensor]]:
    torch.manual_seed(71)
    model = _ToyModel()
    state = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }
    path = run_dir / "weights" / "candidate.payload"
    path.parent.mkdir(parents=True)
    torch.save(state, path)
    return path, state


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _runtime_descriptor(
    *,
    task_version: str = "1",
    objective_version: str = "1",
) -> CheckpointRuntimeDescriptor:
    """构造与 toy task/runtime 实现绑定的显式 checkpoint runtime identity。"""

    return CheckpointRuntimeDescriptor.create(
        task_definition_id="portable.generic_regression",
        task_definition_version=task_version,
        objective_executor_version=objective_version,
        capability_descriptors=(
            {
                "name": "checkpoint_model_loader",
                "definition_version": 1,
            },
            {
                "name": "sample_objective_executor",
                "definition_version": 1,
            },
        ),
    )


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_base_cold_import_runs_while_host_and_extensions_are_blocked() -> None:
    code = """
import builtins
import sys

real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == "src" or name.startswith("src."):
        raise AssertionError(name)
    if name == "model_diagnostics.extensions" or name.startswith(
        "model_diagnostics.extensions."
    ):
        raise AssertionError(name)
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded
import model_diagnostics.base as base
import model_diagnostics.base.runtime as runtime
definition = base.AnalyzerDefinition(
    name="cold_import_example",
    evidence_kind="example",
    execution_mode="aggregate",
)
assert definition.name == "cold_import_example"
assert "Controller" not in runtime.__all__
assert "setup_training_diagnostics" not in runtime.__all__
assert "ModuleDiagnosticsPlan" not in runtime.__all__
assert not any(name == "src" or name.startswith("src.") for name in sys.modules)
assert not any(
    name == "model_diagnostics.extensions"
    or name.startswith("model_diagnostics.extensions.")
    for name in sys.modules
)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_checkpoint_stages_are_mutually_exclusive() -> None:
    sweep = DiagnosticsRecipe.checkpoint_sweep(
        analyzer_catalog=_ToyAdapter.analyzer_catalog,
    )
    final = DiagnosticsRecipe.final_selected(
        "model.pt",
        analyzer_catalog=_ToyAdapter.analyzer_catalog,
    )

    assert sweep.analyzers == ("checkpoint_sweep",)
    assert final.analyzers == (
        "final_module_influence",
        "example_analyzer",
    )
    assert sweep.to_dict()["stage"] == "checkpoint_sweep"
    assert sweep.to_dict() == DiagnosticsRecipe.checkpoint_sweep(
        analyzer_catalog=_ToyAdapter.analyzer_catalog,
    ).to_dict()


def test_base_checkpoint_runs_identity_and_scale_conditions_on_arbitrary_file(
    tmp_path: Path,
) -> None:
    checkpoint_path, initial_state = _write_arbitrary_checkpoint(tmp_path)
    adapter = _ToyAdapter(checkpoint_path)
    recipe = DiagnosticsRecipe.final_selected(
        checkpoint_path,
        analyzer_catalog=adapter.analyzer_catalog,
    )

    output = run_checkpoint_diagnostics(
        adapter=adapter,
        run_dir=tmp_path,
        runtime_descriptor=_runtime_descriptor(),
        recipe=recipe,
    )
    validation = validate_checkpoint_analysis(output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    cohort = json.loads((output / "cohort.json").read_text(encoding="utf-8"))
    component_catalog = json.loads(
        (output / "component_catalog.json").read_text(encoding="utf-8")
    )
    root_index = json.loads(
        (tmp_path / "diagnostics" / "v2" / "index.json").read_text(
            encoding="utf-8"
        )
    )
    rows = _jsonl(output / "analyzers" / "_base" / "streams" / "conditions.jsonl")
    response_rows = _jsonl(
        output / "analyzers" / "_base" / "streams" / "module_effects.jsonl"
    )

    assert validation.valid is True
    assert validation.status == "success"
    assert manifest["format_version"] == CHECKPOINT_FORMAT_VERSION
    assert root_index["format_version"] == "diagnostics_v2"
    assert manifest["runtime_descriptor_digest"] == (
        _runtime_descriptor().runtime_descriptor_digest
    )
    assert manifest["task_definition_id"] == "portable.generic_regression"
    assert manifest["task_definition_version"] == "1"
    assert manifest["objective_executor_version"] == "1"
    assert manifest["capability_descriptors"] == [
        {
            "definition_version": 1,
            "name": "checkpoint_model_loader",
        },
        {
            "definition_version": 1,
            "name": "sample_objective_executor",
        },
    ]
    assert manifest["status"] == "complete"
    assert "component_catalog" not in cohort["metadata"]
    assert cohort["metadata"]["component_catalog_path"] == (
        "component_catalog.json"
    )
    assert manifest["component_catalog_digest"] == (
        cohort["metadata"]["component_catalog_digest"]
    )
    assert cohort["metadata"]["hierarchy_discovery_status"] == (
        "insufficient_observations"
    )
    assert len(cohort["metadata"]["hierarchy_catalog"]) == 3
    assert {row["display_label"] for row in component_catalog} == {
        "输入特征 0",
        "输入特征 1",
        "预测值",
    }
    assert all(
        "component_shape" in row and "reduction_semantics" in row
        for row in component_catalog
    )
    assert manifest["checkpoints"][0]["path"] == str(checkpoint_path.resolve())
    assert {row["method"] for row in rows} == {
        "identity",
        "output_scale",
        "mean_patch",
    }
    assert all(
        row["identity_control_exact"] is True
        for row in rows
        if row["method"] == "identity"
        or row["method"] == "output_scale" and row["scale"] == 1.0
    )
    module_response_rows = [
        row
        for row in response_rows
        if row.get("record_kind") == "module_output_influence"
    ]
    assert module_response_rows
    assert {
        row["response_component_id"] for row in module_response_rows
    } == {"prediction"}
    assert all(
        "effect_value" in row
        and "normalized_effect" in row
        and "degradation_effect" not in row
        for row in module_response_rows
    )
    # 每个样本执行一次 Taylor baseline、一次 response baseline 和三个干预。
    assert adapter.objective_call_count == 2 * (2 + 3)
    assert all(
        row["predictive_dependence_only"] is True
        and row["physical_causality_claimed"] is False
        for row in module_response_rows
    )
    assert adapter.closed_states
    assert all(
        torch.equal(adapter.closed_states[-1][name], value)
        for name, value in initial_state.items()
    )
    report_dir = generate_diagnostics_report(tmp_path)
    report_summary = json.loads(
        (report_dir / "summary.json").read_text(encoding="utf-8")
    )
    assert report_summary["hierarchy_node_count"] >= 3
    dashboard = (
        tmp_path
        / "diagnostics"
        / "v2"
        / "diagnostics-report.html"
    ).read_text(encoding="utf-8")
    figure = next(
        item
        for item in report_summary["evidence_figures"]
        if item["evidence_kind"]
        == "checkpoint_conditioned_activation_intervention"
    )
    assert figure["renderer_kind"] == "configured"
    assert figure["null_control_status"] == "passed"
    assert {
        metric["path"] for metric in figure["metrics"]
    } >= {"effect_value", "objective_normalized_local_taylor"}
    assert (report_dir / figure["figure_path"]).is_file()
    assert "输入特征 0" in dashboard
    assert "预测值" in dashboard
    assert "目标 1" not in dashboard
    assert 'coordinate:"checkpoint_update"' in dashboard
    assert (
        'temporalCondition=!["update","checkpoint_update"].includes('
        "coordinateName)"
        in dashboard
    )
    assert (
        "temporalSeriesIdentity("
        "item.row,guide.category,componentAliases,xName)"
        in dashboard
    )
    assert "至少需要 2 个不同的" not in dashboard
    assert 'kind:"objective_pair"' not in dashboard
    assert 'id="zoom-y-scale"' in dashboard
    assert 'data-role="y-scale"' not in dashboard
    assert "ZOOM_Y_SCALE.onchange" in dashboard
    assert 'id="run"' not in dashboard
    assert "只展示原始证据、对照、有效观测和解释边界" not in dashboard
    assert "0 与负值保留" in dashboard
    assert "niceLinearTicks" in dashboard
    assert "主刻度固定为 10 的整数次幂" in dashboard
    assert "hasSiteIdentity=rows.some" in dashboard
    assert "!hasOutputIdentity&&!hasSiteIdentity" in dashboard
    assert 'class="chart-viewport"' in dashboard
    assert 'kind:"objective_overview"' in dashboard
    assert 'kind:"rollout_overview"' in dashboard
    assert 'kind:"input_sensitivity"' in dashboard
    assert "扰动幅度—相对响应曲线" in dashboard
    assert 'kind:"rollout_heatmap"' in dashboard
    assert 'kind:"rollout_coverage"' in dashboard
    assert "预测 ${horizon} 步" in dashboard
    assert "汇总误差指标" in dashboard
    assert "滚动样本覆盖与缺失原因" in dashboard
    assert "有限记录 / 全部记录" in dashboard
    assert "有效输出元素" in dashboard
    assert "改动元素（实际/计划）" in dashboard
    assert "未干预路径核对" in dashboard
    assert "共同活跃参数比例" in dashboard


def test_runtime_metric_catalog_excludes_redundant_processed_microbatch_count() -> None:
    paths = {
        guide["path"]
        for guide in build_metric_guide_catalog(
            BASE_EVIDENCE_RENDERER_CATALOG
        )
    }

    assert "rejected_microbatch_count" in paths
    assert "processed_microbatch_count" not in paths
    assert "grad_abs_max" in paths
    assert "no_grad_parameter_tensor_count" in paths


def test_objective_identity_must_describe_the_actual_backward_objective() -> None:
    with pytest.raises(ValueError, match="objective_identity"):
        ObjectiveResult(
            total=torch.tensor(1.0),
            objective_identity="unspecified",
        )


def test_activation_capture_keeps_repeated_invocation_and_output_path_identity() -> None:
    class Split(torch.nn.Module):
        def forward(self, value: torch.Tensor):
            return value + 1.0, {"residual": value - 1.0}

    module = Split()
    site = ModuleSite(
        site_id="split",
        node_id="split",
        module_path="split",
        module_type="Split",
    )
    with ModuleActivationCapture(
        module,
        site,
        capture_gradients=False,
    ) as capture:
        module(torch.tensor([1.0]))
        module(torch.tensor([2.0]))

    assert set(capture.samples) == {
        (0, "$[0]"),
        (0, "$[1].residual"),
        (1, "$[0]"),
        (1, "$[1].residual"),
    }
    identities = {
        sample.activation_site["activation_site_id"]
        for sample in capture.samples.values()
    }
    assert len(identities) == 4


def test_activation_capture_preserves_unused_output_as_no_gradient() -> None:
    class Split(torch.nn.Module):
        def forward(self, value: torch.Tensor):
            return value.square(), value.sin()

    module = Split()
    site = ModuleSite(
        site_id="split",
        node_id="split",
        module_path="split",
        module_type="Split",
    )
    with ModuleActivationCapture(
        module,
        site,
        capture_gradients=True,
    ) as capture:
        used, _unused = module(torch.tensor([1.0], requires_grad=True))
        used.sum().backward()

    assert capture.samples[(0, "$[0]")].gradient is not None
    assert capture.samples[(0, "$[1]")].gradient is None


@pytest.mark.parametrize(
    ("failure_mode", "expected_skip_reason"),
    (
        ("contract_mismatch", "objective_contract_mismatch"),
        (
            "vector_objective",
            "non_scalar_backward_objective_requires_explicit_cotangent",
        ),
    ),
)
def test_checkpoint_branch_contract_failures_are_structured_skips(
    tmp_path: Path,
    failure_mode: str,
    expected_skip_reason: str,
) -> None:
    checkpoint_path, _state = _write_arbitrary_checkpoint(tmp_path)
    adapter = _ToyAdapter(checkpoint_path)
    adapter.alternate_objective_identity = failure_mode == "contract_mismatch"
    adapter.vector_objective = failure_mode == "vector_objective"
    recipe = DiagnosticsRecipe.final_selected(
        checkpoint_path,
        analyzer_catalog=adapter.analyzer_catalog,
    )

    output = run_checkpoint_diagnostics(
        adapter=adapter,
        run_dir=tmp_path,
        runtime_descriptor=_runtime_descriptor(),
        recipe=recipe,
    )
    rows = _jsonl(
        output / "analyzers" / "_base" / "streams" / "conditions.jsonl"
    )

    assert any(
        row.get("status") == "skipped"
        and row.get("skip_reason") == expected_skip_reason
        for row in rows
    )
    assert all(
        "objective_delta" not in row
        for row in rows
        if row.get("skip_reason") == expected_skip_reason
    )


def test_checkpoint_branch_accepts_explicit_vector_vjp(
    tmp_path: Path,
) -> None:
    checkpoint_path, _state = _write_arbitrary_checkpoint(tmp_path)
    adapter = _ToyAdapter(checkpoint_path)
    adapter.vector_objective = True
    adapter.vector_cotangent = True
    recipe = DiagnosticsRecipe.final_selected(
        checkpoint_path,
        analyzer_catalog=adapter.analyzer_catalog,
    )

    output = run_checkpoint_diagnostics(
        adapter=adapter,
        run_dir=tmp_path,
        runtime_descriptor=_runtime_descriptor(),
        recipe=recipe,
    )
    rows = _jsonl(
        output / "analyzers" / "_base" / "streams" / "conditions.jsonl"
    )

    assert any(row.get("status") == "success" for row in rows)
    assert not any(
        row.get("skip_reason")
        == "non_scalar_backward_objective_requires_explicit_cotangent"
        for row in rows
    )


def test_runtime_descriptor_changes_analysis_identity_and_is_validated(
    tmp_path: Path,
) -> None:
    """task/objective runtime identity 漂移必须创建新 analysis，而非错误 resume。"""

    checkpoint_path, _state = _write_arbitrary_checkpoint(tmp_path)
    adapter = _ToyAdapter(checkpoint_path)
    recipe = DiagnosticsRecipe.final_selected(
        checkpoint_path,
        analyzer_catalog=adapter.analyzer_catalog,
    )

    first = run_checkpoint_diagnostics(
        adapter=adapter,
        run_dir=tmp_path,
        runtime_descriptor=_runtime_descriptor(objective_version="1"),
        recipe=recipe,
    )
    second = run_checkpoint_diagnostics(
        adapter=adapter,
        run_dir=tmp_path,
        runtime_descriptor=_runtime_descriptor(objective_version="2"),
        recipe=recipe,
    )

    assert first != second
    assert validate_checkpoint_analysis(first).valid is True
    assert validate_checkpoint_analysis(second).valid is True

    original_manifest = json.loads(
        (second / "manifest.json").read_text(encoding="utf-8")
    )
    tampered_manifest = dict(original_manifest)
    tampered_manifest["objective_executor_version"] = "3"
    _write_json(second / "manifest.json", tampered_manifest)
    tampered = validate_checkpoint_analysis(second)

    assert tampered.valid is False
    assert {
        issue.code for issue in tampered.issues
    } >= {"invalid_runtime_descriptor"}

    missing_field_manifest = dict(original_manifest)
    missing_field_manifest.pop("task_definition_id")
    _write_json(second / "manifest.json", missing_field_manifest)
    missing_field = validate_checkpoint_analysis(second)

    assert missing_field.valid is False
    assert {
        issue.code for issue in missing_field.issues
    } >= {"missing_runtime_descriptor_fields"}


def test_noncurrent_checkpoint_format_fails_closed_before_writer_setup(
    tmp_path: Path,
) -> None:
    """非当前 format 既不能通过 validator，也不能进入 writer 初始化。"""

    analysis = tmp_path / "analysis"
    analysis.mkdir()
    _write_json(
        analysis / "manifest.json",
        {
            "analysis_id": analysis.name,
            "format_version": "unsupported",
            "status": "complete",
        },
    )

    validation = validate_checkpoint_analysis(analysis)
    assert validation.valid is False
    assert validation.status == "corrupt"
    assert {
        issue.code for issue in validation.issues
    } >= {"format_version_mismatch"}
    with pytest.raises(
        RuntimeError,
        match="unsupported checkpoint format version",
    ):
        DiagnosticsArtifactStore(analysis, rank=0, world_size=1)
    assert not (analysis / "commits").exists()
    assert not (analysis / "analyzers").exists()

    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    _write_json(
        incomplete / "run_state.json",
        {
            "analysis_id": incomplete.name,
            "format_version": "unsupported",
            "status": "running",
        },
    )
    incomplete_validation = validate_checkpoint_analysis(incomplete)
    assert incomplete_validation.valid is False
    assert incomplete_validation.status == "corrupt"
    assert {
        issue.code for issue in incomplete_validation.issues
    } >= {"format_version_mismatch"}


def test_custom_analyzer_descriptor_drives_artifact_validator_and_report(
    tmp_path: Path,
) -> None:
    checkpoint_path, _state = _write_arbitrary_checkpoint(tmp_path)
    adapter = _ToyAdapter(checkpoint_path)
    recipe = DiagnosticsRecipe.final_selected(
        checkpoint_path,
        analyzer_catalog=adapter.analyzer_catalog,
    )

    output = run_checkpoint_diagnostics(
        adapter=adapter,
        run_dir=tmp_path,
        runtime_descriptor=_runtime_descriptor(),
        recipe=recipe,
    )
    validation = validate_checkpoint_analysis(output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    records = _jsonl(
        output
        / "analyzers"
        / "example_analyzer"
        / "streams"
        / "records.jsonl"
    )
    report_dir = generate_diagnostics_report(tmp_path)
    report_summary = json.loads(
        (report_dir / "summary.json").read_text(encoding="utf-8")
    )

    descriptor = next(
        item
        for item in manifest["analyzer_definitions"]
        if item["name"] == "example_analyzer"
    )
    assert descriptor == _EXAMPLE_ANALYZER.to_manifest()
    assert records[0]["value"] == pytest.approx(2.0)
    assert validation.valid is True
    assert report_summary["status"] == "success"
    checkpoint_summary = report_summary["checkpoint_analyses"][0]
    example_evidence = next(
        item
        for item in checkpoint_summary["analyzer_evidence"]
        if item["name"] == "example_analyzer"
    )
    assert example_evidence["claim_boundaries"] == [
        "example evidence is descriptive only"
    ]
    figure = next(
        item
        for item in report_summary["evidence_figures"]
        if item["analyzer_name"] == "example_analyzer"
    )
    assert figure["evidence_kind"] == "example_scalar_evidence"
    assert figure["renderer_kind"] == "generic"
    assert figure["claim_boundaries"] == [
        "example evidence is descriptive only"
    ]
    assert next(
        metric
        for metric in figure["metrics"]
        if metric["path"] == "value"
    )["mean"] == pytest.approx(2.0)
    assert (report_dir / figure["figure_path"]).is_file()
    assert adapter.module_site_call_count > 0
    assert adapter.analyzer_calls == [
        {
            "model_name": "generic_mlp",
            "sample_ids": ("sample-0", "sample-1"),
            "options": {},
        }
    ]


@pytest.mark.parametrize(
    ("mode", "skipped_count", "missing_count"),
    (("skipped", 1, 0), ("missing", 0, 1)),
)
def test_aggregate_analyzer_inventory_distinguishes_skip_from_missing(
    tmp_path: Path,
    mode: str,
    skipped_count: int,
    missing_count: int,
) -> None:
    """expected inventory 必须区分合法跳过与 producer 缺失。"""

    run_dir = tmp_path / mode
    checkpoint_path, _state = _write_arbitrary_checkpoint(run_dir)
    adapter = _ToyAdapter(checkpoint_path)
    adapter.inventory_mode = mode
    recipe = DiagnosticsRecipe.final_selected(
        checkpoint_path,
        analyzer_catalog=adapter.analyzer_catalog,
    )

    output = run_checkpoint_diagnostics(
        adapter=adapter,
        run_dir=run_dir,
        runtime_descriptor=_runtime_descriptor(),
        recipe=recipe,
    )
    validation = validate_checkpoint_analysis(output)
    manifest = json.loads(
        (output / "manifest.json").read_text(encoding="utf-8")
    )
    inventory = manifest["analyzer_evidence_inventories"][
        "example_analyzer"
    ][0]

    assert validation.valid is True
    assert validation.status == "degraded"
    assert inventory["expected_evidence_count"] == 2
    assert inventory["observed_evidence_count"] == 1
    assert inventory["skipped_evidence_count"] == skipped_count
    assert inventory["missing_evidence_count"] == missing_count


def test_unbound_aggregate_analyzer_fails_closed_as_structured_skip(
    tmp_path: Path,
) -> None:
    """descriptor 存在但 runner 未绑定时不得回退名称分发或伪造成功。"""

    checkpoint_path, _state = _write_arbitrary_checkpoint(tmp_path)
    adapter = _ToyAdapter(checkpoint_path)
    adapter.analyzer_bindings = AnalyzerBindingCatalog({})
    recipe = DiagnosticsRecipe.final_selected(
        checkpoint_path,
        analyzer_catalog=adapter.analyzer_catalog,
    )

    output = run_checkpoint_diagnostics(
        adapter=adapter,
        run_dir=tmp_path,
        runtime_descriptor=_runtime_descriptor(),
        recipe=recipe,
    )
    records = _jsonl(
        output
        / "analyzers"
        / "example_analyzer"
        / "streams"
        / "records.jsonl"
    )

    assert records[0]["status"] == "skipped"
    assert records[0]["skip_reason"] == (
        "analyzer_binding_unavailable:example_analyzer"
    )


def test_capability_binding_fails_closed_when_adapter_capability_is_missing(
    tmp_path: Path,
) -> None:
    """binding 存在但 adapter capability 缺失时必须产生 structured skip。"""

    checkpoint_path, _state = _write_arbitrary_checkpoint(tmp_path)
    adapter = _ToyAdapter(checkpoint_path)
    adapter.analyzer_bindings = AnalyzerBindingCatalog(
        {
            _EXAMPLE_ANALYZER.name: bind_adapter_capability(
                _EXAMPLE_ANALYZER,
                "missing_example_capability",
            )
        }
    )
    recipe = DiagnosticsRecipe.final_selected(
        checkpoint_path,
        analyzer_catalog=adapter.analyzer_catalog,
    )

    output = run_checkpoint_diagnostics(
        adapter=adapter,
        run_dir=tmp_path,
        runtime_descriptor=_runtime_descriptor(),
        recipe=recipe,
    )
    records = _jsonl(
        output
        / "analyzers"
        / "example_analyzer"
        / "streams"
        / "records.jsonl"
    )

    assert records[0]["status"] == "skipped"
    assert records[0]["skip_reason"] == (
        "analyzer 'example_analyzer' requires adapter capability "
        "'missing_example_capability'"
    )
