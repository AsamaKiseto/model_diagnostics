"""调度任务无关的 checkpoint branch、module intervention 与 analyzer runner。"""

from __future__ import annotations

from collections import Counter
from contextlib import ExitStack
import copy
from dataclasses import dataclass
import gc
from itertools import chain
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch

from ..artifacts.checkpoint import DiagnosticsArtifactStore
from ..artifacts.checkpoint_identity import (
    build_checkpoint_analysis_identity_payload,
)
from ..artifacts.common import stable_json_hash, update_v2_index
from ..hierarchy import (
    HierarchyDiscoveryPolicy,
    HierarchyDiscoveryResult,
    discover_model_hierarchy,
)
from ..interventions import (
    InterventionSpec,
    ModuleOutputIntervention,
    UnsupportedModuleOutput,
    preserve_module_modes,
    resolve_module,
)
from ..measurements.checkpoint import (
    GradientSnapshot,
    ParameterSnapshot,
    aggregate_gradients,
    hierarchy_nodes,
    parameter_delta_metrics,
    scalar_terms,
)
from ..registry import (
    AnalyzerBinding,
    AnalyzerBindingCatalog,
    AnalyzerCapabilityUnavailable,
    AnalyzerCatalog,
    AnalyzerExecutionResult,
    AnalyzerRunContext,
)
from .activation_capture import (
    ActivationCaptureSample,
    ModuleActivationCapture,
)
from .branch_state import (
    capture_canonical_model_state,
    paired_branch_rng,
    restore_canonical_model_state,
)
from .contracts import (
    CheckpointDiagnosticsAdapter,
    CheckpointRef,
    CheckpointRuntimeDescriptor,
    CohortSelection,
    ConditionUnavailable,
    DiagnosticComponent,
    DiagnosticsRecipe,
    FORMAT_VERSION,
    LoadedModel,
    ModulePatchReference,
    ModuleSite,
    ObjectiveResult,
    SampleRef,
)


_TERMINAL_STATUSES = {
    "success",
    "failed",
    "skipped",
    "insufficient_evidence",
}
_ANALYZER_ENGINE_FIELDS = {
    "analyzer",
    "analyzer_record_id",
    "condition_id",
    "checkpoint_identity",
    "checkpoint_path",
    "checkpoint_update",
    "run_id",
    "recipe_hash",
}


@dataclass(frozen=True, slots=True)
class _AnalysisPlan:
    condition_id: str
    analyzer: str
    checkpoint: CheckpointRef
    sample: SampleRef | None = None
    site: ModuleSite | None = None
    method: str | None = None
    scale: float = 1.0
    constant: float | None = None


@dataclass(slots=True)
class _BranchResult:
    status: str
    skip_reason: str | None
    loss: float | None
    terms: dict[str, float | None]
    outputs: dict[str, torch.Tensor]
    component_responses: dict[tuple[str, str], dict[str, Any]]
    gradient: GradientSnapshot | None
    activation_samples: dict[tuple[str, int, str], ActivationCaptureSample]
    local_taylor_rows: list[dict[str, Any]]
    parameter_delta_rows: list[dict[str, Any]]
    objective_identity: str
    objective_contract_digest: str
    objective_metadata: dict[str, Any]
    raw_objective: float | None
    backward_objective: float | None
    optimizer_steps: int
    buffer_restore_status: str


@dataclass(frozen=True, slots=True)
class _AdapterExecution:
    """解析最小 adapter 之外的可选 capability，并提供任务无关默认行为。

    分布式身份在分析开始时冻结；device 随每个 loaded model 解析。宿主未提供
    payload clone 或 module-site catalog 时，Base 分别使用 ``deepcopy`` 与通用
    ``nn.Module`` hierarchy discovery，不推断数据 role 或任务语义。
    """

    adapter: CheckpointDiagnosticsAdapter
    rank: int
    world_size: int
    bindings: AnalyzerBindingCatalog
    execution_precision: str

    @classmethod
    def create(
        cls,
        adapter: CheckpointDiagnosticsAdapter,
        *,
        analyzer_catalog: AnalyzerCatalog,
        analyzer_bindings: (
            AnalyzerBindingCatalog | Mapping[str, AnalyzerBinding] | None
        ),
    ) -> "_AdapterExecution":
        raw_bindings = (
            analyzer_bindings
            if analyzer_bindings is not None
            else getattr(adapter, "analyzer_bindings", None)
        )
        if raw_bindings is None:
            bindings = AnalyzerBindingCatalog({})
        elif isinstance(raw_bindings, AnalyzerBindingCatalog):
            bindings = raw_bindings
        elif isinstance(raw_bindings, Mapping):
            bindings = AnalyzerBindingCatalog(dict(raw_bindings))
        else:
            raise TypeError(
                "analyzer_bindings must be an AnalyzerBindingCatalog or mapping"
            )
        bindings.validate_against(analyzer_catalog)
        rank = int(getattr(adapter, "rank", 0))
        world_size = int(getattr(adapter, "world_size", 1))
        if world_size < 1 or rank < 0 or rank >= world_size:
            raise ValueError("adapter rank/world_size are outside valid ranges")
        if world_size > 1 and not callable(getattr(adapter, "barrier", None)):
            raise TypeError(
                "multi-process checkpoint diagnostics require adapter.barrier"
            )
        execution_precision = str(adapter.execution_precision).strip().lower()
        if not execution_precision:
            raise ValueError("adapter.execution_precision must not be empty")
        return cls(
            adapter=adapter,
            rank=rank,
            world_size=world_size,
            bindings=bindings,
            execution_precision=execution_precision,
        )

    def device(self, loaded: LoadedModel) -> torch.device:
        """返回 paired-RNG 使用的设备；未声明时从 model state 推导。"""

        declared = getattr(self.adapter, "device", None)
        if declared is not None:
            return torch.device(declared)
        for value in chain(
            loaded.model.parameters(),
            loaded.model.buffers(),
        ):
            return value.device
        return torch.device("cpu")

    def barrier(self) -> None:
        """执行可选 distributed barrier；单进程 adapter 默认为 no-op。"""

        barrier = getattr(self.adapter, "barrier", None)
        if callable(barrier):
            barrier()

    def clone_payload(self, payload: Any) -> Any:
        """隔离 condition payload；自定义容器可由宿主覆盖默认 deepcopy。"""

        clone = getattr(self.adapter, "clone_sample_payload", None)
        if callable(clone):
            return clone(payload)
        try:
            return copy.deepcopy(payload)
        except (TypeError, RuntimeError) as error:
            raise ConditionUnavailable(
                "sample_payload_requires_clone_capability"
            ) from error

    def _declared_module_sites(
        self,
        loaded: LoadedModel,
    ) -> tuple[ModuleSite, ...]:
        """取得宿主已验证的增量 module sites，并校验唯一身份。"""

        provider = getattr(self.adapter, "list_module_sites", None)
        if callable(provider):
            sites = tuple(provider(loaded))
        else:
            sites = ()
        if any(not isinstance(site, ModuleSite) for site in sites):
            raise TypeError("module site capability must return ModuleSite values")
        site_ids = tuple(site.site_id for site in sites)
        if len(set(site_ids)) != len(site_ids):
            raise ValueError("module site capability returned duplicate site_id values")
        return sites

    def discover_module_hierarchy(
        self,
        loaded: LoadedModel,
        samples: Sequence[SampleRef],
    ) -> HierarchyDiscoveryResult:
        """在 final cohort 的 3–8 次成功 forward 上确认自动 Block frontier。

        当前三阶段契约禁止训练期 module hook，因此调用确认只发生在 final-selected，
        不改变训练 loop。每次观测前后恢复 model state、module mode 与 payload storage；
        失败的 materialization/condition 不推进稳定窗口。宿主声明只作为自动结果的
        增量 override，不会关闭通用发现。
        """

        overrides = self._declared_module_sites(loaded)
        policy = HierarchyDiscoveryPolicy()
        static = discover_model_hierarchy(
            loaded.model,
            model_name=loaded.model_name,
            overrides=overrides,
            policy=policy,
        )
        if len(samples) < policy.discovery_min_observations:
            return HierarchyDiscoveryResult(
                sites=overrides,
                coverage_gaps=(
                    {
                        "reason": "insufficient_discovery_samples",
                        "required": policy.discovery_min_observations,
                        "available": len(samples),
                    },
                ),
                candidate_paths=static.candidate_paths,
                observed_module_paths=(),
                discovery_status="insufficient_observations",
            )

        modules = dict(loaded.model.named_modules())
        observed_paths: set[str] = set()
        current_paths: set[str] = set()
        handles: list[Any] = []

        def observe(path: str):
            """为每个候选建立独立闭包，避免循环变量晚绑定。"""

            def hook(_module: torch.nn.Module, _inputs: Any, _output: Any) -> None:
                current_paths.add(path)

            return hook

        for path in static.candidate_paths:
            module = modules.get(path)
            if module is not None:
                handles.append(module.register_forward_hook(observe(path)))

        model_state = capture_canonical_model_state(loaded.model)
        successful_observations = 0
        stable_observations = 0
        previous_frontier: tuple[str, ...] | None = None
        result = static
        try:
            for sample in samples[: policy.discovery_max_observations]:
                restore_canonical_model_state(loaded.model, model_state)
                current_paths.clear()
                try:
                    payload = self.adapter.materialize_sample(loaded, sample)
                    branch_payload = self.clone_payload(payload)
                    with preserve_module_modes(loaded.model), paired_branch_rng(
                        int(
                            stable_json_hash(
                                {
                                    "hierarchy_discovery": sample.sample_id,
                                    "model": loaded.model_name,
                                }
                            )[:16],
                            16,
                        ),
                        self.device(loaded),
                    ):
                        loaded.model.zero_grad(set_to_none=True)
                        self.adapter.forward_objective(
                            loaded,
                            branch_payload,
                            objective={},
                            precision=self.execution_precision,
                            return_aux=False,
                        )
                except ConditionUnavailable:
                    continue
                successful_observations += 1
                observed_paths.update(current_paths)
                result = discover_model_hierarchy(
                    loaded.model,
                    model_name=loaded.model_name,
                    observed_module_paths=observed_paths,
                    overrides=overrides,
                    policy=policy,
                )
                frontier = tuple(site.site_id for site in result.sites)
                stable_observations = (
                    stable_observations + 1
                    if frontier == previous_frontier
                    else 1
                )
                previous_frontier = frontier
                if (
                    successful_observations >= policy.discovery_min_observations
                    and stable_observations >= policy.stability_patience
                ):
                    break
        finally:
            for handle in handles:
                handle.remove()
            restore_canonical_model_state(loaded.model, model_state)
            loaded.model.zero_grad(set_to_none=True)

        if successful_observations < policy.discovery_min_observations:
            return HierarchyDiscoveryResult(
                sites=overrides,
                coverage_gaps=(
                    *result.coverage_gaps,
                    {
                        "reason": "insufficient_successful_observations",
                        "required": policy.discovery_min_observations,
                        "observed": successful_observations,
                    },
                ),
                candidate_paths=static.candidate_paths,
                observed_module_paths=tuple(sorted(observed_paths)),
                discovery_status="insufficient_observations",
            )
        traced_sites = tuple(
            ModuleSite(
                site_id=site.site_id,
                node_id=site.node_id,
                module_path=site.module_path,
                module_type=site.module_type,
                parent_node_id=site.parent_node_id,
                stage_id=site.stage_id,
                invocation_index=site.invocation_index,
                output_path=site.output_path,
                alias_node_ids=site.alias_node_ids,
                metadata={
                    **dict(site.metadata),
                    "discovery_successful_observations": successful_observations,
                    "discovery_stable_observations": stable_observations,
                },
            )
            for site in result.sites
        )
        return HierarchyDiscoveryResult(
            sites=traced_sites,
            coverage_gaps=result.coverage_gaps,
            candidate_paths=static.candidate_paths,
            observed_module_paths=tuple(sorted(observed_paths)),
            discovery_status="runtime_confirmed",
        )

    def list_diagnostic_components(
        self,
        loaded: LoadedModel,
        samples: Sequence[SampleRef],
    ) -> tuple[DiagnosticComponent, ...]:
        """取得宿主显式 component catalog；未声明时返回空 catalog。"""

        provider = getattr(
            self.adapter,
            "list_diagnostic_components",
            None,
        )
        if not callable(provider):
            return ()
        components = tuple(provider(loaded, tuple(samples)))
        if any(
            not isinstance(component, DiagnosticComponent)
            for component in components
        ):
            raise TypeError(
                "component catalog capability must return "
                "DiagnosticComponent values"
            )
        component_ids = tuple(
            component.component_id for component in components
        )
        if len(set(component_ids)) != len(component_ids):
            raise ValueError(
                "component catalog returned duplicate component_id values"
            )
        return tuple(
            sorted(
                components,
                key=lambda component: (
                    component.component_kind,
                    component.component_id,
                ),
            )
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
        """调用 task-owned replacement capability；Base 不猜测 mean 或 donor。"""

        provider = getattr(self.adapter, "module_patch_reference", None)
        if not callable(provider):
            raise ConditionUnavailable(
                f"{method}_requires_module_patch_capability"
            )
        replacement = provider(
            loaded,
            site=site,
            method=method,
            sample=sample,
            payload=payload,
        )
        if not isinstance(replacement, ModulePatchReference):
            raise TypeError(
                "module_patch_reference must return ModulePatchReference"
            )
        return replacement

    def run_analyzer(self, context: AnalyzerRunContext) -> Any:
        """运行显式 binding；未绑定 aggregate analyzer 立即 fail closed。"""

        binding = self.bindings.get(context.definition.name)
        if binding is None:
            raise ConditionUnavailable(
                f"analyzer_binding_unavailable:{context.definition.name}"
            )
        return binding.run(context)


def run_checkpoint_diagnostics(
    *,
    adapter: CheckpointDiagnosticsAdapter,
    run_dir: str | Path,
    runtime_descriptor: CheckpointRuntimeDescriptor | Mapping[str, Any],
    recipe: DiagnosticsRecipe,
    analyzer_bindings: (
        AnalyzerBindingCatalog | Mapping[str, AnalyzerBinding] | None
    ) = None,
) -> Path:
    """执行 descriptor-driven checkpoint 分析，不解释宿主 selector 或任务字段。

    ``runtime_descriptor`` 必须由调用方显式给出，并参与 v4 analysis identity。
    ``analyzer_bindings`` 是无副作用 runner catalog；显式参数优先于 adapter 的同名
    catalog。未提供 binding 的 aggregate analyzer fail closed，condition analyzer 仍由
    Base 自己执行。
    """

    resolved_runtime_descriptor = (
        runtime_descriptor
        if isinstance(runtime_descriptor, CheckpointRuntimeDescriptor)
        else CheckpointRuntimeDescriptor.from_mapping(runtime_descriptor)
    )
    catalog = adapter.analyzer_catalog
    if not isinstance(catalog, AnalyzerCatalog):
        raise TypeError("adapter.analyzer_catalog must be an AnalyzerCatalog")
    execution = _AdapterExecution.create(
        adapter,
        analyzer_catalog=catalog,
        analyzer_bindings=analyzer_bindings,
    )
    if not isinstance(recipe, DiagnosticsRecipe):
        raise TypeError(
            "recipe must be created by DiagnosticsRecipe.checkpoint_sweep() "
            "or DiagnosticsRecipe.final_selected()"
        )
    resolved_recipe = recipe
    catalog.validate(
        resolved_recipe.analyzers,
        {},
    )
    adapter.configure(resolved_recipe)
    run_path = Path(run_dir).resolve()
    checkpoints = tuple(
        adapter.resolve_checkpoints(
            run_dir=run_path,
        )
    )
    if not checkpoints:
        raise FileNotFoundError("adapter selected no checkpoints")
    if any(not isinstance(item, CheckpointRef) for item in checkpoints):
        raise TypeError("adapter.resolve_checkpoints must return CheckpointRef values")
    identities = [item.identity for item in checkpoints]
    if len(set(identities)) != len(identities):
        raise ValueError("adapter selected duplicate checkpoint identities")
    if resolved_recipe.stage == "final_selected" and len(checkpoints) != 1:
        raise ValueError(
            "final_selected diagnostics require exactly one checkpoint"
        )

    first_load_started = time.perf_counter()
    first_loaded = adapter.load_model(
        checkpoints[0].path,
        precision=execution.execution_precision,
    )
    first_load_seconds = time.perf_counter() - first_load_started
    _validate_loaded_model(first_loaded, checkpoints[0])
    store: DiagnosticsArtifactStore | None = None
    analysis_path: Path | None = None
    analysis_id: str | None = None
    recipe_hash: str | None = None
    cohort_hash: str | None = None
    try:
        cohort = adapter.build_cohort(
            first_loaded,
            checkpoints=checkpoints,
        )
        components = execution.list_diagnostic_components(
            first_loaded,
            cohort.samples,
        )
        cohort = _bind_component_catalog(cohort, components)
        hierarchy = (
            execution.discover_module_hierarchy(
                first_loaded,
                cohort.samples,
            )
            if "final_module_influence" in resolved_recipe.analyzers
            and cohort.status == "available"
            and cohort.samples
            else HierarchyDiscoveryResult(
                sites=(),
                coverage_gaps=(),
                candidate_paths=(),
                observed_module_paths=(),
                discovery_status="not_requested",
            )
        )
        first_sites = hierarchy.sites
        if resolved_recipe.stage == "final_selected":
            cohort = _bind_hierarchy_catalog(
                cohort,
                model_name=first_loaded.model_name,
                hierarchy=hierarchy,
            )
        descriptors = catalog.selected_descriptors(resolved_recipe.analyzers)
        analysis_id = stable_json_hash(
            build_checkpoint_analysis_identity_payload(
                recipe=resolved_recipe.to_dict(),
                analyzer_definitions=descriptors,
                checkpoint_identities=identities,
                cohort_status=cohort.status,
                cohort_identity=cohort.identity,
                cohort_samples=[
                    sample.to_dict() for sample in cohort.samples
                ],
                runtime_descriptor=resolved_runtime_descriptor.to_dict(),
            )
        )
        analyses_root = (
            run_path / "diagnostics" / "v2" / "checkpoint" / "analyses"
        ).resolve()
        analysis_path = analyses_root / analysis_id
        store = DiagnosticsArtifactStore(
            analysis_path,
            rank=execution.rank,
            world_size=execution.world_size,
        )
        recipe_hash, cohort_hash, completed = store.initialize(
            run_id=run_path.name,
            recipe=resolved_recipe,
            checkpoints=checkpoints,
            cohort=cohort,
            analysis_id=analysis_id,
            runtime_descriptor=resolved_runtime_descriptor,
        )
        if execution.rank == 0:
            update_v2_index(
                run_path,
                "checkpoint_analyses",
                analysis_id,
                {
                    "path": str(analysis_path.relative_to(run_path)),
                    "status": (
                        store.completed_status
                        if store.already_complete
                        else "running"
                    ),
                    "recipe_hash": recipe_hash,
                    "cohort_hash": cohort_hash,
                    "checkpoint_identities": identities,
                    "runtime_descriptor_digest": (
                        resolved_runtime_descriptor.runtime_descriptor_digest
                    ),
                },
            )
        if store.already_complete:
            return analysis_path

        if cohort.status != "available" or not cohort.samples:
            unavailable_id = stable_json_hash(
                {"checkpoint": checkpoints[0].identity, "cohort": "unavailable"}
            )
            plans = ()
            local_plans = ()
            store.register_expected_condition_ids(
                (unavailable_id,) if execution.rank == 0 else ()
            )
        else:
            plans = _build_plans(
                checkpoints,
                cohort.samples,
                first_sites,
                resolved_recipe,
                catalog,
            )
            local_plans = tuple(
                plan
                for index, plan in enumerate(plans)
                if index % execution.world_size == execution.rank
            )
            store.register_expected_condition_ids(
                plan.condition_id for plan in local_plans
            )
        execution.barrier()
        if cohort.status != "available" or not cohort.samples:
            if execution.rank == 0:
                _write_unavailable_cohort(
                    store,
                    checkpoint=checkpoints[0],
                    reason=cohort.skip_reason or "empty_cohort",
                    run_id=run_path.name,
                    recipe_hash=recipe_hash,
                )
        else:
            for checkpoint_index, checkpoint in enumerate(checkpoints):
                if checkpoint_index == 0:
                    loaded = first_loaded
                    checkpoint_load_seconds = first_load_seconds
                else:
                    checkpoint_load_started = time.perf_counter()
                    loaded = adapter.load_model(
                        checkpoint.path,
                        precision=execution.execution_precision,
                    )
                    checkpoint_load_seconds = (
                        time.perf_counter() - checkpoint_load_started
                    )
                try:
                    _validate_loaded_model(loaded, checkpoint)
                    checkpoint_plans = [
                        plan
                        for plan in local_plans
                        if plan.checkpoint.identity == checkpoint.identity
                        and plan.condition_id not in completed
                    ]
                    _execute_checkpoint_plans(
                        execution=execution,
                        loaded=loaded,
                        checkpoint=checkpoint,
                        plans=checkpoint_plans,
                        samples=cohort.samples,
                        recipe=resolved_recipe,
                        recipe_hash=recipe_hash,
                        run_id=run_path.name,
                        store=store,
                        checkpoint_load_seconds=checkpoint_load_seconds,
                    )
                finally:
                    if loaded is not first_loaded:
                        adapter.close_loaded_model(loaded)
                    gc.collect()
                    if (
                        execution.device(loaded).type == "cuda"
                        and torch.cuda.is_available()
                    ):
                        torch.cuda.empty_cache()
        store.flush()
        execution.barrier()
        final_status = store.finalize(
            run_id=run_path.name,
            recipe=resolved_recipe,
            recipe_hash=recipe_hash,
            cohort_hash=cohort_hash,
            checkpoints=checkpoints,
            analysis_id=analysis_id,
            analyzer_catalog=catalog,
            runtime_descriptor=resolved_runtime_descriptor,
        )
        if execution.rank == 0:
            update_v2_index(
                run_path,
                "checkpoint_analyses",
                analysis_id,
                {
                    "path": str(analysis_path.relative_to(run_path)),
                    "status": final_status,
                    "recipe_hash": recipe_hash,
                    "cohort_hash": cohort_hash,
                    "checkpoint_identities": identities,
                    "runtime_descriptor_digest": (
                        resolved_runtime_descriptor.runtime_descriptor_digest
                    ),
                },
            )
        execution.barrier()
        return analysis_path
    except BaseException as error:
        if store is not None:
            store.mark_corrupt(error=error)
        if (
            execution.rank == 0
            and analysis_path is not None
            and analysis_id is not None
        ):
            update_v2_index(
                run_path,
                "checkpoint_analyses",
                analysis_id,
                {
                    "path": str(analysis_path.relative_to(run_path)),
                    "status": "corrupt",
                    "recipe_hash": recipe_hash,
                    "cohort_hash": cohort_hash,
                    "checkpoint_identities": identities,
                    "runtime_descriptor_digest": (
                        resolved_runtime_descriptor.runtime_descriptor_digest
                    ),
                    "failure": f"{type(error).__name__}: {error}",
                },
            )
        raise
    finally:
        adapter.close_loaded_model(first_loaded)


def _bind_component_catalog(
    cohort: CohortSelection,
    components: Sequence[DiagnosticComponent],
) -> CohortSelection:
    """把 component catalog 绑定到已参与 analysis identity 的 cohort 文档。"""

    component_rows = [component.to_dict() for component in components]
    component_catalog_digest = stable_json_hash(component_rows)
    metadata = dict(cohort.metadata)
    reserved = {
        "component_catalog",
        "component_catalog_digest",
        "host_cohort_identity",
    }
    collisions = sorted(reserved & set(metadata))
    if collisions:
        raise ValueError(
            "cohort metadata cannot override Base component catalog fields: "
            f"{collisions}"
        )
    metadata.update(
        {
            "component_catalog": component_rows,
            "component_catalog_digest": component_catalog_digest,
            "host_cohort_identity": cohort.identity,
        }
    )
    return CohortSelection(
        status=cohort.status,
        samples=cohort.samples,
        skip_reason=cohort.skip_reason,
        identity=stable_json_hash(
            {
                "host_cohort_identity": cohort.identity,
                "component_catalog_digest": component_catalog_digest,
            }
        ),
        metadata=metadata,
    )


def _bind_hierarchy_catalog(
    cohort: CohortSelection,
    *,
    model_name: str,
    hierarchy: HierarchyDiscoveryResult,
) -> CohortSelection:
    """把 runtime-confirmed hierarchy 绑定到 cohort identity 与持久化 metadata。

    hierarchy 会改变 condition 计划，因此必须先进入 analysis identity，不能只写一份
    report attachment；这也确保旧的零模块 analysis 不会被错误复用。
    """

    metadata = dict(cohort.metadata)
    reserved = {
        "hierarchy_catalog",
        "hierarchy_catalog_digest",
        "hierarchy_coverage_gaps",
        "hierarchy_discovery_status",
        "hierarchy_candidate_paths",
        "hierarchy_observed_module_paths",
    }
    collisions = sorted(reserved & set(metadata))
    if collisions:
        raise ValueError(
            "cohort metadata cannot override Base hierarchy fields: "
            f"{collisions}"
        )
    stable_model_name = str(model_name).strip()
    hierarchy_rows = [
        {
            "node_id": "all",
            "parent_id": None,
            "hierarchy_level": "All",
            "model_name": stable_model_name,
            "module_path": "",
            "module_type": "All",
            "discovery_source": "base_checkpoint_engine",
        },
        {
            "node_id": f"model:{stable_model_name}",
            "parent_id": "all",
            "hierarchy_level": "Model",
            "model_name": stable_model_name,
            "module_path": "",
            "module_type": "Model",
            "discovery_source": "base_checkpoint_engine",
        },
        *(
            {
                "node_id": site.node_id or site.site_id,
                "parent_id": (
                    site.parent_node_id or f"model:{stable_model_name}"
                ),
                "hierarchy_level": str(
                    site.metadata.get("hierarchy_level", "Block")
                ),
                "model_name": stable_model_name,
                "module_path": site.module_path,
                "module_type": site.module_type,
                "stage_id": site.stage_id,
                "discovery_source": site.metadata.get("discovery_source"),
                "metadata": dict(site.metadata),
            }
            for site in hierarchy.sites
        ),
    ]
    hierarchy_digest = stable_json_hash(hierarchy_rows)
    metadata.update(
        {
            "hierarchy_catalog": hierarchy_rows,
            "hierarchy_catalog_digest": hierarchy_digest,
            "hierarchy_coverage_gaps": [
                dict(gap) for gap in hierarchy.coverage_gaps
            ],
            "hierarchy_discovery_status": hierarchy.discovery_status,
            "hierarchy_candidate_paths": list(hierarchy.candidate_paths),
            "hierarchy_observed_module_paths": list(
                hierarchy.observed_module_paths
            ),
        }
    )
    return CohortSelection(
        status=cohort.status,
        samples=cohort.samples,
        skip_reason=cohort.skip_reason,
        identity=stable_json_hash(
            {
                "component_bound_cohort_identity": cohort.identity,
                "hierarchy_catalog_digest": hierarchy_digest,
                "hierarchy_discovery_status": hierarchy.discovery_status,
            }
        ),
        metadata=metadata,
    )


def _build_plans(
    checkpoints: Sequence[CheckpointRef],
    samples: Sequence[SampleRef],
    sites: Sequence[ModuleSite],
    recipe: DiagnosticsRecipe,
    catalog: AnalyzerCatalog,
) -> tuple[_AnalysisPlan, ...]:
    plans: list[_AnalysisPlan] = []
    for checkpoint in checkpoints:
        for analyzer in recipe.analyzers:
            definition = catalog.definition(analyzer)
            if definition.execution_mode in {"aggregate", "stream"}:
                plans.append(
                    _AnalysisPlan(
                        condition_id=stable_json_hash(
                            {
                                "checkpoint": checkpoint.identity,
                                "analyzer": analyzer,
                            }
                        ),
                        analyzer=analyzer,
                        checkpoint=checkpoint,
                    )
                )
                continue
            if analyzer != "final_module_influence":
                raise ValueError(
                    f"Base has no condition executor for analyzer {analyzer!r}"
                )
            methods = ("identity", "output_scale", "mean_patch")
            selected_sites = sites
            for sample in samples:
                for site in selected_sites:
                    for method in methods:
                        method_scales = (0.0,) if method == "output_scale" else (1.0,)
                        for scale in method_scales:
                            payload = {
                                "checkpoint": checkpoint.identity,
                                "analyzer": analyzer,
                                "sample": sample.sample_id,
                                "site": site.site_id,
                                "method": method,
                                "scale": scale,
                            }
                            plans.append(
                                _AnalysisPlan(
                                    condition_id=stable_json_hash(payload),
                                    analyzer=analyzer,
                                    checkpoint=checkpoint,
                                    sample=sample,
                                    site=site,
                                    method=method,
                                    scale=scale,
                                    constant=(
                                        0.0
                                        if method == "constant_patch" else None
                                    ),
                                )
                            )
    return tuple(plans)


def _execute_checkpoint_plans(
    *,
    execution: _AdapterExecution,
    loaded: LoadedModel,
    checkpoint: CheckpointRef,
    plans: Sequence[_AnalysisPlan],
    samples: Sequence[SampleRef],
    recipe: DiagnosticsRecipe,
    recipe_hash: str,
    run_id: str,
    store: DiagnosticsArtifactStore,
    checkpoint_load_seconds: float,
) -> None:
    adapter = execution.adapter
    requires_module_intervention = any(
        plan.site is not None for plan in plans
    )
    model_state: Mapping[str, torch.Tensor] | None = None
    sites: dict[str, ModuleSite] = {}
    if requires_module_intervention:
        model_state = capture_canonical_model_state(loaded.model)
        sites = {
            plan.site.site_id: plan.site
            for plan in plans
            if plan.site is not None
        }
    taylor_baseline_by_sample: dict[str, _BranchResult] = {}
    response_baseline_by_sample: dict[str, _BranchResult] = {}
    payload_by_sample: dict[str, Any] = {}
    capture_sites = tuple(sites.values())
    for plan in plans:
        if plan.site is None:
            _execute_aggregate_analyzer(
                execution=execution,
                loaded=loaded,
                plan=plan,
                samples=samples,
                recipe=recipe,
                recipe_hash=recipe_hash,
                run_id=run_id,
                store=store,
                checkpoint_load_seconds=checkpoint_load_seconds,
            )
            continue
        assert model_state is not None
        site = sites.get(plan.site.site_id)
        if site is None:
            _write_condition_terminal(
                store,
                plan=plan,
                loaded=loaded,
                recipe_hash=recipe_hash,
                run_id=run_id,
                status="skipped",
                skip_reason="module_site_unavailable_in_checkpoint",
            )
            continue
        assert plan.sample is not None and plan.method is not None
        try:
            sample_id = plan.sample.sample_id
            payload = payload_by_sample.get(sample_id)
            if payload is None:
                payload = adapter.materialize_sample(loaded, plan.sample)
                payload_by_sample[sample_id] = payload
            branch_pair_id = stable_json_hash(
                {
                    "checkpoint": checkpoint.identity,
                    "sample": sample_id,
                    "analyzer": "final_module_influence",
                }
            )
            taylor_baseline = taylor_baseline_by_sample.get(sample_id)
            if taylor_baseline is None:
                taylor_baseline = _run_branch(
                    execution=execution,
                    loaded=loaded,
                    payload=payload,
                    site=site,
                    capture_sites=capture_sites,
                    spec=None,
                    model_state=model_state,
                    condition_id=branch_pair_id,
                )
                taylor_baseline_by_sample[sample_id] = taylor_baseline
            response_baseline = response_baseline_by_sample.get(sample_id)
            if response_baseline is None:
                response_baseline = _run_branch(
                    execution=execution,
                    loaded=loaded,
                    payload=payload,
                    site=site,
                    capture_sites=(),
                    spec=None,
                    model_state=model_state,
                    condition_id=branch_pair_id,
                    perform_backward=False,
                    objective_options={"response_only": True},
                    evaluation_mode=True,
                )
                response_baseline_by_sample[sample_id] = response_baseline
            replacement = _resolve_replacement(
                execution,
                loaded=loaded,
                plan=plan,
                site=site,
                payload=payload,
            )
            spec = InterventionSpec(
                method=plan.method,
                module_site=site,
                scale=plan.scale,
                replacement=replacement,
                provenance={
                    "source": (
                        "adapter"
                        if isinstance(replacement, ModulePatchReference)
                        else "recipe"
                        if plan.method == "constant_patch"
                        else "none"
                    )
                },
            )
            if isinstance(replacement, ModulePatchReference):
                spec = InterventionSpec(
                    method=plan.method,
                    module_site=site,
                    scale=plan.scale,
                    replacement=replacement.replacement,
                    provenance=replacement.provenance,
                )
            intervention = _run_branch(
                execution=execution,
                loaded=loaded,
                payload=payload,
                site=site,
                capture_sites=(site,),
                spec=spec,
                model_state=model_state,
                condition_id=branch_pair_id,
                perform_backward=False,
                objective_options={"response_only": True},
                evaluation_mode=True,
            )
            _write_module_comparison(
                store,
                plan=plan,
                loaded=loaded,
                baseline=response_baseline,
                intervention=intervention,
                taylor_baseline=taylor_baseline,
                recipe_hash=recipe_hash,
                run_id=run_id,
            )
        except ConditionUnavailable as error:
            _write_condition_terminal(
                store,
                plan=plan,
                loaded=loaded,
                recipe_hash=recipe_hash,
                run_id=run_id,
                status="skipped",
                skip_reason=str(error) or "condition_unavailable",
            )
        except (RuntimeError, TypeError, ValueError, UnsupportedModuleOutput) as error:
            _write_condition_terminal(
                store,
                plan=plan,
                loaded=loaded,
                recipe_hash=recipe_hash,
                run_id=run_id,
                status="failed",
                skip_reason=f"{type(error).__name__}: {error}",
            )


def _execute_aggregate_analyzer(
    *,
    execution: _AdapterExecution,
    loaded: LoadedModel,
    plan: _AnalysisPlan,
    samples: Sequence[SampleRef],
    recipe: DiagnosticsRecipe,
    recipe_hash: str,
    run_id: str,
    store: DiagnosticsArtifactStore,
    checkpoint_load_seconds: float,
) -> None:
    expected_evidence_keys: tuple[str, ...]
    try:
        definition = execution.adapter.analyzer_catalog.definition(
            plan.analyzer
        )
        result = execution.run_analyzer(
            AnalyzerRunContext(
                definition=definition,
                loaded=loaded,
                samples=tuple(samples),
                options={},
                adapter=execution.adapter,
                recipe=recipe,
                checkpoint=plan.checkpoint,
                execution_metadata={
                    "checkpoint_load_seconds": float(
                        checkpoint_load_seconds
                    )
                },
            )
        )
        if not isinstance(result, AnalyzerExecutionResult):
            raise TypeError(
                "aggregate analyzer runner must return "
                "AnalyzerExecutionResult"
            )
        rows = [dict(row) for row in result.rows]
        expected_evidence_keys = result.expected_evidence_keys
        row_keys: list[str] = []
        for row in rows:
            collisions = sorted(set(row) & _ANALYZER_ENGINE_FIELDS)
            if collisions:
                raise ValueError(
                    "analyzer row attempted to override engine-owned fields: "
                    f"{collisions}"
                )
            if row.get("record_kind") == "analyzer_terminal":
                raise ValueError(
                    "analyzer row cannot emit the engine-owned analyzer_terminal"
                )
            evidence_key = str(row.get("evidence_key") or "").strip()
            if not evidence_key:
                raise ValueError(
                    "aggregate analyzer detail row requires evidence_key"
                )
            row["evidence_key"] = evidence_key
            row_keys.append(evidence_key)
        if len(set(row_keys)) != len(row_keys):
            raise ValueError(
                "aggregate analyzer detail evidence_key values must be unique"
            )
        unexpected = sorted(set(row_keys) - set(expected_evidence_keys))
        if unexpected:
            raise ValueError(
                "aggregate analyzer returned evidence keys outside its "
                f"expected inventory: {unexpected[:8]}"
            )
    except (ConditionUnavailable, AnalyzerCapabilityUnavailable) as error:
        expected_evidence_keys = (plan.condition_id,)
        rows = [
            {
                "evidence_key": plan.condition_id,
                "status": "skipped",
                "skip_reason": str(error),
            }
        ]
    except (RuntimeError, TypeError, ValueError) as error:
        expected_evidence_keys = (plan.condition_id,)
        rows = [
            {
                "evidence_key": plan.condition_id,
                "status": "failed",
                "skip_reason": f"{type(error).__name__}: {error}",
            }
        ]
    statuses: Counter[str] = Counter()
    keys_by_status: dict[str, set[str]] = {
        status: set() for status in _TERMINAL_STATUSES
    }
    for index, row in enumerate(rows):
        status = str(row.get("status", "success"))
        if status not in _TERMINAL_STATUSES:
            raise ValueError(
                f"invalid analyzer status {status!r} for {plan.analyzer}"
            )
        statuses[status] += 1
        keys_by_status[status].add(str(row["evidence_key"]))
        record = {
            **row,
            "analyzer": plan.analyzer,
            "record_kind": str(row.get("record_kind", "analyzer_detail")),
            "condition_id": plan.condition_id,
            "checkpoint_identity": plan.checkpoint.identity,
            "checkpoint_path": plan.checkpoint.path,
            "checkpoint_update": (
                loaded.checkpoint_update
                if loaded.checkpoint_update is not None
                else plan.checkpoint.update
            ),
            "run_id": run_id,
            "recipe_hash": recipe_hash,
            "catalog_digest": store.component_catalog_digest,
            "status": status,
        }
        record["analyzer_record_id"] = stable_json_hash(
            {
                "condition": plan.condition_id,
                "index": index,
                "record": record,
            }
        )
        store.append("analyzers", record)
    expected_set = set(expected_evidence_keys)
    returned_set = set().union(*keys_by_status.values())
    missing_evidence_keys = sorted(expected_set - returned_set)
    observed_evidence_keys = sorted(keys_by_status["success"])
    skipped_evidence_keys = sorted(
        keys_by_status["skipped"]
        | keys_by_status["insufficient_evidence"]
    )
    failed_evidence_keys = sorted(keys_by_status["failed"])
    terminal_status = (
        "failed"
        if statuses.get("failed")
        else "insufficient_evidence"
        if (
            statuses.get("insufficient_evidence")
            or missing_evidence_keys
            or statuses.get("skipped") and statuses.get("success")
        )
        else "skipped"
        if statuses and statuses.get("skipped") == sum(statuses.values())
        else "success"
    )
    store.append(
        "analyzers",
        {
            "analyzer": plan.analyzer,
            "record_kind": "analyzer_terminal",
            "condition_id": plan.condition_id,
            "checkpoint_identity": plan.checkpoint.identity,
            "checkpoint_path": plan.checkpoint.path,
            "checkpoint_update": (
                loaded.checkpoint_update
                if loaded.checkpoint_update is not None
                else plan.checkpoint.update
            ),
            "run_id": run_id,
            "recipe_hash": recipe_hash,
            "catalog_digest": store.component_catalog_digest,
            "status": terminal_status,
            "skip_reason": (
                "analyzer_detail_rows_not_all_success"
                if terminal_status != "success"
                else None
            ),
            "detail_row_count": sum(statuses.values()),
            "detail_status_counts": dict(statuses),
            "evidence_inventory_contract": "explicit_evidence_keys_v1",
            "expected_evidence_keys": sorted(expected_set),
            "observed_evidence_keys": observed_evidence_keys,
            "skipped_evidence_keys": skipped_evidence_keys,
            "failed_evidence_keys": failed_evidence_keys,
            "missing_evidence_keys": missing_evidence_keys,
            "expected_evidence_count": len(expected_set),
            "observed_evidence_count": len(observed_evidence_keys),
            "skipped_evidence_count": len(skipped_evidence_keys),
            "failed_evidence_count": len(failed_evidence_keys),
            "missing_evidence_count": len(missing_evidence_keys),
            "evidence_inventory_digest": stable_json_hash(
                sorted(expected_set)
            ),
        },
    )


def _run_branch(
    *,
    execution: _AdapterExecution,
    loaded: LoadedModel,
    payload: Any,
    site: ModuleSite,
    capture_sites: Sequence[ModuleSite],
    spec: InterventionSpec | None,
    model_state: Mapping[str, torch.Tensor],
    condition_id: str,
    perform_backward: bool = True,
    objective_options: Mapping[str, Any] | None = None,
    evaluation_mode: bool = False,
) -> _BranchResult:
    adapter = execution.adapter
    model = loaded.model
    restore_canonical_model_state(model, model_state)
    buffer_names = {name for name, _buffer in model.named_buffers()}
    branch_payload = execution.clone_payload(payload)
    module = resolve_module(model, site.module_path)
    initial_parameters = ParameterSnapshot.capture(model)
    latest_result: ObjectiveResult | None = None
    latest_gradient: GradientSnapshot | None = None
    latest_samples: dict[
        tuple[str, int, str],
        ActivationCaptureSample,
    ] = {}
    latest_rows: list[dict[str, Any]] = []
    try:
        with preserve_module_modes(model), paired_branch_rng(
            int(
                stable_json_hash({"condition_id": condition_id})[:16],
                16,
            ),
            execution.device(loaded),
        ):
            if evaluation_mode:
                model.eval()
            model.zero_grad(set_to_none=True)
            stack = ExitStack()
            try:
                intervention = None
                if spec is not None:
                    intervention = stack.enter_context(
                        ModuleOutputIntervention(module, spec)
                    )
                probes = [
                    stack.enter_context(
                        ModuleActivationCapture(
                            resolve_module(model, capture_site.module_path),
                            capture_site,
                            capture_gradients=perform_backward,
                        )
                    )
                    for capture_site in capture_sites
                ]
                result = adapter.forward_objective(
                    loaded,
                    branch_payload,
                    objective=dict(objective_options or {}),
                    precision=execution.execution_precision,
                    return_aux=True,
                )
                if intervention is not None:
                    # non-reentrant activation checkpointing 会在 backward 期间重放
                    # forward；以首次 forward 的调用数作为周期，保持 invocation
                    # selector 在原始计算与重算图上指向同一个逻辑位置。
                    intervention.mark_forward_complete()
                for probe in probes:
                    probe.mark_forward_complete()
                objective = result.effective_backward_objective()
                if perform_backward and objective.requires_grad:
                    objective.backward()
                latest_gradient = (
                    GradientSnapshot.capture(model)
                    if perform_backward
                    else None
                )
                objective_value = float(
                    objective.detach().to(dtype=torch.float64).cpu().item()
                )
                latest_rows = [
                    row
                    for probe in probes
                    for row in probe.rows(
                        objective_value=objective_value,
                        condition_id=condition_id,
                    )
                ]
                latest_samples = {
                    (
                        sample.module_site_id,
                        sample.invocation_index,
                        sample.output_path,
                    ): sample
                    for probe in probes
                    for sample in probe.samples.values()
                }
                latest_result = result
            finally:
                stack.close()
        changed_buffers = [
            name
            for name in buffer_names
            if name in model.state_dict()
            and not torch.equal(
                model.state_dict()[name].detach().cpu(),
                model_state[name],
            )
        ]
        buffer_status = (
            "unchanged"
            if not changed_buffers
            else "restored_after_mutation"
        )
        if latest_result is None:
            raise RuntimeError("branch produced no objective result")
        raw = (
            latest_result.raw_total
            if latest_result.raw_total is not None
            else latest_result.total
        )
        backward = latest_result.effective_backward_objective()
        return _BranchResult(
            status="success",
            skip_reason=None,
            loss=float(latest_result.total.detach().float().cpu().item()),
            terms=scalar_terms(latest_result.terms),
            outputs={
                str(name): value.detach().to("cpu").clone()
                for name, value in latest_result.outputs.items()
            },
            component_responses={
                (
                    response.response_component_id,
                    response.metric_id,
                ): {
                    "value": float(
                        response.value.detach().to(dtype=torch.float64).cpu().item()
                    ),
                    "support_count": float(response.support_count),
                    "normalization_id": response.normalization_id,
                    "higher_is_better": bool(response.higher_is_better),
                    "metadata": dict(response.metadata),
                }
                for response in latest_result.component_responses
            },
            gradient=latest_gradient,
            activation_samples=latest_samples,
            local_taylor_rows=latest_rows,
            parameter_delta_rows=parameter_delta_metrics(
                model,
                initial_parameters,
                hierarchy_nodes(loaded.model_name, capture_sites),
            ),
            objective_identity=latest_result.objective_identity,
            objective_contract_digest=stable_json_hash(
                {
                    "objective_identity": latest_result.objective_identity,
                    "objective_metadata": {
                        key: latest_result.objective_metadata.get(key)
                        for key in (
                            "amp_scale",
                            "actual_precision",
                            "autocast_dtype",
                            "autocast_enabled",
                            "checkpoint_weight_boundary",
                            "chunk_identity",
                            "device_type",
                            "loss_cap",
                            "loss_cap_applied",
                            "normalization_divisor",
                            "objective_executor_version",
                            "objective_schedule_coordinate",
                            "objective_schedule_total",
                            "parameter_dtype",
                            "precision_provenance_status",
                            "requested_precision",
                            "reduction",
                            "backward_objective_shape",
                            "backward_objective_reduction",
                            "cotangent_identity",
                            "segment_identity",
                        )
                        if key in latest_result.objective_metadata
                    },
                }
            ),
            objective_metadata=dict(latest_result.objective_metadata),
            raw_objective=float(raw.detach().float().cpu().item()),
            backward_objective=float(backward.detach().float().cpu().item()),
            optimizer_steps=0,
            buffer_restore_status=buffer_status,
        )
    finally:
        restore_canonical_model_state(model, model_state)
        model.zero_grad(set_to_none=True)


def _resolve_replacement(
    execution: _AdapterExecution,
    *,
    loaded: LoadedModel,
    plan: _AnalysisPlan,
    site: ModuleSite,
    payload: Any,
) -> Any:
    if plan.method == "mean_patch":
        assert plan.sample is not None
        return execution.module_patch_reference(
            loaded,
            site=site,
            method=plan.method,
            sample=plan.sample,
            payload=payload,
        )
    return None


def _write_module_comparison(
    store: DiagnosticsArtifactStore,
    *,
    plan: _AnalysisPlan,
    loaded: LoadedModel,
    baseline: _BranchResult,
    intervention: _BranchResult,
    taylor_baseline: _BranchResult,
    recipe_hash: str,
    run_id: str,
) -> None:
    if baseline.status != "success" or intervention.status != "success":
        _write_condition_terminal(
            store,
            plan=plan,
            loaded=loaded,
            recipe_hash=recipe_hash,
            run_id=run_id,
            status="failed",
            skip_reason=baseline.skip_reason or intervention.skip_reason,
        )
        return
    objective_contract_match = (
        baseline.objective_identity == intervention.objective_identity
        and baseline.objective_contract_digest
        == intervention.objective_contract_digest
    )
    if not objective_contract_match:
        _write_condition_terminal(
            store,
            plan=plan,
            loaded=loaded,
            recipe_hash=recipe_hash,
            run_id=run_id,
            status="skipped",
            skip_reason="objective_contract_mismatch",
        )
        return
    target_site_id = plan.site.site_id if plan.site is not None else None
    if plan.method == "identity":
        matching_taylor_rows = [
            row
            for row in taylor_baseline.local_taylor_rows
            if row.get("module_site_id") == target_site_id
        ]
        local_taylor_abs_sum = sum(
            float(
                row.get("local_taylor_abs_sum")
                or row.get("local_taylor_abs_sum_estimate")
                or 0.0
            )
            for row in matching_taylor_rows
        )
        store.append(
            "local_taylor",
            {
                **_base_record(plan, loaded, recipe_hash, run_id),
                "catalog_digest": store.component_catalog_digest,
                "record_kind": "module_local_taylor_summary",
                "status": "success",
                "module_site_id": target_site_id,
                "local_taylor_abs_sum": local_taylor_abs_sum,
                "objective_normalized_local_taylor": (
                    local_taylor_abs_sum
                    / max(abs(float(taylor_baseline.loss or 0.0)), 1e-12)
                ),
                "captured_output_count": len(matching_taylor_rows),
                "objective_identity": taylor_baseline.objective_identity,
                "claim_boundary": (
                    "local Taylor is a local first-order sensitivity, "
                    "not an intervention effect or physical causality"
                ),
            },
        )
    baseline_response_keys = set(baseline.component_responses)
    intervention_response_keys = set(intervention.component_responses)
    paired_response_keys = baseline_response_keys & intervention_response_keys
    expected_response_component_ids = set(store.output_component_ids)
    returned_response_keys = baseline_response_keys | intervention_response_keys
    baseline_response_component_ids = {
        component_id for component_id, _metric_id in baseline_response_keys
    }
    returned_response_component_ids = {
        component_id for component_id, _metric_id in returned_response_keys
    }
    identity_control = plan.method == "identity" or (
        plan.method == "output_scale" and abs(plan.scale - 1.0) <= 1e-12
    )
    sample_set_digest = stable_json_hash(
        [plan.sample.sample_id] if plan.sample is not None else []
    )
    component_response_contract_match = (
        not expected_response_component_ids and not returned_response_keys
    ) or (
        bool(baseline_response_keys)
        and baseline_response_keys == intervention_response_keys
        and baseline_response_component_ids == expected_response_component_ids
    )
    expected_response_keys = set(returned_response_keys)
    expected_response_keys.update(
        (response_component_id, "__missing_response__")
        for response_component_id in (
            expected_response_component_ids - returned_response_component_ids
        )
    )
    expected_evidence_keys: set[str] = {
        stable_json_hash(
            {
                "evidence_kind": "module_output_influence",
                "condition_id": plan.condition_id,
                "module_site_id": (
                    plan.site.site_id if plan.site is not None else None
                ),
                "response_component_id": response_component_id,
                "metric_id": metric_id,
            }
        )
        for response_component_id, metric_id in expected_response_keys
    }
    observed_evidence_keys: set[str] = set()
    skipped_evidence_keys: set[str] = set()
    for response_component_id, metric_id in sorted(
        paired_response_keys
    ):
        evidence_key = stable_json_hash(
            {
                "evidence_kind": "module_output_influence",
                "condition_id": plan.condition_id,
                "module_site_id": (
                    plan.site.site_id if plan.site is not None else None
                ),
                "response_component_id": response_component_id,
                "metric_id": metric_id,
            }
        )
        baseline_response = baseline.component_responses[
            (response_component_id, metric_id)
        ]
        intervention_response = intervention.component_responses[
            (response_component_id, metric_id)
        ]
        response_contract_match = (
            baseline_response["normalization_id"]
            == intervention_response["normalization_id"]
            and baseline_response["higher_is_better"]
            == intervention_response["higher_is_better"]
            and baseline_response["metadata"].get("response_id")
            == intervention_response["metadata"].get("response_id")
        )
        support_matches = (
            baseline_response["support_count"]
            == intervention_response["support_count"]
        )
        if not response_contract_match or not support_matches:
            component_response_contract_match = False
            skipped_evidence_keys.add(evidence_key)
            store.append(
                "module_effects",
                {
                    **_base_record(plan, loaded, recipe_hash, run_id),
                    "catalog_digest": store.component_catalog_digest,
                    "record_kind": "module_output_influence",
                    "response_component_id": response_component_id,
                    "metric_id": metric_id,
                    "evidence_key": evidence_key,
                    "status": "insufficient_evidence",
                    "baseline_value": float(baseline_response["value"]),
                    "condition_value": float(intervention_response["value"]),
                    "raw_delta": None,
                    "effect_value": None,
                    "normalized_effect": None,
                    "skip_reason": (
                        "output_response_contract_mismatch"
                        if not response_contract_match
                        else "output_response_support_mismatch"
                    ),
                    "baseline_support_count": baseline_response[
                        "support_count"
                    ],
                    "condition_support_count": intervention_response[
                        "support_count"
                    ],
                    "predictive_dependence_only": True,
                    "physical_causality_claimed": False,
                },
            )
            continue
        observed_evidence_keys.add(evidence_key)
        baseline_value = float(baseline_response["value"])
        intervention_value = float(intervention_response["value"])
        raw_delta = intervention_value - baseline_value
        effect_value = (
            -raw_delta
            if bool(intervention_response["higher_is_better"])
            else raw_delta
        )
        effect_normalization_scale = max(abs(baseline_value), 1e-12)
        normalized_effect = effect_value / effect_normalization_scale
        store.append(
            "module_effects",
            {
                **_base_record(plan, loaded, recipe_hash, run_id),
                "catalog_digest": store.component_catalog_digest,
                "record_kind": "module_output_influence",
                "response_component_id": response_component_id,
                "metric_id": metric_id,
                "response_normalization": intervention_response[
                    "normalization_id"
                ],
                "response_id": intervention_response["metadata"].get(
                    "response_id"
                ),
                "baseline_value": baseline_value,
                "condition_value": intervention_value,
                "raw_delta": raw_delta,
                "effect_value": effect_value,
                "normalized_effect": normalized_effect,
                "normalized_effect_denominator": (
                    "max(abs(baseline_value),1e-12)"
                ),
                "normalized_effect_status": (
                    "baseline_floor_applied"
                    if abs(baseline_value) < 1e-12
                    else "valid"
                ),
                "baseline_support_count": baseline_response["support_count"],
                "condition_support_count": intervention_response[
                    "support_count"
                ],
                "support_count": intervention_response["support_count"],
                "higher_is_better": intervention_response[
                    "higher_is_better"
                ],
                "sample_set_digest": sample_set_digest,
                "cohort_policy": "paired_sample",
                "distinct_group_count": (
                    1 if plan.sample is not None else 0
                ),
                "control_status": (
                    "identity_control"
                    if identity_control
                    else "intervention"
                ),
                "evidence_key": evidence_key,
                "status": "success",
                "predictive_dependence_only": True,
                "physical_causality_claimed": False,
            },
        )
    missing_response_keys = sorted(expected_response_keys - paired_response_keys)
    for response_component_id, metric_id in missing_response_keys:
        evidence_key = stable_json_hash(
            {
                "evidence_kind": "module_output_influence",
                "condition_id": plan.condition_id,
                "module_site_id": (
                    plan.site.site_id if plan.site is not None else None
                ),
                "response_component_id": response_component_id,
                "metric_id": metric_id,
            }
        )
        baseline_response = baseline.component_responses.get(
            (response_component_id, metric_id)
        )
        intervention_response = intervention.component_responses.get(
            (response_component_id, metric_id)
        )
        store.append(
            "module_effects",
            {
                **_base_record(plan, loaded, recipe_hash, run_id),
                "catalog_digest": store.component_catalog_digest,
                "record_kind": "module_output_influence",
                "response_component_id": response_component_id,
                "metric_id": metric_id,
                "evidence_key": evidence_key,
                "status": "missing",
                "skip_reason": "output_response_missing_from_branch",
                "baseline_value": (
                    None
                    if baseline_response is None
                    else float(baseline_response["value"])
                ),
                "condition_value": (
                    None
                    if intervention_response is None
                    else float(intervention_response["value"])
                ),
                "raw_delta": None,
                "effect_value": None,
                "normalized_effect": None,
                "baseline_support_count": (
                    0
                    if baseline_response is None
                    else int(baseline_response["support_count"])
                ),
                "condition_support_count": (
                    0
                    if intervention_response is None
                    else int(intervention_response["support_count"])
                ),
                "support_count": 0,
                "sample_set_digest": sample_set_digest,
                "cohort_policy": "paired_sample",
                "control_status": (
                    "identity_control"
                    if identity_control
                    else "intervention"
                ),
                "predictive_dependence_only": True,
                "physical_causality_claimed": False,
            },
        )
    identity_exact = (
        all(
            abs(
                float(intervention.component_responses[key]["value"])
                - float(baseline.component_responses[key]["value"])
            )
            <= 1e-12
            for key in paired_response_keys
        )
    )
    status = (
        "insufficient_evidence"
        if not component_response_contract_match
        else "success"
        if not identity_control or identity_exact
        else "failed"
    )
    missing_evidence_keys = sorted(
        expected_evidence_keys
        - observed_evidence_keys
        - skipped_evidence_keys
    )
    terminal_row = {
        **_base_record(plan, loaded, recipe_hash, run_id),
        "catalog_digest": store.component_catalog_digest,
        "record_kind": "condition_terminal",
        "status": status,
        "skip_reason": (
            None
            if status == "success"
            else "component_response_contract_mismatch"
            if status == "insufficient_evidence"
            else "identity_control_not_exact"
        ),
        "baseline_objective": taylor_baseline.loss,
        "objective_identity": taylor_baseline.objective_identity,
        "objective_contract_digest": taylor_baseline.objective_contract_digest,
        "response_objective_identity": baseline.objective_identity,
        "response_objective_contract_match": objective_contract_match,
        "training_objective_intervention_computed": False,
        "identity_control": identity_control,
        "identity_control_exact": identity_exact if identity_control else None,
        "component_response_contract_match": component_response_contract_match,
        "expected_response_count": len(expected_response_component_ids),
        "observed_response_count": len(
            baseline_response_component_ids
            & {
                component_id
                for component_id, _metric_id in intervention_response_keys
            }
        ),
        "expected_response_component_ids": sorted(
            expected_response_component_ids
        ),
        "observed_response_component_ids": sorted(
            baseline_response_component_ids
            & {
                component_id
                for component_id, _metric_id in intervention_response_keys
            }
        ),
        "missing_response_keys": [list(key) for key in missing_response_keys],
        "missing_response_component_ids": sorted(
            expected_response_component_ids - baseline_response_component_ids
        ),
        "buffer_restore_status": intervention.buffer_restore_status,
        "probe_optimizer_steps": intervention.optimizer_steps,
        "conductance_computed": False,
        "integrated_gradients_computed": False,
        "activation_patching_computed": False,
        "physical_causality_claimed": False,
    }
    if expected_evidence_keys:
        terminal_row.update(
            {
                "evidence_inventory_contract": "explicit_evidence_keys_v1",
                "expected_evidence_keys": sorted(expected_evidence_keys),
                "observed_evidence_keys": sorted(observed_evidence_keys),
                "skipped_evidence_keys": sorted(skipped_evidence_keys),
                "failed_evidence_keys": [],
                "missing_evidence_keys": missing_evidence_keys,
                "expected_evidence_count": len(expected_evidence_keys),
                "observed_evidence_count": len(observed_evidence_keys),
                "skipped_evidence_count": len(skipped_evidence_keys),
                "failed_evidence_count": 0,
                "missing_evidence_count": len(missing_evidence_keys),
                "evidence_inventory_digest": stable_json_hash(
                    sorted(expected_evidence_keys)
                ),
            }
        )
    store.append("conditions", terminal_row)


def _write_condition_terminal(
    store: DiagnosticsArtifactStore,
    *,
    plan: _AnalysisPlan,
    loaded: LoadedModel,
    recipe_hash: str,
    run_id: str,
    status: str,
    skip_reason: str | None,
) -> None:
    evidence_keys: list[str] = []
    if plan.analyzer == "final_module_influence" and plan.site is not None:
        for response_component_id in sorted(store.output_component_ids):
            evidence_key = stable_json_hash(
                {
                    "evidence_kind": "module_output_influence",
                    "condition_id": plan.condition_id,
                    "module_site_id": plan.site.site_id,
                    "response_component_id": response_component_id,
                    "metric_id": "__condition_unavailable__",
                }
            )
            evidence_keys.append(evidence_key)
            store.append(
                "module_effects",
                {
                    **_base_record(plan, loaded, recipe_hash, run_id),
                    "catalog_digest": store.component_catalog_digest,
                    "record_kind": "module_output_influence",
                    "response_component_id": response_component_id,
                    "metric_id": "__condition_unavailable__",
                    "evidence_key": evidence_key,
                    "status": status,
                    "skip_reason": skip_reason,
                    "baseline_value": None,
                    "condition_value": None,
                    "raw_delta": None,
                    "effect_value": None,
                    "normalized_effect": None,
                    "baseline_support_count": 0,
                    "condition_support_count": 0,
                    "support_count": 0,
                    "control_status": (
                        "identity_control_unavailable"
                        if plan.method == "identity"
                        or (
                            plan.method == "output_scale"
                            and abs(plan.scale - 1.0) <= 1e-12
                        )
                        else "intervention_unavailable"
                    ),
                    "predictive_dependence_only": True,
                    "physical_causality_claimed": False,
                },
            )
    row = {
        **_base_record(plan, loaded, recipe_hash, run_id),
        "catalog_digest": store.component_catalog_digest,
        "record_kind": "condition_terminal",
        "status": status,
        "skip_reason": skip_reason,
        "expected_response_component_ids": sorted(
            store.output_component_ids
        ),
        "observed_response_component_ids": [],
        "component_response_contract_match": not store.output_component_ids,
        "physical_causality_claimed": False,
    }
    if evidence_keys:
        evidence_keys = sorted(evidence_keys)
        row.update(
            {
                "evidence_inventory_contract": "explicit_evidence_keys_v1",
                "expected_evidence_keys": evidence_keys,
                "observed_evidence_keys": [],
                "skipped_evidence_keys": (
                    evidence_keys if status == "skipped" else []
                ),
                "failed_evidence_keys": (
                    evidence_keys if status == "failed" else []
                ),
                "missing_evidence_keys": [],
                "expected_evidence_count": len(evidence_keys),
                "observed_evidence_count": 0,
                "skipped_evidence_count": (
                    len(evidence_keys) if status == "skipped" else 0
                ),
                "failed_evidence_count": (
                    len(evidence_keys) if status == "failed" else 0
                ),
                "missing_evidence_count": 0,
                "evidence_inventory_digest": stable_json_hash(evidence_keys),
            }
        )
    if status == "failed":
        store.append("failures", row)
    store.append("conditions", row)


def _write_unavailable_cohort(
    store: DiagnosticsArtifactStore,
    *,
    checkpoint: CheckpointRef,
    reason: str,
    run_id: str,
    recipe_hash: str,
) -> None:
    condition_id = stable_json_hash(
        {"checkpoint": checkpoint.identity, "cohort": "unavailable"}
    )
    row = {
        "run_id": run_id,
        "recipe_hash": recipe_hash,
        "condition_id": condition_id,
        "condition_kind": "cohort",
        "checkpoint_identity": checkpoint.identity,
        "checkpoint_path": checkpoint.path,
        "checkpoint_update": checkpoint.update,
        "catalog_digest": store.component_catalog_digest,
        "record_kind": "condition_terminal",
        "status": "skipped",
        "skip_reason": reason,
    }
    store.append("failures", row)
    store.append("conditions", row)


def _base_record(
    plan: _AnalysisPlan,
    loaded: LoadedModel,
    recipe_hash: str,
    run_id: str,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "recipe_hash": recipe_hash,
        "condition_id": plan.condition_id,
        "condition_kind": plan.analyzer,
        "checkpoint_identity": plan.checkpoint.identity,
        "checkpoint_path": plan.checkpoint.path,
        "checkpoint_update": (
            loaded.checkpoint_update
            if loaded.checkpoint_update is not None
            else plan.checkpoint.update
        ),
        "model_name": loaded.model_name,
        "sample_id": plan.sample.sample_id if plan.sample else None,
        "partition": plan.sample.partition if plan.sample else None,
        "group_id": plan.sample.group_id if plan.sample else None,
        "position": plan.sample.position if plan.sample else None,
        "module_site_id": plan.site.site_id if plan.site else None,
        "node_id": (
            plan.site.node_id or plan.site.site_id
            if plan.site
            else None
        ),
        "parent_node_id": plan.site.parent_node_id if plan.site else None,
        "stage_id": plan.site.stage_id if plan.site else None,
        "module_path": plan.site.module_path if plan.site else None,
        "module_type": plan.site.module_type if plan.site else None,
        "hierarchy_level": (
            plan.site.metadata.get("hierarchy_level")
            if plan.site
            else None
        ),
        "module_discovery_source": (
            plan.site.metadata.get("discovery_source")
            if plan.site
            else None
        ),
        "method": plan.method,
        "scale": plan.scale if plan.method == "output_scale" else None,
    }


def _validate_loaded_model(
    loaded: LoadedModel,
    checkpoint: CheckpointRef,
) -> None:
    """校验 loaded model 类型，不把 selector update 当成 embedded provenance。

    ``CheckpointRef.update`` 可能只是调用方 catalog coordinate；model load 后从
    checkpoint 内已验证 training provenance 解析的 local/global update 更可靠。两者
    可以不同，adapter 必须在 structure/objective metadata 中记录来源，Base 不能用
    selector 值覆盖 embedded 事实。
    """

    if not isinstance(loaded.model, torch.nn.Module):
        raise TypeError("adapter.load_model must return LoadedModel with nn.Module")
    del checkpoint
    if loaded.checkpoint_update is not None and int(loaded.checkpoint_update) < 0:
        raise ValueError("loaded checkpoint update must be non-negative")


__all__ = ["run_checkpoint_diagnostics"]
