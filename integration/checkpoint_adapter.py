"""
把任务中立 HostTaskRuntime 翻译为 checkpoint engine protocol。

adapter 不解析 checkpoint 文件名、不重建 objective、不认识宿主 batch 类型。模型、
sample 与 objective 的唯一执行事实分别来自 runtime providers；本文件只处理两个
公共 DTO 集合之间的边界转换和 checkpoint branch 生命周期。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch

from model_diagnostics.base.artifacts import stable_json_hash
from model_diagnostics.base.checkpoint import (
    CheckpointRef,
    CheckpointRuntimeDescriptor,
    ComponentResponse,
    CohortSelection,
    ConditionUnavailable,
    DiagnosticComponent,
    DiagnosticsRecipe,
    LoadedModel,
    ModulePatchReference,
    ModuleSite,
    ObjectiveResult,
    SampleRef,
)
from model_diagnostics.base.interventions import iter_module_output_tensors
from model_diagnostics.base.registry import AnalyzerBindingCatalog, AnalyzerCatalog
from model_diagnostics.host_runtime.composition import (
    ExecutionSession,
    HostTaskRuntime,
)
from model_diagnostics.host_runtime.contracts import (
    BatchEnvelope,
    CheckpointTrainingProvenance,
    ExecutionPreparationCapability,
    ObjectiveContext,
    ObjectiveExecution,
    ObjectiveUnit,
    RunRef,
    RuntimeCheckpointRef,
    RuntimeInputRef,
    RuntimeOutputRef,
    RuntimeOutputResponse,
    RuntimeSampleRef,
    SampleSelector,
)

_DIAGNOSTIC_MODULE_SITE_ATTRIBUTE = "diagnostic_module_site"
_SWEEP_COHORT_SIZE = 32
_SWEEP_MINIMUM_GROUPS = 8
_FINAL_COHORT_SIZE = 8
_FINAL_MINIMUM_GROUPS = 8


_RUNTIME_OWNED_OBJECTIVE_FIELDS = frozenset(
    {
        "amp_scale",
        "loss_cap",
        "normalization_divisor",
        "objective_id",
        "precision",
        "prefer_chunked",
        "requires_schedule_coordinate",
        "return_aux_enabled",
        "schedule_coordinate",
        "schedule_total",
        "training",
    }
)


@dataclass(slots=True)
class _LoadedRuntimeContext:
    """把 Base LoadedModel 与 owned Host Runtime session 绑定。"""

    session: ExecutionSession
    checkpoint: RuntimeCheckpointRef
    objective_schedule_coordinate: int | float | str | None
    objective_schedule_source: str
    objective_schedule_total: int | float | None
    objective_schedule_total_source: str
    weight_boundary: str | None


@dataclass(frozen=True, slots=True)
class _ExecutedObjective:
    unit: ObjectiveUnit
    execution: ObjectiveExecution
    schedule_coordinate: int | float | str | None
    schedule_source: str
    schedule_total: int | float | None
    schedule_total_source: str


class GenericCheckpointAdapter:
    """实现 Base checkpoint protocol 的 task-neutral Host Runtime adapter."""

    def __init__(
        self,
        *,
        runtime: HostTaskRuntime,
        run_dir: str | Path,
        analyzer_catalog: AnalyzerCatalog,
        analyzer_bindings: AnalyzerBindingCatalog,
        capability_requirements_by_analyzer: Mapping[
            str,
            Sequence[str],
        ] | None = None,
        owns_runtime: bool = False,
    ) -> None:
        if not isinstance(runtime, HostTaskRuntime):
            raise TypeError("runtime must be a HostTaskRuntime")
        if not isinstance(analyzer_catalog, AnalyzerCatalog):
            raise TypeError("analyzer_catalog must be an AnalyzerCatalog")
        if not isinstance(analyzer_bindings, AnalyzerBindingCatalog):
            raise TypeError("analyzer_bindings must be an AnalyzerBindingCatalog")
        analyzer_bindings.validate_against(analyzer_catalog)
        self.runtime = runtime
        self.run_dir = Path(run_dir).resolve()
        self.analyzer_catalog = analyzer_catalog
        self.analyzer_bindings = analyzer_bindings
        self._capability_requirements_by_analyzer = MappingProxyType(
            {
                str(analyzer).strip(): tuple(
                    str(capability).strip()
                    for capability in capabilities
                    if str(capability).strip()
                )
                for analyzer, capabilities in dict(
                    capability_requirements_by_analyzer or {}
                ).items()
            }
        )
        self._owns_runtime = bool(owns_runtime)
        self.runtime_descriptor = _checkpoint_runtime_descriptor(runtime)
        self.rank, self.world_size = _distributed_identity(runtime)
        self.device = _runtime_device(runtime)
        self._recipe: DiagnosticsRecipe | None = None
        self._checkpoint_refs_by_path: dict[str, RuntimeCheckpointRef] = {}
        self._sample_refs_by_id: dict[str, RuntimeSampleRef] = {}
        self._open_sessions: dict[int, ExecutionSession] = {}
        self._component_catalog_by_session: dict[
            int,
            tuple[
                tuple[RuntimeInputRef, ...],
                tuple[RuntimeOutputRef, ...],
                str,
            ],
        ] = {}
        self._frozen_component_catalog: tuple[
            tuple[RuntimeInputRef, ...],
            tuple[RuntimeOutputRef, ...],
            str,
        ] | None = None
        self._closed = False
        self.execution_precision = (
            runtime.descriptor.checkpoint_objective.precision
        )

    def configure(self, recipe: DiagnosticsRecipe) -> None:
        """保存 recipe，并激活其 analyzer 已声明的 execution requirements。"""

        self._ensure_open()
        if self._recipe is not None:
            if self._recipe.to_dict() != recipe.to_dict():
                raise RuntimeError(
                    "GenericCheckpointAdapter cannot be reconfigured"
                )
            return
        required_capabilities = {
            capability_id
            for analyzer in recipe.analyzers
            for capability_id in self._capability_requirements_by_analyzer.get(
                analyzer,
                (),
            )
        }
        for capability_id in sorted(required_capabilities):
            capability = self.runtime.capabilities.get(capability_id)
            if capability is None:
                raise ValueError(
                    "analyzer requires unavailable runtime capability: "
                    f"{capability_id}"
                )
            if isinstance(capability, ExecutionPreparationCapability):
                capability.prepare_execution()
        self._recipe = recipe

    def resolve_checkpoints(
        self,
        *,
        run_dir: Path,
    ) -> Sequence[CheckpointRef]:
        """让 CheckpointProvider 按固定阶段策略解析 checkpoint 身份。"""

        self._ensure_open()
        if self._recipe is None:
            raise RuntimeError("checkpoint adapter must be configured first")
        if Path(run_dir).resolve() != self.run_dir:
            raise ValueError("checkpoint adapter run directory identity changed")
        descriptor_run = self.runtime.descriptor.run
        effective_run = RunRef(
            run_id=descriptor_run.run_id,
            locator=descriptor_run.locator,
            metadata={
                **dict(descriptor_run.metadata),
                "diagnostics_checkpoint_stage": self._recipe.stage,
                "diagnostics_final_checkpoint": (
                    self._recipe.final_checkpoint
                ),
            },
        )
        runtime_refs = tuple(self.runtime.checkpoints.resolve(effective_run))
        if not runtime_refs:
            return ()
        if any(not isinstance(ref, RuntimeCheckpointRef) for ref in runtime_refs):
            raise TypeError(
                "CheckpointProvider.resolve must return RuntimeCheckpointRef values"
            )
        self._checkpoint_refs_by_path.clear()
        converted: list[CheckpointRef] = []
        for runtime_ref in runtime_refs:
            resolved_path = str(Path(runtime_ref.locator).resolve())
            if resolved_path in self._checkpoint_refs_by_path:
                raise ValueError(
                    "CheckpointProvider returned duplicate checkpoint locator"
                )
            self._checkpoint_refs_by_path[resolved_path] = runtime_ref
            metadata = dict(runtime_ref.metadata)
            converted.append(
                CheckpointRef(
                    path=resolved_path,
                    identity=runtime_ref.checkpoint_id,
                    update=_optional_int(metadata.get("update")),
                    kind=str(metadata.get("kind", "runtime_checkpoint")),
                    metadata=metadata,
                )
            )
        return tuple(converted)

    def load_model(self, checkpoint_path: str, *, precision: str) -> LoadedModel:
        """通过 HostTaskRuntime 打开一次 owned checkpoint session。"""

        self._ensure_open()
        requested_precision = str(precision).strip().lower()
        if requested_precision != self.execution_precision:
            raise ValueError(
                "checkpoint model precision does not match the "
                "HostTaskRuntime execution policy"
            )
        resolved_path = str(Path(checkpoint_path).resolve())
        runtime_ref = self._checkpoint_refs_by_path.get(resolved_path)
        if runtime_ref is None:
            raise ValueError(
                "checkpoint path was not returned by resolve_checkpoints"
            )
        session = self.runtime.open_session(runtime_ref, device=self.device)
        self._open_sessions[id(session)] = session
        metadata = dict(session.handle.metadata)
        checkpoint_metadata = dict(
            session.checkpoint_state.metadata
            if session.checkpoint_state is not None
            else {}
        )
        provenance = (
            session.checkpoint_state.training_provenance
            if session.checkpoint_state is not None
            else None
        )
        checkpoint_update = next(
            (
                int(value)
                for value in (
                    (
                        provenance.local_completed_update
                        if provenance is not None
                        else None
                    ),
                    runtime_ref.metadata.get("update"),
                )
                if value is not None
            ),
            None,
        )
        schedule_coordinate, schedule_source = _objective_schedule_coordinate(
            provenance=provenance,
            checkpoint_metadata=runtime_ref.metadata,
        )
        schedule_total, schedule_total_source = _objective_schedule_total(
            provenance=provenance,
            checkpoint_metadata=runtime_ref.metadata,
        )
        weight_boundary = (
            provenance.weight_boundary
            if provenance is not None
            else None
        )
        return LoadedModel(
            model=session.model,
            checkpoint_update=checkpoint_update,
            model_name=session.handle.model_id,
            structure_metadata={
                **metadata,
                "runtime_id": self.runtime.descriptor.runtime_id,
                "requested_precision": requested_precision,
                "checkpoint_ref_update": _optional_int(
                    runtime_ref.metadata.get("update")
                ),
                "embedded_local_completed_update": _optional_int(
                    (
                        provenance.local_completed_update
                        if provenance is not None
                        else None
                    )
                ),
                "checkpoint_update_source": (
                    "checkpoint_training_provenance.local_completed_update"
                    if (
                        provenance is not None
                        and provenance.local_completed_update is not None
                    )
                    else "checkpoint_ref.metadata.update"
                    if runtime_ref.metadata.get("update") is not None
                    else "unavailable"
                ),
                "objective_schedule_coordinate": schedule_coordinate,
                "objective_schedule_source": schedule_source,
                "objective_schedule_total": schedule_total,
                "objective_schedule_total_source": schedule_total_source,
                "checkpoint_weight_boundary": weight_boundary,
            },
            ignored_payload_keys=tuple(
                str(key)
                for key in checkpoint_metadata.get("ignored_payload_keys", ())
            ),
            adapter_context=_LoadedRuntimeContext(
                session=session,
                checkpoint=runtime_ref,
                objective_schedule_coordinate=schedule_coordinate,
                objective_schedule_source=schedule_source,
                objective_schedule_total=schedule_total,
                objective_schedule_total_source=schedule_total_source,
                weight_boundary=weight_boundary,
            ),
        )

    def build_cohort(
        self,
        loaded: LoadedModel,
        *,
        checkpoints: Sequence[CheckpointRef],
    ) -> CohortSelection:
        """按固定分组统计策略建立跨 checkpoint 稳定 cohort。"""

        context = self._loaded_context(loaded)
        self._activate_batch_session(context)
        if self._recipe is None:
            raise RuntimeError("checkpoint adapter must be configured first")
        if self._recipe.stage == "checkpoint_sweep":
            limit = _SWEEP_COHORT_SIZE
            minimum_groups = _SWEEP_MINIMUM_GROUPS
            policy_id = "fixed_group_stratified_sweep"
        else:
            limit = _FINAL_COHORT_SIZE
            minimum_groups = _FINAL_MINIMUM_GROUPS
            policy_id = "fixed_group_stratified_final"
        criteria = {"minimum_distinct_groups": minimum_groups}
        refs = tuple(
            self.runtime.batches.select(
                self.runtime.descriptor.run,
                SampleSelector(
                    partition="test",
                    limit=limit,
                    criteria=criteria,
                ),
            )
        )
        if any(not isinstance(ref, RuntimeSampleRef) for ref in refs):
            raise TypeError(
                "BatchProvider.select must return RuntimeSampleRef values"
            )
        if len({ref.sample_id for ref in refs}) != len(refs):
            raise ValueError("BatchProvider returned duplicate sample_id values")
        self._sample_refs_by_id = {ref.sample_id: ref for ref in refs}
        base_refs = tuple(
            SampleRef(
                sample_id=ref.sample_id,
                partition=ref.partition,
                sample_index=ref.sample_index,
                group_id=ref.group_id,
                position=ref.position,
                metadata=dict(ref.metadata),
            )
            for ref in refs
        )
        if (
            refs
            and self.runtime.capabilities.get("component_catalog") is not None
        ):
            first_batch = self.runtime.batches.materialize(
                (refs[0],),
                device=self.device,
            )
            self._resolve_component_catalog(
                loaded,
                first_batch,
                require_cached=False,
            )
        identity = stable_json_hash(
            {
                "runtime_descriptor_digest": (
                    self.runtime_descriptor.runtime_descriptor_digest
                ),
                "checkpoints": [checkpoint.identity for checkpoint in checkpoints],
                "samples": [sample.to_dict() for sample in base_refs],
            }
        )
        if not base_refs:
            return CohortSelection(
                status="unavailable",
                samples=(),
                skip_reason="runtime_batch_provider_selected_no_samples",
                identity=identity,
                metadata={
                    "selection_policy": policy_id,
                    "requested_sample_count": limit,
                    "minimum_distinct_groups": minimum_groups,
                },
            )
        distinct_groups = len(
            {ref.group_id for ref in refs if ref.group_id is not None}
        )
        if len(refs) < limit or distinct_groups < minimum_groups:
            return CohortSelection(
                status="unavailable",
                samples=base_refs,
                skip_reason="insufficient_group_stratified_support",
                identity=identity,
                metadata={
                    "selection_policy": policy_id,
                    "requested_sample_count": limit,
                    "selected_sample_count": len(refs),
                    "minimum_distinct_groups": minimum_groups,
                    "distinct_group_count": distinct_groups,
                },
            )
        return CohortSelection(
            status="available",
            samples=base_refs,
            identity=identity,
            metadata={
                "selection_policy": policy_id,
                "requested_sample_count": limit,
                "selected_sample_count": len(refs),
                "minimum_distinct_groups": minimum_groups,
                "distinct_group_count": distinct_groups,
                "selection_method": "deterministic_group_even_spacing",
            },
        )

    def list_diagnostic_components(
        self,
        loaded: LoadedModel,
        samples: Sequence[SampleRef],
    ) -> tuple[DiagnosticComponent, ...]:
        """把 Host Runtime catalog 转换为 Base 的 analysis-level 身份表。

        catalog 的持久化、digest 与 analysis identity 由 Base checkpoint engine
        统一负责；adapter 不在 cohort metadata 中创建第二份事实源。
        """

        if (
            not samples
            or self.runtime.capabilities.get("component_catalog") is None
        ):
            return ()
        batch = self.materialize_sample(loaded, samples[0])
        inputs, outputs, _digest = self._resolve_component_catalog(
            loaded,
            batch,
            require_cached=False,
        )
        components: list[DiagnosticComponent] = []
        for ref in (*inputs, *outputs):
            metadata = dict(ref.component.metadata)
            component_shape = metadata.pop("component_shape", None)
            reduction_semantics = metadata.pop(
                "reduction_semantics",
                None,
            )
            components.append(
                DiagnosticComponent(
                    component_id=ref.component.component_id,
                    semantic_id=ref.component.semantic_id,
                    component_kind=ref.component.component_kind,
                    display_label=ref.component.display_label,
                    component_group=ref.component.component_group,
                    component_index=ref.component.component_index,
                    tensor_path=ref.tensor_path,
                    tensor_axis=ref.tensor_axis,
                    tensor_index=ref.tensor_index,
                    normalization_id=(
                        ref.normalization_id
                        if isinstance(ref, RuntimeOutputRef)
                        else None
                    ),
                    component_shape=(
                        tuple(int(value) for value in component_shape)
                        if component_shape is not None
                        else None
                    ),
                    reduction_semantics=reduction_semantics,
                    metadata=metadata,
                )
            )
        return tuple(components)

    def materialize_sample(
        self,
        loaded: LoadedModel,
        sample: SampleRef,
    ) -> BatchEnvelope:
        """物化单个稳定 sample；payload 仍封装在 task-neutral BatchEnvelope。"""

        context = self._loaded_context(loaded)
        self._activate_batch_session(context)
        runtime_ref = self._sample_refs_by_id.get(sample.sample_id)
        if runtime_ref is None:
            raise ValueError("sample was not returned by build_cohort")
        return self.runtime.batches.materialize(
            (runtime_ref,),
            device=self.device,
        )

    def runtime_session(self, loaded: LoadedModel) -> ExecutionSession:
        """向 integration bindings 暴露 task-neutral loaded session。"""

        return self._loaded_context(loaded).session

    def component_catalog(
        self,
        loaded: LoadedModel,
        batch: BatchEnvelope,
    ) -> tuple[
        tuple[RuntimeInputRef, ...],
        tuple[RuntimeOutputRef, ...],
        str,
    ]:
        """返回 cohort 已冻结的 component catalog，并核验当前 batch identity。

        catalog 由 Host Runtime 物化；adapter 只规范化顺序和验证跨 sample 稳定性。
        label、group 和 tensor 坐标仅写入 cohort metadata，analyzer row 只引用
        component ID，避免重复事实源。
        """

        return self._resolve_component_catalog(
            loaded,
            batch,
            require_cached=True,
        )

    def _resolve_component_catalog(
        self,
        loaded: LoadedModel,
        batch: BatchEnvelope,
        *,
        require_cached: bool,
    ) -> tuple[
        tuple[RuntimeInputRef, ...],
        tuple[RuntimeOutputRef, ...],
        str,
    ]:
        """调用 task-neutral catalog capability，并拒绝跨 batch identity 漂移。"""

        capability = self.runtime.capabilities.get("component_catalog")
        if capability is None:
            raise ConditionUnavailable(
                "runtime_component_catalog_capability_unavailable"
            )
        session = self.runtime_session(loaded)
        inputs = tuple(
            capability.list_input_components(session=session, batch=batch)
        )
        outputs = tuple(
            capability.list_output_components(session=session, batch=batch)
        )
        if any(not isinstance(item, RuntimeInputRef) for item in inputs):
            raise TypeError(
                "component catalog must return RuntimeInputRef inputs"
            )
        if any(not isinstance(item, RuntimeOutputRef) for item in outputs):
            raise TypeError(
                "component catalog must return RuntimeOutputRef outputs"
            )
        records = _component_catalog_records(inputs, outputs)
        digest = stable_json_hash(records)
        session_id = id(session)
        cached = self._component_catalog_by_session.get(session_id)
        if cached is None:
            frozen = self._frozen_component_catalog
            if frozen is not None and frozen[2] == digest:
                cached = (inputs, outputs, digest)
                self._component_catalog_by_session[session_id] = cached
            elif require_cached:
                raise ConditionUnavailable(
                    "component_catalog_identity_changed_from_frozen_cohort"
                )
            else:
                cached = (inputs, outputs, digest)
                self._frozen_component_catalog = cached
                self._component_catalog_by_session[session_id] = cached
        elif cached[2] != digest:
            raise ConditionUnavailable(
                "component_catalog_identity_changed_within_cohort"
            )
        return cached

    def runtime_sample_ref(self, sample: SampleRef) -> RuntimeSampleRef:
        """把 Base sample identity 还原为 BatchProvider 选择的 runtime ref。"""

        try:
            return self._sample_refs_by_id[sample.sample_id]
        except KeyError as error:
            raise ValueError("sample was not returned by build_cohort") from error

    def clone_sample_payload(self, payload: Any) -> BatchEnvelope:
        """委托 BatchProvider 创建 storage-isolated condition payload。"""

        if not isinstance(payload, BatchEnvelope):
            raise TypeError(
                "GenericCheckpointAdapter payload must be a BatchEnvelope"
            )
        cloned = self.runtime.batches.clone(payload)
        if not isinstance(cloned, BatchEnvelope):
            raise TypeError("BatchProvider.clone must return a BatchEnvelope")
        return cloned

    def list_module_sites(self, loaded: LoadedModel) -> tuple[ModuleSite, ...]:
        """提供模型类显式标记的增量 module-site override。

        模型可在希望参与最终作用分析的 ``nn.Module`` 类上声明
        ``diagnostic_module_site = True``。Base 仍独立执行通用 Stage/Block 自动发现；
        这些声明只补充或覆盖结构本身无法稳定表达的边界。
        """

        sites: list[ModuleSite] = []
        for module_path, module in loaded.model.named_modules():
            if not module_path or not bool(
                getattr(
                    module.__class__,
                    _DIAGNOSTIC_MODULE_SITE_ATTRIBUTE,
                    False,
                )
            ):
                continue
            sites.append(
                ModuleSite(
                    site_id=f"declared_module:{module_path}",
                    node_id=f"declared_module:{module_path}",
                    module_path=module_path,
                    module_type=module.__class__.__name__,
                    parent_node_id=f"model:{loaded.model_name}",
                    alias_node_ids=(f"declared_module:{module_path}",),
                    metadata={
                        "discovery_source": "explicit_model_declaration",
                        "declaration_attribute": (
                            _DIAGNOSTIC_MODULE_SITE_ATTRIBUTE
                        ),
                    },
                )
            )
        return tuple(sites)

    def forward_objective(
        self,
        loaded: LoadedModel,
        payload: Any,
        *,
        objective: Mapping[str, Any],
        precision: str,
        return_aux: bool,
    ) -> ObjectiveResult:
        """调用唯一 ObjectiveExecutor，并保留实际 backward objective provenance。

        checkpoint branch 由 Base 在返回后执行 backward，因此当前 bridge 只接受一个
        objective unit。需要逐 unit 立即 backward 的 streaming objective 必须由专门
        capability 执行，不能在这里偷偷重建或改变其数学。
        """

        executed = self._execute_objective_unit(
            loaded,
            payload,
            objective=objective,
            precision=precision,
            return_aux=return_aux,
        )
        unit = executed.unit
        loaded_context = self._loaded_context(loaded)
        execution_metadata = executed.execution.metadata
        outputs = {
            path: tensor
            for output_name, output in unit.outputs.items()
            for path, tensor in iter_module_output_tensors(
                output,
                _path=f"$.{output_name}",
            )
        }
        terms = {
            str(name): (
                value
                if torch.is_tensor(value)
                else unit.backward_total.new_tensor(float(value))
            )
            for name, value in unit.terms.items()
        }
        component_responses: tuple[ComponentResponse, ...] = ()
        output_evaluator = self.runtime.capabilities.get("output_evaluation")
        if output_evaluator is not None:
            from model_diagnostics.integration.component_evidence import (
                output_response_map,
            )

            if not isinstance(payload, BatchEnvelope):
                raise TypeError(
                    "output evaluation requires a BatchEnvelope payload"
                )
            _inputs, output_refs, _catalog_digest = self.component_catalog(
                loaded,
                payload,
            )
            available, _unavailable = output_response_map(
                output_evaluator,
                batch=payload,
                unit=unit,
                outputs=output_refs,
            )
            runtime_responses = tuple(
                available[output.component.component_id]
                for output in output_refs
                if output.component.component_id in available
            )
            if any(
                not isinstance(response, RuntimeOutputResponse)
                for response in runtime_responses
            ):
                raise TypeError(
                    "output_evaluation must return RuntimeOutputResponse values"
                )
            component_responses = tuple(
                ComponentResponse(
                    response_component_id=(
                        response.output.component.component_id
                    ),
                    metric_id=response.metric_id,
                    value=response.value,
                    support_count=response.support_count,
                    normalization_id=response.normalization,
                    higher_is_better=response.higher_is_better,
                    gradient_tensor=response.gradient_tensor,
                    gradient_axis=response.output.tensor_axis,
                    gradient_index=response.output.tensor_index,
                    metadata={
                        "response_id": response.response_id,
                        **dict(response.metadata),
                    },
                )
                for response in runtime_responses
            )
        return ObjectiveResult(
            total=unit.raw_total,
            terms=terms,
            outputs=outputs,
            raw_total=unit.raw_total,
            backward_total=unit.backward_total,
            backward_cotangent=unit.backward_cotangent,
            cotangent_identity=unit.cotangent_identity,
            component_responses=component_responses,
            aux={
                "execution_id": executed.execution.execution_id,
                "unit_id": unit.unit_id,
            },
            objective_identity=unit.objective_identity,
            objective_metadata={
                **dict(unit.metadata),
                "runtime_id": self.runtime.descriptor.runtime_id,
                "objective_executor_version": (
                    self.runtime.descriptor.objective_executor_version
                ),
                "objective_schedule_coordinate": executed.schedule_coordinate,
                "objective_schedule_source": executed.schedule_source,
                "objective_schedule_total": (
                    executed.schedule_total
                ),
                "objective_schedule_total_source": (
                    executed.schedule_total_source
                ),
                "checkpoint_weight_boundary": loaded_context.weight_boundary,
                "requested_precision": self.execution_precision,
                "actual_precision": execution_metadata.get(
                    "actual_precision"
                ),
                "parameter_dtype": execution_metadata.get(
                    "parameter_dtype"
                ),
                "device_type": execution_metadata.get("device_type"),
                "autocast_enabled": execution_metadata.get(
                    "autocast_enabled"
                ),
                "autocast_dtype": execution_metadata.get("autocast_dtype"),
                "precision_provenance_status": (
                    "observed"
                    if execution_metadata.get("actual_precision") is not None
                    else "not_reported"
                ),
                "normalization_divisor": unit.normalization_divisor,
                "loss_cap": unit.loss_cap,
                "loss_cap_applied": unit.loss_cap_applied,
                "amp_scale": unit.amp_scale,
                "chunk_identity": unit.chunk_identity,
                "segment_identity": unit.segment_identity,
                "backward_objective_shape": list(unit.backward_total.shape),
                "backward_objective_reduction": (
                    "explicit_vector_jacobian_product"
                    if unit.backward_cotangent is not None
                    else "scalar_identity"
                ),
                "cotangent_identity": unit.cotangent_identity,
            },
        )

    def execute_objective_unit(
        self,
        loaded: LoadedModel,
        batch: BatchEnvelope,
        *,
        objective: Mapping[str, Any] | None = None,
        precision: str | None = None,
        return_aux: bool = True,
    ) -> ObjectiveUnit:
        """为 integration bindings 执行同一 single-unit checkpoint objective。"""

        return self._execute_objective_unit(
            loaded,
            batch,
            objective=(
                dict(objective)
                if objective is not None
                else {}
            ),
            precision=(
                str(precision)
                if precision is not None
                else self.execution_precision
            ),
            return_aux=return_aux,
        ).unit

    def optimization_probe_objective_context(
        self,
        loaded: LoadedModel,
        *,
        probe_step: int,
    ) -> ObjectiveContext:
        """构造 fresh branch 使用的真实 objective context。

        branch 每步只有一个 microbatch，所以 divisor 固定为一；objective identity、
        cap、schedule 与 precision 仍来自 Host Runtime/checkpoint provenance。
        """

        context = self._loaded_context(loaded)
        specification = self.runtime.descriptor.checkpoint_objective
        schedule_total = (
            context.objective_schedule_total
            if context.objective_schedule_total is not None
            else specification.schedule_total
        )
        return ObjectiveContext(
            objective_id=specification.objective_id,
            training=True,
            update_index=int(probe_step),
            microbatch_index=0,
            schedule_coordinate=context.objective_schedule_coordinate,
            schedule_total=schedule_total,
            normalization_divisor=1.0,
            loss_cap=specification.loss_cap,
            amp_scale=1.0,
            options={
                **dict(specification.options),
                "collect_objective_ledger": True,
                "prefer_chunked": False,
                "return_aux_enabled": True,
            },
            metadata={
                "checkpoint_identity": context.checkpoint.checkpoint_id,
                "precision": self.execution_precision,
                "branch_kind": "fresh_optimization_probe",
            },
        )

    def _execute_objective_unit(
        self,
        loaded: LoadedModel,
        payload: Any,
        *,
        objective: Mapping[str, Any],
        precision: str,
        return_aux: bool,
    ) -> _ExecutedObjective:
        """执行并严格验证一个可由 Base branch 负责 backward 的 objective unit。"""

        context = self._loaded_context(loaded)
        if str(precision).strip().lower() != self.execution_precision:
            raise ValueError(
                "checkpoint objective precision does not match the "
                "HostTaskRuntime execution policy"
            )
        if not isinstance(payload, BatchEnvelope):
            raise TypeError(
                "GenericCheckpointAdapter payload must be a BatchEnvelope"
            )
        self._activate_batch_session(context)
        objective_options = _objective_options(
            runtime=self.runtime,
            requested=objective,
            precision=precision,
            return_aux=return_aux,
        )
        runtime_declared_schedule = "schedule_coordinate" in objective_options
        requested_schedule = objective_options.pop(
            "schedule_coordinate",
            context.objective_schedule_coordinate,
        )
        schedule_required = bool(
            objective_options.pop("requires_schedule_coordinate", False)
        )
        if schedule_required and requested_schedule is None:
            raise ConditionUnavailable(
                "checkpoint_objective_schedule_provenance_unavailable"
            )
        requested_schedule_total, schedule_total_source = (
            _requested_schedule_total(
                objective_options=objective_options,
                context=context,
            )
        )
        execution = self.runtime.objectives.execute(
            context.session.handle,
            payload,
            ObjectiveContext(
                objective_id=str(
                    objective_options.pop(
                        "objective_id",
                        "checkpoint_training_objective",
                    )
                ),
                training=bool(objective_options.pop("training", False)),
                update_index=loaded.checkpoint_update,
                schedule_coordinate=requested_schedule,
                schedule_total=requested_schedule_total,
                normalization_divisor=float(
                    objective_options.pop("normalization_divisor", 1.0)
                ),
                loss_cap=objective_options.pop("loss_cap", None),
                amp_scale=float(objective_options.pop("amp_scale", 1.0)),
                options=objective_options,
                metadata={
                    "checkpoint_identity": context.checkpoint.checkpoint_id,
                    "precision": str(precision),
                },
            ),
        )
        units = iter(execution.units)
        try:
            unit = next(units)
        except StopIteration as error:
            raise ConditionUnavailable(
                "objective_executor_returned_no_units"
            ) from error
        try:
            next(units)
        except StopIteration:
            pass
        else:
            raise ConditionUnavailable(
                "checkpoint_branch_requires_single_objective_unit"
            )
        finally:
            close = getattr(units, "close", None)
            if callable(close):
                close()
        return _ExecutedObjective(
            unit=unit,
            execution=execution,
            schedule_coordinate=requested_schedule,
            schedule_source=(
                "runtime_descriptor.checkpoint_objective"
                if runtime_declared_schedule
                else context.objective_schedule_source
            ),
            schedule_total=requested_schedule_total,
            schedule_total_source=schedule_total_source,
        )

    def module_patch_reference(
        self,
        loaded: LoadedModel,
        *,
        site: Any,
        method: str,
        sample: SampleRef,
        payload: Any,
    ) -> ModulePatchReference:
        """委托可选通用 module-patch capability，缺失时 fail closed。"""

        provider = self.runtime.capabilities.get("module_patch")
        if provider is None:
            raise ConditionUnavailable(
                f"{method}_requires_runtime_module_patch_capability"
            )
        context = self._loaded_context(loaded)
        try:
            reference = provider.reference(
                session=context.session,
                site=site,
                method=str(method),
                sample=self._sample_refs_by_id.get(sample.sample_id),
                batch=payload,
            )
        except RuntimeError as error:
            code = getattr(error, "code", None)
            if code is None:
                raise
            raise ConditionUnavailable(
                f"runtime_module_patch_unavailable:{code}"
            ) from error
        if isinstance(reference, ModulePatchReference):
            return reference
        replacement = getattr(reference, "replacement", None)
        provenance = getattr(reference, "provenance", {})
        if replacement is None or not isinstance(provenance, Mapping):
            raise TypeError(
                "module_patch capability must return replacement and provenance"
            )
        return ModulePatchReference(
            replacement=replacement,
            provenance=dict(provenance),
        )

    def barrier(self) -> None:
        """调用可选 distributed capability；单进程 runtime 为 no-op。"""

        distributed = self.runtime.capabilities.get("distributed")
        barrier = getattr(distributed, "barrier", None)
        if callable(barrier):
            barrier()

    def close_loaded_model(self, loaded: LoadedModel) -> None:
        """幂等关闭该 LoadedModel 对应的 owned Host Runtime session。"""

        context = loaded.adapter_context
        if not isinstance(context, _LoadedRuntimeContext):
            raise TypeError("LoadedModel was not created by this adapter")
        context.session.close()
        self._open_sessions.pop(id(context.session), None)
        self._component_catalog_by_session.pop(id(context.session), None)

    def close(self) -> None:
        """关闭 checkpoint sessions，并按显式 ownership 收口 runtime 资源。"""

        if self._closed:
            return
        self._closed = True
        for session in tuple(self._open_sessions.values()):
            session.close()
        self._open_sessions.clear()
        self._component_catalog_by_session.clear()
        self._frozen_component_catalog = None
        if self._owns_runtime:
            lifecycle = self.runtime.capabilities.get("runtime_lifecycle")
            close = getattr(lifecycle, "close", None)
            if callable(close):
                close()

    def _loaded_context(self, loaded: LoadedModel) -> _LoadedRuntimeContext:
        if self._closed:
            raise RuntimeError("GenericCheckpointAdapter is closed")
        context = loaded.adapter_context
        if not isinstance(context, _LoadedRuntimeContext):
            raise TypeError("LoadedModel does not carry a Host Runtime session")
        if context.session.closed:
            raise RuntimeError("checkpoint session is already closed")
        return context

    def _activate_batch_session(
        self,
        context: _LoadedRuntimeContext,
    ) -> None:
        """在依赖 model context 的 provider 操作前显式切换 active session。"""

        activate = getattr(self.runtime.batches, "activate_session", None)
        if callable(activate):
            activate(context.session)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("GenericCheckpointAdapter is closed")


def _checkpoint_runtime_descriptor(
    runtime: HostTaskRuntime,
) -> CheckpointRuntimeDescriptor:
    capabilities = (
        {
            "capability_id": "checkpoint_execution_policy",
            "definition_version": 1,
            "metadata": {
                "precision": (
                    runtime.descriptor.checkpoint_objective.precision
                ),
            },
        },
        *tuple(
            {
                "capability_id": binding.descriptor.capability_id,
                "definition_version": binding.descriptor.definition_version,
                "metadata": dict(binding.descriptor.metadata),
            }
            for _name, binding in sorted(runtime.capabilities.entries.items())
        ),
    )
    return CheckpointRuntimeDescriptor.create(
        task_definition_id=runtime.descriptor.task_definition_id,
        task_definition_version=str(
            runtime.descriptor.task_definition_version
        ),
        objective_executor_version=str(
            runtime.descriptor.objective_executor_version
        ),
        capability_descriptors=capabilities,
    )


def _component_catalog_records(
    inputs: Sequence[RuntimeInputRef],
    outputs: Sequence[RuntimeOutputRef],
) -> list[dict[str, Any]]:
    """序列化一次 analysis 的唯一 component identity 事实源。"""

    records = [
        {
            "component_id": ref.component.component_id,
            "semantic_id": ref.component.semantic_id,
            "component_kind": ref.component.component_kind,
            "display_label": ref.component.display_label,
            "component_group": ref.component.component_group,
            "component_index": ref.component.component_index,
            "tensor_path": ref.tensor_path,
            "tensor_axis": ref.tensor_axis,
            "tensor_index": ref.tensor_index,
            "normalization_id": (
                ref.normalization_id
                if isinstance(ref, RuntimeOutputRef)
                else None
            ),
            "metadata": dict(ref.component.metadata),
        }
        for ref in (*inputs, *outputs)
    ]
    records.sort(
        key=lambda row: (
            str(row["component_kind"]),
            str(row["component_id"]),
        )
    )
    component_ids = [str(row["component_id"]) for row in records]
    if len(set(component_ids)) != len(component_ids):
        raise ValueError("component catalog contains duplicate component_id")
    return records


def _distributed_identity(runtime: HostTaskRuntime) -> tuple[int, int]:
    distributed = runtime.capabilities.get("distributed")
    rank = int(getattr(distributed, "rank", 0))
    world_size = int(getattr(distributed, "world_size", 1))
    if rank < 0 or world_size < 1 or rank >= world_size:
        raise ValueError("distributed capability returned invalid rank identity")
    return rank, world_size


def _runtime_device(runtime: HostTaskRuntime) -> torch.device:
    distributed = runtime.capabilities.get("distributed")
    device = getattr(
        distributed,
        "device",
        runtime.descriptor.device,
    )
    return torch.device(str(device))


def _objective_options(
    *,
    runtime: HostTaskRuntime,
    requested: Mapping[str, Any],
    precision: str,
    return_aux: bool,
) -> dict[str, Any]:
    """合并 task-owned objective spec 与安全的诊断附加选项。

    cap、normalization、schedule、AMP 和 objective identity 都属于 Host Runtime；
    recipe 只能要求诸如 ledger 收集的诊断附加行为，不能重定义实际 backward 口径。
    """

    _validate_objective_overrides(requested)
    specification = runtime.descriptor.checkpoint_objective
    defaults = {
        "objective_id": specification.objective_id,
        "training": specification.training,
        "normalization_divisor": specification.normalization_divisor,
        "loss_cap": specification.loss_cap,
        "amp_scale": specification.amp_scale,
        "schedule_total": specification.schedule_total,
        **dict(specification.options),
    }
    options = {**defaults, **dict(requested)}
    nested = options.pop("options", {})
    if not isinstance(nested, Mapping):
        raise TypeError("objective.options must be a mapping")
    options.update(dict(nested))
    # Base owns backward after forward_objective returns. A streaming objective
    # cannot be consumed faithfully through this branch.
    options["prefer_chunked"] = False
    options["return_aux_enabled"] = bool(return_aux)
    options["precision"] = str(precision)
    return options


def _validate_objective_overrides(requested: Mapping[str, Any]) -> None:
    """拒绝 recipe 在顶层或 ``options`` 中覆盖 Host Runtime objective 事实。"""

    nested = requested.get("options", {})
    if not isinstance(nested, Mapping):
        raise TypeError("objective.options must be a mapping")
    protected = sorted(
        (set(requested) | set(nested)) & _RUNTIME_OWNED_OBJECTIVE_FIELDS
    )
    if protected:
        raise ValueError(
            "recipe.objective cannot override HostTaskRuntime provenance fields: "
            f"{protected}"
        )


def _requested_schedule_total(
    *,
    objective_options: dict[str, Any],
    context: _LoadedRuntimeContext,
) -> tuple[int | float | None, str]:
    """按 checkpoint provenance > runtime descriptor 选择 schedule total。"""

    runtime_default = objective_options.pop("schedule_total", None)
    if context.objective_schedule_total is not None:
        return (
            context.objective_schedule_total,
            context.objective_schedule_total_source,
        )
    if runtime_default is not None:
        return runtime_default, "runtime_descriptor.checkpoint_objective"
    return None, "unavailable"


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _objective_schedule_coordinate(
    *,
    provenance: CheckpointTrainingProvenance | None,
    checkpoint_metadata: Mapping[str, Any],
) -> tuple[int | float | str | None, str]:
    """只从已验证 provenance/explicit ref metadata 解析 schedule，不猜文件名。"""

    if provenance is not None:
        for key in (
            "objective_schedule_update",
            "next_objective_schedule_update",
        ):
            value = getattr(provenance, key)
            if value is not None:
                return value, f"checkpoint_training_provenance.{key}"
    for key in (
        "objective_schedule_update",
        "schedule_coordinate",
        "update",
    ):
        if checkpoint_metadata.get(key) is not None:
            return checkpoint_metadata[key], f"checkpoint_ref.metadata.{key}"
    return None, "unavailable"


def _objective_schedule_total(
    *,
    provenance: CheckpointTrainingProvenance | None,
    checkpoint_metadata: Mapping[str, Any],
) -> tuple[int | float | None, str]:
    """解析 schedule 总量；embedded checkpoint provenance 始终优先。"""

    if (
        provenance is not None
        and provenance.training_updates is not None
    ):
        return _schedule_total_value(provenance.training_updates), (
            "checkpoint_training_provenance.training_updates"
        )
    for key in ("training_updates", "schedule_total"):
        if checkpoint_metadata.get(key) is not None:
            return _schedule_total_value(checkpoint_metadata[key]), (
                f"checkpoint_ref.metadata.{key}"
            )
    return None, "unavailable"


def _schedule_total_value(value: Any) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("objective schedule total must be numeric")
    return value


def create_checkpoint_adapter(
    *,
    runtime: HostTaskRuntime,
    run_dir: str | Path,
    owns_runtime: bool = False,
) -> GenericCheckpointAdapter:
    """为任意 HostTaskRuntime 组合 portable checkpoint adapter。

    宿主只负责构造 runtime；analyzer catalog、binding 与 checkpoint engine 翻译均由
    独立包统一拥有，避免每个仓库复制诊断执行流程。
    """

    from .bindings import compose_checkpoint_binding_plan

    path = Path(run_dir).resolve()
    plan = compose_checkpoint_binding_plan(runtime)
    return GenericCheckpointAdapter(
        runtime=runtime,
        run_dir=path,
        analyzer_catalog=plan.compose_catalog(),
        analyzer_bindings=plan.compose_bindings(runtime),
        capability_requirements_by_analyzer=(
            plan.capability_requirements_by_analyzer
        ),
        owns_runtime=bool(owns_runtime),
    )


__all__ = [
    "GenericCheckpointAdapter",
    "create_checkpoint_adapter",
]
