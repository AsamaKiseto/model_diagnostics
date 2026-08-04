"""从通用 ``nn.Module`` 结构发现可干预的 Stage/Block frontier。

本模块只依赖 PyTorch 结构与 package 原生 ``ModuleSite``，不读取宿主类型、模型类名、
module path 白名单或 forward aux。静态候选可先用于注册轻量调用 hook；调用方再把有限
观测窗口内实际执行的 module path 传回同一入口，得到非重叠的最终 Block frontier。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from torch import nn

from ..interventions import ModuleSite


_ATOMIC_TYPES = (
    nn.Linear,
    nn.Bilinear,
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
    nn.Embedding,
    nn.Identity,
    nn.Flatten,
    nn.Unflatten,
    nn.Dropout,
    nn.AlphaDropout,
    nn.LayerNorm,
    nn.GroupNorm,
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
    nn.ReLU,
    nn.ReLU6,
    nn.GELU,
    nn.SiLU,
    nn.ELU,
    nn.LeakyReLU,
    nn.Sigmoid,
    nn.Tanh,
    nn.Softmax,
)

_STRUCTURAL_BLOCK_TYPES = (
    nn.RNNBase,
    nn.MultiheadAttention,
    nn.TransformerEncoderLayer,
    nn.TransformerDecoderLayer,
)

_TRANSPARENT_CONTAINERS = (
    nn.ModuleDict,
    nn.ModuleList,
    nn.Sequential,
    nn.ParameterDict,
    nn.ParameterList,
)

_SEMANTIC_STAGE_TOKENS = (
    ("encoder", ("encoder", "encoding", "tokenizer", "input")),
    ("core", ("core", "trunk", "backbone", "body")),
    ("decoder", ("decoder", "decoding", "readout", "output")),
    ("head", ("head", "predictor", "projection")),
)


@dataclass(frozen=True, slots=True)
class HierarchyDiscoveryPolicy:
    """定义 final-only 调用确认的稳定窗口。"""

    discovery_min_observations: int = 3
    discovery_max_observations: int = 8
    stability_patience: int = 3

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"HierarchyDiscoveryPolicy.{name} must be a positive integer")
        if self.discovery_min_observations > self.discovery_max_observations:
            raise ValueError(
                "discovery_min_observations must not exceed discovery_max_observations"
            )


@dataclass(frozen=True, slots=True)
class HierarchyDiscoveryResult:
    """一次静态或运行时确认后的站点集合与结构化覆盖缺口。"""

    sites: tuple[ModuleSite, ...]
    coverage_gaps: tuple[dict[str, Any], ...]
    candidate_paths: tuple[str, ...]
    observed_module_paths: tuple[str, ...]
    discovery_status: str


@dataclass(frozen=True, slots=True)
class _BlockCandidate:
    path: str
    module: nn.Module
    stage_path: str
    stage_id: str
    stage_name: str
    reason: str
    priority: int
    depth: int


def discover_model_hierarchy(
    model: nn.Module,
    *,
    model_name: str | None = None,
    observed_module_paths: Iterable[str] | None = None,
    overrides: Sequence[ModuleSite] = (),
    policy: HierarchyDiscoveryPolicy | None = None,
) -> HierarchyDiscoveryResult:
    """发现 Stage 与非重叠 Block，并把宿主 override 作为增量站点合并。

    ``observed_module_paths=None`` 返回宽静态候选，适合调用方注册临时 forward hook；
    传入实际调用并集时只保留被执行的自动节点。override 是宿主已验证边界，始终保留，
    但同一路径只产生一个 site。
    """

    if not isinstance(model, nn.Module):
        raise TypeError("discover_model_hierarchy requires an nn.Module")
    if policy is not None and not isinstance(policy, HierarchyDiscoveryPolicy):
        raise TypeError("policy must be a HierarchyDiscoveryPolicy")
    stable_model_name = _stable_segment(model_name or type(model).__name__)
    observed = (
        None
        if observed_module_paths is None
        else {str(path).strip(".") for path in observed_module_paths}
    )
    named_modules = dict(model.named_modules())
    stage_entries = _stage_entries(model)
    stage_sites: list[ModuleSite] = []
    block_candidates: list[_BlockCandidate] = []
    coverage_gaps: list[dict[str, Any]] = []

    for stage_path, stage_module, stage_name in stage_entries:
        stage_id = _site_id("stage", stable_model_name, stage_path)
        stage_called = observed is None or stage_path in observed
        if stage_called:
            stage_sites.append(
                _module_site(
                    site_id=stage_id,
                    path=stage_path,
                    module=stage_module,
                    parent_node_id=f"model:{stable_model_name}",
                    stage_id=stage_id,
                    level="Stage",
                    stage_name=stage_name,
                    source=(
                        "generic_structure"
                        if observed is None
                        else "runtime_observed_generic_structure"
                    ),
                )
            )
        candidates = _block_candidates(
            stage_path,
            stage_module,
            stage_id=stage_id,
            stage_name=stage_name,
        )
        block_candidates.extend(candidates)
        if observed is not None and stage_called and not any(
            candidate.path in observed for candidate in candidates
        ):
            coverage_gaps.append(
                {
                    "stage_id": stage_id,
                    "module_path": stage_path,
                    "reason": "no_executable_block_observed",
                }
            )

    selected_blocks = _select_block_frontier(
        block_candidates,
        observed=observed,
    )
    automatic_sites = [
        *stage_sites,
        *(
            _module_site(
                site_id=_site_id("block", stable_model_name, candidate.path),
                path=candidate.path,
                module=candidate.module,
                parent_node_id=candidate.stage_id,
                stage_id=candidate.stage_id,
                level="Block",
                stage_name=candidate.stage_name,
                source=(
                    f"generic_{candidate.reason}"
                    if observed is None
                    else f"runtime_observed_{candidate.reason}"
                ),
            )
            for candidate in selected_blocks
        ),
    ]
    sites_by_path = {site.module_path: site for site in automatic_sites}
    for override in overrides:
        if not isinstance(override, ModuleSite):
            raise TypeError("hierarchy overrides must contain ModuleSite values")
        if override.module_path not in named_modules:
            raise ValueError(
                "hierarchy override module path does not exist: "
                f"{override.module_path}"
            )
        metadata = {
            **dict(override.metadata),
            "discovery_source": "hierarchy_override",
            "hierarchy_level": str(
                override.metadata.get("hierarchy_level", "Block")
            ),
        }
        sites_by_path[override.module_path] = ModuleSite(
            site_id=override.site_id,
            node_id=override.node_id or override.site_id,
            module_path=override.module_path,
            module_type=override.module_type,
            parent_node_id=(
                override.parent_node_id or f"model:{stable_model_name}"
            ),
            stage_id=override.stage_id,
            invocation_index=override.invocation_index,
            output_path=override.output_path,
            alias_node_ids=override.alias_node_ids,
            metadata=metadata,
        )

    candidate_paths = tuple(
        sorted(
            {
                *(path for path, _module, _name in stage_entries),
                *(candidate.path for candidate in block_candidates),
                *(site.module_path for site in overrides),
            }
        )
    )
    return HierarchyDiscoveryResult(
        sites=tuple(
            sorted(
                sites_by_path.values(),
                key=lambda site: (
                    0 if site.metadata.get("hierarchy_level") == "Stage" else 1,
                    site.module_path,
                ),
            )
        ),
        coverage_gaps=tuple(coverage_gaps),
        candidate_paths=candidate_paths,
        observed_module_paths=tuple(sorted(observed or ())),
        discovery_status="static_candidates" if observed is None else "runtime_confirmed",
    )


def _stage_entries(model: nn.Module) -> list[tuple[str, nn.Module, str]]:
    """把顶层语义模块或其直接 routed ModuleDict entry 作为 Stage。"""

    entries: list[tuple[str, nn.Module, str]] = []
    for child_name, child in model.named_children():
        if _is_atomic(child) or not _has_trainable_content(child):
            continue
        routed: list[tuple[str, nn.Module, str]] = []
        if isinstance(child, nn.ModuleDict):
            for route_name, route_module in child.items():
                if _is_atomic(route_module) or not _has_trainable_content(route_module):
                    continue
                routed.append(
                    (
                        f"{child_name}.{route_name}",
                        route_module,
                        _semantic_stage(child_name),
                    )
                )
        for router_name, router in child.named_children():
            if not isinstance(router, nn.ModuleDict):
                continue
            for route_name, route_module in router.items():
                if _is_atomic(route_module) or not _has_trainable_content(route_module):
                    continue
                path = f"{child_name}.{router_name}.{route_name}"
                routed.append((path, route_module, _semantic_stage(child_name)))
        entries.extend(routed or [(child_name, child, _semantic_stage(child_name))])
    if not entries and _has_trainable_content(model):
        entries.append(("", model, "core"))
    return entries


def _block_candidates(
    stage_path: str,
    stage_module: nn.Module,
    *,
    stage_id: str,
    stage_name: str,
) -> list[_BlockCandidate]:
    """递归产生结构候选；原子层与纯容器永不成为自动 Block。"""

    candidates: dict[str, _BlockCandidate] = {}

    def visit(module: nn.Module, path: str, depth: int, routed: bool, repeated: bool) -> None:
        executable = type(module).forward is not nn.Module.forward
        if path != stage_path and not _is_atomic(module) and _has_trainable_content(module):
            reason: str | None = None
            priority = 99
            if isinstance(module, _STRUCTURAL_BLOCK_TYPES):
                reason, priority = "structural", 0
            elif repeated and executable:
                reason, priority = "repeated", 1
            elif routed and executable:
                reason, priority = "routed_component", 2
            elif _is_leaf_composite(module):
                reason, priority = "leaf_composite", 3
            elif executable and not isinstance(module, _TRANSPARENT_CONTAINERS):
                reason, priority = "composite_fallback", 4
            if reason is not None:
                candidates[path] = _BlockCandidate(
                    path=path,
                    module=module,
                    stage_path=stage_path,
                    stage_id=stage_id,
                    stage_name=stage_name,
                    reason=reason,
                    priority=priority,
                    depth=depth,
                )
        for name, child in module.named_children():
            child_path = f"{path}.{name}" if path else name
            if _is_atomic(child):
                continue
            visit(
                child,
                child_path,
                depth + 1,
                isinstance(module, nn.ModuleDict),
                isinstance(module, (nn.ModuleList, nn.Sequential)),
            )

    visit(stage_module, stage_path, 0, False, False)
    return list(candidates.values())


def _select_block_frontier(
    candidates: Sequence[_BlockCandidate],
    *,
    observed: set[str] | None,
) -> list[_BlockCandidate]:
    """按结构优先级选择同一 Stage 内不互相包含的实际执行 Block。"""

    eligible = [
        candidate
        for candidate in candidates
        if observed is None or candidate.path in observed
    ]
    selected: list[_BlockCandidate] = []
    for candidate in sorted(
        eligible,
        key=lambda item: (item.priority, item.depth, item.path),
    ):
        if any(
            item.stage_id == candidate.stage_id
            and _paths_overlap(item.path, candidate.path)
            for item in selected
        ):
            continue
        selected.append(candidate)
    return selected


def _module_site(
    *,
    site_id: str,
    path: str,
    module: nn.Module,
    parent_node_id: str,
    stage_id: str,
    level: str,
    stage_name: str,
    source: str,
) -> ModuleSite:
    return ModuleSite(
        site_id=site_id,
        node_id=site_id,
        module_path=path,
        module_type=type(module).__name__,
        parent_node_id=parent_node_id,
        stage_id=stage_id,
        alias_node_ids=(site_id,),
        metadata={
            "discovery_source": source,
            "hierarchy_level": level,
            "stage": stage_name,
        },
    )


def _has_trainable_content(module: nn.Module) -> bool:
    return any(parameter.requires_grad for parameter in module.parameters())


def _is_leaf_composite(module: nn.Module) -> bool:
    if _is_atomic(module) or isinstance(module, _TRANSPARENT_CONTAINERS):
        return False
    children = tuple(module.children())
    return bool(children) and all(_contains_only_atomic_leaves(child) for child in children)


def _contains_only_atomic_leaves(module: nn.Module) -> bool:
    if _is_atomic(module):
        return True
    if not isinstance(module, _TRANSPARENT_CONTAINERS):
        return False
    return all(_contains_only_atomic_leaves(child) for child in module.children())


def _is_atomic(module: nn.Module) -> bool:
    return isinstance(module, _ATOMIC_TYPES)


def _paths_overlap(left: str, right: str) -> bool:
    if left == right:
        return True
    if not left or not right:
        return True
    return left.startswith(f"{right}.") or right.startswith(f"{left}.")


def _semantic_stage(name: str) -> str:
    lowered = str(name).lower()
    for stage, tokens in _SEMANTIC_STAGE_TOKENS:
        if any(token in lowered for token in tokens):
            return stage
    return _stable_segment(name)


def _site_id(level: str, model_name: str, path: str) -> str:
    digest = sha256(
        f"{level}:{model_name}:{path}".encode("utf-8")
    ).hexdigest()[:16]
    return f"{level}:{digest}"


def _stable_segment(value: str) -> str:
    return "".join(
        character.lower() if character.isalnum() else "_"
        for character in str(value)
    ).strip("_") or "unnamed"


__all__ = [
    "HierarchyDiscoveryPolicy",
    "HierarchyDiscoveryResult",
    "discover_model_hierarchy",
]
