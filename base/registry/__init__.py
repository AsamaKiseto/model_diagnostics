"""定义无全局副作用的 analyzer 描述符与显式 catalog 组合。"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class AnalyzerDefinition:
    """描述 analyzer 的证据边界、执行形态和严格 option schema。"""

    name: str
    evidence_kind: str
    execution_mode: str
    option_keys: frozenset[str] = frozenset()
    definition_version: int = 1
    claim_boundaries: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        name = str(self.name).strip().lower()
        if not name:
            raise ValueError("analyzer name must not be empty")
        if not str(self.evidence_kind).strip():
            raise ValueError("analyzer evidence_kind must not be empty")
        execution_mode = str(self.execution_mode).strip().lower()
        if execution_mode not in {"condition", "aggregate", "stream"}:
            raise ValueError(
                "analyzer execution_mode must be condition, aggregate, or stream"
            )
        if int(self.definition_version) < 1:
            raise ValueError("analyzer definition_version must be positive")
        option_keys = frozenset(str(key).strip() for key in self.option_keys)
        if "" in option_keys:
            raise ValueError("analyzer option_keys must not contain empty keys")
        boundaries = tuple(
            str(boundary).strip() for boundary in self.claim_boundaries
        )
        if any(not boundary for boundary in boundaries):
            raise ValueError("analyzer claim_boundaries must not contain empty values")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "execution_mode", execution_mode)
        object.__setattr__(self, "option_keys", option_keys)
        object.__setattr__(self, "claim_boundaries", boundaries)

    def to_manifest(self) -> dict[str, Any]:
        """返回 validator 与 report 可独立消费的稳定描述符。"""

        return {
            "name": self.name,
            "evidence_kind": self.evidence_kind,
            "execution_mode": self.execution_mode,
            "option_keys": sorted(self.option_keys),
            "definition_version": int(self.definition_version),
            "claim_boundaries": list(self.claim_boundaries),
        }


@dataclass(frozen=True, slots=True)
class AnalyzerExecutionResult:
    """声明 aggregate analyzer 的完整 evidence inventory。

    runner 只返回 detail rows；checkpoint engine 独占 terminal。每个 detail row 的
    ``evidence_key`` 必须属于 ``expected_evidence_keys``，使 validator 能区分真实
    skip 与 producer 未生成的缺口。
    """

    rows: tuple[Mapping[str, Any], ...]
    expected_evidence_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        rows = tuple(dict(row) for row in self.rows)
        expected = tuple(
            str(key).strip() for key in self.expected_evidence_keys
        )
        if not expected or any(not key for key in expected):
            raise ValueError(
                "expected_evidence_keys must contain non-empty keys"
            )
        if len(set(expected)) != len(expected):
            raise ValueError("expected_evidence_keys must be unique")
        object.__setattr__(
            self,
            "rows",
            tuple(MappingProxyType(row) for row in rows),
        )
        object.__setattr__(self, "expected_evidence_keys", expected)


class AnalyzerCapabilityUnavailable(RuntimeError):
    """表示 runner 缺少宿主可选 capability，应产生 structured skip。"""


@dataclass(frozen=True, slots=True)
class AnalyzerRunContext:
    """向显式 analyzer runner 传递一次 checkpoint-conditioned 执行上下文。

    Base 只装配任务无关的 checkpoint、sample、recipe 与 adapter capability；runner
    自行决定需要哪些可选 capability。上下文不触发注册，也不保存跨 condition 状态。
    """

    definition: AnalyzerDefinition
    loaded: Any
    samples: tuple[Any, ...]
    options: Mapping[str, Any]
    adapter: Any
    recipe: Any
    checkpoint: Any
    execution_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.definition, AnalyzerDefinition):
            raise TypeError("AnalyzerRunContext.definition must be AnalyzerDefinition")
        object.__setattr__(self, "samples", tuple(self.samples))
        object.__setattr__(
            self,
            "options",
            MappingProxyType(dict(self.options)),
        )
        object.__setattr__(
            self,
            "execution_metadata",
            MappingProxyType(dict(self.execution_metadata)),
        )

    def require_capability(self, name: str) -> Callable[..., Any]:
        """取得宿主显式 capability；缺失时 fail-fast 而不是猜测任务语义。"""

        normalized = str(name).strip()
        capability = getattr(self.adapter, normalized, None)
        if not normalized or not callable(capability):
            raise AnalyzerCapabilityUnavailable(
                f"analyzer {self.definition.name!r} requires adapter capability "
                f"{normalized or '<empty>'!r}"
            )
        return capability


@dataclass(frozen=True, slots=True)
class AnalyzerBinding:
    """把一个 versioned analyzer definition 与纯显式 runner 绑定。"""

    definition: AnalyzerDefinition
    runner: Callable[[AnalyzerRunContext], AnalyzerExecutionResult]

    def __post_init__(self) -> None:
        if not isinstance(self.definition, AnalyzerDefinition):
            raise TypeError("AnalyzerBinding.definition must be AnalyzerDefinition")
        if not callable(self.runner):
            raise TypeError("AnalyzerBinding.runner must be callable")

    @property
    def name(self) -> str:
        """返回与 definition catalog 共用的稳定 analyzer 名称。"""

        return self.definition.name

    def run(self, context: AnalyzerRunContext) -> AnalyzerExecutionResult:
        """执行 runner，并拒绝 descriptor 版本错配。"""

        if context.definition != self.definition:
            raise ValueError(
                f"analyzer binding definition mismatch for {self.name!r}"
            )
        return self.runner(context)


@dataclass(frozen=True, slots=True)
class AnalyzerBindingCatalog:
    """保存一次运行可用的 analyzer runners，不执行 import-time 注册。"""

    bindings: Mapping[str, AnalyzerBinding]

    def __post_init__(self) -> None:
        normalized: dict[str, AnalyzerBinding] = {}
        for raw_name, binding in self.bindings.items():
            if not isinstance(binding, AnalyzerBinding):
                raise TypeError(
                    "binding catalog values must be AnalyzerBinding instances"
                )
            name = str(raw_name).strip().lower()
            if name != binding.name:
                raise ValueError(
                    f"analyzer binding catalog key does not match binding name: "
                    f"{raw_name}"
                )
            if name in normalized:
                raise ValueError(f"duplicate analyzer binding: {name}")
            normalized[name] = binding
        object.__setattr__(self, "bindings", MappingProxyType(normalized))

    def get(self, name: str) -> AnalyzerBinding | None:
        """按 analyzer 名称取得 runner binding；未绑定时返回 ``None``。"""

        return self.bindings.get(str(name).strip().lower())

    def validate_against(self, catalog: "AnalyzerCatalog") -> None:
        """确认每个 runner 与有效 descriptor catalog 使用同一 versioned definition。"""

        for name, binding in self.bindings.items():
            if catalog.definition(name) != binding.definition:
                raise ValueError(
                    f"analyzer binding definition does not match catalog: {name}"
                )


@dataclass(frozen=True, slots=True)
class AnalyzerCatalog:
    """保存一次运行允许的 analyzer 集合，不触发 import-time 注册。"""

    definitions: Mapping[str, AnalyzerDefinition]

    def __post_init__(self) -> None:
        normalized: dict[str, AnalyzerDefinition] = {}
        for raw_name, definition in self.definitions.items():
            if not isinstance(definition, AnalyzerDefinition):
                raise TypeError("catalog definitions must be AnalyzerDefinition values")
            name = str(raw_name).strip().lower()
            if name != definition.name:
                raise ValueError(
                    f"analyzer catalog key does not match definition name: {raw_name}"
                )
            if name in normalized:
                raise ValueError(f"duplicate analyzer definition: {name}")
            normalized[name] = definition
        object.__setattr__(self, "definitions", MappingProxyType(normalized))

    def definition(self, name: str) -> AnalyzerDefinition:
        """取得 analyzer 定义；未注册名称立即失败。"""

        normalized = str(name).strip().lower()
        try:
            return self.definitions[normalized]
        except KeyError as error:
            raise ValueError(f"Unknown analyzer: {normalized}") from error

    def validate(
        self,
        analyzers: Sequence[str],
        options_by_name: Mapping[str, Mapping[str, Any]],
    ) -> None:
        """严格校验 analyzer 选择和各自 option keys。"""

        selected = tuple(str(name).strip().lower() for name in analyzers)
        if not selected:
            raise ValueError("analyzers must select at least one analyzer")
        if len(set(selected)) != len(selected):
            raise ValueError("analyzers must not contain duplicates")
        unknown_option_owners = sorted(
            set(str(name).strip().lower() for name in options_by_name) - set(selected)
        )
        if unknown_option_owners:
            raise ValueError(
                "options provided for unselected analyzers: "
                f"{unknown_option_owners}"
            )
        for name in selected:
            definition = self.definition(name)
            options = dict(options_by_name.get(name, {}) or {})
            unknown = sorted(set(str(key) for key in options) - definition.option_keys)
            if unknown:
                raise ValueError(f"Unknown analyzers.{name} fields: {unknown}")

    def selected_descriptors(
        self,
        analyzers: Sequence[str],
    ) -> list[dict[str, Any]]:
        """按 recipe 顺序返回 manifest descriptors。"""

        return [self.definition(name).to_manifest() for name in analyzers]


def compose_catalogs(
    *catalogs: AnalyzerCatalog | Iterable[AnalyzerDefinition],
) -> AnalyzerCatalog:
    """显式合并 Base、extension 与宿主 analyzer，重复名称立即失败。"""

    merged: dict[str, AnalyzerDefinition] = {}
    for source in catalogs:
        definitions = (
            source.definitions.values()
            if isinstance(source, AnalyzerCatalog)
            else source
        )
        for definition in definitions:
            if definition.name in merged:
                raise ValueError(
                    f"analyzer definition already registered: {definition.name}"
                )
            merged[definition.name] = definition
    return AnalyzerCatalog(merged)


def compose_bindings(
    *catalogs: AnalyzerBindingCatalog | Iterable[AnalyzerBinding],
) -> AnalyzerBindingCatalog:
    """显式合并 extension 与宿主 runner bindings，重复名称立即失败。"""

    merged: dict[str, AnalyzerBinding] = {}
    for source in catalogs:
        bindings = (
            source.bindings.values()
            if isinstance(source, AnalyzerBindingCatalog)
            else source
        )
        for binding in bindings:
            if not isinstance(binding, AnalyzerBinding):
                raise TypeError(
                    "compose_bindings inputs must contain AnalyzerBinding values"
                )
            if binding.name in merged:
                raise ValueError(
                    f"analyzer binding already registered: {binding.name}"
                )
            merged[binding.name] = binding
    return AnalyzerBindingCatalog(merged)


def bind_adapter_capability(
    definition: AnalyzerDefinition,
    capability_name: str,
) -> AnalyzerBinding:
    """把 aggregate analyzer 绑定到同签名的显式 adapter capability。

    capability 接收 ``loaded/samples/options`` 三个 keyword。该 helper 只负责通用调用
    约定，不决定任务数据、target、scenario 或 analyzer 结果。
    """

    normalized = str(capability_name).strip()
    if not normalized:
        raise ValueError("capability_name must not be empty")

    def run(context: AnalyzerRunContext) -> AnalyzerExecutionResult:
        capability = context.require_capability(normalized)
        result = capability(
            loaded=context.loaded,
            samples=context.samples,
            options=context.options,
        )
        if not isinstance(result, AnalyzerExecutionResult):
            raise TypeError(
                f"capability {normalized!r} must return AnalyzerExecutionResult"
            )
        return result

    return AnalyzerBinding(definition=definition, runner=run)


BASE_ANALYZER_CATALOG = AnalyzerCatalog(
    {
        "checkpoint_sweep": AnalyzerDefinition(
            name="checkpoint_sweep",
            evidence_kind="checkpoint_training_health",
            execution_mode="aggregate",
            option_keys=frozenset(),
            definition_version=1,
            claim_boundaries=(
                "checkpoint sweeps monitor training health and do not measure module or input influence",
                "checkpoint observations do not identify physical causality",
            ),
        ),
        "final_module_influence": AnalyzerDefinition(
            name="final_module_influence",
            evidence_kind="checkpoint_conditioned_activation_intervention",
            execution_mode="condition",
            option_keys=frozenset(),
            definition_version=2,
            claim_boundaries=(
                "local Taylor is a local first-order approximation",
                "module intervention effects use response-only evaluation and are not training-loss deltas or physical causality",
            ),
        )
    }
)

FLIGHT_RECORDER_ANALYZER_DEFINITION = AnalyzerDefinition(
    name="flight_recorder",
    evidence_kind="training_attempt_provenance",
    execution_mode="stream",
    definition_version=2,
    claim_boundaries=(
        "flight-recorder rows preserve observed training facts, not a causal root cause",
        "parameter localization is only performed after a nonfinite-gradient trigger",
        "per-output objective trends are optimization evidence, not physical causality",
    ),
)

__all__ = [
    "AnalyzerBinding",
    "AnalyzerBindingCatalog",
    "AnalyzerCapabilityUnavailable",
    "AnalyzerCatalog",
    "AnalyzerDefinition",
    "AnalyzerExecutionResult",
    "AnalyzerRunContext",
    "BASE_ANALYZER_CATALOG",
    "FLIGHT_RECORDER_ANALYZER_DEFINITION",
    "bind_adapter_capability",
    "compose_bindings",
    "compose_catalogs",
]
