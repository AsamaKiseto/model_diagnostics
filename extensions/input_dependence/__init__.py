"""声明最终 input × output 预测依赖证据及其报告口径。"""

from __future__ import annotations

from model_diagnostics.base import (
    AnalyzerDefinition,
    EvidenceMetric,
    EvidenceRendererDefinition,
    compose_catalogs,
    compose_renderer_catalogs,
)


ANALYZER_DEFINITIONS = (
    AnalyzerDefinition(
        name="final_channel_influence",
        evidence_kind="predictive_dependence_intervention",
        execution_mode="aggregate",
        definition_version=1,
        claim_boundaries=(
            "mean replacement measures model response under that intervention",
            "input-output response does not establish physical causality",
        ),
    ),
)
ANALYZER_CATALOG = compose_catalogs(ANALYZER_DEFINITIONS)

REPORT_RENDERER_DEFINITIONS = (
    EvidenceRendererDefinition(
        evidence_kind="predictive_dependence_intervention",
        title="输入通道对输出通道的影响",
        analyzer_definition_version=1,
        metrics=(
            EvidenceMetric(
                "effect_value",
                "输入对输出的影响",
                role="intervention",
                visualization="matrix",
                category="input",
                priority="P0",
                description=(
                    "记未替换输入时的输出评价指标为 B，替换指定输入通道后的同一"
                    "指标为 C。对于越小越好的指标，effect_value = C - B；对于"
                    "越大越好的指标，effect_value = B - C。因此 effect_value "
                    "始终满足：正值表示替换后任务表现变差，负值表示改善，0 表示"
                    "在该指标精度下没有测得响应。"
                ),
                reading=(
                    "固定 replacement、cohort、输出指标和归一化方式，比较不同"
                    "输入通道对同一输出的响应幅度，或比较同一输入影响了哪些输出。"
                ),
                reference=(
                    "恒等替换对照应接近 0；实际替换只有明显超过恒等对照波动时"
                    "才说明该干预产生了可测的模型响应。"
                ),
                invalid_when=(
                    "基线与干预没有使用同一样本、support、输出归一化或 validity "
                    "mask，或者替换产生严重分布外输入。"
                ),
            ),
            EvidenceMetric(
                "normalized_effect",
                "输入对输出的相对影响",
                role="context",
                visualization="matrix",
                category="input",
                priority="P1",
                description=(
                    "normalized_effect = effect_value / "
                    "max(abs(B), 1e-12)，其中 B 是同一样本、同一输出指标的基线值。"
                    "它表示干预响应相对基线指标量级的大小。"
                ),
                reading=(
                    "用于比较原始指标量级不同的输出，但只应在输出评价指标语义、"
                    "normalization 和 cohort 一致时比较。"
                ),
                reference="恒等替换对照应接近 0；绝对值越大表示相对基线的响应越强。",
                invalid_when=(
                    "基线值接近 0 时分母会使用 1e-12 下限，结果可能被极度放大；"
                    "不同 metric 的相对响应也不能直接比较。"
                ),
            ),
        ),
        null_control_fields=("control_status",),
        status_fields=("status",),
    ),
)
REPORT_RENDERER_CATALOG = compose_renderer_catalogs(
    REPORT_RENDERER_DEFINITIONS
)


__all__ = [
    "ANALYZER_CATALOG",
    "ANALYZER_DEFINITIONS",
    "REPORT_RENDERER_CATALOG",
    "REPORT_RENDERER_DEFINITIONS",
]
