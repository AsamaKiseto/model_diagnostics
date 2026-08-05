"""声明最终 input × output 预测依赖与多尺度敏感度证据。"""

from __future__ import annotations

import math

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
    AnalyzerDefinition(
        name="final_input_sensitivity",
        evidence_kind="predictive_input_sensitivity",
        execution_mode="aggregate",
        definition_version=1,
        claim_boundaries=(
            "symmetric relative response is specific to the declared scale "
            "and perturbation distribution",
            "input-output sensitivity does not establish physical causality",
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
    EvidenceRendererDefinition(
        evidence_kind="predictive_input_sensitivity",
        title="输入扰动敏感度",
        analyzer_definition_version=1,
        metrics=(
            EvidenceMetric(
                "symmetric_relative_output_response",
                "对称相对输出响应",
                role="intervention",
                visualization="matrix",
                category="input",
                priority="P0",
                description=(
                    "对输入通道 i 在模型实际输入坐标中施加幅值为 s 的确定性扰动，"
                    "并以正负两个方向配对。对每个方向 d，令 y0 为未扰动输出、"
                    "yd 为扰动后输出，则 R_ij,d(s) = 2 × RMS(yd - y0) / "
                    "[RMS(yd) + RMS(y0)]。先在同一 validity mask 上计算每个样本的"
                    "R_ij,d，再以 S_ij(s) = RMS(R_ij,-, R_ij,+) 配对，并对 cohort "
                    "做等样本 RMS。报告字段 symmetric_relative_output_response "
                    "就是 S_ij(s)。输入扰动 RMS、训练集标准差和整段样本特征都"
                    "不进入这个分母。"
                ),
                reading=(
                    "在固定输入 i、输出 j 时查看 S_ij(s) 随 s 的曲线：近似水平"
                    "表示输出相对自身幅度的变化比例在该尺度范围内接近不变；随"
                    "尺度上升表示相对响应增强，随尺度下降表示相对响应减弱。"
                    "同一尺度的矩阵用于比较哪些输入令哪些输出发生更强的相对变化。"
                ),
                reference=(
                    "S_ij(s) 的理论范围是 0 到 2。0 表示扰动前后输出相同；数值"
                    "越大表示变化相对于输出自身 RMS 越强。它没有正负号，也没有"
                    "统一的正常/异常阈值；当扰动前后输出都接近 0 时，应谨慎解释"
                    "较大的相对响应。"
                ),
                invalid_when=(
                    "扰动前后输出没有使用相同输出坐标、validity mask 和 support，"
                    "正负方向未配对，不同尺度使用了不同扰动方向，有效 support "
                    "为空，或者扰动已经严重离开模型训练分布。不同输入坐标本身"
                    "量纲不一致时，也不能仅凭相同数值 s 比较输入的物理敏感度。"
                ),
            ),
        ),
        status_fields=("status",),
    ),
)
REPORT_RENDERER_CATALOG = compose_renderer_catalogs(
    REPORT_RENDERER_DEFINITIONS
)


def symmetric_relative_output_response(
    *,
    baseline_output_square_sum: float,
    condition_output_square_sum: float,
    output_difference_square_sum: float,
    support_count: int,
) -> float:
    """计算单个扰动方向相对于输出自身幅度的对称 RMS 响应。

    三个平方和必须在同一有效元素集合上计算。分母同时包含 baseline 和
    condition RMS，因此无需训练集统计、逐 shot 特征或输入扰动 RMS；两个输出均为
    0 时按无响应返回 0。
    """

    sums = (
        float(baseline_output_square_sum),
        float(condition_output_square_sum),
        float(output_difference_square_sum),
    )
    count = int(support_count)
    if count <= 0 or any(not math.isfinite(value) or value < 0 for value in sums):
        raise ValueError(
            "symmetric relative response requires finite non-empty support"
        )
    baseline_rms, condition_rms, difference_rms = (
        math.sqrt(value / count) for value in sums
    )
    denominator = baseline_rms + condition_rms
    if denominator == 0:
        if difference_rms == 0:
            return 0.0
        raise ValueError("zero output magnitude cannot have a non-zero difference")
    return 2.0 * difference_rms / denominator


__all__ = [
    "ANALYZER_CATALOG",
    "ANALYZER_DEFINITIONS",
    "REPORT_RENDERER_CATALOG",
    "REPORT_RENDERER_DEFINITIONS",
    "symmetric_relative_output_response",
]
