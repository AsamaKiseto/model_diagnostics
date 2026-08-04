"""通用诊断图的只读视觉参考，不属于 active diagnostics runtime。

这里仅保留原报告常用的颜色、网格、ranked bar 和 trend line 呈现方式。调用方需要自行
提供已经整理好的 labels/values；本文件不读取 artifact、不加载 checkpoint、不执行诊断。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


PALETTE = {
    "accent": "#5267df",
    "accent_secondary": "#16a085",
    "warning": "#d9822b",
    "critical": "#c23b4a",
    "grid": "#dce2ec",
    "ink": "#172033",
}


def plot_ranked_bars(ax: Any, labels: Sequence[str], values: Sequence[float], *, title: str) -> None:
    """保留横向影响排名的视觉形式；数据 contract 由使用者决定。"""

    positions = list(range(len(labels)))
    colors = [PALETTE["accent"] if value >= 0 else PALETTE["critical"] for value in values]
    ax.barh(positions, values, color=colors, alpha=0.9)
    ax.set_yticks(positions, labels)
    ax.axvline(0.0, color=PALETTE["ink"], linewidth=0.8)
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.7, alpha=0.8)
    ax.set_title(title, loc="left", fontweight="semibold")


def plot_metric_trends(ax: Any, series: Mapping[str, Sequence[float]], *, title: str) -> None:
    """保留多节点趋势线和紧凑图例样式，不绑定任何指标字段。"""

    for label, values in series.items():
        ax.plot(range(len(values)), values, linewidth=1.5, label=label)
    ax.grid(color=PALETTE["grid"], linewidth=0.7, alpha=0.8)
    ax.set_title(title, loc="left", fontweight="semibold")
    ax.legend(frameon=False, fontsize=8, ncols=2)
