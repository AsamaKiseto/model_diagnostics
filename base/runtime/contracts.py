"""三阶段诊断仍使用的精简运行策略和 objective provenance。"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class ObjectiveTrace:
    """描述真实 backward objective 的 identity 与缩放口径。

    该 DTO 只表达来源，不执行 backward，也不把 local Taylor 解释为完整路径归因。
    """

    objective_identity: str
    normalization_divisor: float = 1.0
    loss_cap_enabled: bool = False
    loss_cap_value: float | None = None
    loss_cap_hit: bool = False
    amp_scale: float = 1.0
    chunk_identity: str | None = None
    segment_identity: str | None = None
    cotangent_identity: str | None = None

    def __post_init__(self) -> None:
        identity = str(self.objective_identity).strip()
        if not identity:
            raise ValueError("objective_identity must be non-empty")
        object.__setattr__(self, "objective_identity", identity)
        if (
            not math.isfinite(float(self.normalization_divisor))
            or float(self.normalization_divisor) <= 0
        ):
            raise ValueError("normalization_divisor must be finite and positive")
        if not math.isfinite(float(self.amp_scale)) or float(self.amp_scale) <= 0:
            raise ValueError("amp_scale must be finite and positive")
        if self.loss_cap_value is not None and (
            not math.isfinite(float(self.loss_cap_value))
            or float(self.loss_cap_value) <= 0
        ):
            raise ValueError("loss_cap_value must be finite and positive")


__all__ = ["ObjectiveTrace"]
