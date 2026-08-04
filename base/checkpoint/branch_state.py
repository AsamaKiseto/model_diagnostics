"""
提供 checkpoint condition 共用的 canonical model restore 与 paired RNG 隔离。
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
import random

import torch


def capture_canonical_model_state(
    model: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    """复制完整 parameter/buffer state，供多个隔离 condition 反复恢复。"""

    return {
        name: value.detach().to("cpu").clone()
        for name, value in model.state_dict().items()
    }


def restore_canonical_model_state(
    model: torch.nn.Module,
    state: Mapping[str, torch.Tensor],
) -> None:
    """严格恢复 canonical state；任何缺失或额外 key 都视为隔离失败。"""

    result = model.load_state_dict(dict(state), strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("canonical model state restore was not exact")


@contextmanager
def paired_branch_rng(seed: int, device: torch.device) -> Iterator[None]:
    """隔离 Python/PyTorch RNG，使 baseline 与 intervention 可使用同一随机流。"""

    python_state = random.getstate()
    devices = (
        [
            device.index
            if device.index is not None
            else torch.cuda.current_device()
        ]
        if device.type == "cuda" and torch.cuda.is_available()
        else []
    )
    try:
        with torch.random.fork_rng(devices=devices, enabled=True):
            random.seed(int(seed))
            torch.manual_seed(int(seed))
            if devices:
                torch.cuda.manual_seed_all(int(seed))
            yield
    finally:
        random.setstate(python_state)


__all__ = [
    "capture_canonical_model_state",
    "paired_branch_rng",
    "restore_canonical_model_state",
]
