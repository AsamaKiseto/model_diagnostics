"""
采集精确 module invocation/output-path activation，并可选保留对应 gradient。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch

from ..interventions import (
    ModuleSite,
    canonical_activation_output_path,
    iter_module_output_tensors,
)
from .contracts import (
    build_activation_record_identity,
    build_activation_site_identity,
)


@dataclass(slots=True)
class ActivationCaptureSample:
    """保存一个精确 activation site 的完整 CPU tensor 与可选 gradient。"""

    activation_site: dict[str, Any]
    module_site_id: str
    invocation_index: int
    output_path: str
    activation: torch.Tensor
    gradient: torch.Tensor | None
    source_numel: int
    checkpoint_recompute_observed: bool = False


class ModuleActivationCapture:
    """观察一个 module 的全部精确 invocation/output Tensor site。

    ``ModuleSite`` 可以约束 invocation 或 output path；未约束时仍为每个实际 site
    生成独立 identity，禁止把 structured outputs 拼接或让 repeated invocation
    覆盖。调用方应在原始 forward 结束后调用 ``mark_forward_complete()``，这样
    activation checkpoint 的 backward recompute 会复用逻辑 invocation identity，
    但不会伪装成新的 forward invocation。
    """

    def __init__(
        self,
        module: torch.nn.Module,
        site: ModuleSite,
        *,
        capture_gradients: bool,
    ) -> None:
        self._module = module
        self._site = site
        self._capture_gradients = bool(capture_gradients)
        self._raw_call_count = 0
        self._forward_invocation_count: int | None = None
        self._handle: Any = None
        self._gradient_handles: list[Any] = []
        self.samples: dict[tuple[int, str], ActivationCaptureSample] = {}

    def __enter__(self) -> "ModuleActivationCapture":
        self._handle = self._module.register_forward_hook(self._observe)
        return self

    def mark_forward_complete(self) -> None:
        """冻结原始 forward invocation 周期，供 backward recompute 对齐。"""

        if self._forward_invocation_count is None:
            self._forward_invocation_count = self._raw_call_count

    def _observe(
        self,
        _module: torch.nn.Module,
        _inputs: tuple[Any, ...],
        output: Any,
    ) -> None:
        raw_invocation = self._raw_call_count
        self._raw_call_count += 1
        recompute = (
            self._forward_invocation_count is not None
            and self._forward_invocation_count > 0
            and raw_invocation >= self._forward_invocation_count
        )
        invocation = (
            raw_invocation % self._forward_invocation_count
            if recompute
            else raw_invocation
        )
        if (
            self._site.invocation_index is not None
            and invocation != int(self._site.invocation_index)
        ):
            return

        selected_path = canonical_activation_output_path(
            self._site.output_path
        )
        for output_path, tensor in iter_module_output_tensors(output):
            canonical_path = canonical_activation_output_path(output_path)
            if canonical_path is None:
                continue
            if selected_path not in {None, "$"} and canonical_path != selected_path:
                continue
            key = (invocation, canonical_path)
            sample = self.samples.get(key)
            if sample is None:
                flat = tensor.reshape(-1)
                activation_site = build_activation_site_identity(
                    node_id=self._site.node_id or self._site.site_id,
                    stage_id=self._site.stage_id,
                    parent_node_id=self._site.parent_node_id,
                    invocation_index=invocation,
                    output_path=canonical_path,
                    alias_node_ids=self._site.alias_node_ids,
                )
                sample = ActivationCaptureSample(
                    activation_site=activation_site,
                    module_site_id=self._site.site_id,
                    invocation_index=invocation,
                    output_path=canonical_path,
                    activation=(
                        flat.detach()
                        .to(device="cpu", dtype=torch.float64)
                        .clone()
                    ),
                    gradient=None,
                    source_numel=int(flat.numel()),
                    checkpoint_recompute_observed=recompute,
                )
                self.samples[key] = sample
            elif recompute:
                sample.checkpoint_recompute_observed = True

            if self._capture_gradients and tensor.requires_grad:

                def capture(
                    gradient: torch.Tensor | None,
                    *,
                    target: ActivationCaptureSample = sample,
                ) -> None:
                    if gradient is None:
                        return
                    target.gradient = (
                        gradient.detach()
                        .reshape(-1)
                        .to(device="cpu", dtype=torch.float64)
                        .clone()
                    )

                self._gradient_handles.append(tensor.register_hook(capture))

    def rows(
        self,
        *,
        objective_value: float | None,
        condition_id: str,
    ) -> list[dict[str, Any]]:
        """把精确 site tensor 转为充分统计，不持久化完整 activation。"""

        rows: list[dict[str, Any]] = []
        for sample in self.samples.values():
            gradient = sample.gradient
            activation = sample.activation
            finite_activation = torch.isfinite(activation)
            finite_gradient = (
                torch.isfinite(gradient)
                if gradient is not None
                else torch.zeros_like(finite_activation)
            )
            joint = finite_activation & finite_gradient
            activation_energy = float(
                torch.square(activation[finite_activation]).sum().item()
            )
            gradient_energy = (
                float(torch.square(gradient[finite_gradient]).sum().item())
                if gradient is not None
                else 0.0
            )
            dot = (
                float((activation[joint] * gradient[joint]).sum().item())
                if gradient is not None
                else 0.0
            )
            abs_product = (
                float((activation[joint] * gradient[joint]).abs().sum().item())
                if gradient is not None
                else None
            )
            denominator = math.sqrt(activation_energy * gradient_energy)
            rows.append(
                {
                    **build_activation_record_identity(
                        observed_site=sample.activation_site,
                        condition_target_site=sample.activation_site,
                        record_context={"condition_id": condition_id},
                    ),
                    "record_kind": "local_taylor",
                    "condition_id": condition_id,
                    "module_site_id": sample.module_site_id,
                    "sampled_element_count": int(activation.numel()),
                    "source_element_count": sample.source_numel,
                    "activation_rms": math.sqrt(
                        activation_energy
                        / max(int(finite_activation.sum().item()), 1)
                    ),
                    "gradient_rms": (
                        math.sqrt(
                            gradient_energy
                            / max(int(finite_gradient.sum().item()), 1)
                        )
                        if gradient is not None
                        else None
                    ),
                    "activation_gradient_cosine": (
                        dot / denominator if denominator > 1e-12 else None
                    ),
                    "local_taylor_signed_sum": (
                        dot if gradient is not None else None
                    ),
                    "local_taylor_abs_sum": abs_product,
                    "objective_normalized_local_taylor": (
                        abs_product / max(abs(objective_value), 1e-12)
                        if abs_product is not None
                        and objective_value is not None
                        and math.isfinite(objective_value)
                        else None
                    ),
                    "activation_checkpoint_recompute_status": (
                        "observed"
                        if sample.checkpoint_recompute_observed
                        else "not_observed"
                    ),
                    "conductance_computed": False,
                    "integrated_gradients_computed": False,
                    "activation_patching_computed": False,
                    "physical_causality_claimed": False,
                }
            )
        return rows

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._handle is not None:
            self._handle.remove()
        for handle in self._gradient_handles:
            handle.remove()
        self._gradient_handles.clear()


__all__ = ["ActivationCaptureSample", "ModuleActivationCapture"]
