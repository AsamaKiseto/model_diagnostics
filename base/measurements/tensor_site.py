"""
为调用方显式暴露的 functional Tensor site 建立任务无关 activation/gradient 探针。
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from typing import Any, Callable

import torch

from ..interventions import TensorSite
from .probes import ModuleDiagnosticsProbe


_EPS = 1e-12


def tensor_site_identity(site: TensorSite) -> str:
    """仅用稳定定位字段生成 identity，避免 metadata 改变 artifact 主键。"""

    payload = json.dumps(
        {
            "path": site.path,
            "axis": site.axis,
            "index": site.index,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"tensor/{sha256(payload.encode('utf-8')).hexdigest()[:16]}"


@dataclass(slots=True)
class TensorSiteProbe:
    """拥有一次显式 functional site invocation 的有界 autograd probe。"""

    site: TensorSite
    invocation_index: int
    microbatch_id: str
    probe: ModuleDiagnosticsProbe

    @classmethod
    def from_tensor(
        cls,
        *,
        site: TensorSite,
        invocation_index: int,
        microbatch_id: str,
        tensor: torch.Tensor,
        with_grad: bool,
        failure_callback: Callable[[str, Exception], None] | None = None,
        propagate_hook_failures: bool = True,
    ) -> "TensorSiteProbe | None":
        """直接观察调用方传入 Tensor；返回值不会替换或包装训练 Tensor。"""

        probe = ModuleDiagnosticsProbe.from_output(
            tensor,
            with_grad=with_grad,
            failure_callback=failure_callback,
            propagate_hook_failures=propagate_hook_failures,
        )
        if probe is None:
            return None
        return cls(
            site=site,
            invocation_index=int(invocation_index),
            microbatch_id=microbatch_id,
            probe=probe,
        )

    @property
    def sample_element_count(self) -> int:
        return self.probe.sample_element_count

    @property
    def disabled(self) -> bool:
        return self.probe.disabled

    def bind_amp_scale(self, amp_scale: float) -> None:
        self.probe.bind_amp_scale(amp_scale)

    def finalize(
        self,
        *,
        update: int,
        rank: int,
        objective_summary: dict[str, Any],
        objective_provenance: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """在通用充分统计上追加 site/invocation identity 与 objective provenance。"""

        records = self.probe.finalize(
            update=update,
            rank=rank,
            objective_raw_sum=objective_summary.get("objective_raw_sum"),
            backward_objective_sum=objective_summary.get(
                "backward_objective_sum"
            ),
            objective_normalization_divisor=objective_summary.get(
                "objective_normalization_divisor"
            ),
            loss_cap_hit_fraction=objective_summary.get(
                "loss_cap_hit_fraction"
            ),
            objective_trace_status=str(
                objective_summary.get("objective_trace_status") or "missing"
            ),
        )
        identity = tensor_site_identity(self.site)
        for record in records:
            record.update(
                {
                    "record_kind": "tensor_site_sufficient_statistics",
                    "tensor_site_id": identity,
                    "tensor_site_path": self.site.path,
                    "tensor_site_axis": self.site.axis,
                    "tensor_site_index": self.site.index,
                    "tensor_site_invocation_index": self.invocation_index,
                    "microbatch_id": self.microbatch_id,
                    "activation_source": "explicit_functional_tensor_site",
                    "tensor_site_metadata_keys": sorted(
                        str(key) for key in self.site.metadata
                    ),
                    **objective_provenance,
                }
            )
        return records

    def close(self) -> None:
        self.probe.close()


def aggregate_tensor_site_records(
    records: list[dict[str, Any]],
    *,
    rank_objectives: dict[int, dict[str, Any]],
    world_size: int,
    update: int,
) -> list[dict[str, Any]]:
    """按 site 汇总 invocation/rank 充分统计，不把 functional site 伪装成 module node。"""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        identity = str(record["tensor_site_id"])
        grouped.setdefault(identity, []).append(record)
    return [
        _aggregate_one_tensor_site(
            identity,
            matching,
            rank_objectives=rank_objectives,
            world_size=world_size,
            update=update,
        )
        for identity, matching in sorted(grouped.items())
    ]


def _aggregate_one_tensor_site(
    identity: str,
    matching: list[dict[str, Any]],
    *,
    rank_objectives: dict[int, dict[str, Any]],
    world_size: int,
    update: int,
) -> dict[str, Any]:
    """重建 pooled RMS/cosine，并用 objective 的 rank reduction 计算 Taylor mass。"""

    def additive(field_name: str) -> float:
        return sum(
            float(record.get(field_name) or 0.0) for record in matching
        )

    activation_count = int(additive("activation_count"))
    activation_finite_count = int(additive("activation_finite_count"))
    activation_sum = additive("activation_sum")
    activation_square_sum = additive("activation_square_sum")
    activation_zero_count = int(additive("activation_zero_count"))
    activation_nonfinite_count = int(additive("activation_nonfinite_count"))
    gradient_count = int(additive("activation_grad_count"))
    gradient_finite_count = int(additive("activation_grad_finite_count"))
    gradient_square_sum = additive("activation_grad_square_sum")
    gradient_zero_count = int(additive("activation_grad_zero_count"))
    gradient_nonfinite_count = int(additive("activation_grad_nonfinite_count"))
    joint_count = int(additive("joint_finite_count"))
    dot_sum = additive("dot_sum")
    left_square_sum = additive("activation_joint_square_sum")
    right_square_sum = additive("gradient_joint_square_sum")
    cosine_denominator = math.sqrt(
        max(left_square_sum, 0.0) * max(right_square_sum, 0.0)
    )
    source_numel = int(additive("num_elements"))
    sample_numel = int(additive("sample_num_elements"))

    records_by_rank: dict[int, list[dict[str, Any]]] = {}
    for record in matching:
        records_by_rank.setdefault(int(record.get("rank") or 0), []).append(
            record
        )
    reduced_ranks = sorted(rank_objectives) or list(range(world_size))
    taylor_ranks = {
        rank
        for rank, rank_records in records_by_rank.items()
        if any(int(record.get("joint_finite_count") or 0) > 0 for record in rank_records)
    }

    def rank_mean(field_name: str) -> float | None:
        if not reduced_ranks or not taylor_ranks:
            return None
        return sum(
            sum(
                float(record.get(field_name) or 0.0)
                for record in records_by_rank.get(rank, ())
            )
            for rank in reduced_ranks
        ) / len(reduced_ranks)

    exact = bool(
        matching
        and all(
            record.get("local_taylor_abs_sum") is not None
            and int(record.get("sample_num_elements") or 0)
            == int(record.get("num_elements") or -1)
            for record in matching
        )
    )
    estimated = bool(
        matching
        and all(
            record.get("local_taylor_abs_sum_estimate") is not None
            for record in matching
        )
    )
    sampled_abs = rank_mean("abs_product_sum")
    sampled_signed = rank_mean("signed_product_sum")
    exact_abs = sampled_abs if exact else None
    exact_signed = sampled_signed if exact else None
    estimated_abs = (
        rank_mean("local_taylor_abs_sum_estimate") if estimated else None
    )
    estimated_signed = (
        rank_mean("local_taylor_signed_sum_estimate") if estimated else None
    )

    objective = _aggregate_tensor_site_objective(rank_objectives, matching)
    denominator = objective["backward_objective_sum"]
    objective_complete = objective["objective_trace_status"] == "complete"
    called_ranks = sorted(records_by_rank)
    invocation_counts = [
        len(
            {
                (
                    str(record.get("microbatch_id")),
                    int(record.get("tensor_site_invocation_index") or 0),
                )
                for record in records_by_rank.get(rank, ())
            }
        )
        for rank in range(world_size)
    ]
    first = matching[0]
    return {
        "record_kind": "tensor_site",
        "update": int(update),
        "rank": None,
        "world_size": int(world_size),
        "tensor_site_id": identity,
        "tensor_site_path": first.get("tensor_site_path"),
        "tensor_site_axis": first.get("tensor_site_axis"),
        "tensor_site_index": first.get("tensor_site_index"),
        "tensor_site_metadata_keys": first.get("tensor_site_metadata_keys", []),
        "activation_source": "explicit_functional_tensor_site",
        "activation_probe_count": len(matching),
        "called_rank_count": len(called_ranks),
        "collected_rank_count": len(taylor_ranks),
        "call_count_min": min(invocation_counts, default=0),
        "call_count_max": max(invocation_counts, default=0),
        "not_called_any_rank": len(called_ranks) < world_size,
        "not_called_all_ranks": not called_ranks,
        "rank_divergence": len(set(invocation_counts)) > 1,
        "activation_count": activation_count,
        "activation_finite_count": activation_finite_count,
        "activation_sum": activation_sum,
        "activation_square_sum": activation_square_sum,
        "activation_rms": (
            math.sqrt(max(activation_square_sum, 0.0) / activation_finite_count)
            if activation_finite_count > 0
            else None
        ),
        "activation_std": (
            math.sqrt(
                max(
                    activation_square_sum / activation_finite_count
                    - (activation_sum / activation_finite_count) ** 2,
                    0.0,
                )
            )
            if activation_finite_count > 0
            else None
        ),
        "activation_abs_max": max(
            (
                float(record["activation_abs_max"])
                for record in matching
                if record.get("activation_abs_max") is not None
            ),
            default=None,
        ),
        "activation_zero_fraction": (
            activation_zero_count / activation_finite_count
            if activation_finite_count > 0
            else None
        ),
        "activation_nonfinite_fraction": (
            activation_nonfinite_count / activation_count
            if activation_count > 0
            else None
        ),
        "activation_grad_rms": (
            math.sqrt(max(gradient_square_sum, 0.0) / gradient_finite_count)
            if gradient_finite_count > 0
            else None
        ),
        "activation_grad_abs_max": max(
            (
                float(record["activation_grad_abs_max"])
                for record in matching
                if record.get("activation_grad_abs_max") is not None
            ),
            default=None,
        ),
        "activation_grad_zero_fraction": (
            gradient_zero_count / gradient_finite_count
            if gradient_finite_count > 0
            else None
        ),
        "activation_grad_nonfinite_fraction": (
            gradient_nonfinite_count / gradient_count
            if gradient_count > 0
            else None
        ),
        "activation_grad_cosine": (
            dot_sum / cosine_denominator if cosine_denominator > 0.0 else None
        ),
        "sampled_local_taylor_abs_sum": sampled_abs,
        "sampled_local_taylor_signed_sum": sampled_signed,
        "local_taylor_abs_sum": exact_abs,
        "local_taylor_signed_sum": exact_signed,
        "local_taylor_abs_sum_estimate": estimated_abs,
        "local_taylor_signed_sum_estimate": estimated_signed,
        "local_taylor_abs_mean": (
            additive("abs_product_sum") / joint_count if joint_count > 0 else None
        ),
        "local_taylor_source_numel": source_numel,
        "local_taylor_sample_numel": sample_numel,
        "local_taylor_element_coverage": (
            sample_numel / source_numel if source_numel > 0 else None
        ),
        "local_taylor_sum_status": (
            "exact_full_coverage"
            if exact
            else "deterministic_expansion_estimate"
            if estimated
            else "unavailable"
        ),
        "objective_normalized_local_taylor": (
            exact_abs / abs(float(denominator))
            if exact_abs is not None
            and denominator is not None
            and abs(float(denominator)) > _EPS
            and objective_complete
            else None
        ),
        "objective_normalized_local_taylor_estimate": (
            estimated_abs / abs(float(denominator))
            if estimated_abs is not None
            and denominator is not None
            and abs(float(denominator)) > _EPS
            and objective_complete
            else None
        ),
        "local_taylor_rank_reduction": "rank_mean",
        "local_taylor_collected_rank_count": len(taylor_ranks),
        "local_taylor_reduced_rank_count": len(reduced_ranks),
        "conductance_computed": False,
        "integrated_gradients_computed": False,
        "activation_patching_computed": False,
        **objective,
    }


def _aggregate_tensor_site_objective(
    rank_objectives: dict[int, dict[str, Any]],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """要求所有 rank 使用相同 objective 与 VJP provenance 后才允许归一化。"""

    summaries = [rank_objectives[rank] for rank in sorted(rank_objectives)]
    backward = [
        float(summary["backward_objective_sum"])
        for summary in summaries
        if summary.get("backward_objective_sum") is not None
    ]
    raw = [
        float(summary["objective_raw_sum"])
        for summary in summaries
        if summary.get("objective_raw_sum") is not None
    ]
    divisors = {
        float(summary["objective_normalization_divisor"])
        for summary in summaries
        if summary.get("objective_normalization_divisor") is not None
    }
    cap_hits = [
        float(summary["loss_cap_hit_fraction"])
        for summary in summaries
        if summary.get("loss_cap_hit_fraction") is not None
    ]
    identities = {
        tuple(str(value) for value in summary.get("objective_identities", ()))
        for summary in summaries
    }
    provenance = {
        (
            record.get("objective_reduction"),
            tuple(record.get("objective_shape") or ()),
            record.get("cotangent_identity"),
        )
        for record in records
    }
    complete = bool(
        summaries
        and len(backward) == len(summaries)
        and len(raw) == len(summaries)
        and len(cap_hits) == len(summaries)
        and {summary.get("objective_trace_status") for summary in summaries}
        == {"complete"}
        and {record.get("objective_trace_status") for record in records}
        == {"complete"}
        and len(divisors) == 1
        and len(identities) == 1
        and len(provenance) == 1
    )
    reduction, objective_shape, cotangent_identity = (
        next(iter(provenance)) if len(provenance) == 1 else (None, (), None)
    )
    cotangent_norms = {
        float(record["cotangent_norm"])
        for record in records
        if record.get("cotangent_norm") is not None
    }
    return {
        "objective_raw_sum": sum(raw) / len(raw) if raw else None,
        "backward_objective_sum": (
            sum(backward) / len(backward) if backward else None
        ),
        "objective_normalization_divisor": (
            next(iter(divisors)) if len(divisors) == 1 else None
        ),
        "loss_cap_hit_fraction": (
            sum(cap_hits) / len(cap_hits) if cap_hits else None
        ),
        "objective_trace_status": (
            "complete" if complete else "mixed_or_incomplete"
        ),
        "objective_reduction": reduction,
        "objective_shape": list(objective_shape),
        "cotangent_identity": cotangent_identity,
        "cotangent_norm": (
            next(iter(cotangent_norms)) if len(cotangent_norms) == 1 else None
        ),
    }


__all__ = [
    "TensorSiteProbe",
    "aggregate_tensor_site_records",
    "tensor_site_identity",
]
