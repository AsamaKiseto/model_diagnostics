"""通用 PyTorch Stage/Block 自动发现回归。"""

from __future__ import annotations

import torch
from torch import nn

from model_diagnostics.base.hierarchy import (
    discover_model_hierarchy,
)
from model_diagnostics.base.interventions import ModuleSite


class _ActionEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_projection = nn.Linear(4, 8)
        self.future_attention = nn.MultiheadAttention(8, 2, batch_first=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(value)
        attended, _weights = self.future_attention(hidden, hidden, hidden)
        return attended


class _StateEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(4, 8), nn.GELU())

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.projection(value)


class _RoutedEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.role_encoders = nn.ModuleDict(
            {
                "action": _ActionEncoder(),
                "state": _StateEncoder(),
            }
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.role_encoders["action"](value) + self.role_encoders["state"](value)


class _ToyCore(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.TransformerEncoderLayer(8, 2, batch_first=True)]
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            value = layer(value)
        return value


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _RoutedEncoder()
        self.core = _ToyCore()
        self.decoder = nn.Sequential(nn.Linear(8, 4), nn.Tanh())

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.core(self.encoder(value)))


class _RoutedDecoderModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decoders = nn.ModuleDict(
            {
                "state": _StateEncoder(),
                "action": _ActionEncoder(),
            }
        )


def test_discovery_promotes_routed_stages_and_structural_blocks() -> None:
    model = _ToyModel()
    static = discover_model_hierarchy(model, model_name="toy")
    assert {
        "encoder.role_encoders.action",
        "encoder.role_encoders.state",
        "core",
        "decoder",
    }.issubset(set(static.candidate_paths))

    confirmed = discover_model_hierarchy(
        model,
        model_name="toy",
        observed_module_paths=static.candidate_paths,
    )
    levels = {
        site.module_path: site.metadata["hierarchy_level"]
        for site in confirmed.sites
    }
    assert levels["encoder.role_encoders.action"] == "Stage"
    assert levels["encoder.role_encoders.state"] == "Stage"
    assert levels["core"] == "Stage"
    assert levels["decoder"] == "Stage"
    assert levels[
        "encoder.role_encoders.action.future_attention"
    ] == "Block"
    assert levels["core.layers.0"] == "Block"
    assert all(
        not isinstance(dict(model.named_modules())[site.module_path], nn.Linear)
        for site in confirmed.sites
    )


def test_override_is_additive() -> None:
    model = _ToyModel()
    override = ModuleSite(
        site_id="explicit-decoder-linear",
        node_id="explicit-decoder-linear",
        module_path="decoder.0",
        module_type="Linear",
        metadata={"hierarchy_level": "Block"},
    )
    result = discover_model_hierarchy(
        model,
        model_name="toy",
        observed_module_paths=dict(model.named_modules()),
        overrides=(override,),
    )
    by_path = {site.module_path: site for site in result.sites}
    assert by_path["decoder.0"].metadata["discovery_source"] == "hierarchy_override"


def test_discovery_promotes_top_level_module_dict_routes_to_stages() -> None:
    result = discover_model_hierarchy(
        _RoutedDecoderModel(),
        model_name="routed_decoder",
    )

    stages = {
        site.module_path
        for site in result.sites
        if site.metadata["hierarchy_level"] == "Stage"
    }
    assert stages == {"decoders.action", "decoders.state"}
