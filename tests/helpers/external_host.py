"""提供完全独立于仓库源码的 MLP/Transformer checkpoint 宿主。

该文件既能生成自包含的外部运行目录，也能直接调用 portable checkpoint engine。
测试会把它单独复制到临时目录，并只使用已安装 wheel。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from model_diagnostics.base import BASE_ANALYZER_CATALOG, ModuleSite
from model_diagnostics.base.artifacts import file_sha256, stable_json_hash
from model_diagnostics.base.checkpoint import (
    CheckpointRef,
    CheckpointRuntimeDescriptor,
    CohortSelection,
    DiagnosticsRecipe,
    LoadedModel,
    ObjectiveResult,
    SampleRef,
    run_checkpoint_diagnostics,
)


class GenericMLP(torch.nn.Module):
    """不含时序或宿主语义的普通多层感知机。"""

    def __init__(self) -> None:
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Linear(4, 6),
            torch.nn.Tanh(),
            torch.nn.Linear(6, 5),
            torch.nn.ReLU(),
        )
        self.head = torch.nn.Linear(5, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """返回批量标量预测。"""

        return self.head(self.features(features))


class GenericTransformerClassifier(torch.nn.Module):
    """只假设 token tensor 的通用 Transformer 分类器。"""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(31, 8)
        layer = torch.nn.TransformerEncoderLayer(
            d_model=8,
            nhead=2,
            dim_feedforward=16,
            dropout=0.0,
            batch_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(
            layer,
            num_layers=1,
            enable_nested_tensor=False,
        )
        self.head = torch.nn.Linear(8, 1)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """编码 token 并对序列维做平均池化。"""

        encoded = self.encoder(self.embedding(token_ids))
        return self.head(encoded.mean(dim=1))


def _make_model(model_kind: str) -> torch.nn.Module:
    """按外部配置构造模型；模型选择由宿主而非 Base 完成。"""

    if model_kind == "mlp":
        return GenericMLP()
    if model_kind == "transformer":
        return GenericTransformerClassifier()
    raise ValueError(f"unsupported external model kind: {model_kind}")


def _make_payload(
    model_kind: str,
    sample_index: int,
) -> dict[str, torch.Tensor]:
    """构造 deterministic 样本，且不依赖任何仓库数据类型。"""

    if model_kind == "mlp":
        features = (
            torch.tensor(
                [
                    [0.2, -0.4, 0.8, 1.0],
                    [-0.3, 0.7, 0.1, -0.5],
                ],
                dtype=torch.float32,
            )
            + float(sample_index) * 0.05
        )
        target = torch.tensor([[0.25], [-0.15]], dtype=torch.float32)
        return {"features": features, "target": target}
    if model_kind == "transformer":
        token_ids = torch.tensor(
            [
                [1, 4, 9, 16, 25],
                [2, 3, 5, 7, 11],
            ],
            dtype=torch.int64,
        )
        token_ids = (token_ids + int(sample_index)) % 31
        target = torch.tensor([[0.1], [-0.2]], dtype=torch.float32)
        return {"token_ids": token_ids, "target": target}
    raise ValueError(f"unsupported external model kind: {model_kind}")


def prepare_external_run(run_dir: Path, model_kind: str) -> None:
    """创建任意命名 checkpoint、adapter 配置和严格 recipe。"""

    run_dir = run_dir.resolve()
    payload_dir = run_dir / "payloads"
    payload_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(3107 if model_kind == "mlp" else 3108)
    model = _make_model(model_kind)
    checkpoint_path = payload_dir / "candidate.weights"
    torch.save(model.state_dict(), checkpoint_path)
    (run_dir / "host_config.json").write_text(
        json.dumps(
            {
                "model_kind": model_kind,
                "checkpoint": str(checkpoint_path.relative_to(run_dir)),
            },
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
class ExternalHostAdapter:
    """实现 Base 最小协议，证明 checkpoint 布局和任务执行由宿主拥有。"""

    execution_precision = "fp32"
    analyzer_catalog = BASE_ANALYZER_CATALOG

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir.resolve()
        config = json.loads(
            (self.run_dir / "host_config.json").read_text(encoding="utf-8")
        )
        self.model_kind = str(config["model_kind"])
        self.checkpoint_path = (
            self.run_dir / str(config["checkpoint"])
        ).resolve()
        self.device = torch.device("cpu")
        self.rank = 0
        self.world_size = 1
        self.runtime_descriptor = CheckpointRuntimeDescriptor.create(
            task_definition_id=f"portable.external_{self.model_kind}",
            task_definition_version="1",
            objective_executor_version="1",
            capability_descriptors=(
                {
                    "name": "external_checkpoint_host",
                    "definition_version": 1,
                },
            ),
        )
        self._recipe: DiagnosticsRecipe | None = None

    def configure(self, recipe: DiagnosticsRecipe) -> None:
        """保存 Base 已验证的唯一 effective recipe。"""

        if self._recipe is not None and self._recipe.to_dict() != recipe.to_dict():
            raise RuntimeError("external adapter cannot change recipe in one run")
        self._recipe = recipe

    def resolve_checkpoints(
        self,
        *,
        run_dir: Path,
    ) -> Sequence[CheckpointRef]:
        """把宿主配置的任意文件名转换为通用 ``CheckpointRef``。"""

        if run_dir.resolve() != self.run_dir:
            raise ValueError("adapter run directory identity changed")
        return (
            CheckpointRef(
                path=str(self.checkpoint_path),
                identity=stable_json_hash(
                    {
                        "sha256": file_sha256(self.checkpoint_path),
                        "model_kind": self.model_kind,
                    }
                ),
                kind="external_config",
                metadata={"filename_contract": "host_owned"},
            ),
        )

    def load_model(self, checkpoint_path: str, *, precision: str) -> LoadedModel:
        """恢复 model-only checkpoint，不依赖特定文件后缀或目录。"""

        if precision != "fp32":
            raise ValueError("external conformance fixture only supports fp32")
        model = _make_model(self.model_kind)
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        return LoadedModel(
            model=model,
            checkpoint_update=None,
            model_name=f"external_{self.model_kind}",
            structure_metadata={"task_family": self.model_kind},
        )

    def build_cohort(
        self,
        loaded: LoadedModel,
        *,
        checkpoints: Sequence[CheckpointRef],
    ) -> CohortSelection:
        """建立通用 sample/group identity，不使用任何领域字段。"""

        del loaded, checkpoints
        size = 2
        samples = tuple(
            SampleRef(
                sample_id=f"external-sample-{index}",
                partition="validation",
                sample_index=index,
                group_id=f"group-{index}",
                position=index,
            )
            for index in range(size)
        )
        return CohortSelection(
            status="available",
            samples=samples,
            identity=stable_json_hash([sample.to_dict() for sample in samples]),
        )

    def materialize_sample(
        self,
        loaded: LoadedModel,
        sample: SampleRef,
    ) -> dict[str, torch.Tensor]:
        """由稳定 sample identity 重新物化独立 tensor payload。"""

        del loaded
        if sample.sample_index is None:
            raise ValueError("external sample requires sample_index")
        return _make_payload(self.model_kind, sample.sample_index)

    def clone_sample_payload(
        self,
        payload: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """隔离不同 condition 使用的可变 tensor storage。"""

        return {
            name: value.detach().clone()
            for name, value in payload.items()
        }

    def list_module_sites(self, loaded: LoadedModel) -> Sequence[ModuleSite]:
        """显式选择一个模型内部 site，不让 Base 推断任务结构。"""

        del loaded
        module_path = (
            "features"
            if self.model_kind == "mlp"
            else "encoder.layers.0"
        )
        return (
            ModuleSite(
                site_id=f"{self.model_kind}-hidden",
                module_path=module_path,
                module_type=(
                    "Sequential"
                    if self.model_kind == "mlp"
                    else "TransformerEncoderLayer"
                ),
                node_id=f"{self.model_kind}-hidden",
                stage_id="feature_extractor",
                invocation_index=0,
                output_path="$",
            ),
        )

    def forward_objective(
        self,
        loaded: LoadedModel,
        payload: Mapping[str, torch.Tensor],
        *,
        objective: Mapping[str, Any],
        precision: str,
        return_aux: bool,
    ) -> ObjectiveResult:
        """执行宿主 forward，并显式返回 scalar MSE objective。"""

        del objective, return_aux
        if precision != "fp32":
            raise ValueError("external conformance fixture only supports fp32")
        model_input = (
            payload["features"]
            if self.model_kind == "mlp"
            else payload["token_ids"]
        )
        prediction = loaded.model(model_input)
        loss = torch.square(prediction - payload["target"]).mean()
        return ObjectiveResult(
            total=loss,
            raw_total=loss,
            backward_total=loss,
            terms={"mse": loss},
            outputs={"prediction": prediction},
            objective_identity="external_mse",
            objective_metadata={"reduction": "mean"},
        )

    def close_loaded_model(self, loaded: LoadedModel) -> None:
        """释放 checkpoint-local 引用；CPU fixture 无额外资源。"""

        del loaded

    def barrier(self) -> None:
        """单进程 adapter 的同步 no-op。"""

    def close(self) -> None:
        """CLI 生命周期结束时的幂等 no-op。"""


def main(argv: list[str] | None = None) -> int:
    """生成或诊断一个外部 conformance run directory。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--model-kind", choices=("mlp", "transformer"))
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir)
    if args.command == "prepare":
        if args.model_kind is None:
            parser.error("prepare requires --model-kind")
        prepare_external_run(run_dir, args.model_kind)
        return 0
    adapter = ExternalHostAdapter(run_dir)
    recipe = DiagnosticsRecipe.final_selected(
        adapter.checkpoint_path,
        analyzer_catalog=adapter.analyzer_catalog,
    )
    run_checkpoint_diagnostics(
        adapter=adapter,
        run_dir=run_dir,
        runtime_descriptor=adapter.runtime_descriptor,
        recipe=recipe,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
