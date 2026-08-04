"""提供宿主只需绑定 adapter factory 的标准 checkpoint diagnostics CLI。"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path

from .base.checkpoint import (
    CheckpointDiagnosticsAdapter,
    DiagnosticsRecipe,
    run_checkpoint_diagnostics,
)
from .base.reporting import generate_diagnostics_report

AdapterFactory = Callable[..., CheckpointDiagnosticsAdapter]


def build_diagnostics_parser(
    *,
    prog: str = "python -m diagnostics",
) -> argparse.ArgumentParser:
    """构造固定三阶段 parser；宿主不能覆盖 analyzer、cohort 或统计参数。"""

    parser = argparse.ArgumentParser(prog=prog)
    commands = parser.add_subparsers(dest="command", required=True)
    sweep = commands.add_parser(
        "checkpoint-sweep",
        help="scan every trajectory checkpoint for training-health anomalies",
    )
    sweep.add_argument("--run-dir", required=True)
    final = commands.add_parser(
        "final",
        help="run all supported analyses on one selected checkpoint",
    )
    final.add_argument("--run-dir", required=True)
    final.add_argument("--checkpoint", required=True)
    report = commands.add_parser(
        "report",
        help="build one self-contained interactive HTML report",
    )
    report.add_argument("--run-dir", required=True)
    return parser


def run_diagnostics_cli(
    create_adapter: AdapterFactory,
    argv: Sequence[str] | None = None,
    *,
    prog: str = "python -m diagnostics",
) -> int:
    """运行标准 CLI，并把唯一的宿主差异限制在 adapter factory。

    factory 只接收 ``run_dir``。checkpoint 选择、recipe 构造、引擎调用、资源关闭和
    report renderer 组合由本函数统一完成，宿主 CLI 不应复制这些步骤。
    """

    args = build_diagnostics_parser(prog=prog).parse_args(argv)
    run_dir = Path(args.run_dir).resolve()
    if args.command == "report":
        from .extensions import build_report_renderer_catalog

        print(
            generate_diagnostics_report(
                run_dir,
                renderer_catalog=build_report_renderer_catalog(),
            )
        )
        return 0

    adapter = create_adapter(run_dir=run_dir)
    try:
        checkpoint = None if args.command == "checkpoint-sweep" else Path(args.checkpoint)
        if checkpoint is not None and not checkpoint.is_absolute():
            checkpoint = run_dir / checkpoint
        recipe = (
            DiagnosticsRecipe.checkpoint_sweep(
                analyzer_catalog=adapter.analyzer_catalog,
            )
            if args.command == "checkpoint-sweep"
            else DiagnosticsRecipe.final_selected(
                checkpoint,
                analyzer_catalog=adapter.analyzer_catalog,
            )
        )
        output = run_checkpoint_diagnostics(
            adapter=adapter,
            run_dir=run_dir,
            runtime_descriptor=adapter.runtime_descriptor,
            recipe=recipe,
        )
        if int(getattr(adapter, "rank", 0)) == 0:
            print(output)
    finally:
        adapter.close()
    return 0


__all__ = [
    "AdapterFactory",
    "build_diagnostics_parser",
    "run_diagnostics_cli",
]
