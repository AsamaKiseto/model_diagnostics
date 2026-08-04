"""提供任务无关的 Base report CLI。"""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    """解析 Base 子命令；task-specific extension 不会被隐式加载。"""

    parser = argparse.ArgumentParser(prog="python -m model_diagnostics.base")
    subparsers = parser.add_subparsers(dest="command", required=True)

    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--run-dir", required=True)

    args = parser.parse_args(argv)

    from .reporting import generate_diagnostics_report

    print(generate_diagnostics_report(args.run_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
