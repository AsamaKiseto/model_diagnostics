"""提供 renderer-aware report CLI。"""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m model_diagnostics.extensions"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    report = subparsers.add_parser("report")
    report.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    from model_diagnostics.base.reporting import (
        generate_diagnostics_report,
    )
    from . import build_report_renderer_catalog

    output = generate_diagnostics_report(
        args.run_dir,
        renderer_catalog=build_report_renderer_catalog(),
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
