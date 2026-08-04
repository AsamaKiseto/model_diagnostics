"""验证 ``model_diagnostics`` 可由复制目录独立构建、安装和接入外部宿主。

测试不使用仓库根 ``PYTHONPATH``：wheel 安装到临时 venv 后，在另一个目录加载通用
MLP/Transformer adapter，并运行 checkpoint intervention 与统一 report。
"""

from __future__ import annotations

from email.parser import BytesParser
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from typing import Iterator
import venv
import zipfile

import pytest


PACKAGE_VERSION = "0.1.0"


def _sanitized_environment() -> dict[str, str]:
    """移除可能让隔离进程回退到仓库源码的 Python 路径。"""

    environment = dict(os.environ)
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def _run(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    """运行发行 conformance 命令，并在失败时保留完整 stdout/stderr。"""

    return subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


@pytest.fixture(scope="module")
def installed_distribution(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, Path]:
    """从复制后的 package 目录构建 wheel/sdist，并仅安装 wheel。"""

    source_root = Path(__file__).resolve().parents[2]
    workspace_root = source_root.parent.resolve()
    root = tmp_path_factory.mktemp("distribution")
    copied_source = root / "portable-source"

    def ignored(_directory: str, names: list[str]) -> set[str]:
        ignored_names = {
            name
            for name in names
            if name in {
                ".cache",
                "build",
                "dist",
                "__pycache__",
            }
            or name.endswith((".egg-info", ".pyc", ".pyo"))
        }
        return ignored_names

    shutil.copytree(source_root, copied_source, ignore=ignored)
    dist_dir = copied_source / "dist"
    dist_dir.mkdir()
    environment = _sanitized_environment()
    backend_script = (
        "from setuptools.build_meta import build_sdist, build_wheel\n"
        "build_sdist('dist')\n"
        "build_wheel('dist')\n"
    )
    _run(
        [sys.executable, "-c", backend_script],
        cwd=copied_source,
        environment=environment,
    )

    wheel = next(dist_dir.glob("*.whl"))
    sdist = next(dist_dir.glob("*.tar.gz"))
    venv_dir = root / "venv"
    venv.EnvBuilder(
        with_pip=True,
        system_site_packages=True,
        clear=True,
    ).create(venv_dir)
    venv_python = (
        venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    )
    _run(
        [
            str(venv_python),
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            str(wheel),
        ],
        cwd=root,
        environment=environment,
    )
    return {
        "repository_source": source_root,
        "source": copied_source,
        "workspace": workspace_root,
        "root": root,
        "wheel": wheel,
        "sdist": sdist,
        "python": venv_python,
    }


def _archive_members(path: Path) -> Iterator[str]:
    """统一枚举 wheel 与 sdist 的 POSIX archive member。"""

    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            yield from archive.namelist()
        return
    with tarfile.open(path, mode="r:gz") as archive:
        yield from archive.getnames()


def test_wheel_contains_runtime_only_and_declares_compatibility(
    installed_distribution: dict[str, Path],
) -> None:
    """wheel 必须声明版本、Python/PyTorch 范围和非猜测许可证。"""

    wheel = installed_distribution["wheel"]
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        metadata_name = next(
            name for name in names if name.endswith(".dist-info/METADATA")
        )
        metadata = BytesParser().parsebytes(archive.read(metadata_name))

    assert "model_diagnostics/base/runtime/contracts.py" in names
    assert "model_diagnostics/base/runtime/controller.py" not in names
    assert "model_diagnostics/extensions/rollout/__init__.py" in names
    assert any(name.endswith(".dist-info/licenses/LICENSE") for name in names)
    assert not any(
        part in name
        for name in names
        for part in (
            "model_diagnostics/docs/",
            "model_diagnostics/tests/",
            "model_diagnostics/src/",
            "__pycache__",
        )
    )
    assert not any(name.endswith((".pyc", ".pyo")) for name in names)
    assert metadata["Name"] == "model-diagnostics"
    assert metadata["Version"] == PACKAGE_VERSION
    assert metadata["License-Expression"] == "LicenseRef-Proprietary"
    assert {
        clause.strip()
        for clause in metadata["Requires-Python"].split(",")
    } == {">=3.10", "<3.13"}
    assert {
        clause.strip()
        for clause in metadata.get_all("Requires-Dist")[0]
        .removeprefix("torch")
        .split(",")
    } == {">=2.5", "<2.11"}


def test_sdist_is_self_contained_without_host_or_cache(
    installed_distribution: dict[str, Path],
) -> None:
    """sdist 保留 portable docs/tests，但不携带宿主源码或构建缓存。"""

    names = tuple(_archive_members(installed_distribution["sdist"]))
    root = f"model_diagnostics-{PACKAGE_VERSION}/"
    required = {
        root + "LICENSE",
        root + "README.md",
        root + "pyproject.toml",
        root + "tox.ini",
        root + ".github/scripts/audit_portability.py",
        root + ".github/workflows/portable-ci.yml",
        root + "base/runtime/contracts.py",
        root + "extensions/rollout/__init__.py",
        root + "docs/README.md",
        root + "tests/base/test_distribution.py",
        root + "tests/release/test_release_contract.py",
    }
    assert required.issubset(names)
    assert not any(
        part in name
        for name in names
        for part in ("/src/", "/.cache/", "/__pycache__/")
    )
    assert not any(name.endswith((".pyc", ".pyo")) for name in names)


def test_built_distributions_pass_portability_audit(
    installed_distribution: dict[str, Path],
) -> None:
    """真实 wheel/sdist member 必须满足正向发行 allow-list。"""

    source = installed_distribution["source"]
    _run(
        [
            sys.executable,
            str(source / ".github/scripts/audit_portability.py"),
            "--source-root",
            str(source),
            "--dist-dir",
            str(source / "dist"),
        ],
        cwd=source,
        environment=_sanitized_environment(),
    )


def test_extracted_sdist_runs_release_contract(
    installed_distribution: dict[str, Path],
) -> None:
    """sdist 必须携带 release test 所依赖的全部 CI contract 资产。"""

    root = installed_distribution["root"] / "extracted-sdist"
    root.mkdir()
    with tarfile.open(
        installed_distribution["sdist"],
        mode="r:gz",
    ) as archive:
        members = archive.getmembers()
        assert all(
            not Path(member.name).is_absolute()
            and ".." not in Path(member.name).parts
            for member in members
        )
        kwargs = (
            {"filter": "data"}
            if "filter" in inspect.signature(archive.extractall).parameters
            else {}
        )
        archive.extractall(root, **kwargs)
    source = next(root.iterdir())
    nested_tmp = root / "release-test-tmp"
    nested_tmp.mkdir()
    environment = _sanitized_environment()
    environment["TMPDIR"] = str(nested_tmp)
    _run(
        [
            str(installed_distribution["python"]),
            "-m",
            "pytest",
            "-q",
            "-c",
            "pytest.ini",
            "tests/release/test_release_contract.py",
        ],
        cwd=source,
        environment=environment,
    )


def test_clean_install_import_and_cli_do_not_use_repository(
    installed_distribution: dict[str, Path],
) -> None:
    """隔离解释器必须从 site-packages 加载 package 并可运行 CLI。"""

    python = installed_distribution["python"]
    environment = _sanitized_environment()
    root = installed_distribution["root"]
    probe = _run(
        [
            str(python),
            "-I",
            "-c",
            (
                "import importlib.metadata, json, model_diagnostics, sys;"
                "print(json.dumps({"
                "'module': model_diagnostics.__file__,"
                "'version': importlib.metadata.version('model-diagnostics'),"
                "'sys_path': sys.path"
                "}, sort_keys=True))"
            ),
        ],
        cwd=root,
        environment=environment,
    )
    payload = json.loads(probe.stdout)
    workspace = installed_distribution["workspace"].resolve()
    repository_source = installed_distribution["repository_source"].resolve()
    copied_source = installed_distribution["source"].resolve()
    module_path = Path(payload["module"]).resolve()
    assert payload["version"] == PACKAGE_VERSION
    assert "site-packages" in payload["module"]
    assert not module_path.is_relative_to(copied_source)
    resolved_sys_path = {
        Path(entry).resolve()
        for entry in payload["sys_path"]
        if entry
    }
    assert workspace not in resolved_sys_path
    assert repository_source not in resolved_sys_path
    assert copied_source not in resolved_sys_path

    help_result = _run(
        [str(python), "-I", "-m", "model_diagnostics.base", "--help"],
        cwd=root,
        environment=environment,
    )
    assert "report" in help_result.stdout


@pytest.mark.parametrize("model_kind", ["mlp", "transformer"])
def test_external_host_checkpoint_intervention_and_report(
    installed_distribution: dict[str, Path],
    model_kind: str,
) -> None:
    """普通外部模型通过 wheel 执行 identity/scale condition 和 report。"""

    python = installed_distribution["python"]
    environment = _sanitized_environment()
    external_root = installed_distribution["root"] / f"external-{model_kind}"
    external_root.mkdir()
    helper_source = (
        Path(__file__).resolve().parents[1] / "helpers" / "external_host.py"
    )
    shutil.copy2(helper_source, external_root / "external_host.py")
    run_dir = external_root / "run"

    _run(
        [
            str(python),
            "external_host.py",
            "prepare",
            "--run-dir",
            str(run_dir),
            "--model-kind",
            model_kind,
        ],
        cwd=external_root,
        environment=environment,
    )
    _run(
        [
            str(python),
            "external_host.py",
            "run",
            "--run-dir",
            str(run_dir),
        ],
        cwd=external_root,
        environment=environment,
    )
    _run(
        [
            str(python),
            "-m",
            "model_diagnostics.base",
            "report",
            "--run-dir",
            str(run_dir),
        ],
        cwd=external_root,
        environment=environment,
    )

    analysis = next(
        (run_dir / "diagnostics" / "v2" / "checkpoint" / "analyses").iterdir()
    )
    manifest = json.loads(
        (analysis / "manifest.json").read_text(encoding="utf-8")
    )
    condition_path = (
        analysis
        / "analyzers"
        / "_base"
        / "streams"
        / "conditions.jsonl"
    )
    conditions = [
        json.loads(line)
        for line in condition_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report_summary = json.loads(
        (
            run_dir / "diagnostics" / "v2" / "report" / "summary.json"
        ).read_text(encoding="utf-8")
    )

    assert manifest["status"] == "complete"
    assert {row["method"] for row in conditions} == {
        "identity",
        "output_scale",
        "mean_patch",
    }
    exact_controls = [
        row
        for row in conditions
        if row["method"] == "identity"
    ]
    assert exact_controls
    assert all(row["identity_control_exact"] is True for row in exact_controls)
    assert report_summary["status"] == "success"
