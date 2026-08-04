"""
验证独立发行的 CI 矩阵、静态边界审计和手动发布门槛没有漂移。
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import subprocess
import sys
import tarfile
import zipfile

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


_ROOT = Path(__file__).resolve().parents[2]


def test_release_metadata_matches_executable_version_matrix() -> None:
    """发行 metadata 与 tox/hosted CI 必须覆盖相同的两端版本。"""

    project = tomllib.loads(
        (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]
    workflow = (
        _ROOT / ".github/workflows/portable-ci.yml"
    ).read_text(encoding="utf-8")
    tox = (_ROOT / "tox.ini").read_text(encoding="utf-8")

    assert project["requires-python"] == ">=3.10,<3.13"
    assert project["dependencies"] == ["torch>=2.5,<2.11"]
    for python_version in ("3.10", "3.11", "3.12"):
        for torch_version in ("2.5.1", "2.10.0"):
            assert f'python: "{python_version}"' in workflow
            assert f'torch: "{torch_version}"' in workflow
    for environment in (
        "py310-torch25",
        "py311-torch25",
        "py312-torch25",
        "py310-torch210",
        "py311-torch210",
        "py312-torch210",
    ):
        assert environment in tox


def test_production_source_passes_portability_audit() -> None:
    """任务无关生产代码不得导入宿主或携带具体宿主字段。"""

    result = subprocess.run(
        [
            sys.executable,
            str(_ROOT / ".github/scripts/audit_portability.py"),
            "--source-root",
            str(_ROOT),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_portability_audit_covers_documentation_code_and_text(
    tmp_path: Path,
) -> None:
    """文档示例 import 与非 Python 文本路径同样属于发行边界。"""

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "example.py").write_text(
        "import foreign_runtime\n",
        encoding="utf-8",
    )
    absolute_path = "".join(("/", "private", "/", "current", "/", "task"))
    (docs / "guide.md").write_text(
        f"Do not embed `{absolute_path}` in a portable document.\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(_ROOT / ".github/scripts/audit_portability.py"),
            "--source-root",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "unsupported dependency 'foreign_runtime'" in result.stderr
    assert "absolute local path is not portable" in result.stderr


def test_distribution_audit_rejects_members_outside_positive_allow_list(
    tmp_path: Path,
) -> None:
    """未知 wheel package member 与 sdist 根目录必须被正向 allow-list 拒绝。"""

    wheel = tmp_path / "model_diagnostics-0.1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, mode="w") as archive:
        archive.writestr("model_diagnostics/unexpected.py", "")
        absolute_path = "".join(
            ("/", "private", "/", "current", "/", "task")
        )
        archive.writestr(
            "model_diagnostics/base/leak.py",
            f"PATH = {absolute_path!r}\n",
        )
    sdist = tmp_path / "model_diagnostics-0.1.0.tar.gz"
    with tarfile.open(sdist, mode="w:gz") as archive:
        payload = b"not part of the source contract"
        member = tarfile.TarInfo(
            "model_diagnostics-0.1.0/private/payload.txt"
        )
        member.size = len(payload)
        archive.addfile(member, BytesIO(payload))

    result = subprocess.run(
        [
            sys.executable,
            str(_ROOT / ".github/scripts/audit_portability.py"),
            "--source-root",
            str(_ROOT),
            "--dist-dir",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "leaves runtime allow-list" in result.stderr
    assert "leaves source allow-list" in result.stderr
    assert "absolute local path is not portable" in result.stderr
