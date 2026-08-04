"""
审计独立发行源码和构建产物的依赖边界与宿主知识隔离。
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
from pathlib import Path, PurePosixPath
import re
import sys
import tarfile
import zipfile


_TEXT_SUFFIXES = frozenset(
    {".css", ".ini", ".md", ".toml", ".txt", ".yaml", ".yml"}
)
_TEXT_FILENAMES = frozenset({"LICENSE", "MANIFEST.in"})
_POSIX_LOCAL_PATH = re.compile(
    # `${VAR}/...`、`{tox_root}/...` 和 `<run_dir>/...` 都由调用环境解析，
    # 不属于写死的本地绝对路径；真实的多段绝对路径仍会命中。
    r"(?<![A-Za-z0-9_:/.\-}>$)\]])"
    r"/(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+"
)
_WINDOWS_LOCAL_PATH = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"[A-Za-z]:[\\/](?:[^\\/:*?\"<>|\r\n]+[\\/])+"
    r"[^\\/:*?\"<>|\r\n]+"
)


def _python_files(source_root: Path) -> tuple[Path, ...]:
    """枚举 production、portable tests、文档示例与 CI helper 的 Python 源码。"""

    roots = (
        source_root / "__init__.py",
        source_root / "cli.py",
        source_root / "base",
        source_root / "extensions",
        source_root / "host_runtime",
        source_root / "integration",
        source_root / "tests",
        source_root / "docs",
        source_root / ".github" / "scripts",
    )
    files: list[Path] = []
    for root in roots:
        if root.is_file():
            files.append(root)
        elif root.is_dir():
            files.extend(root.rglob("*.py"))
    return tuple(sorted(files))


def _non_python_text_files(source_root: Path) -> tuple[Path, ...]:
    """枚举会进入源码发行或发布配置的非 Python 文本。"""

    roots = (
        source_root / "README.md",
        source_root / "LICENSE",
        source_root / "MANIFEST.in",
        source_root / "pyproject.toml",
        source_root / "pytest.ini",
        source_root / "tox.ini",
        source_root / "base",
        source_root / "extensions",
        source_root / "host_runtime",
        source_root / "integration",
        source_root / "tests",
        source_root / "docs",
        source_root / ".github",
    )
    files: set[Path] = set()
    for root in roots:
        candidates = (root,) if root.is_file() else root.rglob("*") if root.is_dir() else ()
        for path in candidates:
            if (
                path.is_file()
                and path.suffix != ".py"
                and (
                    path.suffix.lower() in _TEXT_SUFFIXES
                    or path.name in _TEXT_FILENAMES
                )
            ):
                files.add(path)
    return tuple(sorted(files))


def _current_package(source_root: Path, path: Path) -> str:
    """把源码路径转换为 import 解析所需的当前 package。"""

    relative = path.relative_to(source_root)
    if relative.name == "__init__.py":
        suffix = relative.parent.parts
    else:
        suffix = relative.parent.parts
    return ".".join(("model_diagnostics", *suffix))


def _resolved_imports(source_root: Path, path: Path) -> tuple[str, ...]:
    """把相对/绝对 import 都解析为可比较的绝对 module。"""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[str] = []
    package = _current_package(source_root, path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module:
                    modules.append(node.module)
                continue
            relative_name = "." * node.level + (node.module or "")
            try:
                modules.append(importlib.util.resolve_name(relative_name, package))
            except (ImportError, ValueError):
                modules.append("<distribution-root-escape>")
    return tuple(modules)


def _audit_imports(source_root: Path, files: tuple[Path, ...]) -> list[str]:
    """验证 portable 子包的正向依赖 contract。"""

    errors: list[str] = []
    stdlib = sys.stdlib_module_names
    for path in files:
        relative = path.relative_to(source_root)
        in_base = relative.parts[0] == "base"
        in_extensions = relative.parts[0] == "extensions"
        in_host_runtime = relative.parts[0] == "host_runtime"
        in_integration = relative.parts[0] == "integration"
        in_cli = relative == Path("cli.py")
        in_tests = relative.parts[0] == "tests"
        for module in _resolved_imports(source_root, path):
            root = module.partition(".")[0]
            if (
                root in stdlib
                or root == "torch"
                or in_tests and root in {"pytest", "tomli", "tomllib"}
            ):
                continue
            if module == "<distribution-root-escape>":
                errors.append(f"{relative}: relative import escapes distribution root")
            elif root != "model_diagnostics":
                errors.append(f"{relative}: unsupported dependency {module!r}")
            elif in_base and not (
                module == "model_diagnostics.base"
                or module.startswith("model_diagnostics.base.")
            ):
                errors.append(f"{relative}: Base leaves its package boundary via {module!r}")
            elif in_extensions and not (
                module == "model_diagnostics.base"
                or module.startswith("model_diagnostics.base.")
                or module == "model_diagnostics.extensions"
                or module.startswith("model_diagnostics.extensions.")
            ):
                errors.append(
                    f"{relative}: extension leaves its package boundary via {module!r}"
                )
            elif in_host_runtime and not (
                module == "model_diagnostics.host_runtime"
                or module.startswith("model_diagnostics.host_runtime.")
            ):
                errors.append(
                    f"{relative}: Host Runtime leaves its package boundary via {module!r}"
                )
            elif in_integration and not (
                module == "model_diagnostics.base"
                or module.startswith("model_diagnostics.base.")
                or module == "model_diagnostics.extensions"
                or module.startswith("model_diagnostics.extensions.")
                or module == "model_diagnostics.host_runtime"
                or module.startswith("model_diagnostics.host_runtime.")
                or module == "model_diagnostics.integration"
                or module.startswith("model_diagnostics.integration.")
            ):
                errors.append(
                    f"{relative}: integration leaves its package boundary via {module!r}"
                )
            elif in_cli and not (
                module == "model_diagnostics.base"
                or module.startswith("model_diagnostics.base.")
                or module == "model_diagnostics.extensions"
                or module.startswith("model_diagnostics.extensions.")
                or module == "model_diagnostics.cli"
                or module.startswith("model_diagnostics.cli.")
            ):
                errors.append(
                    f"{relative}: CLI leaves its package boundary via {module!r}"
                )
            elif not any(
                (in_base, in_extensions, in_host_runtime, in_integration, in_cli, in_tests)
            ):
                errors.append(
                    f"{relative}: top-level/CI code imports runtime package {module!r}"
                )
    return errors


def _audit_path_boundaries(
    source_root: Path,
    files: tuple[Path, ...],
) -> list[str]:
    """拒绝源码中越出 package 的相对 import 和绝对本地文件路径。"""

    errors: list[str] = []
    for path in files:
        relative = path.relative_to(source_root)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(
                node.value,
                str,
            ):
                continue
            value = node.value.strip()
            posix_parts = PurePosixPath(value).parts
            posix_absolute = (
                value.startswith("/")
                and len(posix_parts) >= 3
                and not value.endswith("/")
                and not value.startswith("//")
            )
            windows_absolute = (
                len(value) >= 3
                and value[0].isalpha()
                and value[1] == ":"
                and value[2] in {"\\", "/"}
            )
            if posix_absolute or windows_absolute:
                errors.append(
                    f"{relative}: absolute local path literal is not portable"
                )
    return errors


def _absolute_local_path_matches(text: str) -> tuple[str, ...]:
    """返回文本中的本地绝对路径；URL 与 package-relative 路径不会命中。"""

    matches = {
        match.group(0)
        for pattern in (_POSIX_LOCAL_PATH, _WINDOWS_LOCAL_PATH)
        for match in pattern.finditer(text)
    }
    return tuple(sorted(matches))


def _audit_text_path_boundaries(
    source_root: Path,
    files: tuple[Path, ...],
) -> list[str]:
    """拒绝文档、配置和 workflow 中不可迁移的本地绝对路径。"""

    errors: list[str] = []
    for path in files:
        relative = path.relative_to(source_root)
        text = path.read_text(encoding="utf-8")
        for match in _absolute_local_path_matches(text):
            errors.append(
                f"{relative}: absolute local path is not portable: {match!r}"
            )
    return errors


def _archive_members(path: Path) -> tuple[str, ...]:
    """读取 wheel 或 gzip sdist 的 POSIX member 名。"""

    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            return tuple(archive.namelist())
    if path.name.endswith(".tar.gz"):
        with tarfile.open(path, mode="r:gz") as archive:
            return tuple(archive.getnames())
    raise ValueError(f"unsupported distribution artifact: {path}")


def _archive_text_members(path: Path) -> tuple[tuple[str, str], ...]:
    """读取 archive 中可审计的 UTF-8 文本，不把 member 解压到文件系统。"""

    records: list[tuple[str, str]] = []
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                name = PurePosixPath(member.filename)
                if (
                    member.is_dir()
                    or (
                        name.suffix.lower() not in _TEXT_SUFFIXES | {".py"}
                        and name.name not in _TEXT_FILENAMES | {"METADATA", "WHEEL"}
                    )
                ):
                    continue
                records.append(
                    (
                        member.filename,
                        archive.read(member).decode("utf-8"),
                    )
                )
        return tuple(records)
    if path.name.endswith(".tar.gz"):
        with tarfile.open(path, mode="r:gz") as archive:
            for member in archive.getmembers():
                name = PurePosixPath(member.name)
                if (
                    not member.isfile()
                    or (
                        name.suffix.lower() not in _TEXT_SUFFIXES | {".py"}
                        and name.name
                        not in _TEXT_FILENAMES | {"PKG-INFO", "METADATA", "WHEEL"}
                    )
                ):
                    continue
                handle = archive.extractfile(member)
                if handle is None:
                    continue
                records.append(
                    (
                        member.name,
                        handle.read().decode("utf-8"),
                    )
                )
        return tuple(records)
    raise ValueError(f"unsupported distribution artifact: {path}")


def _audit_archive(path: Path) -> list[str]:
    """按发行 allow-list 验证构建产物边界和文件类型。"""

    errors: list[str] = []
    is_wheel = path.suffix == ".whl"
    members = _archive_members(path)
    sdist_root = (
        PurePosixPath(members[0]).parts[0]
        if members and not is_wheel
        else None
    )
    source_top_level = {
        "LICENSE",
        "README.md",
        "MANIFEST.in",
        "pyproject.toml",
        "pytest.ini",
        "__init__.py",
        "cli.py",
        "AGENTS.md",
        "base",
        "extensions",
        "host_runtime",
        "integration",
        "docs",
        "tests",
        "tox.ini",
        ".github",
        "PKG-INFO",
        "setup.cfg",
    }
    for raw_name in members:
        name = PurePosixPath(raw_name)
        parts = set(name.parts)
        if ".cache" in parts or "__pycache__" in parts:
            errors.append(f"{path.name}: contains cache path {raw_name}")
        if name.suffix in {".pyc", ".pyo"}:
            errors.append(f"{path.name}: contains bytecode {raw_name}")
        if is_wheel:
            first = name.parts[0] if name.parts else ""
            runtime_member = (
                first == "model_diagnostics"
                and len(name.parts) >= 2
                and name.parts[1]
                in {
                    "__init__.py",
                    "cli.py",
                    "base",
                    "extensions",
                    "host_runtime",
                    "integration",
                }
            )
            metadata_member = (
                re.fullmatch(
                    r"model_diagnostics-[0-9][A-Za-z0-9_.+!-]*\.dist-info",
                    first,
                )
                is not None
            )
            if metadata_member:
                metadata_parts = name.parts[1:]
                metadata_member = (
                    len(metadata_parts) == 1
                    and metadata_parts[0]
                    in {"METADATA", "WHEEL", "top_level.txt", "RECORD"}
                    or metadata_parts == ("licenses", "LICENSE")
                )
            if not runtime_member and not metadata_member:
                errors.append(
                    f"{path.name}: wheel member leaves runtime allow-list {raw_name}"
                )
            if {"tests", "docs"} & parts:
                errors.append(
                    f"{path.name}: wheel contains non-runtime file {raw_name}"
                )
            continue
        relative_parts = name.parts[1:] if name.parts else ()
        if (
            not name.parts
            or name.parts[0] != sdist_root
            or (
                relative_parts
                and relative_parts[0] not in source_top_level
                and not relative_parts[0].endswith(".egg-info")
            )
        ):
            errors.append(
                f"{path.name}: sdist member leaves source allow-list {raw_name}"
            )
    for raw_name, text in _archive_text_members(path):
        for match in _absolute_local_path_matches(text):
            errors.append(
                f"{path.name}:{raw_name}: absolute local path is not portable: "
                f"{match!r}"
            )
    return errors


def audit(source_root: Path, dist_dir: Path | None) -> tuple[str, ...]:
    """运行源码与可选发行 artifact 审计并返回全部错误。"""

    source_root = source_root.resolve()
    files = _python_files(source_root)
    errors = _audit_imports(source_root, files)
    errors.extend(_audit_path_boundaries(source_root, files))
    errors.extend(
        _audit_text_path_boundaries(
            source_root,
            _non_python_text_files(source_root),
        )
    )
    if dist_dir is not None:
        artifacts = sorted(dist_dir.resolve().glob("*.whl"))
        artifacts.extend(sorted(dist_dir.resolve().glob("*.tar.gz")))
        if not artifacts:
            errors.append(f"no wheel or sdist found under {dist_dir}")
        for artifact in artifacts:
            errors.extend(_audit_archive(artifact))
    return tuple(errors)


def main(argv: list[str] | None = None) -> int:
    """解析 CLI 并以非零状态拒绝任何边界违规。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("--dist-dir", type=Path, default=None)
    arguments = parser.parse_args(argv)
    errors = audit(arguments.source_root, arguments.dist_dir)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("portable source and distribution audit passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
