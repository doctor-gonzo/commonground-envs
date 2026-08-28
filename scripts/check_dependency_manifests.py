"""Generate or verify the environment dependency manifests from ``uv.lock``."""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "uv.lock"
ROOT_PYPROJECT = ROOT / "pyproject.toml"
MANIFEST_NAME = "dependency-manifest.txt"
_EDITABLE_REQUIREMENT = re.compile(r"^-e\s+(?P<path>\S+)\s*$")
_WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:[/\\]")
_WINDOWS_DIRECT_PATH = re.compile(r"\s@\s+[A-Za-z]:[/\\]")


@dataclass(frozen=True)
class ManifestTarget:
    distribution: str
    project_dir: Path
    import_package: str

    @property
    def manifest_path(self) -> Path:
        return self.project_dir / self.import_package / MANIFEST_NAME


TARGETS = (
    ManifestTarget(
        distribution="commonground-predict",
        project_dir=ROOT / "environments" / "commonground_predict",
        import_package="commonground_predict",
    ),
    ManifestTarget(
        distribution="commonground-elicit",
        project_dir=ROOT / "environments" / "commonground_elicit",
        import_package="commonground_elicit",
    ),
)


def _normalized_workspace_path(path: str) -> str:
    normalized = PurePosixPath(path.replace("\\", "/")).as_posix()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def workspace_pins(lock_document: Mapping[str, Any]) -> dict[str, str]:
    """Map each locked workspace source path to its immutable distribution pin."""

    packages = lock_document.get("package")
    if not isinstance(packages, list):
        raise ValueError("uv.lock has no package array")

    pins: dict[str, str] = {}
    for package in packages:
        if not isinstance(package, dict):
            raise ValueError("uv.lock contains a non-table package entry")
        source = package.get("source")
        if not isinstance(source, dict):
            continue
        workspace_path = source.get("editable")
        if workspace_path is None:
            workspace_path = source.get("virtual")
        if workspace_path is None:
            continue
        name = package.get("name")
        version = package.get("version")
        if not all(isinstance(value, str) for value in (workspace_path, name, version)):
            raise ValueError(
                "workspace lock entries require string path, name, and version"
            )
        path_key = _normalized_workspace_path(workspace_path)
        pin = f"{name}=={version}"
        existing = pins.get(path_key)
        if existing is not None and existing != pin:
            raise ValueError(f"workspace path {path_key!r} has conflicting pins")
        pins[path_key] = pin
    return pins


def canonicalize_export(exported: str, pins: Mapping[str, str]) -> str:
    """Replace uv workspace paths with exact pins and reject local references."""

    lines: list[str] = []
    replaced_paths: set[str] = set()
    for raw_line in exported.replace("\r\n", "\n").splitlines():
        line = raw_line.rstrip()
        editable = _EDITABLE_REQUIREMENT.fullmatch(line)
        if editable is not None:
            path_key = _normalized_workspace_path(editable.group("path"))
            pin = pins.get(path_key)
            if pin is None:
                raise ValueError(
                    f"uv export contains unknown workspace path {path_key!r}"
                )
            line = pin
            replaced_paths.add(path_key)
        lines.append(line)

    if not replaced_paths:
        raise ValueError("uv export did not contain any workspace paths to pin")
    canonical = "\n".join(lines).rstrip() + "\n"
    assert_no_local_paths(canonical)
    return canonical


def assert_no_local_paths(requirements: str) -> None:
    """Fail if canonical requirements retain a relative or absolute local path."""

    for line in requirements.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        lowered = stripped.casefold()
        if (
            stripped.startswith(("/", "./", "../", "\\"))
            or _WINDOWS_ABSOLUTE_PATH.match(stripped)
            or lowered.startswith("-e ")
            or "file:" in lowered
            or " @ /" in stripped
            or " @ \\" in stripped
            or _WINDOWS_DIRECT_PATH.search(stripped)
        ):
            raise ValueError(f"dependency manifest contains a local path: {stripped!r}")


def _project_scope(target: ManifestTarget) -> tuple[str, str]:
    document = tomllib.loads(
        (target.project_dir / "pyproject.toml").read_text(encoding="utf-8")
    )
    project = document.get("project")
    if not isinstance(project, dict):
        raise ValueError(f"{target.distribution}: pyproject.toml has no project table")
    name = project.get("name")
    version = project.get("version")
    requires_python = project.get("requires-python")
    if name != target.distribution or not isinstance(version, str):
        raise ValueError(f"{target.distribution}: unexpected project name or version")
    if not isinstance(requires_python, str):
        raise ValueError(f"{target.distribution}: requires-python must be a string")
    return version, requires_python


def _required_uv_version() -> str:
    document = tomllib.loads(ROOT_PYPROJECT.read_text(encoding="utf-8"))
    tool = document.get("tool")
    uv = tool.get("uv") if isinstance(tool, dict) else None
    required_version = uv.get("required-version") if isinstance(uv, dict) else None
    if not isinstance(required_version, str) or not required_version.startswith("=="):
        raise ValueError("tool.uv.required-version must be an exact == pin")
    version = required_version.removeprefix("==")
    if not version or any(character.isspace() for character in version):
        raise ValueError("tool.uv.required-version contains an invalid version")
    return version


def _export_requirements(distribution: str) -> str:
    command = [
        "uv",
        "export",
        "--no-cache",
        "--offline",
        "--locked",
        "--package",
        distribution,
        "--no-dev",
        "--format",
        "requirements.txt",
        "--no-header",
    ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout


def render_manifest(
    target: ManifestTarget,
    *,
    lock_document: Mapping[str, Any],
    exported: str,
) -> str:
    """Render one deterministic manifest without timestamps or host details."""

    version, requires_python = _project_scope(target)
    requirements = canonicalize_export(exported, workspace_pins(lock_document))
    expected_self_pin = f"{target.distribution}=={version}"
    if expected_self_pin not in requirements.splitlines():
        raise ValueError(
            f"{target.distribution}: export is missing {expected_self_pin}"
        )
    resolution_sha256 = hashlib.sha256(requirements.encode("utf-8")).hexdigest()
    uv_version = _required_uv_version()
    header = (
        "# Common Ground resolved dependency manifest\n"
        "#\n"
        f"# Distribution: {expected_self_pin}\n"
        f"# Python-Requires: {requires_python}\n"
        "# Lock-File: uv.lock\n"
        f"# Resolution-SHA256: {resolution_sha256}\n"
        f"# uv-Version: {uv_version}\n"
        "# Generated-By: scripts/check_dependency_manifests.py\n"
        "#\n"
        "# Exact no-dev resolution for this release. Workspace members are\n"
        "# represented as immutable distribution pins; this file is provenance,\n"
        "# not a substitute for the platform-specific installation resolver.\n"
    )
    return header + requirements


def expected_manifests() -> dict[Path, str]:
    lock_bytes = LOCK_PATH.read_bytes()
    lock_document = tomllib.loads(lock_bytes.decode("utf-8"))
    return {
        target.manifest_path: render_manifest(
            target,
            lock_document=lock_document,
            exported=_export_requirements(target.distribution),
        )
        for target in TARGETS
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="replace the manifests")
    mode.add_argument("--check", action="store_true", help="verify committed bytes")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    failures: list[Path] = []
    for path, expected in expected_manifests().items():
        relative_path = path.relative_to(ROOT)
        if args.write:
            path.write_text(expected, encoding="utf-8", newline="\n")
            print(f"wrote {relative_path}")
        elif not path.is_file() or path.read_bytes() != expected.encode("utf-8"):
            failures.append(relative_path)

    if failures:
        for path in failures:
            print(
                f"{path}: missing or stale; regenerate with --write",
                file=sys.stderr,
            )
        return 1
    if args.check:
        print("dependency manifests match the exact locked environment")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
