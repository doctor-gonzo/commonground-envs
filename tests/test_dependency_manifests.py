from __future__ import annotations

import importlib.util
import subprocess
import sys
import tomllib
from copy import deepcopy
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_manifest_script() -> ModuleType:
    script_path = ROOT / "scripts" / "check_dependency_manifests.py"
    spec = importlib.util.spec_from_file_location(
        "commonground_check_dependency_manifests", script_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


manifests = _load_manifest_script()


def test_workspace_paths_are_replaced_with_exact_locked_pins() -> None:
    exported = (
        "-e ./environments/commonground_predict\r\n"
        "-e .\\packages\\commonground-score\r\n"
        "    # via commonground-predict\r\n"
        "verifiers==0.3.0\r\n"
    )

    rendered = manifests.canonicalize_export(
        exported,
        {
            "environments/commonground_predict": "commonground-predict==0.2.5",
            "packages/commonground-score": "commonground-score==0.1.1",
        },
    )

    assert rendered == (
        "commonground-predict==0.2.5\n"
        "commonground-score==0.1.1\n"
        "    # via commonground-predict\n"
        "verifiers==0.3.0\n"
    )


@pytest.mark.parametrize(
    "requirement",
    [
        "-e ./packages/private",
        "package @ file:///private/tmp/package",
        "package @ /private/tmp/package",
        "package @ C:\\private\\package",
    ],
)
def test_manifest_rejects_local_dependency_paths(requirement: str) -> None:
    with pytest.raises(ValueError, match="local path"):
        manifests.assert_no_local_paths(requirement)


def test_checked_in_manifests_match_the_exact_lock() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "check_dependency_manifests.py"),
            "--check",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "match the exact locked environment" in result.stdout


@pytest.mark.parametrize(
    ("distribution", "import_package", "expected_version"),
    [
        ("commonground-predict", "commonground_predict", "0.2.5"),
        ("commonground-elicit", "commonground_elicit", "0.2.5"),
    ],
)
def test_manifest_records_release_scope_without_local_paths(
    distribution: str, import_package: str, expected_version: str
) -> None:
    manifest_path = (
        ROOT
        / "environments"
        / import_package
        / import_package
        / manifests.MANIFEST_NAME
    )
    content = manifest_path.read_text(encoding="utf-8")
    assert f"# Distribution: {distribution}=={expected_version}" in content
    assert "# Python-Requires: >=3.12,<3.13" in content
    assert "# Resolution-SHA256: " in content
    assert "# Lock-SHA256: " not in content
    assert "# uv-Version: 0.10.9" in content
    assert "commonground-scenarios==0.1.1" in content
    assert "commonground-score==0.1.1" in content
    assert "datasets==5.0.1" in content
    assert "verifiers==0.3.0" in content
    assert "--hash=sha256:" in content
    assert "-e " not in content
    assert " @ file:" not in content.casefold()
    assert str(ROOT) not in content
    manifests.assert_no_local_paths(content)


def test_unrelated_workspace_version_does_not_change_manifest() -> None:
    lock_document = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    changed_lock_document = deepcopy(lock_document)
    for package in changed_lock_document["package"]:
        if package.get("name") == "commonground-predict":
            package["version"] = "99.99.99"
            break
    else:
        raise AssertionError("commonground-predict is missing from uv.lock")

    target = next(
        target
        for target in manifests.TARGETS
        if target.distribution == "commonground-elicit"
    )
    exported = "-e ./environments/commonground_elicit\nverifiers==0.3.0\n"

    before = manifests.render_manifest(
        target,
        lock_document=lock_document,
        exported=exported,
    )
    after = manifests.render_manifest(
        target,
        lock_document=changed_lock_document,
        exported=exported,
    )

    assert before == after


def test_ci_and_release_checker_enforce_manifest_freshness() -> None:
    workflow = (ROOT / ".github" / "workflows" / "test.yml").read_text(encoding="utf-8")
    release_checker = (ROOT / "scripts" / "check_release_wheel.py").read_text(
        encoding="utf-8"
    )

    expected = "python scripts/check_dependency_manifests.py --check"
    assert expected in workflow
    assert workflow.count('version: "0.10.9"') == 4
    assert "check_dependency_manifests.py" in release_checker
    assert manifests.MANIFEST_NAME in release_checker

    root_project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'required-version = "==0.10.9"' in root_project
