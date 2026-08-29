from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_URL = "https://github.com/doctor-gonzo/commonground-envs"
PROJECTS = {
    "commonground-predict": (
        ROOT / "environments" / "commonground_predict",
        "commonground_predict",
        "0.4.1",
        [
            "commonground-scenarios==0.3.0",
            "commonground-score==0.3.0",
            "datasets>=5.0.1,<6.0.0",
            "verifiers==0.3.0",
        ],
    ),
    "commonground-elicit": (
        ROOT / "environments" / "commonground_elicit",
        "commonground_elicit",
        "0.4.1",
        [
            "commonground-scenarios==0.3.0",
            "commonground-score==0.3.0",
            "datasets>=5.0.1,<6.0.0",
            "verifiers==0.3.0",
        ],
    ),
    "commonground-score": (
        ROOT / "packages" / "commonground-score",
        "commonground_score",
        "0.3.0",
        [],
    ),
    "commonground-scenarios": (
        ROOT / "packages" / "commonground-scenarios",
        "commonground_scenarios",
        "0.3.0",
        [],
    ),
}

EXPECTED_ENVIRONMENT_DESCRIPTIONS = {
    "commonground-predict": (
        "Probabilistic masked-vote prediction over synthetic stakeholder panels."
    ),
    "commonground-elicit": (
        "Structured policy-issue diagnosis and top-k clarification over synthetic "
        "stakeholder scenarios."
    ),
}


@pytest.mark.parametrize(("distribution", "project"), PROJECTS.items())
def test_release_metadata_is_immutable_and_publication_ready(
    distribution: str,
    project: tuple[Path, str, str, list[str]],
) -> None:
    project_dir, _, expected_version, expected_dependencies = project
    metadata = tomllib.loads(
        (project_dir / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]

    assert metadata["name"] == distribution
    assert metadata["version"] == expected_version
    assert metadata["dependencies"] == expected_dependencies
    assert metadata["urls"]["Repository"] == REPOSITORY_URL
    if distribution in EXPECTED_ENVIRONMENT_DESCRIPTIONS:
        assert (
            metadata["description"] == EXPECTED_ENVIRONMENT_DESCRIPTIONS[distribution]
        )
        assert "Context Engine" not in metadata["description"]
    assert (project_dir / "LICENSE").read_bytes() == (ROOT / "LICENSE").read_bytes()


@pytest.mark.parametrize(("distribution", "project"), PROJECTS.items())
def test_runtime_version_comes_from_installed_metadata(
    distribution: str,
    project: tuple[Path, str, str, list[str]],
) -> None:
    project_dir, module_name, _, _ = project
    source_root = project_dir
    if distribution in {"commonground-score", "commonground-scenarios"}:
        source_root /= "src"
    init_source = (source_root / module_name / "__init__.py").read_text(
        encoding="utf-8"
    )

    assert f'version("{distribution}")' in init_source
    assert '__version__ = "0+unknown"' in init_source
    assert '__version__ = "0.' not in init_source


def test_predict_declares_complete_multi_license_boundary() -> None:
    project_dir = ROOT / "environments" / "commonground_predict"
    metadata = tomllib.loads(
        (project_dir / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]

    assert metadata["license-files"] == [
        "LICENSE",
        "LICENSES/Apache-2.0.txt",
        "LICENSES/MPL-2.0.txt",
        "LICENSES/NOTICE.txt",
        "NOTICE",
    ]
    assert metadata["license"] == "Apache-2.0 AND MPL-2.0"
    assert (project_dir / "LICENSES" / "Apache-2.0.txt").read_bytes() == (
        project_dir / "LICENSE"
    ).read_bytes()
    assert (project_dir / "LICENSES" / "MPL-2.0.txt").is_file()
    assert (project_dir / "LICENSES" / "NOTICE.txt").read_bytes() == (
        project_dir / "NOTICE"
    ).read_bytes()
    assert (project_dir / "NOTICE").is_file()


def _prime_hub_source_archive_members(project_dir: Path) -> set[str]:
    """Model the source collector shipped by Prime CLI 0.6.28 and 0.6.29."""

    members = {
        path.relative_to(project_dir).as_posix()
        for pattern in ("README.md", "pyproject.toml", "*.py")
        for path in project_dir.glob(pattern)
        if path.is_file() and not path.is_symlink() and not path.name.startswith(".")
    }
    excluded_directories = {"dist", "__pycache__", "build", "outputs"}
    for path in project_dir.rglob("*"):
        relative = path.relative_to(project_dir)
        if len(relative.parts) < 2 or not path.is_file() or path.is_symlink():
            continue
        parent_parts = relative.parts[:-1]
        if any(part.startswith(".") for part in relative.parts):
            continue
        if any(
            part in excluded_directories or part.endswith(".egg-info")
            for part in parent_parts
        ):
            continue
        members.add(relative.as_posix())
    return members


@pytest.mark.parametrize(
    ("environment_name", "required_legal_files"),
    [
        (
            "commonground_predict",
            {
                "LICENSES/Apache-2.0.txt",
                "LICENSES/MPL-2.0.txt",
                "LICENSES/NOTICE.txt",
            },
        ),
        ("commonground_elicit", {"LICENSES/Apache-2.0.txt"}),
    ],
)
def test_prime_hub_source_archive_contains_legal_files(
    environment_name: str, required_legal_files: set[str]
) -> None:
    project_dir = ROOT / "environments" / environment_name
    archive_members = _prime_hub_source_archive_members(project_dir)

    assert required_legal_files <= archive_members
    assert (project_dir / "LICENSES" / "Apache-2.0.txt").read_bytes() == (
        project_dir / "LICENSE"
    ).read_bytes()


def test_predict_notice_pins_demo_fixture_provenance_and_content() -> None:
    project_dir = ROOT / "environments" / "commonground_predict"
    notice = (project_dir / "NOTICE").read_text(encoding="utf-8")
    fixture = (
        project_dir / "commonground_predict" / "data" / "eval_ce_demo.jsonl"
    ).read_bytes()
    fixture_sha256 = hashlib.sha256(fixture).hexdigest()

    assert "7650531e926181d1ed24b25da085b34f4200eb7b" in notice
    assert "client/src/variables/demo/demo_polis_data.json" in notice
    assert "commonground-ce-demo-v1" in notice
    assert "255f6cad34d482aa36abdcb0828365ab256cb811fed04cb036375cfcb5cd02f0" in notice
    assert fixture_sha256 == (
        "f459c6bb2524806e17648fae737a8aae18b58402a81de9b4d4920b23ee7eb909"
    )
    assert fixture_sha256 in notice
    assert "67267c93457252b6bcb730818933e8a94f1332c2" in notice
    assert "Mozilla Public License, Version 2.0" in notice


def test_local_smoke_runs_offline_tests_before_authentication() -> None:
    script = (ROOT / "scripts" / "local_smoke.sh").read_text(encoding="utf-8")

    assert script.index("uv sync --all-packages --locked") < script.index(
        "uv run pytest -q"
    )
    assert script.index("uv run pytest -q") < script.index("prime --plain whoami")
    assert script.index("uv run validate commonground-predict") < script.index(
        "prime --plain whoami"
    )
    assert 'source "$HOME/.config/prime/env"' not in script
    assert "PASTE-KEY-HERE" not in script
    assert "prime config set-api-key" in script
    assert "prime --plain eval run" not in script
    assert script.count("uv run eval commonground-") == 3
    assert script.count("--no-push") == 3
    assert "--runtime.type subprocess" in script


def test_ci_audits_the_locked_third_party_dependency_set() -> None:
    workflow = (ROOT / ".github" / "workflows" / "test.yml").read_text(encoding="utf-8")

    assert "uv export --locked --all-packages --no-dev --no-emit-workspace" in workflow
    assert "uvx --from pip-audit==2.10.1 pip-audit" in workflow
    assert "--no-deps --disable-pip" in workflow


@pytest.mark.parametrize(
    "environment_dir",
    [
        ROOT / "environments" / "commonground_predict",
        ROOT / "environments" / "commonground_elicit",
    ],
)
def test_every_shipped_test_passes_from_detached_environment_source(
    tmp_path: Path, environment_dir: Path
) -> None:
    detached_source = tmp_path / environment_dir.name
    shutil.copytree(
        environment_dir,
        detached_source,
        ignore=shutil.ignore_patterns(".prime", "outputs", "__pycache__", "*.pyc"),
    )

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests"],
        cwd=detached_source,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 skipped" in result.stdout


def test_built_release_artifacts_pass_policy_checks() -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_release_wheel.py")],
        cwd=ROOT,
        check=True,
    )
