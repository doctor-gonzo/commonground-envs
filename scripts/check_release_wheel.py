"""Build and inspect every Common Ground release wheel."""

from __future__ import annotations

import argparse
import email
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANONICAL_LICENSE = ROOT / "LICENSE"
REQUIRES_PYTHON_PARTS = frozenset({">=3.12", "<3.13"})
EXPECTED_AUTHOR = "Common Ground contributors"
EXPECTED_LICENSE = "Apache-2.0"
EXPECTED_REPOSITORY_URL = "https://github.com/doctor-gonzo/commonground-envs"
REQUIRED_CLASSIFIERS = frozenset(
    {
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    }
)
REPOSITORY_TOOLING = frozenset(
    {
        "aggregate_baselines.py",
        "baseline-sweep.toml",
    }
)
DEPENDENCY_MANIFEST_NAME = "dependency-manifest.txt"
DEPENDENCY_MANIFEST_PATHS = (
    ROOT
    / "environments"
    / "commonground_predict"
    / "commonground_predict"
    / DEPENDENCY_MANIFEST_NAME,
    ROOT
    / "environments"
    / "commonground_elicit"
    / "commonground_elicit"
    / DEPENDENCY_MANIFEST_NAME,
)


@dataclass(frozen=True)
class WheelTarget:
    name: str
    version: str
    source_dir: Path
    requirements: frozenset[str]
    bundled_files: frozenset[str]
    license_expression: str = EXPECTED_LICENSE
    legal_files: frozenset[str] = frozenset({"LICENSE"})


def _data_files(source_dir: Path, package_name: str) -> frozenset[str]:
    data_dir = source_dir / package_name / "data"
    return frozenset(
        f"{package_name}/data/{source_path.name}"
        for source_path in data_dir.glob("*.jsonl")
    )


TARGETS = (
    WheelTarget(
        name="commonground-predict",
        version="0.6.1",
        source_dir=ROOT / "environments" / "commonground_predict",
        requirements=frozenset(
            {
                "commonground-scenarios==0.6.1",
                "commonground-score==0.6.1",
                "datasets<6.0.0,>=5.0.1",
                "verifiers==0.3.0",
            }
        ),
        bundled_files=_data_files(
            ROOT / "environments" / "commonground_predict", "commonground_predict"
        )
        | frozenset({f"commonground_predict/{DEPENDENCY_MANIFEST_NAME}"}),
        license_expression="Apache-2.0 AND MPL-2.0",
        legal_files=frozenset(
            {
                "LICENSE",
                "LICENSES/Apache-2.0.txt",
                "LICENSES/MPL-2.0.txt",
                "LICENSES/NOTICE.txt",
                "NOTICE",
            }
        ),
    ),
    WheelTarget(
        name="commonground-elicit",
        version="0.6.1",
        source_dir=ROOT / "environments" / "commonground_elicit",
        requirements=frozenset(
            {
                "commonground-scenarios==0.6.1",
                "commonground-score==0.6.1",
                "datasets<6.0.0,>=5.0.1",
                "verifiers==0.3.0",
            }
        ),
        bundled_files=_data_files(
            ROOT / "environments" / "commonground_elicit", "commonground_elicit"
        )
        | frozenset(
            {
                "commonground_elicit/data/README.md",
                f"commonground_elicit/{DEPENDENCY_MANIFEST_NAME}",
            }
        ),
        legal_files=frozenset({"LICENSE", "LICENSES/Apache-2.0.txt"}),
    ),
    WheelTarget(
        name="commonground-scenarios",
        version="0.6.1",
        source_dir=ROOT / "packages" / "commonground-scenarios",
        requirements=frozenset(),
        bundled_files=frozenset({"commonground_scenarios/schema/scenario.schema.json"}),
    ),
    WheelTarget(
        name="commonground-score",
        version="0.6.1",
        source_dir=ROOT / "packages" / "commonground-score",
        requirements=frozenset(),
        bundled_files=frozenset(),
    ),
)


def _bundled_source_path(target: WheelTarget, relative_path: str) -> Path:
    direct_path = target.source_dir / relative_path
    if direct_path.is_file():
        return direct_path
    src_path = target.source_dir / "src" / relative_path
    if src_path.is_file():
        return src_path
    raise AssertionError(
        f"{target.name}: bundled source file does not exist: {relative_path}"
    )


def _normalized_requirement(requirement: str) -> str:
    return requirement.lower().replace(" ", "")


def _inspect_wheel(target: WheelTarget, wheel_path: Path) -> int:
    with zipfile.ZipFile(wheel_path) as wheel:
        names = set(wheel.namelist())
        metadata_names = sorted(
            name for name in names if name.endswith(".dist-info/METADATA")
        )
        if len(metadata_names) != 1:
            raise AssertionError(
                f"{target.name}: expected one METADATA file, found {len(metadata_names)}"
            )

        metadata = email.message_from_bytes(wheel.read(metadata_names[0]))
        if metadata["Name"] != target.name:
            raise AssertionError(
                f"expected wheel for {target.name}, found {metadata['Name']!r}"
            )
        if metadata["Version"] != target.version:
            raise AssertionError(
                f"{target.name}: expected version {target.version!r}; "
                f"found {metadata['Version']!r}"
            )
        if not metadata["Summary"] or metadata["Summary"] == "UNKNOWN":
            raise AssertionError(f"{target.name}: wheel metadata has no description")
        if metadata["Author"] != EXPECTED_AUTHOR:
            raise AssertionError(
                f"{target.name}: expected author {EXPECTED_AUTHOR!r}; "
                f"found {metadata['Author']!r}"
            )
        license_value = metadata["License-Expression"] or metadata["License"]
        if license_value != target.license_expression:
            raise AssertionError(
                f"{target.name}: expected license {target.license_expression!r}; "
                f"found {license_value!r}"
            )
        classifiers = frozenset(metadata.get_all("Classifier", []))
        missing_classifiers = sorted(REQUIRED_CLASSIFIERS - classifiers)
        if missing_classifiers:
            raise AssertionError(
                f"{target.name}: wheel metadata is missing classifiers: "
                f"{missing_classifiers!r}"
            )
        project_urls = metadata.get_all("Project-URL", [])
        repository_entry = f"Repository, {EXPECTED_REPOSITORY_URL}"
        if repository_entry not in project_urls:
            raise AssertionError(
                f"{target.name}: expected project URL {repository_entry!r}; "
                f"found {project_urls!r}"
            )
        python_parts = frozenset(
            part.strip() for part in metadata["Requires-Python"].split(",")
        )
        if python_parts != REQUIRES_PYTHON_PARTS:
            raise AssertionError(
                f"{target.name}: unexpected Requires-Python {metadata['Requires-Python']!r}"
            )
        requirements = metadata.get_all("Requires-Dist", [])
        normalized_requirements = {
            _normalized_requirement(requirement) for requirement in requirements
        }
        expected_requirements = {
            _normalized_requirement(requirement) for requirement in target.requirements
        }
        if expected_requirements != normalized_requirements:
            raise AssertionError(
                f"{target.name}: expected requirements {sorted(target.requirements)!r}; "
                f"found {requirements!r}"
            )
        if any(
            "workspace" in requirement or "file:" in requirement
            for requirement in normalized_requirements
        ):
            raise AssertionError(
                f"{target.name}: wheel contains a local dependency reference: "
                f"{requirements!r}"
            )
        pyproject_files = sorted(
            name for name in names if Path(name).name == "pyproject.toml"
        )
        if pyproject_files:
            raise AssertionError(
                f"{target.name}: wheel contains project configuration: "
                f"{pyproject_files!r}"
            )
        license_prefixes = {
            name[: name.index(".dist-info/licenses/") + len(".dist-info/licenses/")]
            for name in names
            if ".dist-info/licenses/" in name
        }
        if len(license_prefixes) != 1:
            raise AssertionError(
                f"{target.name}: expected one dist-info license directory; "
                f"found {sorted(license_prefixes)!r}"
            )
        license_prefix = next(iter(license_prefixes))
        expected_legal_members = {
            f"{license_prefix}{relative_path}" for relative_path in target.legal_files
        }
        actual_legal_members = {
            name
            for name in names
            if name.startswith(license_prefix) and not name.endswith("/")
        }
        if actual_legal_members != expected_legal_members:
            raise AssertionError(
                f"{target.name}: expected wheel legal files "
                f"{sorted(expected_legal_members)!r}; found "
                f"{sorted(actual_legal_members)!r}"
            )
        for relative_path in target.legal_files:
            if (
                wheel.read(f"{license_prefix}{relative_path}")
                != (target.source_dir / relative_path).read_bytes()
            ):
                raise AssertionError(
                    f"{target.name}: wheel {relative_path} differs from Hub source"
                )
        metadata_license_files = frozenset(metadata.get_all("License-File", []))
        if metadata_license_files != target.legal_files:
            raise AssertionError(
                f"{target.name}: expected License-File metadata "
                f"{sorted(target.legal_files)!r}; found "
                f"{sorted(metadata_license_files)!r}"
            )
        missing_files = sorted(target.bundled_files - names)
        if missing_files:
            raise AssertionError(
                f"{target.name}: release wheel is missing bundled files: {missing_files!r}"
            )
        for relative_path in target.bundled_files:
            if (
                wheel.read(relative_path)
                != _bundled_source_path(target, relative_path).read_bytes()
            ):
                raise AssertionError(
                    f"{target.name}: wheel {relative_path} differs from Hub source"
                )
        if target.name in {"commonground-predict", "commonground-elicit"} and not any(
            name.endswith(".jsonl") for name in target.bundled_files
        ):
            raise AssertionError(f"{target.name}: source package has no JSONL data")
        shipped_tooling = sorted(
            name for name in names if Path(name).name in REPOSITORY_TOOLING
        )
        if shipped_tooling:
            raise AssertionError(
                f"{target.name}: release wheel contains repository tooling: "
                f"{shipped_tooling!r}"
            )
        return len(target.bundled_files)


def _inspect_sdist(target: WheelTarget, sdist_path: Path) -> None:
    with tarfile.open(sdist_path, "r:gz") as archive:
        members = {member.name: member for member in archive.getmembers()}
        for relative_path in target.legal_files:
            legal_names = sorted(
                name for name in members if name.endswith(f"/{relative_path}")
            )
            if len(legal_names) != 1:
                raise AssertionError(
                    f"{target.name}: expected one sdist {relative_path}; "
                    f"found {legal_names!r}"
                )
            legal_file = archive.extractfile(members[legal_names[0]])
            if (
                legal_file is None
                or legal_file.read() != (target.source_dir / relative_path).read_bytes()
            ):
                raise AssertionError(
                    f"{target.name}: sdist {relative_path} differs from Hub source"
                )
        for relative_path in target.bundled_files:
            bundled_names = sorted(
                name for name in members if name.endswith(f"/{relative_path}")
            )
            if len(bundled_names) != 1:
                raise AssertionError(
                    f"{target.name}: expected one sdist {relative_path}; "
                    f"found {bundled_names!r}"
                )
            bundled_file = archive.extractfile(members[bundled_names[0]])
            if (
                bundled_file is None
                or bundled_file.read()
                != _bundled_source_path(target, relative_path).read_bytes()
            ):
                raise AssertionError(
                    f"{target.name}: sdist {relative_path} differs from Hub source"
                )


def _fresh_install(wheel_paths: list[Path], output_dir: Path) -> None:
    """Install built wheels together and load every taskset outside the source tree."""

    with tempfile.TemporaryDirectory(
        prefix="commonground-fresh-install-"
    ) as temporary_dir:
        _fresh_install_in_venv(wheel_paths, output_dir, Path(temporary_dir))


def _fresh_install_in_venv(
    wheel_paths: list[Path], output_dir: Path, venv_dir: Path
) -> None:
    """Run the fresh-install probe in an automatically discarded environment."""

    subprocess.run(
        ["uv", "venv", "--python", "3.12", str(venv_dir)],
        cwd=output_dir,
        check=True,
    )
    venv_python = venv_dir / (
        "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
    )
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(venv_python),
            "--find-links",
            str(output_dir),
            *(
                argument
                for manifest_path in DEPENDENCY_MANIFEST_PATHS
                for argument in ("--requirement", str(manifest_path))
            ),
            *(str(path) for path in wheel_paths),
        ],
        cwd=output_dir,
        check=True,
    )
    probe = """
import asyncio
import json
from importlib.metadata import version

import commonground_elicit
import commonground_predict
import commonground_scenarios
import commonground_score

assert version("commonground-predict") == "0.6.1"
assert version("commonground-elicit") == "0.6.1"
assert version("commonground-scenarios") == "0.6.1"
assert version("commonground-score") == "0.6.1"

assert len(commonground_predict.load_taskset().load()) == 100
assert len(commonground_elicit.load_taskset().load()) == 100
assert len(commonground_elicit.load_taskset(task="elicit-ask").load()) == 100

import verifiers as legacy_vf
from verifiers.types import State
from verifiers.v1.harnesses.null import NullHarness
from verifiers.v1.utils.loaders import default_harness_id, harness_class, taskset_class

predict_legacy = legacy_vf.load_environment("commonground-predict", split="eval")
elicit_find_legacy = legacy_vf.load_environment(
    "commonground-elicit", task="find", split="eval"
)
elicit_ask_legacy = legacy_vf.load_environment(
    "commonground-elicit", task="elicit-ask", split="eval"
)
assert isinstance(predict_legacy, legacy_vf.SingleTurnEnv)
assert isinstance(elicit_find_legacy, legacy_vf.SingleTurnEnv)
assert isinstance(elicit_ask_legacy, legacy_vf.SingleTurnEnv)
assert len(predict_legacy.get_eval_dataset()) == 100
assert len(elicit_find_legacy.get_eval_dataset()) == 100
assert len(elicit_ask_legacy.get_eval_dataset()) == 100

assert taskset_class("commonground-predict") is commonground_predict.CommonGroundPredictTaskset
assert taskset_class("commonground-elicit") is commonground_elicit.ElicitTaskset
assert default_harness_id("commonground-predict") == "commonground-predict"
assert default_harness_id("commonground-elicit") == "commonground-elicit"
assert issubclass(harness_class("commonground-predict"), NullHarness)
assert issubclass(harness_class("commonground-elicit"), NullHarness)

def score(env, row, response):
    task = {
        key: row[key]
        for key in ("prompt", "answer", "info", "example_id")
        if key in row
    }
    state = State.for_task(task)
    state["completion"] = [
        {"role": "assistant", "content": json.dumps(response, sort_keys=True)}
    ]
    asyncio.run(env.rubric.score_rollout(state))
    return state["reward"]

predict_row = dict(predict_legacy.get_eval_dataset()[0])
predict_answer = json.loads(predict_row["answer"])
predict_probabilities = {
    cell: {
        "agree": 1.0 if label == 1 else 0.0,
        "disagree": 1.0 if label == -1 else 0.0,
        "pass": 1.0 if label == 0 else 0.0,
    }
    for cell, label in predict_answer.items()
}
assert score(predict_legacy, predict_row, {
    "predictions": predict_probabilities
}) == 1.0

def elicit_response(row):
    answer = json.loads(row["answer"])
    info = json.loads(row["info"])
    questions = [
        {
            key: question[key]
            for key in (
                "doc_id",
                "quote",
                "type",
                "question",
                "decision",
                "yes_choice",
                "related_evidence",
                "target_stances",
            )
        }
        for question in answer.get("questions", [])[:info["question_count"]]
    ]
    response = {"questions": questions}
    if info["allow_combined_questions"]:
        response["findings"] = [
            {
                key: finding[key]
                for key in (
                    "doc_id",
                    "quote",
                    "type",
                    "diagnosis",
                    "related_evidence",
                )
            }
            for finding in answer["findings"]
        ]
    return response

elicit_find_row = dict(elicit_find_legacy.get_eval_dataset()[0])
elicit_ask_row = dict(elicit_ask_legacy.get_eval_dataset()[0])
assert score(
    elicit_find_legacy, elicit_find_row, elicit_response(elicit_find_row)
) == 1.0
assert score(
    elicit_ask_legacy, elicit_ask_row, elicit_response(elicit_ask_row)
) == 1.0
"""
    subprocess.run(
        [str(venv_python), "-I", "-c", probe],
        cwd=output_dir,
        check=True,
    )


def _build_and_check(output_dir: Path) -> list[str]:
    """Build, inspect, and fresh-install one exact artifact set."""

    summaries: list[str] = []
    wheel_paths: list[Path] = []
    for target in TARGETS:
        source_license = target.source_dir / "LICENSE"
        if source_license.read_bytes() != CANONICAL_LICENSE.read_bytes():
            raise AssertionError(
                f"{target.name}: Hub source LICENSE differs from repository"
            )
        subprocess.run(
            [
                "uv",
                "build",
                "--wheel",
                "--out-dir",
                str(output_dir),
                str(target.source_dir),
            ],
            cwd=ROOT,
            check=True,
        )
        subprocess.run(
            [
                "uv",
                "build",
                "--sdist",
                "--out-dir",
                str(output_dir),
                str(target.source_dir),
            ],
            cwd=ROOT,
            check=True,
        )
        wheel_prefix = target.name.replace("-", "_") + "-"
        wheels = sorted(output_dir.glob(f"{wheel_prefix}*.whl"))
        if len(wheels) != 1:
            raise AssertionError(
                f"{target.name}: expected one release wheel, found {len(wheels)}"
            )
        wheel_paths.append(wheels[0])
        bundled_count = _inspect_wheel(target, wheels[0])
        sdists = sorted(output_dir.glob(f"{wheel_prefix}*.tar.gz"))
        if len(sdists) != 1:
            raise AssertionError(
                f"{target.name}: expected one release sdist, found {len(sdists)}"
            )
        _inspect_sdist(target, sdists[0])
        summaries.append(
            f"{wheels[0].name} + {sdists[0].name} "
            f"({bundled_count} required bundled files)"
        )
    _fresh_install(wheel_paths, output_dir)
    return summaries


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Retain the exact verified wheels and source distributions here.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "check_dependency_manifests.py"),
            "--check",
        ],
        cwd=ROOT,
        check=True,
    )
    if args.output_dir is None:
        with tempfile.TemporaryDirectory(prefix="commonground-wheel-") as temporary_dir:
            summaries = _build_and_check(Path(temporary_dir))
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        summaries = _build_and_check(args.output_dir)

    print(
        "release artifact check passed (licenses present; project configuration and "
        "repository tooling excluded; exact-manifest fresh wheel install and taskset "
        "loads passed): " + "; ".join(summaries)
    )


if __name__ == "__main__":
    main()
