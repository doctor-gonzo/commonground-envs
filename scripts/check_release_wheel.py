"""Build and inspect every Common Ground release wheel."""

from __future__ import annotations

import email
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
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


@dataclass(frozen=True)
class WheelTarget:
    name: str
    version: str
    source_dir: Path
    requirements: frozenset[str]
    bundled_files: frozenset[str]


def _data_files(source_dir: Path, package_name: str) -> frozenset[str]:
    data_dir = source_dir / package_name / "data"
    return frozenset(
        f"{package_name}/data/{source_path.name}"
        for source_path in data_dir.glob("*.jsonl")
    )


TARGETS = (
    WheelTarget(
        name="commonground-predict",
        version="0.1.2",
        source_dir=ROOT / "environments" / "commonground_predict",
        requirements=frozenset({"commonground-score<0.2,>=0.1.0", "verifiers==0.1.14"}),
        bundled_files=_data_files(
            ROOT / "environments" / "commonground_predict", "commonground_predict"
        ),
    ),
    WheelTarget(
        name="commonground-elicit",
        version="0.1.2",
        source_dir=ROOT / "environments" / "commonground_elicit",
        requirements=frozenset(
            {
                "commonground-scenarios<0.2,>=0.1.0",
                "commonground-score<0.2,>=0.1.0",
                "verifiers==0.1.14",
            }
        ),
        bundled_files=_data_files(
            ROOT / "environments" / "commonground_elicit", "commonground_elicit"
        )
        | frozenset({"commonground_elicit/data/README.md"}),
    ),
    WheelTarget(
        name="commonground-scenarios",
        version="0.1.0",
        source_dir=ROOT / "packages" / "commonground-scenarios",
        requirements=frozenset(),
        bundled_files=frozenset({"commonground_scenarios/schema/scenario.schema.json"}),
    ),
    WheelTarget(
        name="commonground-score",
        version="0.1.0",
        source_dir=ROOT / "packages" / "commonground-score",
        requirements=frozenset(),
        bundled_files=frozenset(),
    ),
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
        if license_value != EXPECTED_LICENSE:
            raise AssertionError(
                f"{target.name}: expected license {EXPECTED_LICENSE!r}; "
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
        missing_files = sorted(target.bundled_files - names)
        if missing_files:
            raise AssertionError(
                f"{target.name}: release wheel is missing bundled files: {missing_files!r}"
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


def main() -> None:
    summaries: list[str] = []
    with tempfile.TemporaryDirectory(prefix="commonground-wheel-") as temporary_dir:
        output_dir = Path(temporary_dir)
        for target in TARGETS:
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
            wheel_prefix = target.name.replace("-", "_") + "-"
            wheels = sorted(output_dir.glob(f"{wheel_prefix}*.whl"))
            if len(wheels) != 1:
                raise AssertionError(
                    f"{target.name}: expected one release wheel, found {len(wheels)}"
                )
            bundled_count = _inspect_wheel(target, wheels[0])
            summaries.append(
                f"{wheels[0].name} ({bundled_count} required bundled files)"
            )

    print(
        "release wheel check passed (repository tooling excluded): "
        + "; ".join(summaries)
    )


if __name__ == "__main__":
    main()
