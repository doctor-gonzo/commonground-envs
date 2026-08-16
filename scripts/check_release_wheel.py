"""Build and inspect every Common Ground release wheel."""

from __future__ import annotations

from dataclasses import dataclass
import email
from pathlib import Path
import subprocess
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
REQUIRES_PYTHON_PARTS = frozenset({">=3.12", "<3.13"})
REPOSITORY_TOOLING = frozenset(
    {
        "aggregate_baselines.py",
        "baseline-sweep.toml",
    }
)


@dataclass(frozen=True)
class WheelTarget:
    name: str
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
        source_dir=ROOT / "environments" / "commonground_predict",
        requirements=frozenset(
            {"commonground-score<0.2,>=0.1.0", "verifiers==0.1.14"}
        ),
        bundled_files=_data_files(
            ROOT / "environments" / "commonground_predict", "commonground_predict"
        ),
    ),
    WheelTarget(
        name="commonground-elicit",
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
        source_dir=ROOT / "packages" / "commonground-scenarios",
        requirements=frozenset(),
        bundled_files=frozenset(
            {"commonground_scenarios/schema/scenario.schema.json"}
        ),
    ),
    WheelTarget(
        name="commonground-score",
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
