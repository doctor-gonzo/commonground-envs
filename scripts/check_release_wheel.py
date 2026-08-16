"""Build and inspect the commonground-predict release wheel."""

from __future__ import annotations

import email
from pathlib import Path
import subprocess
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT_DIR = ROOT / "environments" / "commonground_predict"
DATA_DIR = ENVIRONMENT_DIR / "commonground_predict" / "data"
EXPECTED_SCORE_RANGE = "commonground-score<0.2,>=0.1.0"


def _normalized_requirement(requirement: str) -> str:
    return requirement.lower().replace(" ", "")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="commonground-wheel-") as temporary_dir:
        output_dir = Path(temporary_dir)
        subprocess.run(
            [
                "uv",
                "build",
                "--wheel",
                "--out-dir",
                str(output_dir),
                str(ENVIRONMENT_DIR),
            ],
            cwd=ROOT,
            check=True,
        )

        wheels = sorted(output_dir.glob("*.whl"))
        if len(wheels) != 1:
            raise AssertionError(f"expected one release wheel, found {len(wheels)}")

        with zipfile.ZipFile(wheels[0]) as wheel:
            names = set(wheel.namelist())
            metadata_names = sorted(name for name in names if name.endswith(".dist-info/METADATA"))
            if len(metadata_names) != 1:
                raise AssertionError(f"expected one METADATA file, found {len(metadata_names)}")

            metadata = email.message_from_bytes(wheel.read(metadata_names[0]))
            requirements = metadata.get_all("Requires-Dist", [])
            normalized_requirements = {_normalized_requirement(requirement) for requirement in requirements}
            if EXPECTED_SCORE_RANGE not in normalized_requirements:
                raise AssertionError(
                    "release wheel must require "
                    f"{EXPECTED_SCORE_RANGE}; found {requirements!r}"
                )
            if any("workspace" in requirement or "file:" in requirement for requirement in normalized_requirements):
                raise AssertionError(f"release wheel contains a local dependency reference: {requirements!r}")

            expected_data = {
                f"commonground_predict/data/{source_path.name}"
                for source_path in DATA_DIR.glob("*.jsonl")
            }
            missing_data = sorted(expected_data - names)
            if not expected_data:
                raise AssertionError("source package has no bundled JSONL data")
            if missing_data:
                raise AssertionError(f"release wheel is missing bundled data: {missing_data!r}")

        print(
            f"release wheel check passed: {wheels[0].name}; "
            f"{len(expected_data)} data files bundled"
        )


if __name__ == "__main__":
    main()
