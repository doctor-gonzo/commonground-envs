"""Generate the bundled synthetic splits for commonground-elicit."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from commonground_scenarios import (
    HELDOUT_TEMPLATES,
    TRAIN_TEMPLATES,
    DomainTemplate,
    generate_scenario,
    scenario_to_bytes,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = (
    ROOT / "environments" / "commonground_elicit" / "commonground_elicit" / "data"
)
GENERATED_AT = "2026-08-15"
TRAIN_SEED_BASE = 8100
EVAL_SEED_BASE = 8200
REPETITIONS_PER_TEMPLATE = 2


def build_split_bytes(
    templates: Sequence[DomainTemplate],
    *,
    seed_base: int,
    generated_at: str = GENERATED_AT,
) -> bytes:
    """Return canonical JSONL for a fixed template/seed/date matrix."""

    return b"".join(
        scenario_to_bytes(
            generate_scenario(
                seed=seed_base + template_index * 10 + repetition,
                domain_template=template,
                generated_at=generated_at,
            )
        )
        for template_index, template in enumerate(templates)
        for repetition in range(REPETITIONS_PER_TEMPLATE)
    )


def write_splits(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    generated_at: str = GENERATED_AT,
) -> tuple[Path, Path]:
    """Write train and held-out files, returning their paths."""

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train_synthetic.jsonl"
    eval_path = output_dir / "eval_synthetic_heldout.jsonl"
    train_path.write_bytes(
        build_split_bytes(
            TRAIN_TEMPLATES,
            seed_base=TRAIN_SEED_BASE,
            generated_at=generated_at,
        )
    )
    eval_path.write_bytes(
        build_split_bytes(
            HELDOUT_TEMPLATES,
            seed_base=EVAL_SEED_BASE,
            generated_at=generated_at,
        )
    )
    return train_path, eval_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--generated-at", default=GENERATED_AT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    train_path, eval_path = write_splits(
        args.output_dir,
        generated_at=args.generated_at,
    )
    print(f"wrote {train_path}")
    print(f"wrote {eval_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
