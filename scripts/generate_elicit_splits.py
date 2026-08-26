"""Generate the bundled synthetic splits for commonground-elicit."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

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
TRAIN_SCENARIOS_PER_TEMPLATE = 10
EVAL_SCENARIOS_PER_TEMPLATE = 1


def build_split_bytes(
    templates: Sequence[DomainTemplate],
    *,
    seed_base: int,
    generated_at: str = GENERATED_AT,
    scenarios_per_template: int | None = None,
) -> bytes:
    """Return canonical JSONL for a fixed template/seed/date matrix."""

    if scenarios_per_template is None:
        scenarios_per_template = (
            EVAL_SCENARIOS_PER_TEMPLATE
            if templates
            and all(template.template_set == "heldout" for template in templates)
            else TRAIN_SCENARIOS_PER_TEMPLATE
        )
    if scenarios_per_template <= 0:
        raise ValueError("scenarios_per_template must be positive")
    scenarios = [
        generate_scenario(
            seed=seed_base + template_index * 10 + repetition,
            domain_template=template,
            generated_at=generated_at,
        )
        for template_index, template in enumerate(templates)
        for repetition in range(scenarios_per_template)
    ]
    assert_unique_semantic_tasks(scenarios)
    return b"".join(scenario_to_bytes(scenario) for scenario in scenarios)


def scenario_semantic_key(scenario: dict[str, Any]) -> str:
    """Hash evaluation-relevant semantics while ignoring identity-only seed noise."""

    documents = sorted(
        (
            {field: document[field] for field in ("doc_id", "title", "style", "text")}
            for document in scenario["documents"]
        ),
        key=lambda document: str(document["doc_id"]),
    )
    plants = sorted(
        (
            {
                field: plant[field]
                for field in (
                    "plant_id",
                    "doc_id",
                    "anchor_quote",
                    "type",
                    "canonical_question",
                    "canonical_question_aliases",
                    "target_stances",
                )
            }
            for plant in scenario["planted_items"]
        ),
        key=lambda plant: str(plant["plant_id"]),
    )
    payload = json.dumps(
        {"documents": documents, "planted_items": plants},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def assert_unique_semantic_tasks(scenarios: Sequence[dict[str, Any]]) -> None:
    """Reject rows that differ only in seed, order, or organization name."""

    seen: dict[str, str] = {}
    for scenario in scenarios:
        semantic_key = scenario_semantic_key(scenario)
        scenario_id = str(scenario["scenario_id"])
        previous = seen.get(semantic_key)
        if previous is not None:
            raise ValueError(f"duplicate semantic task: {previous} and {scenario_id}")
        seen[semantic_key] = scenario_id


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
            scenarios_per_template=TRAIN_SCENARIOS_PER_TEMPLATE,
        )
    )
    eval_path.write_bytes(
        build_split_bytes(
            HELDOUT_TEMPLATES,
            seed_base=EVAL_SEED_BASE,
            generated_at=generated_at,
            scenarios_per_template=EVAL_SCENARIOS_PER_TEMPLATE,
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
