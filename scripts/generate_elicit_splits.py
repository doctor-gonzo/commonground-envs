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
TRAIN_SCENARIOS_PER_TEMPLATE = 25
EVAL_SCENARIOS_PER_TEMPLATE = 5


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
            seed=seed_base + template_index * scenarios_per_template + repetition,
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


def prompt_fingerprint(scenario: dict[str, Any]) -> str:
    """Hash only model-visible documents and faction descriptions."""

    payload = {
        "documents": sorted(scenario["documents"], key=lambda item: item["doc_id"]),
        "factions": [
            {
                "faction_id": faction["faction_id"],
                "name": faction["name"],
                "summary": faction["summary"],
            }
            for faction in scenario["factions"]
        ],
    }
    return _payload_hash(payload)


def answer_fingerprint(scenario: dict[str, Any]) -> str:
    """Hash the hidden answer contract independently of the prompt hash."""

    return _payload_hash(scenario["planted_items"])


def structural_signature(scenario: dict[str, Any]) -> str:
    """Describe task structure without domain wording or random opaque IDs."""

    documents = {document["doc_id"]: document for document in scenario["documents"]}
    issue_layout = []
    for plant in scenario["planted_items"]:
        sentences = [
            sentence.strip()
            for sentence in documents[plant["doc_id"]]["text"].split(".")
            if sentence.strip()
        ]
        anchor_prefix = plant["anchor_quote"].removesuffix(".")
        anchor_position = next(
            (
                index
                for index, sentence in enumerate(sentences)
                if sentence.startswith(anchor_prefix)
            ),
            -1,
        )
        issue_layout.append(
            {
                "type": plant["type"],
                "anchor_position": anchor_position,
                "document_sentence_count": len(sentences),
                "stance_multiset": sorted(plant["target_stances"].values()),
                "has_related_evidence": plant["related_evidence"] is not None,
            }
        )
    return _payload_hash(
        {
            "generator_family": scenario["provenance"]["generator_family"],
            "document_count": len(scenario["documents"]),
            "faction_count": len(scenario["factions"]),
            "issue_layout": sorted(
                issue_layout, key=lambda item: json.dumps(item, sort_keys=True)
            ),
        }
    )


def assert_split_integrity(
    train_scenarios: Sequence[dict[str, Any]],
    eval_scenarios: Sequence[dict[str, Any]],
) -> None:
    """Block exact prompt/answer overlap and shared generator families."""

    for label, scenarios in (("train", train_scenarios), ("eval", eval_scenarios)):
        for fingerprint_name, fingerprint in (
            ("prompt", prompt_fingerprint),
            ("answer", answer_fingerprint),
        ):
            values = [fingerprint(scenario) for scenario in scenarios]
            if len(values) != len(set(values)):
                raise ValueError(f"duplicate {fingerprint_name} fingerprint in {label}")
    train_prompt_keys = {prompt_fingerprint(scenario) for scenario in train_scenarios}
    eval_prompt_keys = {prompt_fingerprint(scenario) for scenario in eval_scenarios}
    if train_prompt_keys & eval_prompt_keys:
        raise ValueError("train/eval prompt fingerprint overlap")
    train_answer_keys = {answer_fingerprint(scenario) for scenario in train_scenarios}
    eval_answer_keys = {answer_fingerprint(scenario) for scenario in eval_scenarios}
    if train_answer_keys & eval_answer_keys:
        raise ValueError("train/eval answer fingerprint overlap")
    train_families = {
        scenario["provenance"]["generator_family"] for scenario in train_scenarios
    }
    eval_families = {
        scenario["provenance"]["generator_family"] for scenario in eval_scenarios
    }
    if train_families & eval_families:
        raise ValueError("train/eval generator families must be disjoint")
    legacy_ids = {"scope-note", "authority-bulletin", "exception-card"}
    if any(
        legacy_ids & {document["doc_id"] for document in scenario["documents"]}
        for scenario in eval_scenarios
    ):
        raise ValueError("legacy prompt-visible document codebook detected")


def _payload_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


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
    train_bytes = build_split_bytes(
        TRAIN_TEMPLATES,
        seed_base=TRAIN_SEED_BASE,
        generated_at=generated_at,
        scenarios_per_template=TRAIN_SCENARIOS_PER_TEMPLATE,
    )
    eval_bytes = build_split_bytes(
        HELDOUT_TEMPLATES,
        seed_base=EVAL_SEED_BASE,
        generated_at=generated_at,
        scenarios_per_template=EVAL_SCENARIOS_PER_TEMPLATE,
    )
    train_scenarios = [json.loads(line) for line in train_bytes.splitlines()]
    eval_scenarios = [json.loads(line) for line in eval_bytes.splitlines()]
    assert_split_integrity(train_scenarios, eval_scenarios)
    train_path.write_bytes(train_bytes)
    eval_path.write_bytes(eval_bytes)
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
