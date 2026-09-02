"""Generate the bundled synthetic splits for commonground-elicit."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from collections.abc import Sequence
from itertools import pairwise
from math import log, sqrt
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
GENERATED_AT = "2026-08-30"
TRAIN_SEED_BASE = 8100
EVAL_SEED_BASE = 8200
TRAIN_SCENARIOS_PER_TEMPLATE = 25
EVAL_SCENARIOS_PER_TEMPLATE = 5
MAX_CROSS_SPLIT_TFIDF_COSINE = 0.90


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
    assert_unique_policy_issue_semantics(scenarios)
    return b"".join(scenario_to_bytes(scenario) for scenario in scenarios)


def instance_fingerprint(scenario: dict[str, Any]) -> str:
    """Hash one exact visible-layout and hidden-answer instance."""

    return _payload_hash(
        {
            "scenario_id": scenario["scenario_id"],
            "documents": scenario["documents"],
            "factions": scenario["factions"],
            "planted_items": scenario["planted_items"],
        }
    )


def policy_issue_semantics(scenario: dict[str, Any]) -> str:
    """Hash policy propositions, unresolved decisions, relations, and stances.

    Opaque document, plant, and faction IDs as well as neutral title/style fields
    are deliberately excluded. Stances are keyed by stable faction names rather
    than randomized identifiers.
    """

    documents_by_id = {
        str(document["doc_id"]): document for document in scenario["documents"]
    }
    faction_names = {
        str(faction["faction_id"]): _normalized_text(str(faction["name"]))
        for faction in scenario["factions"]
    }
    plants = []
    for plant in scenario["planted_items"]:
        related = plant.get("related_evidence")
        plants.append(
            {
                "document_propositions": sorted(
                    _normalized_propositions(
                        str(documents_by_id[str(plant["doc_id"])]["text"])
                    )
                ),
                "anchor": _normalized_text(str(plant["anchor_quote"])),
                "type": str(plant["type"]),
                "question": _normalized_text(str(plant["canonical_question"])),
                "related_quote": (
                    _normalized_text(str(related["quote"]))
                    if isinstance(related, dict)
                    else None
                ),
                "target_stances": sorted(
                    (faction_names[str(faction_id)], str(stance))
                    for faction_id, stance in plant["target_stances"].items()
                ),
            }
        )
    return _payload_hash(
        sorted(plants, key=lambda plant: json.dumps(plant, sort_keys=True))
    )


def semantic_word_ngrams(scenario: dict[str, Any]) -> Counter[str]:
    """Return identity-free word unigram/bigram counts for similarity audit."""

    texts = [str(document["text"]) for document in scenario["documents"]]
    texts.extend(str(faction["summary"]) for faction in scenario["factions"])
    texts.extend(
        str(plant["canonical_question"]) for plant in scenario["planted_items"]
    )
    tokens = [
        token
        for text in texts
        for token in re.findall(r"[^\W_]+", text.casefold())
        if len(token) >= 3
    ]
    features = tokens + [f"{left}::{right}" for left, right in pairwise(tokens)]
    return Counter(features)


def maximum_cross_split_tfidf_cosine(
    train_scenarios: Sequence[dict[str, Any]],
    eval_scenarios: Sequence[dict[str, Any]],
) -> tuple[float, str, str]:
    """Return closest train/eval pair under word-ngram TF-IDF cosine."""

    scenarios = [*train_scenarios, *eval_scenarios]
    counts = [semantic_word_ngrams(scenario) for scenario in scenarios]
    document_frequency: Counter[str] = Counter()
    for vector in counts:
        document_frequency.update(vector.keys())
    sample_count = len(counts)
    weighted: list[dict[str, float]] = []
    for vector in counts:
        raw = {
            feature: frequency
            * (log((sample_count + 1) / (document_frequency[feature] + 1)) + 1.0)
            for feature, frequency in vector.items()
        }
        norm = sqrt(sum(value * value for value in raw.values()))
        weighted.append(
            {feature: value / norm for feature, value in raw.items()} if norm else {}
        )
    train_vectors = weighted[: len(train_scenarios)]
    eval_vectors = weighted[len(train_scenarios) :]
    best = (0.0, "", "")
    for train_scenario, left in zip(train_scenarios, train_vectors, strict=True):
        for eval_scenario, right in zip(eval_scenarios, eval_vectors, strict=True):
            smaller, larger = (
                (left, right) if len(left) <= len(right) else (right, left)
            )
            score = sum(
                value * larger.get(feature, 0.0) for feature, value in smaller.items()
            )
            candidate = (
                score,
                str(train_scenario["scenario_id"]),
                str(eval_scenario["scenario_id"]),
            )
            if candidate > best:
                best = candidate
    return best


def _normalized_text(text: str) -> str:
    return " ".join(re.findall(r"[^\W_]+", text.casefold()))


def _normalized_propositions(text: str) -> list[str]:
    return [
        _normalized_text(sentence)
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if sentence.strip()
    ]


def assert_split_integrity(
    train_scenarios: Sequence[dict[str, Any]],
    eval_scenarios: Sequence[dict[str, Any]],
) -> None:
    """Block exact/canonical overlap and overly close train/eval semantics."""

    for label, scenarios in (("train", train_scenarios), ("eval", eval_scenarios)):
        for fingerprint_name, fingerprint in (
            ("instance", instance_fingerprint),
            ("policy issue semantics", policy_issue_semantics),
        ):
            values = [fingerprint(scenario) for scenario in scenarios]
            if len(values) != len(set(values)):
                raise ValueError(f"duplicate {fingerprint_name} fingerprint in {label}")
    train_instances = {instance_fingerprint(scenario) for scenario in train_scenarios}
    eval_instances = {instance_fingerprint(scenario) for scenario in eval_scenarios}
    if train_instances & eval_instances:
        raise ValueError("train/eval exact instance overlap")
    train_semantics = {policy_issue_semantics(scenario) for scenario in train_scenarios}
    eval_semantics = {policy_issue_semantics(scenario) for scenario in eval_scenarios}
    if train_semantics & eval_semantics:
        raise ValueError("train/eval policy issue semantics overlap")
    train_families = {
        scenario["provenance"]["generator_family"] for scenario in train_scenarios
    }
    eval_families = {
        scenario["provenance"]["generator_family"] for scenario in eval_scenarios
    }
    if train_families & eval_families:
        raise ValueError("train/eval generator-profile labels must be disjoint")
    similarity, train_id, eval_id = maximum_cross_split_tfidf_cosine(
        train_scenarios, eval_scenarios
    )
    if similarity > MAX_CROSS_SPLIT_TFIDF_COSINE:
        raise ValueError(
            "train/eval TF-IDF near-duplicate exceeds threshold: "
            f"{similarity:.3f} for {train_id} and {eval_id}"
        )
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


def assert_unique_policy_issue_semantics(
    scenarios: Sequence[dict[str, Any]],
) -> None:
    """Reject rows that differ only in opaque identity or visible layout."""

    seen: dict[str, str] = {}
    for scenario in scenarios:
        semantic_key = policy_issue_semantics(scenario)
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
