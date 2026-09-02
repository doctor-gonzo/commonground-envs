"""Compute the small deterministic baseline suite for commonground-elicit."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from commonground_elicit.environment import (
    BUNDLED_EVAL_PATH,
    match_findings,
    panel_disagreement,
    question_utility_score,
    scenario_to_row,
)

QUESTION_COUNT = 1
FINDING_BUDGET = 3
FINDING_TYPES = ("ambiguity", "contradiction", "gap")
RANDOM_SEED = 20260831


def load_scenarios(path: Path) -> list[dict[str, Any]]:
    scenarios = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not scenarios:
        raise ValueError(f"no scenarios loaded from {path}")
    return scenarios


def public_documents(scenario: Mapping[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "doc_id": str(document["doc_id"]),
            "title": str(document["title"]),
            "style": str(document["style"]),
            "text": str(document["text"]),
        }
        for document in scenario["documents"]
    ]


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def _generic_finding(
    document: Mapping[str, str],
    quote: str,
    issue_type: str,
) -> dict[str, Any]:
    return {
        "doc_id": document["doc_id"],
        "quote": quote,
        "type": issue_type,
        "diagnosis": "Should this documented rule be clarified?",
        "decision": {
            "actor": "the documented decision maker",
            "action": "apply the documented rule",
            "condition": "the documented condition occurs",
            "anchor_outcome": "preserve the documented rule",
            "alternative_outcome": "clarify the documented rule",
        },
        "related_evidence": None,
    }


def random_span_findings(
    documents: Sequence[Mapping[str, str]],
    rng: random.Random,
) -> list[dict[str, Any]]:
    candidates = [
        (document, sentence)
        for document in documents
        for sentence in split_sentences(document["text"])
    ]
    selected = rng.sample(candidates, k=min(FINDING_BUDGET, len(candidates)))
    return [
        _generic_finding(document, sentence, FINDING_TYPES[index % 3])
        for index, (document, sentence) in enumerate(selected)
    ]


def longest_visible_sentence_findings(
    documents: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    candidates = sorted(
        (
            (len(sentence), document["doc_id"], sentence, document)
            for document in documents
            for sentence in split_sentences(document["text"])
        ),
        key=lambda item: (-item[0], item[1], item[2]),
    )[:FINDING_BUDGET]
    return [
        _generic_finding(document, sentence, FINDING_TYPES[index % 3])
        for index, (_, _, sentence, document) in enumerate(candidates)
    ]


def exact_answer_for_scenario(
    scenario: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    row = scenario_to_row(
        scenario,
        docs_count=None,
        docs_length=None,
        planted_density=1.0,
        distractor_density=1.0,
        panel_polarization=1.0,
        question_count=QUESTION_COUNT,
        task="find",
    )
    answer = json.loads(str(row["answer"]))
    return {
        "findings": list(answer["findings"]),
        "questions": list(answer["questions"]),
    }


def _candidate_question(plant: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: plant[key]
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


def _question_score(
    candidate: Mapping[str, Any], plants: Sequence[Mapping[str, Any]]
) -> float:
    return question_utility_score(
        [candidate],
        plants,
        panel_polarization=1.0,
        question_count=QUESTION_COUNT,
    )


def _candidate_utilities(plants: Sequence[Mapping[str, Any]]) -> list[float]:
    return [
        float(plant["decision_value"]) * panel_disagreement(plant["target_stances"])
        for plant in plants
    ]


def compute_elicit_floors(path: Path = BUNDLED_EVAL_PATH) -> dict[str, float]:
    scenarios = load_scenarios(path)
    rng = random.Random(RANDOM_SEED)
    random_find: list[float] = []
    longest_find: list[float] = []
    exact_find: list[float] = []
    random_ask: list[float] = []
    runner_up_ask: list[float] = []
    exact_ask: list[float] = []
    issue_counts: Counter[str] = Counter()
    normalized_margins: list[float] = []
    tie_count = 0

    for scenario in scenarios:
        documents = public_documents(scenario)
        answer = exact_answer_for_scenario(scenario)
        findings = answer["findings"]
        plants = answer["questions"]
        if len(findings) != 3 or len(plants) != 3:
            raise ValueError("each release scenario must contain exactly three issues")
        contradiction = next(
            finding for finding in findings if finding["type"] == "contradiction"
        )
        if contradiction["related_evidence"] is None:
            raise ValueError("every contradiction must name authored opposing evidence")

        random_find.append(
            match_findings(random_span_findings(documents, rng), findings)["f1"]
        )
        longest_find.append(
            match_findings(longest_visible_sentence_findings(documents), findings)["f1"]
        )
        exact_find.append(match_findings(findings, findings)["f1"])

        exact_candidates = [_candidate_question(plant) for plant in plants]
        candidate_scores = [
            _question_score(candidate, plants) for candidate in exact_candidates
        ]
        random_ask.append(sum(candidate_scores) / len(candidate_scores))
        runner_up_ask.append(candidate_scores[1])
        exact_ask.append(candidate_scores[0])

        utilities = sorted(_candidate_utilities(plants), reverse=True)
        tie_count += int(abs(utilities[0] - utilities[1]) <= 1e-12)
        normalized_margins.append(
            (utilities[0] - utilities[1]) / utilities[0] if utilities[0] > 0 else 0.0
        )
        issue_counts.update(str(finding["type"]) for finding in findings)

    total_issues = sum(issue_counts.values())
    proportions = [
        issue_counts[issue_type] / total_issues for issue_type in FINDING_TYPES
    ]

    def mean(values: Sequence[float]) -> float:
        return sum(values) / len(values)

    return {
        "find/random-visible-sentences": mean(random_find),
        "find/longest-visible-sentences": mean(longest_find),
        "find/exact-ceiling": mean(exact_find),
        "elicit-ask/uniform-random-candidate": mean(random_ask),
        "elicit-ask/runner-up-candidate": mean(runner_up_ask),
        "elicit-ask/public-profile-ceiling": mean(exact_ask),
        "audit/top1-tie-rate": tie_count / len(scenarios),
        "audit/top1-normalized-margin-min": min(normalized_margins),
        "audit/issue-class-min-proportion": min(proportions),
        "audit/issue-class-max-proportion": max(proportions),
    }


def render_markdown(floors: Mapping[str, float]) -> str:
    labels = (
        ("find/random-visible-sentences", "Find", "Random visible sentences"),
        ("find/longest-visible-sentences", "Find", "Longest visible sentences"),
        ("find/exact-ceiling", "Find", "Exact authored answer (ceiling)"),
        (
            "elicit-ask/uniform-random-candidate",
            "Ask",
            "Uniform candidate + exact components",
        ),
        ("elicit-ask/runner-up-candidate", "Ask", "Exact runner-up candidate"),
        (
            "elicit-ask/public-profile-ceiling",
            "Ask",
            "Public-profile composition (ceiling)",
        ),
        ("audit/top1-tie-rate", "Audit", "Top-1 tie rate"),
        ("audit/top1-normalized-margin-min", "Audit", "Minimum top-1 margin"),
        ("audit/issue-class-min-proportion", "Audit", "Minimum issue-class share"),
        ("audit/issue-class-max-proportion", "Audit", "Maximum issue-class share"),
    )
    lines = [
        "| Task | Comparator or diagnostic | mean |",
        "| --- | --- | ---: |",
    ]
    lines.extend(
        f"| {task} | {label} | {floors[key]:.3f} |" for key, task, label in labels
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", type=Path, nargs="?", default=BUNDLED_EVAL_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(render_markdown(compute_elicit_floors(args.split)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
