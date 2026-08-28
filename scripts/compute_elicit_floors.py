"""Compute deterministic model-free floors for commonground-elicit."""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from commonground_elicit.environment import (
    BUNDLED_EVAL_PATH,
    ElicitJsonParser,
    finding_f1,
    panel_disagreement,
    question_utility,
)
from commonground_scenarios import validate_scenario

QUESTION_COUNT = 2
FINDING_BUDGET = 3
FINDING_TYPES = ("ambiguity", "contradiction", "gap")
STANCES = ("agree", "disagree", "pass")
VAGUE_MARKERS = (
    "adequate",
    "appropriate",
    "as soon as practical",
    "material",
    "meaningful",
    "reasonable",
    "significant",
    "small",
    "substantive",
    "sufficient",
    "timely",
    "unsafe",
    "urgent",
)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def load_scenarios(path: Path) -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            scenario = json.loads(line)
            validate_scenario(scenario)
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            raise ValueError(
                f"invalid scenario at {path}:{line_number}: {error}"
            ) from error
        scenarios.append(scenario)
    if not scenarios:
        raise ValueError(f"no scenarios loaded from {path}")
    return scenarios


def public_documents(scenario: Mapping[str, Any]) -> list[dict[str, str]]:
    """Copy only document fields that a baseline may observe."""

    return [
        {"doc_id": str(document["doc_id"]), "text": str(document["text"])}
        for document in scenario["documents"]
    ]


def public_factions(scenario: Mapping[str, Any]) -> list[dict[str, str]]:
    """Copy only faction fields rendered in the prompt."""

    return [
        {
            "faction_id": str(faction["faction_id"]),
            "name": str(faction["name"]),
            "summary": str(faction["summary"]),
        }
        for faction in scenario["factions"]
    ]


def random_span_findings(
    documents: Sequence[Mapping[str, str]], rng: random.Random
) -> list[dict[str, Any]]:
    """Flag a fixed budget of visible sentences with random finding types."""

    spans = [
        (str(document["doc_id"]), sentence)
        for document in documents
        for sentence in split_sentences(str(document["text"]))
    ]
    rng.shuffle(spans)
    return [
        {
            "doc_id": doc_id,
            "quote": quote,
            "type": rng.choice(FINDING_TYPES),
            "diagnosis": sentence_question(quote),
            "related_evidence": None,
        }
        for doc_id, quote in spans[:FINDING_BUDGET]
    ]


def vague_sounding_findings(
    documents: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    """Flag every visible sentence containing a fixed vague-term marker."""

    findings: list[dict[str, Any]] = []
    for document in documents:
        for sentence in split_sentences(str(document["text"])):
            folded = sentence.casefold()
            if any(marker in folded for marker in VAGUE_MARKERS):
                findings.append(
                    {
                        "doc_id": str(document["doc_id"]),
                        "quote": sentence,
                        "type": "ambiguity",
                        "diagnosis": sentence_question(sentence),
                        "related_evidence": None,
                    }
                )
    return findings


def template_questions(
    documents: Sequence[Mapping[str, str]],
    factions: Sequence[Mapping[str, str]],
    *,
    question_count: int,
) -> list[dict[str, Any]]:
    """Ask a generic clarity question using only prompt-visible context."""

    return [
        {
            "doc_id": str(document["doc_id"]),
            "quote": split_sentences(str(document["text"]))[0],
            "question": f"Is the policy in document {document['doc_id']} clear?",
            "target_stances": {
                str(faction["faction_id"]): STANCES[
                    (question_index + faction_index) % len(STANCES)
                ]
                for faction_index, faction in enumerate(factions)
            },
        }
        for question_index, document in enumerate(documents[:question_count])
    ]


def randomly_targeted_questions(
    documents: Sequence[Mapping[str, str]],
    factions: Sequence[Mapping[str, str]],
    rng: random.Random,
    *,
    question_count: int,
) -> list[dict[str, Any]]:
    """Randomly ground finite question templates without consulting plantings."""

    visible_spans = [
        (str(document["doc_id"]), sentence)
        for document in documents
        for sentence in split_sentences(str(document["text"]))
    ]
    rng.shuffle(visible_spans)
    questions: list[dict[str, Any]] = []
    for question_index, (doc_id, quote) in enumerate(visible_spans[:question_count]):
        questions.append(
            {
                "doc_id": doc_id,
                "quote": quote,
                "question": (
                    f"Should the rule quoted from document {doc_id} be clarified "
                    f"for case {question_index + 1}?"
                ),
                "target_stances": {
                    str(faction["faction_id"]): rng.choice(STANCES)
                    for faction in factions
                },
            }
        )
    return questions


def exact_issue_component_questions(
    planted: Sequence[Mapping[str, Any]],
    factions: Sequence[Mapping[str, str]],
    *,
    question_count: int,
    rng: random.Random | None,
) -> list[dict[str, Any]]:
    """Use exact top-K issue targets while isolating faction-stance inference."""

    selected = sorted(
        planted,
        key=lambda plant: (
            -float(plant["decision_value"])
            * panel_disagreement(plant["target_stances"]),
            str(plant["doc_id"]),
            str(plant["quote"]),
        ),
    )[:question_count]
    return [
        {
            "doc_id": str(plant["doc_id"]),
            "quote": str(plant["quote"]),
            "question": str(plant["question"]),
            "target_stances": {
                str(faction["faction_id"]): (
                    rng.choice(STANCES)
                    if rng is not None
                    else stance_from_visible_summary(faction["summary"], plant)
                )
                for faction in factions
            },
        }
        for plant in selected
    ]


def stance_from_visible_summary(
    summary: str,
    plant: Mapping[str, Any],
) -> str:
    """Read one issue-specific yes/no tendency from a public faction summary."""

    terms = ", ".join(str(term) for term in plant["decision_terms"])
    marker = f"For decisions involving {terms}, "
    _, separator, suffix = summary.partition(marker)
    if not separator:
        return "pass"
    if suffix.startswith("leans toward yes"):
        return "agree"
    if suffix.startswith("leans toward no"):
        return "disagree"
    return "pass"


def sentence_question(sentence: str) -> str:
    """Turn visible policy terms into a deterministic yes/no decision probe."""

    words = re.findall(r"[^\W_]+", sentence, flags=re.UNICODE)
    body = " ".join(words[:12]).casefold()
    return f"Should {body} be made explicit?"


def legacy_codebook_findings(
    documents: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    """Reproduce the exact 0.2.5 document-ID/position shortcut if present."""

    issue_types = {
        "scope-note": "ambiguity",
        "authority-bulletin": "contradiction",
        "exception-card": "gap",
    }
    if {document["doc_id"] for document in documents} != set(issue_types):
        return []
    findings = []
    for document in documents:
        sentences = split_sentences(document["text"])
        quote = sentences[-2] if len(sentences) > 1 else sentences[-1]
        findings.append(
            {
                "doc_id": document["doc_id"],
                "quote": quote,
                "type": issue_types[document["doc_id"]],
                "diagnosis": sentence_question(quote),
                "related_evidence": None,
            }
        )
    return findings


def compute_elicit_floors(path: Path = BUNDLED_EVAL_PATH) -> dict[str, float]:
    """Score prompt baselines and component oracles on a validated split."""

    totals = {
        "find/random-span": 0.0,
        "find/vague-sounding": 0.0,
        "find/legacy-0.2-codebook": 0.0,
        "elicit-ask/template-question": 0.0,
        "elicit-ask/randomly-targeted": 0.0,
        "elicit-ask/exact-issue-random-stance": 0.0,
        "elicit-ask/exact-issue-summary-stance": 0.0,
    }
    scenarios = load_scenarios(path)
    for scenario in scenarios:
        documents = public_documents(scenario)
        factions = public_factions(scenario)
        scenario_id = str(scenario["scenario_id"])
        random_find_rng = random.Random(f"{scenario_id}:random-span")
        random_ask_rng = random.Random(f"{scenario_id}:randomly-targeted")
        component_rng = random.Random(f"{scenario_id}:component-random-stance")
        planted_findings = [
            {
                "doc_id": str(plant["doc_id"]),
                "quote": str(plant["anchor_quote"]),
                "type": str(plant["type"]),
                "diagnosis": str(plant["canonical_question"]),
                "decision_terms": list(plant["decision_terms"]),
                "related_evidence": plant["related_evidence"],
            }
            for plant in scenario["planted_items"]
        ]
        documents_by_id = {
            document["doc_id"]: document["text"] for document in documents
        }
        planted_questions = [
            {
                "doc_id": str(plant["doc_id"]),
                "quote": str(plant["anchor_quote"]),
                "question": str(plant["canonical_question"]),
                "question_aliases": list(plant["canonical_question_aliases"]),
                "target_stances": dict(plant["target_stances"]),
                "decision_terms": list(plant["decision_terms"]),
                "decision_value": float(plant["decision_value"]),
                "document_text": documents_by_id[str(plant["doc_id"])],
            }
            for plant in scenario["planted_items"]
        ]
        answer = {
            "findings": planted_findings,
            "questions": planted_questions,
        }
        info = {
            "panel_polarization": 1.0,
            "question_count": QUESTION_COUNT,
            "allow_combined_questions": False,
        }

        totals["find/random-span"] += float(
            asyncio.run(
                finding_f1(
                    completion_for(
                        {"findings": random_span_findings(documents, random_find_rng)}
                    ),
                    answer,
                    ElicitJsonParser(),
                )
            )
        )
        totals["find/vague-sounding"] += float(
            asyncio.run(
                finding_f1(
                    completion_for({"findings": vague_sounding_findings(documents)}),
                    answer,
                    ElicitJsonParser(),
                )
            )
        )
        totals["find/legacy-0.2-codebook"] += float(
            asyncio.run(
                finding_f1(
                    completion_for({"findings": legacy_codebook_findings(documents)}),
                    answer,
                    ElicitJsonParser(),
                )
            )
        )
        totals["elicit-ask/template-question"] += asyncio.run(
            question_utility(
                completion_for(
                    {
                        "questions": template_questions(
                            documents,
                            factions,
                            question_count=QUESTION_COUNT,
                        )
                    }
                ),
                answer,
                info,
                ElicitJsonParser("questions"),
            )
        )
        totals["elicit-ask/randomly-targeted"] += asyncio.run(
            question_utility(
                completion_for(
                    {
                        "questions": randomly_targeted_questions(
                            documents,
                            factions,
                            random_ask_rng,
                            question_count=QUESTION_COUNT,
                        )
                    }
                ),
                answer,
                info,
                ElicitJsonParser("questions"),
            )
        )
        totals["elicit-ask/exact-issue-random-stance"] += asyncio.run(
            question_utility(
                completion_for(
                    {
                        "questions": exact_issue_component_questions(
                            planted_questions,
                            factions,
                            question_count=QUESTION_COUNT,
                            rng=component_rng,
                        )
                    }
                ),
                answer,
                info,
                ElicitJsonParser("questions"),
            )
        )
        totals["elicit-ask/exact-issue-summary-stance"] += asyncio.run(
            question_utility(
                completion_for(
                    {
                        "questions": exact_issue_component_questions(
                            planted_questions,
                            factions,
                            question_count=QUESTION_COUNT,
                            rng=None,
                        )
                    }
                ),
                answer,
                info,
                ElicitJsonParser("questions"),
            )
        )
    return {name: total / len(scenarios) for name, total in totals.items()}


def split_sentences(text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in _SENTENCE_BOUNDARY.split(text)
        if sentence.strip()
    ]


def completion_for(response: Mapping[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "role": "assistant",
            "content": json.dumps(response, sort_keys=True),
        }
    ]


def render_markdown(floors: Mapping[str, float]) -> str:
    labels = {
        "find/random-span": ("Prompt-observable", "find", "Random visible spans"),
        "find/vague-sounding": (
            "Prompt-observable",
            "find",
            "Flag vague-sounding spans",
        ),
        "find/legacy-0.2-codebook": (
            "Prompt-observable",
            "find",
            "Legacy 0.2 document-ID/position codebook",
        ),
        "elicit-ask/template-question": (
            "Prompt-observable",
            "elicit-ask",
            "Template clarity questions",
        ),
        "elicit-ask/randomly-targeted": (
            "Prompt-observable",
            "elicit-ask",
            "Randomly targeted questions",
        ),
        "elicit-ask/exact-issue-random-stance": (
            "Component oracle",
            "elicit-ask",
            "Exact top-K issues + random stances",
        ),
        "elicit-ask/exact-issue-summary-stance": (
            "Component oracle",
            "elicit-ask",
            "Exact top-K issues + visible-summary stances",
        ),
    }
    lines = [
        "| Comparator class | Task | Comparator | mean reward |",
        "| --- | --- | --- | ---: |",
    ]
    lines.extend(
        f"| {labels[name][0]} | {labels[name][1]} | {labels[name][2]} | {score:.3f} |"
        for name, score in floors.items()
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
