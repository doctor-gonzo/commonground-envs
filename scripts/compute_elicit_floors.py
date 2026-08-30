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
from commonground_scenarios import HELDOUT_TEMPLATES, TRAIN_TEMPLATES, validate_scenario

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
_LEAKED_TENDENCY_PATTERN = re.compile(
    r"For decisions involving (?P<terms>[^.]+), "
    r"(?P<tendency>leans toward yes|leans toward no|has no settled position)\."
)
_LEAKED_TENDENCY_TO_STANCE = {
    "leans toward yes": "agree",
    "leans toward no": "disagree",
    "has no settled position": "pass",
}
_REMOVED_0_4_PRINCIPLES: dict[tuple[str, str], tuple[str, ...]] = {
    ("ambiguity", "agree"): (
        "For open thresholds, usually favors case-specific judgment over one fixed cutoff.",
        "When a standard is vague, tends to leave room for local discretion.",
    ),
    ("ambiguity", "disagree"): (
        "For open thresholds, usually favors one fixed cutoff over case-specific judgment.",
        "When a standard is vague, tends to prefer a centrally defined boundary.",
    ),
    ("ambiguity", "pass"): (
        "For open thresholds, weighs local judgment and fixed cutoffs case by case.",
        "Has no general preference between discretion and a fixed boundary.",
    ),
    ("contradiction", "agree"): (
        "When written rules conflict, tends to give a practical exception more weight.",
        "In authority conflicts, usually favors the instruction that preserves operations.",
    ),
    ("contradiction", "disagree"): (
        "When written rules conflict, tends to give the default authority rule more weight.",
        "In authority conflicts, usually resists exceptions to the stated control.",
    ),
    ("contradiction", "pass"): (
        "When written rules conflict, reviews which authority should control case by case.",
        "Has no general preference between a default rule and an operational exception.",
    ),
    ("gap", "agree"): (
        "When a standard channel fails, tends to favor a workable alternate path.",
        "Usually supports a fallback for people unable to use the normal process.",
    ),
    ("gap", "disagree"): (
        "When a standard channel fails, tends to retain the normal requirement.",
        "Usually resists creating a fallback outside the established process.",
    ),
    ("gap", "pass"): (
        "When a standard channel fails, decides whether to allow a fallback case by case.",
        "Has no general preference about alternatives to the normal process.",
    ),
}


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
            "type": "ambiguity",
            "question": f"Is the policy in document {document['doc_id']} clear?",
            "yes_choice": "alternative",
            "related_evidence": None,
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
                "type": "ambiguity",
                "question": (
                    f"Should the rule quoted from document {doc_id} be clarified "
                    f"for case {question_index + 1}?"
                ),
                "yes_choice": "alternative",
                "related_evidence": None,
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
    rng: random.Random,
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
            "type": str(plant["type"]),
            "question": str(plant["question"]),
            "yes_choice": str(plant["yes_choice"]),
            "related_evidence": plant["related_evidence"],
            "target_stances": {
                str(faction["faction_id"]): rng.choice(STANCES) for faction in factions
            },
        }
        for plant in selected
    ]


def exact_issue_exact_stance_questions(
    planted: Sequence[Mapping[str, Any]],
    *,
    question_count: int,
) -> list[dict[str, Any]]:
    """Expose the attainable ceiling when issue selection and stances are exact."""

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
            "type": str(plant["type"]),
            "question": str(plant["question"]),
            "yes_choice": str(plant["yes_choice"]),
            "related_evidence": plant["related_evidence"],
            "target_stances": dict(plant["target_stances"]),
        }
        for plant in selected
    ]


def removed_0_4_principle_codebook_questions(
    planted: Sequence[Mapping[str, Any]],
    factions: Sequence[Mapping[str, str]],
    *,
    question_count: int,
) -> list[dict[str, Any]]:
    """Try the removed 0.4 ``(issue type, stance) -> phrase`` decoder.

    Exact issue locations are supplied as a component oracle so this baseline
    isolates whether public faction prose still reveals each issue-specific
    stance. It scores perfectly on a 0.4-style fixture and zero on the current
    general-value renderer.
    """

    decoded = _decode_removed_0_4_principles(factions)
    if decoded is None:
        return []

    faction_ids = {str(faction["faction_id"]) for faction in factions}
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
            "type": str(plant["type"]),
            "question": str(plant["question"]),
            "yes_choice": str(plant["yes_choice"]),
            "related_evidence": plant["related_evidence"],
            "target_stances": {
                faction_id: decoded[faction_id][str(plant["type"])]
                for faction_id in sorted(faction_ids)
            },
        }
        for plant in selected
    ]


def _decode_removed_0_4_principles(
    factions: Sequence[Mapping[str, str]],
) -> dict[str, dict[str, str]] | None:
    """Decode the removed finite phrase table from prompt-visible summaries."""

    decoded: dict[str, dict[str, str]] = {}
    for faction in factions:
        faction_id = str(faction["faction_id"])
        summary = str(faction["summary"])
        by_type: dict[str, str] = {}
        for (issue_type, stance), phrases in _REMOVED_0_4_PRINCIPLES.items():
            matches = [phrase for phrase in phrases if phrase in summary]
            if not matches:
                continue
            if issue_type in by_type or len(matches) != 1:
                return None
            by_type[issue_type] = stance
        if by_type:
            decoded[faction_id] = by_type

    faction_ids = {str(faction["faction_id"]) for faction in factions}
    if set(decoded) != faction_ids:
        return None
    if any(set(by_type) != set(FINDING_TYPES) for by_type in decoded.values()):
        return None
    return decoded


def source_template_0_4_codebook_questions(
    documents: Sequence[Mapping[str, str]],
    factions: Sequence[Mapping[str, str]],
    *,
    question_count: int,
) -> list[dict[str, Any]]:
    """Combine public template anchors with the removed 0.4 phrase decoder.

    This source-aware adversary reads only prompt fields at runtime, but knows
    the public template registry. Exact anchors recover issue locations and
    authored relationships; the removed summary phrases recover all stances.
    """

    decoded = _decode_removed_0_4_principles(factions)
    if decoded is None:
        return []
    faction_ids = tuple(sorted(decoded))
    detected: list[dict[str, Any]] = []
    for template in (*TRAIN_TEMPLATES, *HELDOUT_TEMPLATES):
        for authored in template.planted_items:
            primary_matches = [
                str(document["doc_id"])
                for document in documents
                if str(authored["anchor_quote"]) in str(document["text"])
            ]
            if len(primary_matches) != 1:
                continue
            related_evidence: dict[str, str] | None = None
            if authored["type"] == "contradiction":
                related_matches = [
                    str(document["doc_id"])
                    for document in documents
                    if str(authored["related_anchor_quote"]) in str(document["text"])
                ]
                if len(related_matches) != 1:
                    continue
                related_evidence = {
                    "doc_id": related_matches[0],
                    "quote": str(authored["related_anchor_quote"]),
                }
            target_stances = {
                faction_id: decoded[faction_id][str(authored["type"])]
                for faction_id in faction_ids
            }
            decision_value = max(
                0.25,
                sum(stance != "pass" for stance in target_stances.values())
                / len(target_stances),
            )
            detected.append(
                {
                    "doc_id": primary_matches[0],
                    "quote": str(authored["anchor_quote"]),
                    "type": str(authored["type"]),
                    "question": sentence_question(str(authored["anchor_quote"])),
                    "yes_choice": str(authored["canonical_yes_choice"]),
                    "related_evidence": related_evidence,
                    "target_stances": target_stances,
                    "decision_value": decision_value,
                }
            )

    if len(detected) != len(FINDING_TYPES):
        return []
    detected.sort(
        key=lambda plant: (
            -float(plant["decision_value"])
            * panel_disagreement(plant["target_stances"]),
            str(plant["doc_id"]),
            str(plant["quote"]),
        )
    )
    return [
        {
            "doc_id": str(plant["doc_id"]),
            "quote": str(plant["quote"]),
            "type": str(plant["type"]),
            "question": str(plant["question"]),
            "yes_choice": str(plant["yes_choice"]),
            "related_evidence": plant["related_evidence"],
            "target_stances": dict(plant["target_stances"]),
        }
        for plant in detected[:question_count]
    ]


def leaked_summary_codebook_questions(
    documents: Sequence[Mapping[str, str]],
    factions: Sequence[Mapping[str, str]],
    *,
    question_count: int,
) -> list[dict[str, Any]]:
    """Exploit the removed 0.3 issue-signature/stance summary clauses.

    The baseline observes only prompt fields. It extracts repeated decision-term
    signatures and exact stance labels, grounds each signature to the visible
    sentence with greatest token overlap, and ranks candidates with the public
    disagreement and keyword rules used by 0.3.0.
    """

    signatures: dict[tuple[str, ...], dict[str, str]] = {}
    for faction in factions:
        faction_id = str(faction["faction_id"])
        for match in _LEAKED_TENDENCY_PATTERN.finditer(str(faction["summary"])):
            terms = tuple(
                term.strip().casefold()
                for term in match.group("terms").split(",")
                if term.strip()
            )
            signatures.setdefault(terms, {})[faction_id] = _LEAKED_TENDENCY_TO_STANCE[
                match.group("tendency")
            ]
    faction_ids = {str(faction["faction_id"]) for faction in factions}
    visible_spans = [
        (str(document["doc_id"]), sentence)
        for document in documents
        for sentence in split_sentences(str(document["text"]))
    ]
    candidates: list[tuple[float, dict[str, Any]]] = []
    for terms, stances in signatures.items():
        if set(stances) != faction_ids:
            continue
        term_set = set(terms)
        overlap, doc_id, quote = max(
            (
                len(term_set & set(re.findall(r"[^\W_]+", sentence.casefold()))),
                candidate_doc_id,
                sentence,
            )
            for candidate_doc_id, sentence in visible_spans
        )
        if overlap < 2:
            continue
        question = f"Should {' '.join(terms)} be clarified?"
        impact = leaked_keyword_decision_value(question, quote)
        candidates.append(
            (
                impact * panel_disagreement(stances),
                {
                    "doc_id": doc_id,
                    "quote": quote,
                    "type": "ambiguity",
                    "question": question,
                    "yes_choice": "alternative",
                    "related_evidence": None,
                    "target_stances": stances,
                },
            )
        )
    candidates.sort(
        key=lambda item: (
            -item[0],
            str(item[1]["doc_id"]),
            str(item[1]["quote"]),
        )
    )
    return [candidate for _, candidate in candidates[:question_count]]


def leaked_keyword_decision_value(question: str, anchor: str) -> float:
    """Reproduce the removed 0.3 keyword value rule using visible text only."""

    text = f"{question} {anchor}".casefold()
    if any(term in text for term in ("safety", "emergency", "urgent", "harm")):
        return 1.0
    if any(
        term in text
        for term in ("authority", "authorize", "approval", "prohibit", "require")
    ):
        return 0.85
    if any(
        term in text
        for term in ("access", "unavailable", "unreachable", "offline", "cannot")
    ):
        return 0.7
    return 0.55


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
        "elicit-ask/legacy-0.3-summary-codebook": 0.0,
        "elicit-ask/legacy-0.4-principle-codebook": 0.0,
        "elicit-ask/source-template-0.4-principle-codebook": 0.0,
        "elicit-ask/exact-issue-random-stance": 0.0,
        "elicit-ask/exact-issue-exact-stance": 0.0,
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
                "type": str(plant["type"]),
                "question": str(plant["canonical_question"]),
                "question_aliases": list(plant["canonical_question_aliases"]),
                "yes_choice": str(plant["canonical_yes_choice"]),
                "target_stances": dict(plant["target_stances"]),
                "alternative_stances": dict(plant["alternative_stances"]),
                "related_evidence": plant["related_evidence"],
                "related_document_text": (
                    documents_by_id[str(plant["related_evidence"]["doc_id"])]
                    if plant["related_evidence"] is not None
                    else None
                ),
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
        totals["elicit-ask/legacy-0.3-summary-codebook"] += asyncio.run(
            question_utility(
                completion_for(
                    {
                        "questions": leaked_summary_codebook_questions(
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
        totals["elicit-ask/legacy-0.4-principle-codebook"] += asyncio.run(
            question_utility(
                completion_for(
                    {
                        "questions": removed_0_4_principle_codebook_questions(
                            planted_questions,
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
        totals["elicit-ask/source-template-0.4-principle-codebook"] += asyncio.run(
            question_utility(
                completion_for(
                    {
                        "questions": source_template_0_4_codebook_questions(
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
        totals["elicit-ask/exact-issue-exact-stance"] += asyncio.run(
            question_utility(
                completion_for(
                    {
                        "questions": exact_issue_exact_stance_questions(
                            planted_questions,
                            question_count=QUESTION_COUNT,
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
        "elicit-ask/legacy-0.3-summary-codebook": (
            "Prompt-observable",
            "elicit-ask",
            "Removed 0.3 summary/stance codebook",
        ),
        "elicit-ask/legacy-0.4-principle-codebook": (
            "Component oracle",
            "elicit-ask",
            "Exact issues + removed 0.4 principle-table parser",
        ),
        "elicit-ask/source-template-0.4-principle-codebook": (
            "Source-aware prompt-only",
            "elicit-ask",
            "Public template detector + removed 0.4 principle-table parser",
        ),
        "elicit-ask/exact-issue-random-stance": (
            "Component oracle",
            "elicit-ask",
            "Exact top-K issues + random stances",
        ),
        "elicit-ask/exact-issue-exact-stance": (
            "Component oracle",
            "elicit-ask",
            "Exact top-K issues + exact stances (ceiling)",
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
