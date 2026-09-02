"""Compute deterministic model-free floors for commonground-elicit."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from commonground_elicit.environment import (
    BUNDLED_EVAL_PATH,
    PLANT_COVERAGE_THRESHOLD,
    QUOTE_OVERLAP_THRESHOLD,
    QUOTE_PRECISION_THRESHOLD,
    ElicitJsonParser,
    _canonical_decision_frame,
    _oriented_decision_question,
    finding_f1,
    finding_localization_recall,
    finding_type_accuracy,
    match_findings,
    normalized_contiguous_quote,
    normalized_plant_coverage,
    normalized_quote_overlap,
    normalized_quote_precision,
    panel_disagreement,
    question_utility,
)
from commonground_scenarios import HELDOUT_TEMPLATES, TRAIN_TEMPLATES, validate_scenario
from commonground_scenarios.generator import (
    _SHARED_PROCEDURAL_PREDICATES,
    _authored_decision_frame,
    generate_scenario,
    orient_stances,
)
from commonground_scenarios.templates import VALUE_DIMENSIONS
from commonground_scenarios.validation import (
    PASS_THRESHOLD,
    preference_tradeoff_value,
)

QUESTION_COUNT = 1
MIN_TOP1_NORMALIZED_MARGIN = 0.10
FINDING_BUDGET = 3
FINDING_TYPES = ("ambiguity", "contradiction", "gap")
STANCES = ("agree", "disagree", "pass")
HELPER_HELDOUT_TEMPLATE_IDS = frozenset(
    template.template_id
    for template in HELDOUT_TEMPLATES
    if template.balance_type_neutral_distractors
)
DOCUMENT_ROLE_LABELS = (
    "ambiguity-primary",
    "contradiction-primary",
    "contradiction-related",
    "gap-primary",
    "plant-free",
)
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
_OLD_FIXED_FILLERS = frozenset(
    {
        "The document owner records each revision.",
        "Approved copies carry a control code.",
        "Archived versions remain read-only.",
        "Completed reviews are dated in the case record.",
        "The responsible desk logs each completed handoff.",
    }
)
_OLD_TYPE_MARKERS = {
    "Completed reviews are dated in the case record.": "ambiguity",
    "The responsible desk logs each completed handoff.": "contradiction",
}
_VALUE_PROFILE_PATTERN = re.compile(
    r"Value profile used for this panel: "
    r"(?P<profile>.*?safety=[+-]?\d+(?:\.\d+)?)\."
)
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
        {
            "doc_id": str(document["doc_id"]),
            "title": str(document["title"]),
            "style": str(document["style"]),
            "text": str(document["text"]),
        }
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
            "decision": visible_decision_frame(quote, sentence_question(quote)),
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
                        "decision": visible_decision_frame(
                            sentence, sentence_question(sentence)
                        ),
                        "related_evidence": None,
                    }
                )
    return findings


def old_fixed_marker_findings(
    documents: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    """Exploit the removed fixed filler-to-type association from 0.4/0.5."""

    findings: list[dict[str, Any]] = []
    for document in documents:
        sentences = split_sentences(str(document["text"]))
        marker = next(
            (candidate for candidate in sentences if candidate in _OLD_TYPE_MARKERS),
            None,
        )
        if marker is None:
            continue
        candidates = [
            sentence for sentence in sentences if sentence not in _OLD_FIXED_FILLERS
        ]
        if len(candidates) != 1:
            continue
        quote = candidates[0]
        diagnosis = sentence_question(quote)
        findings.append(
            {
                "doc_id": str(document["doc_id"]),
                "quote": quote,
                "type": _OLD_TYPE_MARKERS[marker],
                "diagnosis": diagnosis,
                "decision": visible_decision_frame(quote, diagnosis),
                "related_evidence": None,
            }
        )
    return findings


def layout_position_length_findings(
    documents: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    """Probe a length-rank and fixed-position layout shortcut.

    The adversary uses no policy vocabulary. It maps the three longest
    documents to fixed issue labels and selects a deterministic relative
    sentence position. Balanced layouts should keep this near a weak floor.
    """

    ranked = sorted(
        documents,
        key=lambda document: (
            -len(re.findall(r"[^\W_]+", str(document["text"]))),
            str(document["doc_id"]),
        ),
    )[:FINDING_BUDGET]
    findings: list[dict[str, Any]] = []
    for rank, (document, issue_type) in enumerate(
        zip(ranked, FINDING_TYPES, strict=True)
    ):
        sentences = split_sentences(str(document["text"]))
        quote = sentences[min(rank, len(sentences) - 1)]
        diagnosis = sentence_question(quote)
        findings.append(
            {
                "doc_id": str(document["doc_id"]),
                "quote": quote,
                "type": issue_type,
                "diagnosis": diagnosis,
                "decision": visible_decision_frame(quote, diagnosis),
                "related_evidence": None,
            }
        )
    return findings


def longest_visible_sentence_findings(
    documents: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    """Select the three longest prompt-visible sentences without source access."""

    remaining: list[tuple[str, str]] = []
    for document in documents:
        for sentence in split_sentences(str(document["text"])):
            remaining.append((str(document["doc_id"]), sentence))
    # Respect the same three-finding budget as the other baselines. This is a
    # genuine prompt-only structural probe: it has no template, source, label,
    # relationship, or retired-filler knowledge.
    remaining.sort(
        key=lambda item: (
            -len(re.findall(r"[^\W_]+", item[1], flags=re.UNICODE)),
            item[0],
            item[1],
        )
    )
    findings: list[dict[str, Any]] = []
    for doc_id, sentence in remaining[:FINDING_BUDGET]:
        diagnosis = sentence_question(sentence)
        issue_type = FINDING_TYPES[
            hashlib.sha256(f"{doc_id}\0{sentence}".encode()).digest()[0]
            % len(FINDING_TYPES)
        ]
        findings.append(
            {
                "doc_id": doc_id,
                "quote": sentence,
                "type": issue_type,
                "diagnosis": diagnosis,
                "decision": visible_decision_frame(sentence, diagnosis),
                "related_evidence": None,
            }
        )
    return findings


def shared_predicate_exclusion_findings(
    documents: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    """Strip sentences carrying the current procedural component vocabulary.

    In the vulnerable construction this recovered every authored candidate,
    because only generated distractors used the fixed predicates. Version 0.6
    deliberately renders authored and distractor spans from the same component
    distribution, so the source-aware exclusion attack has no privileged set.
    """

    remaining = [
        (str(document["doc_id"]), sentence)
        for document in documents
        for sentence in split_sentences(str(document["text"]))
        if not any(predicate in sentence for predicate in _SHARED_PROCEDURAL_PREDICATES)
    ]
    remaining.sort(key=lambda item: (item[0], item[1]))
    findings: list[dict[str, Any]] = []
    for doc_id, sentence in remaining[:FINDING_BUDGET]:
        diagnosis = sentence_question(sentence)
        issue_type = FINDING_TYPES[
            hashlib.sha256(f"{doc_id}\0{sentence}".encode()).digest()[0]
            % len(FINDING_TYPES)
        ]
        findings.append(
            {
                "doc_id": doc_id,
                "quote": sentence,
                "type": issue_type,
                "diagnosis": diagnosis,
                "decision": visible_decision_frame(sentence, diagnosis),
                "related_evidence": None,
            }
        )
    return findings


def sector_team_marker_findings(
    documents: Sequence[Mapping[str, str]],
    *,
    sector: str,
) -> list[dict[str, Any]]:
    """Flag every span carrying the source-visible organization-team marker.

    The 0.4.1 generator stamped this phrase on every planted anchor but only a
    minority of neutral spans. Keeping this source-aware adversary in the floor
    report makes a recurrence visible even if strict semantic scoring remains
    low because its issue-type and diagnosis guesses are deliberately generic.
    """

    marker = f"the {sector} team"
    findings: list[dict[str, Any]] = []
    for document in documents:
        for sentence in split_sentences(str(document["text"])):
            if marker not in sentence.casefold():
                continue
            diagnosis = sentence_question(sentence)
            issue_type = FINDING_TYPES[
                hashlib.sha256(f"{document['doc_id']}\0{sentence}".encode()).digest()[0]
                % len(FINDING_TYPES)
            ]
            findings.append(
                {
                    "doc_id": str(document["doc_id"]),
                    "quote": sentence,
                    "type": issue_type,
                    "diagnosis": diagnosis,
                    "decision": visible_decision_frame(sentence, diagnosis),
                    "related_evidence": None,
                }
            )
    return findings


def visible_decision_frame(
    quote: str,
    question: str,
    related_quote: str | None = None,
) -> dict[str, str]:
    """Build a deterministic frame using only prompt-visible candidate text."""

    tokens = re.findall(r"[^\W_]+", f"{question} {quote}", flags=re.UNICODE)
    if not tokens:
        tokens = ["policy", "decision", "scope", "record"]
    while len(tokens) < 8:
        tokens.extend(tokens[: 8 - len(tokens)])
    return {
        "actor": " ".join(tokens[:2]),
        "action": " ".join(tokens[2:4]),
        "condition": " ".join(tokens[4:8]),
        "anchor_outcome": quote,
        "alternative_outcome": related_quote or question,
    }


def localization_component_oracle_findings(
    predicted: Sequence[Mapping[str, Any]],
    planted: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Fill only non-localization fields from gold for already located spans.

    A locator can otherwise look harmless merely because its deliberately
    generic type or diagnosis fails before the strict matcher reaches the
    span. This component oracle preserves every predicted span and every false
    positive. It supplies exact type, diagnosis, decision, and relationship
    fields only when the predicted span passes the strict scorer's document,
    contiguity, coverage, precision, and overlap gates, so strict F1 becomes a
    precision-sensitive measure of localization alone.
    """

    non_localization_fields = (
        "type",
        "diagnosis",
        "decision",
        "related_evidence",
    )
    completed: list[dict[str, Any]] = []
    for candidate in predicted:
        localized: list[tuple[float, int, Mapping[str, Any]]] = []
        for plant_index, plant in enumerate(planted):
            if candidate.get("doc_id") != plant.get("doc_id"):
                continue
            candidate_quote = str(candidate.get("quote", ""))
            plant_quote = str(plant.get("quote", ""))
            if not normalized_contiguous_quote(
                candidate_quote,
                str(plant.get("document_text", plant_quote)),
            ):
                continue
            if (
                normalized_plant_coverage(candidate_quote, plant_quote)
                < PLANT_COVERAGE_THRESHOLD
                or normalized_quote_precision(candidate_quote, plant_quote)
                < QUOTE_PRECISION_THRESHOLD
            ):
                continue
            overlap = normalized_quote_overlap(candidate_quote, plant_quote)
            if overlap >= QUOTE_OVERLAP_THRESHOLD:
                localized.append((overlap, plant_index, plant))
        if not localized:
            completed.append(dict(candidate))
            continue
        _, _, gold = min(localized, key=lambda match: (-match[0], match[1]))
        completed.append(
            {
                "doc_id": str(candidate["doc_id"]),
                "quote": str(candidate["quote"]),
                **{field: gold[field] for field in non_localization_fields},
            }
        )
    return completed


def template_questions(
    documents: Sequence[Mapping[str, str]],
    factions: Sequence[Mapping[str, str]],
    *,
    question_count: int,
) -> list[dict[str, Any]]:
    """Ask a generic clarity question using only prompt-visible context."""

    questions: list[dict[str, Any]] = []
    for question_index, document in enumerate(documents[:question_count]):
        quote = split_sentences(str(document["text"]))[0]
        question = f"Is the policy in document {document['doc_id']} clear?"
        questions.append(
            {
                "doc_id": str(document["doc_id"]),
                "quote": quote,
                "type": "ambiguity",
                "question": question,
                "decision": visible_decision_frame(quote, question),
                "yes_choice": "alternative",
                "related_evidence": None,
                "target_stances": {
                    str(faction["faction_id"]): STANCES[
                        (question_index + faction_index) % len(STANCES)
                    ]
                    for faction_index, faction in enumerate(factions)
                },
            }
        )
    return questions


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
        question = (
            f"Should the rule quoted from document {doc_id} be clarified "
            f"for case {question_index + 1}?"
        )
        questions.append(
            {
                "doc_id": doc_id,
                "quote": quote,
                "type": "ambiguity",
                "question": question,
                "decision": visible_decision_frame(quote, question),
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
            "decision": dict(plant["decision"]),
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
            "decision": dict(plant["decision"]),
            "yes_choice": str(plant["yes_choice"]),
            "related_evidence": plant["related_evidence"],
            "target_stances": dict(plant["target_stances"]),
        }
        for plant in selected
    ]


def random_issue_exact_components_questions(
    planted: Sequence[Mapping[str, Any]],
    *,
    question_count: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Select random exact issues while keeping every per-issue component exact."""

    selected = list(planted)
    rng.shuffle(selected)
    return [
        {
            "doc_id": str(plant["doc_id"]),
            "quote": str(plant["quote"]),
            "type": str(plant["type"]),
            "question": str(plant["question"]),
            "decision": dict(plant["decision"]),
            "yes_choice": str(plant["yes_choice"]),
            "related_evidence": plant["related_evidence"],
            "target_stances": dict(plant["target_stances"]),
        }
        for plant in selected[:question_count]
    ]


def runner_up_exact_components_questions(
    planted: Sequence[Mapping[str, Any]],
    *,
    question_count: int,
) -> list[dict[str, Any]]:
    """Select the exact runner-up with every per-issue component exact."""

    ranked = sorted(planted, key=_question_rank_key)
    return [
        _candidate_question(plant)
        for plant in ranked[question_count : 2 * question_count]
    ]


def public_profile_composition_questions(
    planted: Sequence[Mapping[str, Any]],
    factions: Sequence[Mapping[str, str]],
    *,
    question_count: int,
) -> list[dict[str, Any]]:
    """Compose ranks/stances from public profiles with exact grounding supplied.

    This is deliberately a component oracle, not a prompt-only baseline: issue
    locations and semantic frames come from ``planted``. The target-determining
    stance and ranking values are recomputed solely from faction value summaries
    plus the candidate weights rendered in the Ask prompt.
    """

    value_factions = _visible_value_factions(factions)
    if value_factions is None or any(
        set(faction["values"]) != set(VALUE_DIMENSIONS) for faction in value_factions
    ):
        return []
    composed: list[dict[str, Any]] = []
    for plant in planted:
        raw_weights = plant.get("value_weights")
        if not isinstance(raw_weights, Mapping) or set(raw_weights) != set(
            VALUE_DIMENSIONS
        ):
            return []
        weights = {
            dimension: float(raw_weights[dimension]) for dimension in VALUE_DIMENSIONS
        }
        scale = sum(abs(weight) for weight in weights.values())
        if scale <= 0:
            return []
        alternative_stances: dict[str, str] = {}
        for faction in value_factions:
            preference = (
                sum(
                    float(faction["values"][dimension]) * weights[dimension]
                    for dimension in VALUE_DIMENSIONS
                )
                / scale
            )
            alternative_stances[str(faction["faction_id"])] = (
                "agree"
                if preference >= PASS_THRESHOLD
                else "disagree"
                if preference <= -PASS_THRESHOLD
                else "pass"
            )
        target_stances = orient_stances(
            alternative_stances,
            yes_choice=str(plant["yes_choice"]),
        )
        composed.append(
            {
                **plant,
                "target_stances": target_stances,
                "decision_value": preference_tradeoff_value(
                    value_factions,
                    weights,
                ),
            }
        )
    return [
        _candidate_question(plant)
        for plant in sorted(composed, key=_question_rank_key)[:question_count]
    ]


def exact_uniform_selection_expected_reward(
    planted: Sequence[Mapping[str, Any]],
) -> float:
    """Return the analytic reward of uniform top-one issue selection."""

    utilities = [_question_attainable_utility(plant) for plant in planted]
    denominator = max(utilities, default=0.0)
    if denominator <= 0 or not utilities:
        return 0.0
    return sum(utility / denominator for utility in utilities) / len(utilities)


def top1_selection_accuracy(
    selected: Sequence[Mapping[str, Any]],
    planted: Sequence[Mapping[str, Any]],
) -> float:
    """Measure whether the submitted K=1 evidence identifies the true top issue."""

    if len(selected) != QUESTION_COUNT or not planted:
        return 0.0
    expected = sorted(planted, key=_question_rank_key)[:QUESTION_COUNT]
    selected_keys = {
        (str(candidate.get("doc_id", "")), str(candidate.get("quote", "")))
        for candidate in selected
    }
    expected_keys = {
        (str(candidate["doc_id"]), str(candidate["quote"])) for candidate in expected
    }
    return len(selected_keys & expected_keys) / QUESTION_COUNT


def _candidate_question(plant: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "doc_id": str(plant["doc_id"]),
        "quote": str(plant["quote"]),
        "type": str(plant["type"]),
        "question": str(plant["question"]),
        "decision": dict(plant["decision"]),
        "yes_choice": str(plant["yes_choice"]),
        "related_evidence": plant["related_evidence"],
        "target_stances": dict(plant["target_stances"]),
    }


def _question_attainable_utility(plant: Mapping[str, Any]) -> float:
    return float(plant["decision_value"]) * panel_disagreement(plant["target_stances"])


def _question_rank_key(plant: Mapping[str, Any]) -> tuple[float, str, str]:
    return (
        -_question_attainable_utility(plant),
        str(plant["doc_id"]),
        str(plant["quote"]),
    )


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
            "decision": dict(plant["decision"]),
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
    value_factions = _visible_value_factions(factions)
    if value_factions is None:
        return []
    faction_ids = tuple(sorted(decoded))
    detected: list[dict[str, Any]] = []
    for template in (*TRAIN_TEMPLATES, *HELDOUT_TEMPLATES):
        for authored in template.planted_items:
            # Current rows attach a shared procedural clause to every authored
            # sentence, replacing its terminal period with a semicolon.  The
            # Source-aware replay should still recognize the authored policy
            # clause without depending on that neutral rendering detail.
            primary_anchor = str(authored["anchor_quote"]).removesuffix(".")
            primary_matches = [
                (str(document["doc_id"]), sentence)
                for document in documents
                for sentence in split_sentences(str(document["text"]))
                if primary_anchor in sentence
            ]
            if len(primary_matches) != 1:
                continue
            primary_doc_id, primary_quote = primary_matches[0]
            related_evidence: dict[str, str] | None = None
            if authored["type"] == "contradiction":
                related_anchor = str(authored["related_anchor_quote"]).removesuffix(".")
                related_matches = [
                    (str(document["doc_id"]), sentence)
                    for document in documents
                    for sentence in split_sentences(str(document["text"]))
                    if related_anchor in sentence
                ]
                if len(related_matches) != 1:
                    continue
                related_evidence = {
                    "doc_id": related_matches[0][0],
                    "quote": related_matches[0][1],
                }
            target_stances = {
                faction_id: decoded[faction_id][str(authored["type"])]
                for faction_id in faction_ids
            }
            reference_plant = {
                **authored,
                "doc_id": primary_doc_id,
                "anchor_quote": primary_quote,
                "related_evidence": related_evidence,
                "decision": _authored_decision_frame(
                    template.template_id,
                    str(authored["plant_id"]),
                ),
            }
            base_anchor = primary_anchor
            policy_clause = primary_quote.split(";", maxsplit=1)[0].strip()
            scoped_suffix = policy_clause.removeprefix(base_anchor).removesuffix(".")
            if scoped_suffix:
                reference_plant["decision"]["condition"] = (
                    f"{reference_plant['decision']['condition']} {scoped_suffix.strip()}"
                )
            decision_value = preference_tradeoff_value(
                value_factions,
                authored["value_weights"],
            )
            decision = _canonical_decision_frame(reference_plant, documents)
            yes_choice = str(authored["canonical_yes_choice"])
            question = str(authored["canonical_question"])
            if scoped_suffix:
                question = f"{question.removesuffix('?')} {scoped_suffix.strip()}?"
            detected.append(
                {
                    "doc_id": primary_doc_id,
                    "quote": primary_quote,
                    "type": str(authored["type"]),
                    "question": _oriented_decision_question(
                        question, decision, yes_choice
                    ),
                    "decision": decision,
                    "yes_choice": yes_choice,
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
            "decision": dict(plant["decision"]),
            "yes_choice": str(plant["yes_choice"]),
            "related_evidence": plant["related_evidence"],
            "target_stances": dict(plant["target_stances"]),
        }
        for plant in detected[:question_count]
    ]


def _visible_value_factions(
    factions: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]] | None:
    """Parse the exact value profile deliberately rendered in public summaries."""

    parsed: list[dict[str, Any]] = []
    for faction in factions:
        match = _VALUE_PROFILE_PATTERN.search(str(faction["summary"]))
        if match is None:
            return None
        values: dict[str, float] = {}
        for assignment in match.group("profile").split(","):
            dimension, separator, raw_value = assignment.strip().partition("=")
            if not separator:
                return None
            try:
                values[dimension] = float(raw_value)
            except ValueError:
                return None
        parsed.append(
            {
                "faction_id": str(faction["faction_id"]),
                "values": values,
            }
        )
    return parsed


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
                    "decision": visible_decision_frame(quote, question),
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
        diagnosis = sentence_question(quote)
        findings.append(
            {
                "doc_id": document["doc_id"],
                "quote": quote,
                "type": issue_types[document["doc_id"]],
                "diagnosis": diagnosis,
                "decision": visible_decision_frame(quote, diagnosis),
                "related_evidence": None,
            }
        )
    return findings


def exact_answer_for_scenario(
    scenario: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Construct the exact scorer oracle for one validated scenario."""

    documents = public_documents(scenario)
    documents_by_id = {document["doc_id"]: document["text"] for document in documents}
    findings: list[dict[str, Any]] = []
    questions: list[dict[str, Any]] = []
    for plant in scenario["planted_items"]:
        decision = _canonical_decision_frame(plant, scenario["documents"])
        yes_choice = str(plant["canonical_yes_choice"])
        oriented_question = _oriented_decision_question(
            str(plant["canonical_question"]), decision, yes_choice
        )
        findings.append(
            {
                "doc_id": str(plant["doc_id"]),
                "quote": str(plant["anchor_quote"]),
                "type": str(plant["type"]),
                "diagnosis": oriented_question,
                "decision": decision,
                "related_evidence": plant["related_evidence"],
                "document_text": documents_by_id[str(plant["doc_id"])],
                "related_document_text": (
                    documents_by_id[str(plant["related_evidence"]["doc_id"])]
                    if plant["related_evidence"] is not None
                    else None
                ),
            }
        )
        questions.append(
            {
                "doc_id": str(plant["doc_id"]),
                "quote": str(plant["anchor_quote"]),
                "type": str(plant["type"]),
                "question": oriented_question,
                "question_aliases": [
                    _oriented_decision_question(str(alias), decision, yes_choice)
                    for alias in plant["canonical_question_aliases"]
                ],
                "decision": decision,
                "value_weights": {
                    dimension: float(plant["value_weights"][dimension])
                    for dimension in VALUE_DIMENSIONS
                },
                "yes_choice": yes_choice,
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
        )
    return {"findings": findings, "questions": questions}


def candidate_response_from_exact_answer(
    answer: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    question_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Strip hidden analysis fields from a replayed exact answer."""

    finding_fields = (
        "doc_id",
        "quote",
        "type",
        "diagnosis",
        "decision",
        "related_evidence",
    )
    question_fields = (
        "doc_id",
        "quote",
        "type",
        "question",
        "decision",
        "yes_choice",
        "related_evidence",
        "target_stances",
    )
    findings = [
        {field: finding[field] for field in finding_fields}
        for finding in answer["findings"]
    ]
    questions = [
        {field: question[field] for field in question_fields}
        for question in answer["questions"][:question_count]
    ]
    return findings, questions


def compute_elicit_floors(path: Path = BUNDLED_EVAL_PATH) -> dict[str, float]:
    """Score prompt baselines and component oracles on a validated split."""

    totals = {
        "find/random-span": 0.0,
        "find/vague-sounding": 0.0,
        "find/removed-fixed-filler-marker": 0.0,
        "find/removed-fixed-filler-marker-localization-recall": 0.0,
        "find/removed-fixed-filler-marker-type-accuracy": 0.0,
        "find/removed-fixed-filler-marker-localization-component-oracle-f1": 0.0,
        "find/layout-position-length": 0.0,
        "find/layout-position-length-localization-recall": 0.0,
        "find/layout-position-length-type-accuracy": 0.0,
        "find/layout-position-length-localization-component-oracle-f1": 0.0,
        "find/longest-visible-sentences": 0.0,
        "find/longest-visible-sentences-localization-recall": 0.0,
        "find/longest-visible-sentences-type-accuracy": 0.0,
        "find/longest-visible-sentences-localization-component-oracle-f1": 0.0,
        "find/shared-predicate-exclusion": 0.0,
        "find/shared-predicate-exclusion-localization-recall": 0.0,
        "find/shared-predicate-exclusion-type-accuracy": 0.0,
        "find/sector-team-marker": 0.0,
        "find/sector-team-marker-localization-recall": 0.0,
        "find/sector-team-marker-localization-f1": 0.0,
        "find/sector-team-marker-localization-component-oracle-f1": 0.0,
        "find/public-source-replay": 0.0,
        "find/legacy-0.2-codebook": 0.0,
        "elicit-ask/template-question": 0.0,
        "elicit-ask/randomly-targeted": 0.0,
        "elicit-ask/legacy-0.3-summary-codebook": 0.0,
        "elicit-ask/legacy-0.4-principle-codebook": 0.0,
        "elicit-ask/source-template-0.4-principle-codebook": 0.0,
        "elicit-ask/random-issue-exact-components": 0.0,
        "elicit-ask/random-issue-top1-selection-accuracy": 0.0,
        "elicit-ask/runner-up-exact-components": 0.0,
        "elicit-ask/runner-up-top1-selection-accuracy": 0.0,
        "elicit-ask/public-profile-composition": 0.0,
        "elicit-ask/public-profile-top1-selection-accuracy": 0.0,
        "elicit-ask/exact-issue-random-stance": 0.0,
        "elicit-ask/exact-issue-exact-stance": 0.0,
        "elicit-ask/exact-issue-top1-selection-accuracy": 0.0,
        "elicit-ask/public-source-replay": 0.0,
    }
    scenarios = load_scenarios(path)
    for scenario in scenarios:
        documents = public_documents(scenario)
        factions = public_factions(scenario)
        scenario_id = str(scenario["scenario_id"])
        random_find_rng = random.Random(f"{scenario_id}:random-span")
        random_ask_rng = random.Random(f"{scenario_id}:randomly-targeted")
        component_rng = random.Random(f"{scenario_id}:component-random-stance")
        answer = exact_answer_for_scenario(scenario)
        planted_questions = answer["questions"]
        provenance = scenario["provenance"]
        replayed_scenario = generate_scenario(
            int(provenance["seed"]),
            str(provenance["template_id"]),
            generated_at=str(provenance["generated_at"]),
        )
        if replayed_scenario != scenario:
            raise ValueError(
                f"bundled scenario is not reproducible from public source: {scenario_id}"
            )
        replayed_answer = exact_answer_for_scenario(replayed_scenario)
        replayed_findings, _ = candidate_response_from_exact_answer(
            replayed_answer,
            question_count=QUESTION_COUNT,
        )
        replayed_questions = exact_issue_exact_stance_questions(
            replayed_answer["questions"],
            question_count=QUESTION_COUNT,
        )
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
        marker_findings = old_fixed_marker_findings(documents)
        marker_completion = completion_for({"findings": marker_findings})
        totals["find/removed-fixed-filler-marker"] += float(
            asyncio.run(
                finding_f1(
                    marker_completion,
                    answer,
                    ElicitJsonParser(),
                )
            )
        )
        totals["find/removed-fixed-filler-marker-localization-recall"] += float(
            asyncio.run(
                finding_localization_recall(
                    marker_completion,
                    answer,
                    ElicitJsonParser(),
                )
            )
        )
        totals["find/removed-fixed-filler-marker-type-accuracy"] += float(
            asyncio.run(
                finding_type_accuracy(
                    marker_completion,
                    answer,
                    ElicitJsonParser(),
                )
            )
        )
        totals["find/removed-fixed-filler-marker-localization-component-oracle-f1"] += (
            float(
                asyncio.run(
                    finding_f1(
                        completion_for(
                            {
                                "findings": localization_component_oracle_findings(
                                    marker_findings, answer["findings"]
                                )
                            }
                        ),
                        answer,
                        ElicitJsonParser(),
                    )
                )
            )
        )
        layout_findings = layout_position_length_findings(documents)
        layout_completion = completion_for({"findings": layout_findings})
        totals["find/layout-position-length"] += float(
            asyncio.run(
                finding_f1(
                    layout_completion,
                    answer,
                    ElicitJsonParser(),
                )
            )
        )
        totals["find/layout-position-length-localization-recall"] += float(
            asyncio.run(
                finding_localization_recall(
                    layout_completion,
                    answer,
                    ElicitJsonParser(),
                )
            )
        )
        totals["find/layout-position-length-type-accuracy"] += float(
            asyncio.run(
                finding_type_accuracy(
                    layout_completion,
                    answer,
                    ElicitJsonParser(),
                )
            )
        )
        totals["find/layout-position-length-localization-component-oracle-f1"] += float(
            asyncio.run(
                finding_f1(
                    completion_for(
                        {
                            "findings": localization_component_oracle_findings(
                                layout_findings, answer["findings"]
                            )
                        }
                    ),
                    answer,
                    ElicitJsonParser(),
                )
            )
        )
        longest_findings = longest_visible_sentence_findings(documents)
        longest_completion = completion_for({"findings": longest_findings})
        totals["find/longest-visible-sentences"] += float(
            asyncio.run(
                finding_f1(
                    longest_completion,
                    answer,
                    ElicitJsonParser(),
                )
            )
        )
        totals["find/longest-visible-sentences-localization-recall"] += float(
            asyncio.run(
                finding_localization_recall(
                    longest_completion,
                    answer,
                    ElicitJsonParser(),
                )
            )
        )
        totals["find/longest-visible-sentences-type-accuracy"] += float(
            asyncio.run(
                finding_type_accuracy(
                    longest_completion,
                    answer,
                    ElicitJsonParser(),
                )
            )
        )
        totals["find/longest-visible-sentences-localization-component-oracle-f1"] += (
            float(
                asyncio.run(
                    finding_f1(
                        completion_for(
                            {
                                "findings": localization_component_oracle_findings(
                                    longest_findings, answer["findings"]
                                )
                            }
                        ),
                        answer,
                        ElicitJsonParser(),
                    )
                )
            )
        )
        predicate_findings = shared_predicate_exclusion_findings(documents)
        predicate_completion = completion_for({"findings": predicate_findings})
        totals["find/shared-predicate-exclusion"] += float(
            asyncio.run(
                finding_f1(
                    predicate_completion,
                    answer,
                    ElicitJsonParser(),
                )
            )
        )
        totals["find/shared-predicate-exclusion-localization-recall"] += float(
            asyncio.run(
                finding_localization_recall(
                    predicate_completion,
                    answer,
                    ElicitJsonParser(),
                )
            )
        )
        totals["find/shared-predicate-exclusion-type-accuracy"] += float(
            asyncio.run(
                finding_type_accuracy(
                    predicate_completion,
                    answer,
                    ElicitJsonParser(),
                )
            )
        )
        team_marker_findings = sector_team_marker_findings(
            documents,
            sector=str(scenario["organization"]["sector"]),
        )
        team_marker_completion = completion_for({"findings": team_marker_findings})
        totals["find/sector-team-marker"] += float(
            asyncio.run(
                finding_f1(
                    team_marker_completion,
                    answer,
                    ElicitJsonParser(),
                )
            )
        )
        totals["find/sector-team-marker-localization-recall"] += float(
            asyncio.run(
                finding_localization_recall(
                    team_marker_completion,
                    answer,
                    ElicitJsonParser(),
                )
            )
        )
        totals["find/sector-team-marker-localization-f1"] += float(
            match_findings(team_marker_findings, answer["findings"])["localization_f1"]
        )
        totals["find/sector-team-marker-localization-component-oracle-f1"] += float(
            asyncio.run(
                finding_f1(
                    completion_for(
                        {
                            "findings": localization_component_oracle_findings(
                                team_marker_findings, answer["findings"]
                            )
                        }
                    ),
                    answer,
                    ElicitJsonParser(),
                )
            )
        )
        totals["find/public-source-replay"] += float(
            asyncio.run(
                finding_f1(
                    completion_for({"findings": replayed_findings}),
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
        totals["elicit-ask/random-issue-exact-components"] += (
            exact_uniform_selection_expected_reward(planted_questions)
        )
        totals["elicit-ask/random-issue-top1-selection-accuracy"] += (
            QUESTION_COUNT / len(planted_questions)
        )
        runner_up_questions = runner_up_exact_components_questions(
            planted_questions,
            question_count=QUESTION_COUNT,
        )
        totals["elicit-ask/runner-up-exact-components"] += asyncio.run(
            question_utility(
                completion_for({"questions": runner_up_questions}),
                answer,
                info,
                ElicitJsonParser("questions"),
            )
        )
        totals["elicit-ask/runner-up-top1-selection-accuracy"] += (
            top1_selection_accuracy(runner_up_questions, planted_questions)
        )
        public_profile_questions = public_profile_composition_questions(
            planted_questions,
            factions,
            question_count=QUESTION_COUNT,
        )
        totals["elicit-ask/public-profile-composition"] += asyncio.run(
            question_utility(
                completion_for({"questions": public_profile_questions}),
                answer,
                info,
                ElicitJsonParser("questions"),
            )
        )
        totals["elicit-ask/public-profile-top1-selection-accuracy"] += (
            top1_selection_accuracy(public_profile_questions, planted_questions)
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
        exact_top1_questions = exact_issue_exact_stance_questions(
            planted_questions,
            question_count=QUESTION_COUNT,
        )
        totals["elicit-ask/exact-issue-top1-selection-accuracy"] += (
            top1_selection_accuracy(exact_top1_questions, planted_questions)
        )
        totals["elicit-ask/public-source-replay"] += asyncio.run(
            question_utility(
                completion_for({"questions": replayed_questions}),
                answer,
                info,
                ElicitJsonParser("questions"),
            )
        )
    floors = {name: total / len(scenarios) for name, total in totals.items()}
    floors.update(compute_corpus_audits(scenarios))
    return floors


def compute_corpus_audits(
    scenarios: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    """Measure residual prompt-layout signal and top-K selection degeneracy."""

    if not scenarios:
        raise ValueError("corpus audit requires at least one scenario")
    feature_labels: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    combined_structure_rows: list[
        tuple[str, tuple[str, ...], tuple[float, ...], str]
    ] = []
    document_role_rows: list[tuple[str, tuple[str, ...], tuple[float, ...], str]] = []
    related_document_rows: list[
        tuple[str, tuple[str, ...], tuple[float, ...], str]
    ] = []
    issue_labels: list[str] = []
    document_role_labels: list[str] = []
    related_document_labels: list[str] = []
    marker_rows = 0
    marked_plants = 0
    total_plants = 0
    marked_distractors = 0
    total_distractors = 0
    team_marked_plants = 0
    team_marked_distractors = 0
    top_k_ties = 0
    boundary_gaps: list[float] = []
    normalized_boundary_margins: list[float] = []
    for scenario in scenarios:
        template_id = str(scenario["provenance"]["template_id"])
        documents = {
            str(document["doc_id"]): document for document in scenario["documents"]
        }
        visible_text = " ".join(
            str(document["text"]) for document in scenario["documents"]
        )
        marker_rows += any(marker in visible_text for marker in _OLD_TYPE_MARKERS)
        total_plants += len(scenario["planted_items"])
        team_marker = f"the {scenario['organization']['sector']} team"
        team_marked_plants += sum(
            team_marker in str(plant["anchor_quote"]).casefold()
            for plant in scenario["planted_items"]
        )
        marked_plants += sum(
            any(
                predicate in str(plant["anchor_quote"])
                for predicate in _SHARED_PROCEDURAL_PREDICATES
            )
            for plant in scenario["planted_items"]
        )
        total_distractors += len(scenario["distractors"])
        team_marked_distractors += sum(
            team_marker in str(distractor["anchor_quote"]).casefold()
            for distractor in scenario["distractors"]
        )
        marked_distractors += sum(
            any(
                predicate in str(distractor["anchor_quote"])
                for predicate in _SHARED_PROCEDURAL_PREDICATES
            )
            for distractor in scenario["distractors"]
        )
        utilities: list[float] = []
        for plant in scenario["planted_items"]:
            issue_type = str(plant["type"])
            document = documents[str(plant["doc_id"])]
            sentences = split_sentences(str(document["text"]))
            anchor_position = sentences.index(str(plant["anchor_quote"]))
            word_count = len(
                re.findall(r"[^\W_]+", str(document["text"]), flags=re.UNICODE)
            )
            title_root = re.sub(r"\s+\d+$", "", str(document["title"]))
            feature_labels["title"].append((template_id, title_root, issue_type))
            feature_labels["style"].append(
                (template_id, str(document["style"]), issue_type)
            )
            feature_labels["sentence-position"].append(
                (template_id, str(anchor_position), issue_type)
            )
            feature_labels["sentence-count"].append(
                (template_id, str(len(sentences)), issue_type)
            )
            feature_labels["document-length"].append(
                (template_id, str(word_count // 10), issue_type)
            )
            issue_labels.append(issue_type)
            combined_structure_rows.append(
                (
                    template_id,
                    (title_root,),
                    (
                        anchor_position / max(1, len(sentences) - 1),
                        float(len(sentences)),
                        float(word_count),
                    ),
                    issue_type,
                )
            )
            utilities.append(
                float(plant["decision_value"])
                * panel_disagreement(plant["target_stances"])
            )
        utilities.sort(reverse=True)
        if len(utilities) > QUESTION_COUNT:
            boundary_gap = utilities[QUESTION_COUNT - 1] - utilities[QUESTION_COUNT]
            boundary_gaps.append(boundary_gap)
            leading_utility = utilities[QUESTION_COUNT - 1]
            normalized_boundary_margins.append(
                boundary_gap / leading_utility if leading_utility > 0 else 0.0
            )
            top_k_ties += abs(boundary_gap) <= 1e-12

        if template_id in HELPER_HELDOUT_TEMPLATE_IDS:
            scenario_documents = list(scenario["documents"])
            if len(scenario_documents) != len(DOCUMENT_ROLE_LABELS):
                raise ValueError(
                    f"helper-built template {template_id} must render five documents"
                )
            primary_roles: dict[str, str] = {}
            related_doc_ids: set[str] = set()
            for plant in scenario["planted_items"]:
                doc_id = str(plant["doc_id"])
                role = f"{plant['type']}-primary"
                if doc_id in primary_roles:
                    raise ValueError(
                        f"helper-built template {template_id} reuses a primary document"
                    )
                primary_roles[doc_id] = role
                related = plant.get("related_evidence")
                if related is not None:
                    related_doc_ids.add(str(related["doc_id"]))
            if len(related_doc_ids) != 1 or related_doc_ids & set(primary_roles):
                raise ValueError(
                    f"helper-built template {template_id} must use one separate relationship document"
                )

            observed_roles: list[str] = []
            for document_index, document in enumerate(scenario_documents):
                doc_id = str(document["doc_id"])
                role = primary_roles.get(
                    doc_id,
                    (
                        "contradiction-related"
                        if doc_id in related_doc_ids
                        else "plant-free"
                    ),
                )
                observed_roles.append(role)
                sentences = split_sentences(str(document["text"]))
                word_count = len(
                    re.findall(r"[^\W_]+", str(document["text"]), flags=re.UNICODE)
                )
                title_root = re.sub(r"\s+\d+$", "", str(document["title"]))
                structural_features = (
                    template_id,
                    (title_root,),
                    (
                        document_index / max(1, len(scenario_documents) - 1),
                        float(len(sentences)),
                        float(word_count),
                    ),
                )
                document_role_rows.append((*structural_features, role))
                related_label = (
                    "related" if role == "contradiction-related" else "other"
                )
                related_document_rows.append((*structural_features, related_label))
                document_role_labels.append(role)
                related_document_labels.append(related_label)
            if Counter(observed_roles) != Counter(DOCUMENT_ROLE_LABELS):
                raise ValueError(
                    f"helper-built template {template_id} does not expose one document per authored role"
                )

    plant_marker_rate = marked_plants / total_plants if total_plants else 0.0
    distractor_marker_rate = (
        marked_distractors / total_distractors if total_distractors else 0.0
    )
    team_plant_rate = team_marked_plants / total_plants if total_plants else 0.0
    team_distractor_rate = (
        team_marked_distractors / total_distractors if total_distractors else 0.0
    )
    issue_class_min, issue_class_max = _class_proportion_range(issue_labels)
    role_class_min, role_class_max = _class_proportion_range(document_role_labels)
    related_class_min, related_class_max = _class_proportion_range(
        related_document_labels
    )
    audits = {
        "audit/find-old-fixed-marker-scenario-rate": marker_rows / len(scenarios),
        "audit/find-shared-predicate-plant-rate": plant_marker_rate,
        "audit/find-shared-predicate-distractor-rate": distractor_marker_rate,
        "audit/find-shared-predicate-exclusive-advantage": abs(
            plant_marker_rate - distractor_marker_rate
        ),
        "audit/find-sector-team-plant-rate": team_plant_rate,
        "audit/find-sector-team-distractor-rate": team_distractor_rate,
        "audit/find-sector-team-exclusive-advantage": abs(
            team_plant_rate - team_distractor_rate
        ),
        "audit/find-issue-class-min-proportion": issue_class_min,
        "audit/find-issue-class-max-proportion": issue_class_max,
        "audit/find-issue-balanced-chance": (
            1 / len(set(issue_labels)) if issue_labels else 0.0
        ),
        "audit/find-combined-title-length-position-loto-balanced-accuracy": (
            _leave_one_template_out_knn_balanced_accuracy(combined_structure_rows)
        ),
        "audit/find-document-role-class-min-proportion": role_class_min,
        "audit/find-document-role-class-max-proportion": role_class_max,
        "audit/find-document-role-balanced-chance": (
            1 / len(set(document_role_labels)) if document_role_labels else 0.0
        ),
        "audit/find-document-role-structure-loto-balanced-accuracy": (
            _leave_one_template_out_knn_balanced_accuracy(document_role_rows)
        ),
        "audit/find-related-document-positive-rate": (
            related_document_labels.count("related") / len(related_document_labels)
            if related_document_labels
            else 0.0
        ),
        "audit/find-related-document-class-min-proportion": related_class_min,
        "audit/find-related-document-class-max-proportion": related_class_max,
        "audit/find-related-document-balanced-chance": (
            1 / len(set(related_document_labels)) if related_document_labels else 0.0
        ),
        "audit/find-related-document-structure-loto-balanced-accuracy": (
            _leave_one_template_out_knn_balanced_accuracy(related_document_rows)
        ),
        "audit/elicit-ask-top-k-tie-rate": (
            top_k_ties / len(boundary_gaps) if boundary_gaps else 0.0
        ),
        "audit/elicit-ask-top-k-gap-mean": (
            sum(boundary_gaps) / len(boundary_gaps) if boundary_gaps else 0.0
        ),
        "audit/elicit-ask-top-k-gap-min": min(boundary_gaps, default=0.0),
        "audit/elicit-ask-top1-normalized-margin-mean": (
            sum(normalized_boundary_margins) / len(normalized_boundary_margins)
            if normalized_boundary_margins
            else 0.0
        ),
        "audit/elicit-ask-top1-normalized-margin-min": min(
            normalized_boundary_margins,
            default=0.0,
        ),
    }
    audits.update(
        {
            f"audit/find-{feature}-majority-label-accuracy": (
                _cross_template_majority_label_accuracy(pairs)
            )
            for feature, pairs in feature_labels.items()
        }
    )
    return audits


def _cross_template_majority_label_accuracy(
    rows: Sequence[tuple[str, str, str]],
) -> float:
    """Return leave-one-template-out accuracy for a discrete public feature.

    In-sample majority lookup overstates leakage whenever a feature value is
    rare. Holding out complete template families instead tests whether the
    shortcut transfers to unseen policy domains, which is the relevant split
    claim for these structural diagnostics.
    """

    if not rows:
        return 0.0
    correct = 0
    for heldout_template in sorted({template_id for template_id, _, _ in rows}):
        training = [row for row in rows if row[0] != heldout_template]
        testing = [row for row in rows if row[0] == heldout_template]
        by_feature: dict[str, Counter[str]] = defaultdict(Counter)
        global_counts: Counter[str] = Counter()
        for _, feature, label in training:
            by_feature[feature][label] += 1
            global_counts[label] += 1
        if not global_counts:
            continue
        fallback = min(
            global_counts,
            key=lambda label: (-global_counts[label], label),
        )
        for _, feature, label in testing:
            counts = by_feature.get(feature)
            predicted = (
                min(counts, key=lambda candidate: (-counts[candidate], candidate))
                if counts
                else fallback
            )
            correct += predicted == label
    return correct / len(rows)


def _leave_one_template_out_knn_balanced_accuracy(
    rows: Sequence[tuple[str, tuple[str, ...], tuple[float, ...], str]],
) -> float:
    """Score a frozen one-nearest-neighbor structural attack by template.

    The attacker uses only categorical title roots plus normalized public
    length/position features. Every variant of the held-out template family is
    excluded from fitting. Numeric ranges are learned only from the remaining
    templates, nearest-distance ties use a deterministic majority vote, and
    the final score is macro recall so class imbalance cannot inflate it.
    """

    if not rows:
        return 0.0
    categorical_width = len(rows[0][1])
    numeric_width = len(rows[0][2])
    if any(
        len(categorical) != categorical_width or len(numeric) != numeric_width
        for _, categorical, numeric, _ in rows
    ):
        raise ValueError("structural attack rows have inconsistent feature widths")

    predictions: list[tuple[str, str]] = []
    for heldout_template in sorted({template_id for template_id, *_ in rows}):
        training = [row for row in rows if row[0] != heldout_template]
        testing = [row for row in rows if row[0] == heldout_template]
        if not training:
            continue
        numeric_scales = []
        for feature_index in range(numeric_width):
            values = [row[2][feature_index] for row in training]
            numeric_scales.append(max(values) - min(values) or 1.0)

        for _, test_categorical, test_numeric, actual_label in testing:
            nearest_distance: float | None = None
            nearest_labels: Counter[str] = Counter()
            for _, train_categorical, train_numeric, train_label in training:
                distance = float(
                    sum(
                        left != right
                        for left, right in zip(
                            test_categorical, train_categorical, strict=True
                        )
                    )
                )
                distance += sum(
                    abs(left - right) / scale
                    for left, right, scale in zip(
                        test_numeric,
                        train_numeric,
                        numeric_scales,
                        strict=True,
                    )
                )
                if nearest_distance is None or distance < nearest_distance - 1e-12:
                    nearest_distance = distance
                    nearest_labels = Counter({train_label: 1})
                elif abs(distance - nearest_distance) <= 1e-12:
                    nearest_labels[train_label] += 1
            predicted_label = min(
                nearest_labels,
                key=lambda label: (-nearest_labels[label], label),
            )
            predictions.append((actual_label, predicted_label))

    labels = sorted({label for *_, label in rows})
    if not labels:
        return 0.0
    recalls = []
    for label in labels:
        label_predictions = [
            predicted for actual, predicted in predictions if actual == label
        ]
        recalls.append(
            sum(predicted == label for predicted in label_predictions)
            / len(label_predictions)
            if label_predictions
            else 0.0
        )
    return sum(recalls) / len(recalls)


def _class_proportion_range(labels: Sequence[str]) -> tuple[float, float]:
    """Return smallest/largest observed class share for explicit audit context."""

    if not labels:
        return 0.0, 0.0
    counts = Counter(labels)
    proportions = [count / len(labels) for count in counts.values()]
    return min(proportions), max(proportions)


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
        "find/removed-fixed-filler-marker": (
            "Prompt-observable",
            "find",
            "Removed fixed filler/type marker",
        ),
        "find/removed-fixed-filler-marker-localization-recall": (
            "Prompt-observable diagnostic",
            "find",
            "Removed fixed marker localization recall",
        ),
        "find/removed-fixed-filler-marker-type-accuracy": (
            "Prompt-observable diagnostic",
            "find",
            "Removed fixed marker conditional type accuracy",
        ),
        "find/removed-fixed-filler-marker-localization-component-oracle-f1": (
            "Localization component oracle",
            "find",
            "Removed fixed marker locator + exact non-localization components",
        ),
        "find/layout-position-length": (
            "Prompt-observable",
            "find",
            "Document length rank + fixed sentence position",
        ),
        "find/layout-position-length-localization-recall": (
            "Prompt-observable diagnostic",
            "find",
            "Layout/position/length localization recall",
        ),
        "find/layout-position-length-type-accuracy": (
            "Prompt-observable diagnostic",
            "find",
            "Layout/position/length conditional type accuracy",
        ),
        "find/layout-position-length-localization-component-oracle-f1": (
            "Localization component oracle",
            "find",
            "Layout/position/length locator + exact non-localization components",
        ),
        "find/longest-visible-sentences": (
            "Prompt-observable",
            "find",
            "Select the three longest visible sentences",
        ),
        "find/longest-visible-sentences-localization-recall": (
            "Prompt-observable diagnostic",
            "find",
            "Longest-sentence localization recall",
        ),
        "find/longest-visible-sentences-type-accuracy": (
            "Prompt-observable diagnostic",
            "find",
            "Longest-sentence conditional type accuracy",
        ),
        "find/longest-visible-sentences-localization-component-oracle-f1": (
            "Localization component oracle",
            "find",
            "Longest-sentence locator + exact non-localization components",
        ),
        "find/shared-predicate-exclusion": (
            "Source-aware diagnostic",
            "find",
            "Exclude current shared procedural predicates",
        ),
        "find/shared-predicate-exclusion-localization-recall": (
            "Source-aware diagnostic",
            "find",
            "Shared-predicate exclusion localization recall",
        ),
        "find/shared-predicate-exclusion-type-accuracy": (
            "Source-aware diagnostic",
            "find",
            "Shared-predicate exclusion conditional type accuracy",
        ),
        "find/sector-team-marker": (
            "Source-aware diagnostic",
            "find",
            "Select every sector-team marker",
        ),
        "find/sector-team-marker-localization-recall": (
            "Source-aware diagnostic",
            "find",
            "Sector-team marker localization recall",
        ),
        "find/sector-team-marker-localization-f1": (
            "Source-aware diagnostic",
            "find",
            "Sector-team marker localization F1",
        ),
        "find/sector-team-marker-localization-component-oracle-f1": (
            "Localization component oracle",
            "find",
            "Sector-team locator + exact non-localization components",
        ),
        "find/public-source-replay": (
            "Source-aware memorization ceiling",
            "find",
            "Regenerate exact answer key from candidate template and seed",
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
            "Candidate template detector + removed 0.4 principle-table parser",
        ),
        "elicit-ask/random-issue-exact-components": (
            "Selection diagnostic",
            "elicit-ask",
            "Exact uniform-random issue expectation + exact components",
        ),
        "elicit-ask/random-issue-top1-selection-accuracy": (
            "Selection diagnostic",
            "elicit-ask",
            "Exact uniform-random top-1 selection accuracy",
        ),
        "elicit-ask/runner-up-exact-components": (
            "Selection diagnostic",
            "elicit-ask",
            "Exact runner-up issue + exact per-issue components",
        ),
        "elicit-ask/runner-up-top1-selection-accuracy": (
            "Selection diagnostic",
            "elicit-ask",
            "Runner-up top-1 selection accuracy",
        ),
        "elicit-ask/public-profile-composition": (
            "Component oracle",
            "elicit-ask",
            "Exact grounding + public-profile stance/rank composition",
        ),
        "elicit-ask/public-profile-top1-selection-accuracy": (
            "Component oracle diagnostic",
            "elicit-ask",
            "Public-profile composition top-1 selection accuracy",
        ),
        "elicit-ask/exact-issue-random-stance": (
            "Component oracle",
            "elicit-ask",
            "Exact top-1 issue + random stances",
        ),
        "elicit-ask/exact-issue-exact-stance": (
            "Component oracle",
            "elicit-ask",
            "Exact top-1 issue + exact stances (component ceiling)",
        ),
        "elicit-ask/exact-issue-top1-selection-accuracy": (
            "Component oracle diagnostic",
            "elicit-ask",
            "Exact top-1 selection accuracy",
        ),
        "elicit-ask/public-source-replay": (
            "Source-aware memorization ceiling",
            "elicit-ask",
            "Regenerate exact top-1 answer from candidate template and seed",
        ),
        "audit/find-old-fixed-marker-scenario-rate": (
            "Corpus diagnostic",
            "audit",
            "Rows containing a removed fixed type marker",
        ),
        "audit/find-shared-predicate-plant-rate": (
            "Corpus diagnostic",
            "audit",
            "Planted anchors carrying shared procedural predicates",
        ),
        "audit/find-shared-predicate-distractor-rate": (
            "Corpus diagnostic",
            "audit",
            "Distractors carrying shared procedural predicates",
        ),
        "audit/find-shared-predicate-exclusive-advantage": (
            "Corpus diagnostic",
            "audit",
            "Absolute planted/distractor shared-predicate rate gap",
        ),
        "audit/find-sector-team-plant-rate": (
            "Corpus diagnostic",
            "audit",
            "Planted anchors carrying the sector-team marker",
        ),
        "audit/find-sector-team-distractor-rate": (
            "Corpus diagnostic",
            "audit",
            "Distractors carrying the sector-team marker",
        ),
        "audit/find-sector-team-exclusive-advantage": (
            "Corpus diagnostic",
            "audit",
            "Absolute planted/distractor sector-team rate gap",
        ),
        "audit/find-issue-class-min-proportion": (
            "Corpus diagnostic",
            "audit",
            "Minimum planted issue-class proportion",
        ),
        "audit/find-issue-class-max-proportion": (
            "Corpus diagnostic",
            "audit",
            "Maximum planted issue-class proportion",
        ),
        "audit/find-issue-balanced-chance": (
            "Corpus diagnostic",
            "audit",
            "Issue-type balanced-accuracy chance reference",
        ),
        "audit/find-combined-title-length-position-loto-balanced-accuracy": (
            "Cross-template structural diagnostic",
            "audit",
            "Combined title/length/anchor-position LOTO balanced accuracy",
        ),
        "audit/find-document-role-class-min-proportion": (
            "Corpus diagnostic",
            "audit",
            "Minimum helper document-role class proportion",
        ),
        "audit/find-document-role-class-max-proportion": (
            "Corpus diagnostic",
            "audit",
            "Maximum helper document-role class proportion",
        ),
        "audit/find-document-role-balanced-chance": (
            "Corpus diagnostic",
            "audit",
            "Helper document-role balanced-accuracy chance reference",
        ),
        "audit/find-document-role-structure-loto-balanced-accuracy": (
            "Cross-template structural diagnostic",
            "audit",
            "Helper document-role LOTO balanced accuracy",
        ),
        "audit/find-related-document-positive-rate": (
            "Corpus diagnostic",
            "audit",
            "Helper contradiction-related document prevalence",
        ),
        "audit/find-related-document-class-min-proportion": (
            "Corpus diagnostic",
            "audit",
            "Minimum related-document binary class proportion",
        ),
        "audit/find-related-document-class-max-proportion": (
            "Corpus diagnostic",
            "audit",
            "Maximum related-document binary class proportion",
        ),
        "audit/find-related-document-balanced-chance": (
            "Corpus diagnostic",
            "audit",
            "Related-document balanced-accuracy chance reference",
        ),
        "audit/find-related-document-structure-loto-balanced-accuracy": (
            "Cross-template structural diagnostic",
            "audit",
            "Related-document LOTO balanced accuracy",
        ),
        "audit/find-title-majority-label-accuracy": (
            "Corpus diagnostic",
            "audit",
            "Title-root leave-one-template-out issue-label accuracy",
        ),
        "audit/find-style-majority-label-accuracy": (
            "Corpus diagnostic",
            "audit",
            "Style leave-one-template-out issue-label accuracy",
        ),
        "audit/find-sentence-position-majority-label-accuracy": (
            "Corpus diagnostic",
            "audit",
            "Anchor-position leave-one-template-out issue-label accuracy",
        ),
        "audit/find-sentence-count-majority-label-accuracy": (
            "Corpus diagnostic",
            "audit",
            "Sentence-count leave-one-template-out issue-label accuracy",
        ),
        "audit/find-document-length-majority-label-accuracy": (
            "Corpus diagnostic",
            "audit",
            "Document-length leave-one-template-out issue-label accuracy",
        ),
        "audit/elicit-ask-top-k-tie-rate": (
            "Corpus diagnostic",
            "audit",
            "Top-1 boundary tie rate",
        ),
        "audit/elicit-ask-top-k-gap-mean": (
            "Corpus diagnostic",
            "audit",
            "Mean top-1 boundary utility gap",
        ),
        "audit/elicit-ask-top-k-gap-min": (
            "Corpus diagnostic",
            "audit",
            "Minimum top-1 boundary utility gap",
        ),
        "audit/elicit-ask-top1-normalized-margin-mean": (
            "Corpus diagnostic",
            "audit",
            "Mean normalized top-1 utility margin",
        ),
        "audit/elicit-ask-top1-normalized-margin-min": (
            "Corpus diagnostic",
            "audit",
            "Minimum normalized top-1 utility margin",
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
