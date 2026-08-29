"""Seeded, offline scenario generation."""

from __future__ import annotations

import copy
import hashlib
import json
import random
import re
from collections.abc import Callable, Mapping
from typing import Any

from commonground_scenarios.templates import DomainTemplate, get_template
from commonground_scenarios.validation import (
    PASS_THRESHOLD,
    canonical_date,
    scenario_id_for,
    validate_scenario,
)

DEFAULT_GENERATED_AT = "2026-08-15"
SEMANTIC_SCOPES: tuple[str | None, ...] = (
    None,
    "for after-hours requests",
    "when the primary owner is unavailable",
    "for cases spanning two service regions",
    "when a documented safety risk is active",
    "for first-time participants",
    "during peak-demand periods",
    "when an accessibility accommodation is active",
    "for requests received through an offline channel",
    "while a formal appeal is pending",
    "during a declared service interruption",
    "for requests involving a delegated representative",
    "when two responsible teams disagree",
    "for decisions made outside normal business hours",
    "when the affected person cannot provide the usual record",
    "during a temporary capacity shortage",
    "for cases involving an imminent deadline",
    "when the designated reviewer has a conflict of interest",
    "for decisions that affect more than one jurisdiction",
    "when the standard communication channel is inaccessible",
    "during a documented emergency exception",
    "for first-time appeals of an automated decision",
    "when required evidence arrives after the normal cutoff",
    "for cases transferred between operating teams",
    "when a safety accommodation conflicts with the default process",
)


def generate_scenario(
    seed: int,
    domain_template: DomainTemplate | str,
    *,
    generated_at: str = DEFAULT_GENERATED_AT,
    prose_polisher: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """Generate one canonical planted scenario without network or model calls.

    ``generated_at`` is explicit and never read from the wall clock. An
    operator may inject ``prose_polisher`` at generation time; it is absent and
    therefore off by default.
    """

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    template = (
        get_template(domain_template)
        if isinstance(domain_template, str)
        else domain_template
    )
    canonical_date(generated_at)
    rng = random.Random(seed)

    documents = copy.deepcopy(list(template.documents))
    planted_items = copy.deepcopy(list(template.planted_items))
    distractors = copy.deepcopy(list(template.distractors))
    semantic_scope = SEMANTIC_SCOPES[seed % len(SEMANTIC_SCOPES)]
    if semantic_scope is not None:
        _apply_semantic_scope(documents, planted_items, semantic_scope)
    generator_family = (
        "heldout-template-layout-profile-v3"
        if template.template_set == "heldout"
        else "train-template-layout-profile-v3"
    )
    _randomize_visible_structure(
        rng,
        documents,
        planted_items,
        distractors,
        template_set=template.template_set,
    )
    if prose_polisher is not None:
        for document in documents:
            polished = prose_polisher(document["text"])
            if not isinstance(polished, str):
                raise TypeError("prose_polisher must return text")
            document["text"] = polished

    for planted in planted_items:
        planted["decision_terms"] = _decision_terms(str(planted["canonical_question"]))
        # Final value is computed from simulated faction answers after stance
        # generation. No policy keyword table participates in issue ranking.
        planted["decision_value"] = 1.0
        planted["related_evidence"] = _related_evidence(documents, planted)
    factions = copy.deepcopy(list(template.factions))
    _randomize_faction_structure(rng, factions, planted_items)
    for planted in planted_items:
        dimension = planted["target_dimension"]
        planted["target_stances"] = {
            faction["faction_id"]: _stance_for(float(faction["priors"][dimension]))
            for faction in factions
        }
        planted["decision_value"] = _answer_conditioned_value(planted["target_stances"])
    _add_visible_faction_principles(rng, factions, planted_items)

    scenario = {
        "scenario_id": scenario_id_for(template.template_id, seed),
        "organization": {
            "name": rng.choice(template.organization_names),
            "sector": template.sector,
            "fictional": True,
        },
        "factions": factions,
        "documents": documents,
        "planted_items": planted_items,
        "distractors": distractors,
        "persona_panel": {
            "vote_rule": "dimension-threshold-v1",
            "pass_threshold": PASS_THRESHOLD,
            "faction_ids": [faction["faction_id"] for faction in factions],
        },
        "human_feedback": None,
        "provenance": {
            "seed": seed,
            "template_id": template.template_id,
            "template_set": template.template_set,
            "generated_at": generated_at,
            "synthetic": True,
            "generation_mode": "operator-polished"
            if prose_polisher is not None
            else "template",
            "generator_family": generator_family,
        },
    }
    validate_scenario(scenario)
    return scenario


def scenario_to_bytes(scenario: dict[str, Any]) -> bytes:
    """Validate and serialize a scenario as canonical newline-terminated JSON."""

    validate_scenario(scenario)
    return (
        json.dumps(
            scenario,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _stance_for(prior: float) -> str:
    if prior >= PASS_THRESHOLD:
        return "agree"
    if prior <= -PASS_THRESHOLD:
        return "disagree"
    return "pass"


def _apply_semantic_scope(
    documents: list[dict[str, str]],
    planted_items: list[dict[str, Any]],
    scope: str,
) -> None:
    """Author a substantive, deterministic scope variant into every planted task."""

    documents_by_id = {document["doc_id"]: document for document in documents}
    for plant in planted_items:
        old_anchor = plant["anchor_quote"]
        new_anchor = f"{old_anchor.removesuffix('.')} {scope}."
        document = documents_by_id[plant["doc_id"]]
        document["text"] = document["text"].replace(old_anchor, new_anchor, 1)
        plant["anchor_quote"] = new_anchor
        plant["canonical_question"] = (
            f"{plant['canonical_question'].removesuffix('?')} {scope}?"
        )
        plant["canonical_question_aliases"] = [
            f"{alias.removesuffix('?')} {scope}?"
            for alias in plant["canonical_question_aliases"]
        ]


_QUESTION_STOPWORDS = frozenset(
    {
        "after",
        "allowed",
        "before",
        "could",
        "decide",
        "does",
        "during",
        "each",
        "from",
        "have",
        "instead",
        "into",
        "must",
        "should",
        "than",
        "that",
        "their",
        "them",
        "this",
        "under",
        "what",
        "when",
        "which",
        "with",
        "without",
        "would",
    }
)
_SCOPE_TERMS = frozenset(
    token
    for scope in SEMANTIC_SCOPES
    if scope is not None
    for token in re.findall(r"[^\W_]+", scope.casefold())
    if len(token) >= 4
)


def _randomize_visible_structure(
    rng: random.Random,
    documents: list[dict[str, str]],
    planted_items: list[dict[str, Any]],
    distractors: list[dict[str, str]],
    *,
    template_set: str,
) -> None:
    """Remove stable ID, order, style, and anchor-position codebooks."""

    old_to_new: dict[str, str] = {}
    used_ids: set[str] = set()
    styles = ["brief", "procedure", "guide", "record", "notice", "appendix"]
    title_roots = [
        "Operations note",
        "Service reference",
        "Implementation record",
        "Review packet",
        "Field guidance",
        "Decision memo",
    ]
    for document_index, document in enumerate(documents):
        if template_set == "heldout":
            digest = hashlib.sha256(
                f"{rng.random()}:{document['doc_id']}".encode()
            ).hexdigest()[:8]
            new_id = f"doc-{digest}"
        else:
            new_id = f"train-doc-{document_index + 1}-{rng.randrange(100, 1000)}"
        while new_id in used_ids:
            new_id = f"doc-{rng.randrange(16**8):08x}"
        used_ids.add(new_id)
        old_to_new[document["doc_id"]] = new_id
        document["doc_id"] = new_id
        document["title"] = f"{rng.choice(title_roots)} {rng.randrange(100, 1000)}"
        document["style"] = rng.choice(styles)
        sentences = re.split(r"(?<=[.!?])\s+", document["text"])
        rng.shuffle(sentences)
        document["text"] = " ".join(sentences)
    for plant in planted_items:
        plant["doc_id"] = old_to_new[plant["doc_id"]]
        plant["plant_id"] = f"issue-{rng.randrange(16**8):08x}"
    for distractor in distractors:
        distractor["doc_id"] = old_to_new[distractor["doc_id"]]
    rng.shuffle(documents)
    rng.shuffle(planted_items)


def _randomize_faction_structure(
    rng: random.Random,
    factions: list[dict[str, Any]],
    planted_items: list[dict[str, Any]],
) -> None:
    """Use opaque faction IDs and varying order/count without changing summaries."""

    dimensions = [str(plant["target_dimension"]) for plant in planted_items]
    extra_factions = (
        (
            "Implementation council",
            "Balances operational continuity with reviewable safeguards.",
        ),
        (
            "Access delegates",
            "Prioritizes workable exceptions for people facing access barriers.",
        ),
    )
    for name, summary in extra_factions[: rng.randrange(3)]:
        priors = {}
        for dimension in dimensions:
            magnitude = rng.choice((0.15, 0.45, 0.75))
            priors[dimension] = magnitude if rng.random() < 0.5 else -magnitude
        factions.append(
            {
                "faction_id": "placeholder",
                "name": name,
                "summary": summary,
                "priors": priors,
            }
        )
    rng.shuffle(factions)
    for faction in factions:
        faction["faction_id"] = f"group-{rng.randrange(16**8):08x}"


_PRINCIPLE_EXAMPLES: dict[tuple[str, str], tuple[str, ...]] = {
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


def _add_visible_faction_principles(
    rng: random.Random,
    factions: list[dict[str, Any]],
    planted_items: list[dict[str, Any]],
) -> None:
    """Expose indirect policy principles without issue signatures or stance labels."""

    for faction in factions:
        clauses: list[str] = []
        for plant in planted_items:
            stance = str(plant["target_stances"][faction["faction_id"]])
            clauses.append(
                rng.choice(_PRINCIPLE_EXAMPLES[(str(plant["type"]), stance)])
            )
        rng.shuffle(clauses)
        faction["summary"] = f"{faction['summary']} {' '.join(clauses)}"


def _decision_terms(question: str) -> list[str]:
    """Extract a deterministic multi-token latent decision signature."""

    tokens = re.findall(r"[^\W_]+", question.casefold())
    terms = [
        token
        for token in tokens
        if len(token) >= 4 and token not in _QUESTION_STOPWORDS
    ]
    return list(dict.fromkeys(terms))[:8]


def _answer_conditioned_value(target_stances: Mapping[str, str]) -> float:
    """Value questions by the share of factions with an actionable answer.

    The companion disagreement term rewards a split panel; this term rewards
    questions whose simulated answer would update more than a small minority.
    It replaces the 0.3 policy-keyword lookup with an answer-conditioned signal.
    """

    if not target_stances:
        return 0.25
    answer_coverage = sum(
        stance in {"agree", "disagree"} for stance in target_stances.values()
    ) / len(target_stances)
    return max(0.25, answer_coverage)


def _related_evidence(
    documents: list[dict[str, str]],
    plant: dict[str, Any],
) -> dict[str, str] | None:
    """Identify the second visible rule for a planted contradiction."""

    if plant["type"] != "contradiction":
        return None
    anchor_terms = set(_decision_terms(str(plant["anchor_quote"]))) - _SCOPE_TERMS
    question_terms = (
        set(_decision_terms(str(plant["canonical_question"]))) - _SCOPE_TERMS
    )
    candidates: list[tuple[int, int, str, str]] = []
    for document in documents:
        if document["doc_id"] == plant["doc_id"]:
            continue
        for sentence in re.split(r"(?<=[.!?])\s+", document["text"]):
            sentence_terms = set(_decision_terms(sentence))
            candidates.append(
                (
                    len(anchor_terms & sentence_terms),
                    len(question_terms & sentence_terms),
                    document["doc_id"],
                    sentence,
                )
            )
    if not candidates:
        return None
    _, _, doc_id, quote = max(candidates, key=lambda item: (item[0], item[1], item[3]))
    return {"doc_id": doc_id, "quote": quote}
