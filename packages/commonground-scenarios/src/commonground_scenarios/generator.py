"""Seeded, offline scenario generation."""

from __future__ import annotations

import copy
import hashlib
import json
import random
import re
from collections.abc import Callable, Mapping
from typing import Any

from commonground_scenarios.templates import (
    VALUE_DIMENSIONS,
    DomainTemplate,
    get_template,
)
from commonground_scenarios.validation import (
    PASS_THRESHOLD,
    canonical_date,
    scenario_id_for,
    validate_scenario,
)

DEFAULT_GENERATED_AT = "2026-08-30"
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
        "heldout-template-layout-profile-v4"
        if template.template_set == "heldout"
        else "train-template-layout-profile-v4"
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

    factions = copy.deepcopy(list(template.factions))
    _randomize_faction_structure(rng, factions)
    for planted in planted_items:
        planted["related_evidence"] = _authored_related_evidence(documents, planted)
        planted.pop("related_plant_doc_id", None)
        planted.pop("related_anchor_quote", None)
        planted["alternative_stances"] = {
            faction["faction_id"]: _stance_for(
                _composed_preference(
                    faction["values"],
                    planted["value_weights"],
                )
            )
            for faction in factions
        }
        planted["target_stances"] = orient_stances(
            planted["alternative_stances"],
            yes_choice=str(planted["canonical_yes_choice"]),
        )
        planted["decision_value"] = _answer_conditioned_value(planted["target_stances"])
    _add_visible_faction_values(rng, factions)

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
            "vote_rule": "value-composition-v1",
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


def _composed_preference(
    values: Mapping[str, float], weights: Mapping[str, float]
) -> float:
    """Compose reusable faction values with one issue's alternative trade-off."""

    scale = sum(abs(float(weights[dimension])) for dimension in VALUE_DIMENSIONS)
    if scale == 0:
        raise ValueError("value_weights cannot all be zero")
    return (
        sum(
            float(values[dimension]) * float(weights[dimension])
            for dimension in VALUE_DIMENSIONS
        )
        / scale
    )


def orient_stances(
    alternative_stances: Mapping[str, str], *, yes_choice: str
) -> dict[str, str]:
    """Orient alternative preferences to the yes side of a specific question."""

    if yes_choice == "alternative":
        return dict(alternative_stances)
    if yes_choice != "anchor":
        raise ValueError("yes_choice must be anchor or alternative")
    inverse = {"agree": "disagree", "disagree": "agree", "pass": "pass"}
    return {
        faction_id: inverse[stance]
        for faction_id, stance in alternative_stances.items()
    }


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
        if "related_plant_doc_id" in plant:
            plant["related_plant_doc_id"] = old_to_new[plant["related_plant_doc_id"]]
        plant["plant_id"] = f"issue-{rng.randrange(16**8):08x}"
    for distractor in distractors:
        distractor["doc_id"] = old_to_new[distractor["doc_id"]]
    rng.shuffle(documents)
    rng.shuffle(planted_items)


def _randomize_faction_structure(
    rng: random.Random,
    factions: list[dict[str, Any]],
) -> None:
    """Use opaque faction IDs and varying count without issue-conditioned values."""

    extra_factions = (
        (
            "Implementation council",
            "Balances operational continuity with reviewable safeguards.",
            (0.2, 0.4, 0.8, 0.6, 0.5),
        ),
        (
            "Access delegates",
            "Prioritizes workable exceptions for people facing access barriers.",
            (0.9, 0.6, 0.5, -0.1, 0.2),
        ),
    )
    for name, summary, vector in extra_factions[: rng.randrange(3)]:
        factions.append(
            {
                "faction_id": "placeholder",
                "name": name,
                "summary": summary,
                "values": dict(zip(VALUE_DIMENSIONS, vector, strict=True)),
            }
        )
    rng.shuffle(factions)
    for faction in factions:
        faction["faction_id"] = f"group-{rng.randrange(16**8):08x}"


_VALUE_PRINCIPLES: dict[tuple[str, str], tuple[str, ...]] = {
    ("access", "positive"): (
        "Prioritizes broad access when procedures exclude affected people.",
        "Values making services reachable through more than one channel.",
    ),
    ("access", "negative"): (
        "Accepts narrower access when expansion would weaken the stated process.",
        "Treats broad access as secondary to maintaining bounded eligibility.",
    ),
    ("access", "balanced"): (
        "Balances broader access against the limits of the stated process.",
        "Has no fixed preference for expanding or narrowing access.",
    ),
    ("adaptability", "positive"): (
        "Values discretion to adapt a rule to the facts of a case.",
        "Favors practical flexibility when circumstances differ.",
    ),
    ("adaptability", "negative"): (
        "Values consistent application over case-specific flexibility.",
        "Resists discretionary departures from a common process.",
    ),
    ("adaptability", "balanced"): (
        "Balances consistent rules with case-specific flexibility.",
        "Has no fixed preference between discretion and uniform treatment.",
    ),
    ("continuity", "positive"): (
        "Prioritizes continuity when the normal workflow is interrupted.",
        "Values keeping essential operations moving through disruption.",
    ),
    ("continuity", "negative"): (
        "Accepts interruption rather than preserve service at any cost.",
        "Treats continuity as secondary when the normal workflow cannot be followed.",
    ),
    ("continuity", "balanced"): (
        "Balances operational continuity against reasons to pause.",
        "Has no fixed preference for continuity over interruption.",
    ),
    ("oversight", "positive"): (
        "Prioritizes explicit approval and reviewable decision authority.",
        "Values controls that make exceptions auditable.",
    ),
    ("oversight", "negative"): (
        "Accepts lighter approval controls when they obstruct timely action.",
        "Treats formal oversight as secondary to direct resolution.",
    ),
    ("oversight", "balanced"): (
        "Balances formal oversight with direct operating authority.",
        "Has no fixed preference for more or less approval control.",
    ),
    ("safety", "positive"): (
        "Prioritizes conservative safeguards when consequences are uncertain.",
        "Values reducing safety exposure even when action becomes slower.",
    ),
    ("safety", "negative"): (
        "Accepts bounded safety trade-offs to avoid unnecessary restriction.",
        "Treats precaution as secondary when risks are limited and reviewable.",
    ),
    ("safety", "balanced"): (
        "Balances precaution against the costs of unnecessary restriction.",
        "Has no fixed preference for more or less precaution.",
    ),
}


def _add_visible_faction_values(
    rng: random.Random,
    factions: list[dict[str, Any]],
) -> None:
    """Render each reusable value once, independently of planted issues."""

    for faction in factions:
        clauses: list[str] = []
        for dimension in VALUE_DIMENSIONS:
            value = float(faction["values"][dimension])
            direction = (
                "positive"
                if value >= PASS_THRESHOLD
                else "negative"
                if value <= -PASS_THRESHOLD
                else "balanced"
            )
            clauses.append(rng.choice(_VALUE_PRINCIPLES[(dimension, direction)]))
        rng.shuffle(clauses)
        faction["summary"] = f"{faction['summary']} {' '.join(clauses)}"


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


def _authored_related_evidence(
    documents: list[dict[str, str]],
    plant: dict[str, Any],
) -> dict[str, str] | None:
    """Resolve an explicitly authored second rule for a contradiction.

    Contradiction relationships are semantic authoring data. Inferring them
    from token overlap can silently select a nearby ambiguity or distractor,
    so generation only remaps and verifies the authored document/quote pair.
    """

    if plant["type"] != "contradiction":
        if "related_plant_doc_id" in plant or "related_anchor_quote" in plant:
            raise ValueError("only contradictions may author related evidence")
        return None
    related_doc_id = plant.get("related_plant_doc_id")
    related_quote = plant.get("related_anchor_quote")
    if not isinstance(related_doc_id, str) or not isinstance(related_quote, str):
        raise ValueError("contradictions require authored related evidence")
    if related_doc_id == plant["doc_id"]:
        raise ValueError("contradiction evidence must use another document")
    related_document = next(
        (document for document in documents if document["doc_id"] == related_doc_id),
        None,
    )
    if related_document is None or related_quote not in related_document["text"]:
        raise ValueError("authored contradiction evidence is absent from documents")
    return {"doc_id": related_doc_id, "quote": related_quote}
