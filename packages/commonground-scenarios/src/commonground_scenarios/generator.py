"""Seeded, offline scenario generation."""

from __future__ import annotations

import copy
import hashlib
import json
import random
import re
from collections.abc import Callable, Mapping
from typing import Any

from commonground_scenarios.decision_frames_additional import decision_frame_for
from commonground_scenarios.decision_frames_base import AUTHORED_BASE_DECISION_FRAMES
from commonground_scenarios.templates import (
    VALUE_DIMENSIONS,
    DomainTemplate,
    get_template,
)
from commonground_scenarios.validation import (
    PASS_THRESHOLD,
    canonical_date,
    preference_tradeoff_value,
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

_SHARED_PROCEDURAL_CONTEXTS: tuple[str, ...] = (
    "During a scheduled administrative review for {sector}",
    "Before an ordinary {sector} file is closed",
    "After the operating unit accepts a routine {sector} handoff",
    "When a replacement {sector} document becomes current",
    "As part of monthly {sector} records maintenance",
    "Once the operating unit receives an ordinary {sector} acknowledgement",
    "During the normal {sector} archive cycle",
    "Before a routine {sector} status update is filed",
    "After the receiving desk confirms a {sector} transfer",
    "When the shared {sector} index is refreshed",
    "During an ordinary {sector} edition check",
    "Before the operating unit's next {sector} maintenance review begins",
)
_SHARED_PROCEDURAL_ACTORS: tuple[str, ...] = (
    "the assigned records coordinator",
    "the receiving desk",
    "the document custodian",
    "the service administrator",
    "the review secretary",
    "the operations clerk",
    "the designated liaison",
    "the quality coordinator",
    "the intake team",
    "the current file owner",
)
_SHARED_PROCEDURAL_PREDICATES: tuple[str, ...] = (
    "checks the current document owner against the shared register",
    "records the receiving team and transfer date in the operating ledger",
    "links each routine attachment to its source case in the common index",
    "confirms that the active edition matches the reference catalog",
    "notes ordinary contact updates in the maintenance log",
    "preserves the original filing date in the archive catalog",
    "copies the source reference into the receiving team's case record",
    "reviews the scheduled maintenance date against the current calendar",
    "places the completed acknowledgement beside the related case entry",
    "marks superseded material as read-only in the archive index",
    "verifies the responsible unit in the current service directory",
    "records the edition used for routine processing in the case history",
    "files the routine acknowledgement with its originating record",
    "retains the source identifier when an administrative copy is replaced",
    "matches the completed handoff to the corresponding index entry",
    "updates the responsible contact in the ordinary service register",
)
_SHARED_PROCEDURAL_PURPOSES: tuple[str, ...] = (
    "so the next reviewer can trace the record",
    "to keep the administrative history complete",
    "without changing the underlying case decision",
    "so current and superseded copies remain distinct",
    "for the next scheduled maintenance check",
    "to preserve the source record's filing history",
    "so routine follow-up stays attached to the same case",
    "for ordinary internal recordkeeping",
    "before the next archive review",
    "so the receiving team can verify the handoff",
)
_COMPOSED_DISTRACTOR_REASON = (
    "Seeded compositional administrative context with no ambiguity, conflict, "
    "or uncovered decision."
)
ACTOR_SUPPORT_REASON = (
    "Seeded actor-support context required by an authored candidate decision."
)
_DECISION_ALIAS_FIELDS = (
    "actor",
    "action",
    "condition",
    "anchor_outcome",
    "alternative_outcome",
)
# These aliases are authored from phrases that occur in the visible scenario,
# not learned from model answers.  Most frames need only their canonical form;
# an override records a genuine source-language alternative when the authored
# reference uses a narrower role or a more abstract action label.
_DECISION_ALIAS_OVERRIDES: Mapping[tuple[str, str], Mapping[str, tuple[str, ...]]] = {
    ("community-clinic-scheduling", "scope-threshold"): {
        "actor": ("community health scheduling team",),
        "action": (
            "decide which conditions make a patient eligible for an early appointment",
            "offer priority patients an early appointment",
        ),
        "condition": (
            "conditions make a patient eligible for an early appointment",
            "a patient qualifies as priority",
        ),
    },
    ("customer-support-handbook", "credit-threshold"): {
        "actor": ("customer support",),
        "alternative_outcome": ("set a goodwill credit",),
    },
    ("customer-support-handbook", "refund-window-conflict"): {
        "actor": ("customer support",),
    },
    ("customer-support-handbook", "owner-unavailable-gap"): {
        "actor": ("customer support",),
        "alternative_outcome": ("recover account access",),
    },
    ("learning-platform-operations", "substantive-feedback"): {
        "action": ("provide substantive feedback",),
        "alternative_outcome": ("define substantive feedback",),
    },
    ("learning-platform-operations", "live-access-gap"): {
        "alternative_outcome": ("grant an accessibility request",),
    },
    ("civic-assistant-guidance", "no-address-gap"): {
        "actor": ("case workers",),
    },
    ("creator-marketplace-playbook", "dashboard-access-gap"): {
        "actor": ("creators",),
    },
    ("creator-marketplace-playbook", "timely-response"): {
        "alternative_outcome": ("set a timely response",),
    },
    ("library-digitization-rules", "scope-threshold"): {
        "action": (
            "apply appropriate access restrictions to culturally sensitive materials",
        ),
        "alternative_outcome": ("set access restrictions for sensitive materials",),
    },
    ("library-digitization-rules", "uncovered-exception"): {
        "alternative_outcome": ("accept a takedown request",),
    },
    ("cooperative-housing-maintenance", "uncovered-exception"): {
        "alternative_outcome": ("provide repair notices another way",),
    },
    ("cooperative-housing-maintenance", "scope-threshold"): {
        "actor": ("the cooperative housing maintenance team",),
        "action": ("handle urgent repairs promptly",),
    },
    ("food-bank-distribution", "scope-threshold"): {
        "action": ("give additional staple items to households with high need",),
        "alternative_outcome": ("assign additional staples",),
    },
    ("renewable-microgrid-operations", "scope-threshold"): {
        "action": ("maintain an adequate battery reserve overnight",),
    },
    ("museum-loan-handling", "scope-threshold"): {
        "action": (
            "send significantly damaged loans to immediate conservation review",
        ),
    },
    ("museum-loan-handling", "uncovered-exception"): {
        "action": ("return the borrowed object",),
    },
    ("open-source-grant-program", "scope-threshold"): {
        "action": (
            "give priority review to projects with substantial community benefit",
        ),
    },
    ("agricultural-water-allocation", "scope-threshold"): {
        "action": ("reduce irrigation allocations",),
    },
    ("disaster-shelter-intake", "scope-threshold"): {
        "action": ("prioritize vulnerable residents for available shelter beds",),
    },
    ("translation-quality-service", "scope-threshold"): {
        "action": ("send sensitive content to a senior language review",),
        "alternative_outcome": ("require senior language review",),
    },
    ("research-computing-queue", "scope-threshold"): {
        "action": ("expedite queue placement for high-impact computing runs",),
    },
    ("regional-archives-access", "scope-threshold"): {
        "action": ("apply suitable reading-room restrictions to sensitive files",),
    },
    ("rural-broadband-outages", "scope-threshold"): {
        "action": ("accelerate field response to widespread outages",),
    },
    ("water-quality-laboratory", "scope-threshold"): {
        "alternative_outcome": ("require additional laboratory review",),
    },
}


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
    for planted in planted_items:
        planted["decision"] = _authored_decision_frame(
            template.template_id,
            str(planted["plant_id"]),
        )
    semantic_scope = SEMANTIC_SCOPES[seed % len(SEMANTIC_SCOPES)]
    if semantic_scope is not None:
        _apply_semantic_scope(documents, planted_items, semantic_scope)
    for planted in planted_items:
        planted["decision_aliases"] = _authored_decision_aliases(
            template,
            str(planted["plant_id"]),
            planted["decision"],
            semantic_scope=semantic_scope,
        )
    _blend_authored_and_procedural_language(
        rng,
        documents,
        planted_items,
        distractors,
        sector=template.sector,
    )
    if template.balance_type_neutral_distractors:
        _balance_type_neutral_distractors(
            rng,
            documents,
            distractors,
            sector=template.sector,
        )
    generator_family = (
        "heldout-template-layout-profile-v6"
        if template.template_set == "heldout"
        else "train-template-layout-profile-v6"
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
        planted["decision_value"] = preference_tradeoff_value(
            factions,
            planted["value_weights"],
        )
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
            "vote_rule": "value-composition-v2",
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


def _authored_decision_frame(template_id: str, plant_id: str) -> dict[str, str]:
    """Copy the complete hand-authored frame for one template plant."""

    base = AUTHORED_BASE_DECISION_FRAMES.get((template_id, plant_id))
    if base is not None:
        return dict(base)
    return dict(decision_frame_for(template_id, plant_id))


def _authored_decision_aliases(
    template: DomainTemplate,
    plant_id: str,
    decision: Mapping[str, str],
    *,
    semantic_scope: str | None,
) -> dict[str, list[str]]:
    """Return complete, per-slot accepted concepts for one authored frame.

    Every slot includes its canonical authored form.  Narrow overrides add
    source-observable alternatives without allowing words to migrate between
    actor, action, condition, or outcome slots.
    """

    overrides = _DECISION_ALIAS_OVERRIDES.get((template.template_id, plant_id), {})
    actor_aliases = list(overrides.get("actor", ()))

    aliases: dict[str, list[str]] = {}
    for field in _DECISION_ALIAS_FIELDS:
        field_overrides = (
            actor_aliases if field == "actor" else list(overrides.get(field, ()))
        )
        if field == "condition" and semantic_scope is not None:
            # Regression guard: scope changes define different decisions. An
            # unscoped convenience alias must never bypass the generated scope.
            field_overrides = [
                alias if alias.endswith(semantic_scope) else f"{alias} {semantic_scope}"
                for alias in field_overrides
            ]
        aliases[field] = list(dict.fromkeys((str(decision[field]), *field_overrides)))
    return aliases


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
        # The scope changes the decision, not merely its surface wording. Keep
        # the authored structured reference aligned with the scoped evidence.
        plant["decision"]["condition"] = f"{plant['decision']['condition']} {scope}"


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
    assigned_styles = _balanced_random_choices(rng, styles, len(documents))
    assigned_title_roots = _balanced_random_choices(rng, title_roots, len(documents))
    for document_index, (document, style, title_root) in enumerate(
        zip(documents, assigned_styles, assigned_title_roots, strict=True)
    ):
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
        document["title"] = f"{title_root} {rng.randrange(100, 1000)}"
        document["style"] = style
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


def _balanced_random_choices(
    rng: random.Random, options: list[str], count: int
) -> list[str]:
    """Draw balanced, shuffled labels without tying any label to document type."""

    selected: list[str] = []
    while len(selected) < count:
        batch = list(options)
        rng.shuffle(batch)
        selected.extend(batch)
    return selected[:count]


def _blend_authored_and_procedural_language(
    rng: random.Random,
    documents: list[dict[str, str]],
    planted_items: list[dict[str, Any]],
    distractors: list[dict[str, str]],
    *,
    sector: str,
) -> None:
    """Give authored and neutral sentences the same procedural components.

    A generator-only predicate vocabulary is itself a perfect source marker:
    an attacker can remove every sentence containing those predicates and keep
    only authored issue candidates.  Each original sentence therefore receives
    one clause from the same seeded component distribution used for additional
    distractors.  References are remapped exactly, preserving the policy rule
    while making procedural vocabulary non-exclusive to either class.
    """

    used: set[str] = set()
    remapped: dict[tuple[str, str], str] = {}
    for document in documents:
        sentences = [
            sentence
            for sentence in re.split(r"(?<=[.!?])\s+", document["text"])
            if sentence
        ]
        blended: list[str] = []
        for sentence in sentences:
            clause = _compose_shared_procedural_clause(
                rng,
                sector=sector,
                used=used,
            )
            used.add(clause)
            blended_sentence = (
                f"{sentence.removesuffix('.')}; {clause[0].lower()}{clause[1:]}"
            )
            remapped[(document["doc_id"], sentence)] = blended_sentence
            blended.append(blended_sentence)
        document["text"] = " ".join(blended)

    for planted in planted_items:
        planted["anchor_quote"] = remapped[(planted["doc_id"], planted["anchor_quote"])]
        if "related_anchor_quote" in planted:
            planted["related_anchor_quote"] = remapped[
                (planted["related_plant_doc_id"], planted["related_anchor_quote"])
            ]
    for distractor in distractors:
        distractor["anchor_quote"] = remapped[
            (distractor["doc_id"], distractor["anchor_quote"])
        ]

    # Regression guard: every accepted actor alias is a scored semantic key.
    # Classify generated support separately so view controls may retain it while
    # still removing genuinely optional neutral distractors.
    documents_by_id = {str(document["doc_id"]): document for document in documents}
    for plant in planted_items:
        document = documents_by_id[str(plant["doc_id"])]
        for actor in plant["decision_aliases"]["actor"]:
            actor_is_independently_supported = str(actor).casefold() in str(
                plant["anchor_quote"]
            ).casefold() or any(
                distractor["reason"] == ACTOR_SUPPORT_REASON
                and distractor["doc_id"] == plant["doc_id"]
                and str(actor).casefold() in str(distractor["anchor_quote"]).casefold()
                for distractor in distractors
            )
            if actor_is_independently_supported:
                continue
            support = _compose_neutral_distractor(
                rng,
                sector=sector,
                used=used,
                required_actor=str(actor),
            )
            document["text"] = " ".join((document["text"].strip(), support))
            distractors.append(
                {
                    "doc_id": document["doc_id"],
                    "anchor_quote": support,
                    "reason": ACTOR_SUPPORT_REASON,
                }
            )


def _balance_type_neutral_distractors(
    rng: random.Random,
    documents: list[dict[str, str]],
    distractors: list[dict[str, str]],
    *,
    sector: str,
) -> None:
    """Add varied neutral prose without encoding document or planted-issue roles.

    Counts are sampled independently for every document, and each sentence is
    composed from the same cross-product of procedural clauses already blended
    into authored policy sentences. Some neutral spans combine two independently
    sampled clauses. Both one-clause and two-clause distractors remain visible,
    so there is no single clause-count marker, while their length distribution
    overlaps the authored policy spans instead of making the longest sentences
    a reliable issue locator. The renderer rejects duplicates within a scenario.
    A later pass independently shuffles sentence order.
    """

    used: set[str] = set()
    supported_doc_ids = {
        str(distractor["doc_id"])
        for distractor in distractors
        if distractor["reason"] in {_COMPOSED_DISTRACTOR_REASON, ACTOR_SUPPORT_REASON}
    }
    for document in documents:
        # Plant documents already carry at least one actor-support distractor;
        # other documents receive at least one neutral clause here. Sampling
        # the remaining count preserves the intended two-to-seven sentence
        # range while reducing a fixed sentence-count signal.
        already_supported = str(document["doc_id"]) in supported_doc_ids
        current_sentence_count = len(
            [
                sentence
                for sentence in re.split(r"(?<=[.!?])\s+", document["text"])
                if sentence
            ]
        )
        minimum_extra = 0 if already_supported else 1
        maximum_extra = max(minimum_extra, 7 - current_sentence_count)
        selected = [
            _compose_neutral_distractor(rng, sector=sector, used=used)
            for _ in range(rng.randint(minimum_extra, maximum_extra))
        ]
        document["text"] = " ".join([document["text"].strip(), *selected])
        distractors.extend(
            {
                "doc_id": document["doc_id"],
                "anchor_quote": sentence,
                "reason": _COMPOSED_DISTRACTOR_REASON,
            }
            for sentence in selected
        )


def _compose_neutral_distractor(
    rng: random.Random,
    *,
    sector: str,
    used: set[str],
    required_actor: str | None = None,
) -> str:
    """Render one unique neutral span with overlapping authored-span lengths.

    A single generated clause was systematically shorter than an authored rule
    blended with that same clause. That made the three longest visible sentences
    a strong prompt-only issue locator. Seeded one/two-clause composition keeps
    the vocabulary shared and the lengths overlapping without restoring a fixed
    filler pool or an exclusive lexical marker.
    """

    for _ in range(100):
        first = _compose_shared_procedural_clause(
            rng,
            sector=sector,
            used=used,
            required_actor=required_actor,
        )
        sentence = first
        # Required actor support must remain usable in bounded document views.
        # One shared procedural clause is sufficient evidence; optional neutral
        # distractors still use one/two-clause variation for length overlap.
        if required_actor is None and rng.randrange(3) != 0:
            second = _compose_shared_procedural_clause(
                rng,
                sector=sector,
                used=used | {first},
            )
            sentence = f"{first.removesuffix('.')}; {second[0].lower()}{second[1:]}"
        if sentence not in used:
            used.add(sentence)
            return sentence
    raise RuntimeError("unable to compose a unique neutral distractor")


def _compose_shared_procedural_clause(
    rng: random.Random,
    *,
    sector: str,
    used: set[str],
    required_actor: str | None = None,
) -> str:
    """Render one clause from the distribution shared by all sentence roles."""

    domain_contexts = tuple(
        context.format(sector=sector) for context in _SHARED_PROCEDURAL_CONTEXTS
    )
    for _ in range(100):
        context = rng.choice(domain_contexts)
        actor = required_actor or rng.choice(_SHARED_PROCEDURAL_ACTORS)
        predicate = rng.choice(_SHARED_PROCEDURAL_PREDICATES)
        purpose = rng.choice(_SHARED_PROCEDURAL_PURPOSES)
        if rng.randrange(2):
            sentence = f"{context}, {actor} {predicate} {purpose}."
        else:
            sentence = f"{actor.capitalize()} {predicate} {context.lower()} {purpose}."
        if sentence not in used:
            return sentence
    raise RuntimeError("unable to compose a unique shared procedural clause")


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
        exact_profile = ", ".join(
            f"{dimension}={float(faction['values'][dimension]):+.2f}"
            for dimension in VALUE_DIMENSIONS
        )
        faction["summary"] = (
            f"{faction['summary']} Value profile used for this panel: "
            f"{exact_profile}. {' '.join(clauses)}"
        )


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
