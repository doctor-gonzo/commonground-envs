"""Independently authored, two-sided audit of Elicit decision semantics.

These fixtures are deliberately separate from the scenario templates and their
accepted-alias tables.  They exercise the public semantic contract with fresh
policy domains, then identify every case by issue type and decision-frame slot
so a release failure remains attributable rather than becoming one aggregate
acceptance-rate number.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import pytest
from commonground_elicit import question_utility_score
from commonground_elicit.environment import question_decision_similarity

DECISION_FIELDS = (
    "actor",
    "action",
    "condition",
    "anchor_outcome",
    "alternative_outcome",
)
ISSUE_TYPES = ("ambiguity", "contradiction", "gap")
INVERSE_STANCE = {"agree": "disagree", "disagree": "agree", "pass": "pass"}


@dataclass(frozen=True)
class SemanticAuditFixture:
    issue_type: str
    plant: dict[str, Any]
    paraphrase_decision: dict[str, str]
    passive_question: str


def _stances() -> dict[str, str]:
    return {
        "continuity-group": "agree",
        "oversight-group": "disagree",
        "access-group": "pass",
    }


def _plant(
    *,
    issue_type: str,
    doc_id: str,
    quote: str,
    document_text: str,
    decision: dict[str, str],
    aliases: dict[str, list[str]],
    yes_choice: str,
    related_evidence: dict[str, str] | None = None,
    related_document_text: str = "",
) -> dict[str, Any]:
    stances = _stances()
    return {
        "doc_id": doc_id,
        "quote": quote,
        "type": issue_type,
        # The semantic scorer must not depend on this one reference sentence.
        "question": "Should the authored policy choice be resolved?",
        "question_aliases": ["Must the documented decision be made?"],
        "decision": decision,
        "decision_aliases": aliases,
        "yes_choice": yes_choice,
        "related_evidence": related_evidence,
        "target_stances": stances,
        "alternative_stances": {
            faction_id: INVERSE_STANCE[stance] for faction_id, stance in stances.items()
        },
        "decision_value": 1.0,
        "document_text": document_text,
        "related_document_text": related_document_text,
    }


def _fixtures() -> dict[str, SemanticAuditFixture]:
    ambiguity_quote = (
        "Review leads may release restricted records promptly when a request "
        "seems urgent."
    )
    ambiguity_decision = {
        "actor": "review leads",
        "action": "release restricted records",
        "condition": "a request seems urgent",
        "anchor_outcome": "release restricted records promptly",
        "alternative_outcome": "require documented urgency before release",
    }
    ambiguity_paraphrase = {
        "actor": "records supervisors",
        "action": "disclose restricted records",
        "condition": "urgency is apparent",
        "anchor_outcome": "restricted records are released promptly",
        "alternative_outcome": "document urgency before releasing records",
    }
    ambiguity = SemanticAuditFixture(
        issue_type="ambiguity",
        plant=_plant(
            issue_type="ambiguity",
            doc_id="restricted-records-rule",
            quote=ambiguity_quote,
            document_text=(
                f"{ambiguity_quote} Records supervisors may disclose restricted "
                "records when urgency is apparent. Restricted records are released "
                "promptly unless supervisors document urgency before releasing "
                "records."
            ),
            decision=ambiguity_decision,
            aliases={
                field: [ambiguity_decision[field], ambiguity_paraphrase[field]]
                for field in DECISION_FIELDS
            },
            yes_choice="anchor",
        ),
        paraphrase_decision=ambiguity_paraphrase,
        passive_question=(
            "Could restricted records be disclosed promptly by records supervisors "
            "when urgency is apparent (yes selects the anchor outcome: restricted "
            "records are released promptly)?"
        ),
    )

    contradiction_quote = (
        "Operations directors may authorize emergency purchases above the usual limit."
    )
    contradiction_related = (
        "Only the finance controller may approve purchases above the usual limit."
    )
    contradiction_decision = {
        "actor": "operations directors",
        "action": "authorize emergency purchases",
        "condition": "an emergency purchase exceeds the usual limit",
        "anchor_outcome": "operations directors authorize the purchase",
        "alternative_outcome": "only the finance controller authorizes the purchase",
    }
    contradiction_paraphrase = {
        "actor": "incident commanders",
        "action": "approve emergency purchases",
        "condition": "emergency spending exceeds the routine cap",
        "anchor_outcome": "incident commanders approve the purchase",
        "alternative_outcome": "the finance controller alone approves the purchase",
    }
    contradiction = SemanticAuditFixture(
        issue_type="contradiction",
        plant=_plant(
            issue_type="contradiction",
            doc_id="emergency-purchasing-guide",
            quote=contradiction_quote,
            document_text=(
                f"{contradiction_quote} Incident commanders approve emergency "
                "purchases when emergency spending exceeds the routine cap."
            ),
            decision=contradiction_decision,
            aliases={
                field: [contradiction_decision[field], contradiction_paraphrase[field]]
                for field in DECISION_FIELDS
            },
            yes_choice="anchor",
            related_evidence={
                "doc_id": "finance-control-rule",
                "quote": contradiction_related,
            },
            related_document_text=(
                f"{contradiction_related} The finance controller alone approves "
                "the purchase."
            ),
        ),
        paraphrase_decision=contradiction_paraphrase,
        passive_question=(
            "Would emergency purchases be approved by incident commanders when "
            "emergency spending exceeds the routine cap (yes selects the anchor "
            "outcome: incident commanders approve the purchase)?"
        ),
    )

    gap_quote = "Case officers send all notices through the verified online account."
    gap_decision = {
        "actor": "case officers",
        "action": "deliver official notices",
        "condition": "a recipient has no verified online account",
        "anchor_outcome": "send the notice through the verified online account",
        "alternative_outcome": "use an offline channel for the official notice",
    }
    gap_paraphrase = {
        "actor": "service coordinators",
        "action": "issue official notices",
        "condition": "a recipient does not have a verified online account",
        "anchor_outcome": "official notices use the verified online account",
        "alternative_outcome": "an offline channel is used for the official notice",
    }
    gap = SemanticAuditFixture(
        issue_type="gap",
        plant=_plant(
            issue_type="gap",
            doc_id="notice-delivery-rule",
            quote=gap_quote,
            document_text=(
                f"{gap_quote} Service coordinators issue official notices. If a "
                "recipient does not have a verified online account, the policy does "
                "not say whether an offline channel is used for the official notice."
            ),
            decision=gap_decision,
            aliases={
                field: [gap_decision[field], gap_paraphrase[field]]
                for field in DECISION_FIELDS
            },
            yes_choice="alternative",
        ),
        paraphrase_decision=gap_paraphrase,
        passive_question=(
            "May an offline channel be used by service coordinators to issue "
            "official notices when a recipient does not have a verified online account "
            "(yes selects the alternative outcome: an offline channel is used for "
            "the official notice)?"
        ),
    )
    return {fixture.issue_type: fixture for fixture in (ambiguity, contradiction, gap)}


FIXTURES = _fixtures()


def _render_question(decision: dict[str, str], yes_choice: str) -> str:
    selected = decision[f"{yes_choice}_outcome"]
    return (
        f"Should {decision['actor']} {decision['action']} when "
        f"{decision['condition']} (yes selects the {yes_choice} outcome: "
        f"{selected})?"
    )


def _semantic_score(
    fixture: SemanticAuditFixture,
    *,
    decision: dict[str, str] | None = None,
    question: str | None = None,
    yes_choice: str | None = None,
) -> float:
    submitted = decision or fixture.paraphrase_decision
    choice = yes_choice or str(fixture.plant["yes_choice"])
    return question_decision_similarity(
        question or _render_question(submitted, choice),
        fixture.plant,
        candidate_decision=submitted,
        yes_choice=choice,
    )


@pytest.mark.parametrize("issue_type", ISSUE_TYPES)
@pytest.mark.parametrize("slot", DECISION_FIELDS)
def test_authored_slot_paraphrases_are_accepted(issue_type: str, slot: str) -> None:
    """Each reference slot independently admits its authored source paraphrase."""

    fixture = FIXTURES[issue_type]
    decision = copy.deepcopy(fixture.plant["decision"])
    decision[slot] = fixture.paraphrase_decision[slot]

    assert _semantic_score(fixture, decision=decision) == 1.0


@pytest.mark.parametrize("issue_type", ISSUE_TYPES)
def test_passive_voice_paraphrases_are_accepted(issue_type: str) -> None:
    fixture = FIXTURES[issue_type]

    assert (
        _semantic_score(
            fixture,
            decision=fixture.paraphrase_decision,
            question=fixture.passive_question,
        )
        == 1.0
    )


@pytest.mark.parametrize("issue_type", ISSUE_TYPES)
def test_consistent_yes_no_orientation_reversal_is_accepted(issue_type: str) -> None:
    """Changing the chosen side and prose together is not a polarity error."""

    fixture = FIXTURES[issue_type]
    reversed_choice = (
        "alternative" if fixture.plant["yes_choice"] == "anchor" else "anchor"
    )

    assert (
        _semantic_score(
            fixture,
            decision=fixture.paraphrase_decision,
            question=_render_question(fixture.paraphrase_decision, reversed_choice),
            yes_choice=reversed_choice,
        )
        == 1.0
    )


def test_supported_negative_condition_is_accepted() -> None:
    """A grounded negative is valid; the audit does not reject negation wholesale."""

    fixture = FIXTURES["gap"]

    assert _semantic_score(fixture) == 1.0


def test_hidden_reference_question_wording_does_not_change_credit() -> None:
    """Decision semantics, not one canonical reference sentence, determine credit."""

    fixture = copy.deepcopy(FIXTURES["ambiguity"])
    before = _semantic_score(fixture)
    fixture.plant["question"] = "Might an entirely different editor phrase this?"
    fixture.plant["question_aliases"] = ["Could wording alone decide the score?"]

    assert before == _semantic_score(fixture) == 1.0


def _invalid_slot_value(
    fixture: SemanticAuditFixture,
    slot: str,
) -> str:
    decision = fixture.paraphrase_decision
    if slot == "actor":
        return decision["action"]
    if slot == "action":
        return decision["actor"]
    if slot == "condition":
        return "only when an unaudited revenue threshold exceeds ten thousand"
    if slot == "anchor_outcome":
        return decision["alternative_outcome"]
    return decision["anchor_outcome"]


@pytest.mark.parametrize("issue_type", ISSUE_TYPES)
@pytest.mark.parametrize("slot", DECISION_FIELDS)
def test_slot_corruptions_are_rejected(issue_type: str, slot: str) -> None:
    """Reject role collapse, unsupported scope, and outcome-side substitution."""

    fixture = FIXTURES[issue_type]
    decision = copy.deepcopy(fixture.paraphrase_decision)
    decision[slot] = _invalid_slot_value(fixture, slot)

    assert _semantic_score(fixture, decision=decision) == 0.0


@pytest.mark.parametrize("issue_type", ISSUE_TYPES)
def test_actor_action_swap_is_rejected(issue_type: str) -> None:
    fixture = FIXTURES[issue_type]
    decision = copy.deepcopy(fixture.paraphrase_decision)
    decision["actor"], decision["action"] = decision["action"], decision["actor"]

    assert _semantic_score(fixture, decision=decision) == 0.0


@pytest.mark.parametrize("issue_type", ISSUE_TYPES)
def test_anchor_alternative_outcome_swap_is_rejected(issue_type: str) -> None:
    fixture = FIXTURES[issue_type]
    decision = copy.deepcopy(fixture.paraphrase_decision)
    decision["anchor_outcome"], decision["alternative_outcome"] = (
        decision["alternative_outcome"],
        decision["anchor_outcome"],
    )

    assert _semantic_score(fixture, decision=decision) == 0.0


@pytest.mark.parametrize("issue_type", ISSUE_TYPES)
def test_unsupported_exception_threshold_or_condition_is_rejected(
    issue_type: str,
) -> None:
    fixture = FIXTURES[issue_type]
    additions = {
        "ambiguity": " unless a donor objects",
        "contradiction": " when spending exceeds ten thousand dollars",
        "gap": " only after a court order arrives",
    }
    decision = copy.deepcopy(fixture.paraphrase_decision)
    decision["condition"] += additions[issue_type]

    assert _semantic_score(fixture, decision=decision) == 0.0


@pytest.mark.parametrize("issue_type", ISSUE_TYPES)
def test_polarity_mismatch_is_rejected(issue_type: str) -> None:
    fixture = FIXTURES[issue_type]
    decision = copy.deepcopy(fixture.paraphrase_decision)
    decision["action"] = f"not {decision['action']}"

    assert _semantic_score(fixture, decision=decision) == 0.0


@pytest.mark.parametrize("issue_type", ISSUE_TYPES)
def test_orientation_metadata_mismatch_is_rejected(issue_type: str) -> None:
    fixture = FIXTURES[issue_type]
    declared_choice = (
        "alternative" if fixture.plant["yes_choice"] == "anchor" else "anchor"
    )

    assert (
        _semantic_score(
            fixture,
            question=_render_question(fixture.paraphrase_decision, declared_choice),
            yes_choice=str(fixture.plant["yes_choice"]),
        )
        == 0.0
    )


def test_wrong_contradiction_relationship_is_rejected_end_to_end() -> None:
    fixture = FIXTURES["contradiction"]
    plant = fixture.plant
    candidate = {
        "doc_id": plant["doc_id"],
        "quote": plant["quote"],
        "type": plant["type"],
        "question": fixture.passive_question,
        "decision": fixture.paraphrase_decision,
        "yes_choice": plant["yes_choice"],
        "related_evidence": {
            "doc_id": "unrelated-audit-log",
            "quote": "The audit log is retained for seven years.",
        },
        "target_stances": plant["target_stances"],
    }

    assert (
        question_utility_score(
            [candidate],
            [plant],
            panel_polarization=1.0,
            question_count=1,
        )
        == 0.0
    )
