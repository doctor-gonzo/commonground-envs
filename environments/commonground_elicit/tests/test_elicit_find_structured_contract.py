"""Focused regressions for Find's authored structured-decision contract."""

from __future__ import annotations

import copy
from typing import Any

import pytest
from commonground_elicit.environment import (
    _finding_diagnosis_matches,
    question_decision_similarity,
)

FIELDS = (
    "actor",
    "action",
    "condition",
    "anchor_outcome",
    "alternative_outcome",
)

DECISIONS = {
    "ambiguity": {
        "actor": "review leads",
        "action": "release restricted records",
        "condition": "a request seems urgent",
        "anchor_outcome": "release the records promptly",
        "alternative_outcome": "require documented urgency before release",
    },
    "contradiction": {
        "actor": "operations directors",
        "action": "authorize emergency purchases",
        "condition": "an emergency purchase exceeds the usual limit",
        "anchor_outcome": "operations directors authorize the purchase",
        "alternative_outcome": "only the finance controller authorizes it",
    },
    "gap": {
        "actor": "case officers",
        "action": "deliver official notices",
        "condition": "a recipient has no verified online account",
        "anchor_outcome": "use the verified online account",
        "alternative_outcome": "use an offline notice channel",
    },
}

ALIASES = {
    "ambiguity": {
        "actor": "records supervisors",
        "action": "disclose restricted records",
        "condition": "urgency is apparent",
        "anchor_outcome": "restricted records are released promptly",
        "alternative_outcome": "document urgency before releasing records",
    },
    "contradiction": {
        "actor": "incident commanders",
        "action": "approve emergency purchases",
        "condition": "emergency spending exceeds the routine cap",
        "anchor_outcome": "incident commanders approve the purchase",
        "alternative_outcome": "the finance controller alone approves it",
    },
    "gap": {
        "actor": "service coordinators",
        "action": "issue official notices",
        "condition": "a recipient lacks a verified online account",
        "anchor_outcome": "official notices use the online account",
        "alternative_outcome": "an offline channel carries the notice",
    },
}


def _plant(issue_type: str) -> dict[str, Any]:
    canonical = DECISIONS[issue_type]
    aliases = ALIASES[issue_type]
    return {
        "decision": canonical,
        "decision_aliases": {
            field: [canonical[field], aliases[field]] for field in FIELDS
        },
    }


def _score(
    issue_type: str, decision: Any, diagnosis: str = "Could this be clarified?"
) -> float:
    return question_decision_similarity(
        diagnosis,
        _plant(issue_type),
        candidate_decision=decision,
    )


@pytest.mark.parametrize("issue_type", DECISIONS)
@pytest.mark.parametrize("field", FIELDS)
def test_find_accepts_each_explicitly_authored_slot_alias(
    issue_type: str,
    field: str,
) -> None:
    decision = copy.deepcopy(DECISIONS[issue_type])
    decision[field] = ALIASES[issue_type][field]

    assert _score(issue_type, decision) == 1.0


@pytest.mark.parametrize("issue_type", DECISIONS)
def test_find_prose_wording_is_unscored_beyond_yes_no_form(issue_type: str) -> None:
    decision = ALIASES[issue_type]

    assert _score(issue_type, decision, "Could arbitrary wording be used?") == 1.0
    assert _score(issue_type, decision, "This is a declarative sentence.") == 0.0


@pytest.mark.parametrize("issue_type", DECISIONS)
def test_find_contract_normalizes_unicode_case_and_whitespace(issue_type: str) -> None:
    normalized = {
        field: f"  {value.swapcase().replace(' ', '   ')}  "
        for field, value in ALIASES[issue_type].items()
    }

    assert _score(issue_type, normalized) == 1.0


def test_find_contract_normalizes_unicode_compatibility() -> None:
    normalized = copy.deepcopy(ALIASES["gap"])
    normalized["action"] = normalized["action"].replace("fi", "ﬁ")

    assert normalized["action"] != ALIASES["gap"]["action"]
    assert _score("gap", normalized) == 1.0


@pytest.mark.parametrize("issue_type", DECISIONS)
@pytest.mark.parametrize("field", FIELDS)
def test_find_rejects_unauthored_slot_values(issue_type: str, field: str) -> None:
    decision = copy.deepcopy(ALIASES[issue_type])
    decision[field] = "an unrelated value that is not an authored alias"

    assert _score(issue_type, decision) == 0.0


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: f"{value}?",
        lambda value: f"not {value}",
        lambda value: f"{value} unless a supervisor objects",
    ],
)
def test_find_rejects_high_overlap_unauthored_mutations(mutation: Any) -> None:
    decision = copy.deepcopy(ALIASES["contradiction"])
    decision["condition"] = mutation(decision["condition"])

    assert _score("contradiction", decision) == 0.0


@pytest.mark.parametrize("issue_type", DECISIONS)
def test_find_rejects_swapped_roles_and_collapsed_outcomes(issue_type: str) -> None:
    swapped = copy.deepcopy(ALIASES[issue_type])
    swapped["actor"], swapped["action"] = swapped["action"], swapped["actor"]
    collapsed = copy.deepcopy(ALIASES[issue_type])
    collapsed["alternative_outcome"] = collapsed["anchor_outcome"]

    assert _score(issue_type, swapped) == 0.0
    assert _score(issue_type, collapsed) == 0.0


def test_find_missing_authored_diagnosis_fails_closed() -> None:
    plant = _plant("gap")
    candidate = {
        "diagnosis": "Could this decision be clarified?",
        "decision": copy.deepcopy(ALIASES["gap"]),
    }

    assert not _finding_diagnosis_matches(candidate, plant)
