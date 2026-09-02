"""Focused regressions for Find's evidence-first response contract."""

from __future__ import annotations

from commonground_elicit.environment import (
    _finding_diagnosis_matches,
    _parse_findings_list,
)


def _finding() -> dict[str, object]:
    return {
        "doc_id": "doc-policy",
        "quote": "The team responds promptly.",
        "type": "ambiguity",
        "diagnosis": "Should the team define promptly?",
        "related_evidence": None,
    }


def test_find_contract_requires_only_evidence_type_and_yes_no_diagnosis() -> None:
    assert _parse_findings_list([_finding()]) == [_finding()]


def test_find_contract_rejects_removed_hidden_decision_object() -> None:
    finding = _finding()
    finding["decision"] = {
        "actor": "the team",
        "action": "respond",
        "condition": "a request arrives",
        "anchor_outcome": "respond promptly",
        "alternative_outcome": "define promptly",
    }

    assert _parse_findings_list([finding]) is None


def test_find_diagnosis_scores_form_not_hidden_authored_vocabulary() -> None:
    plant = {"diagnosis": "Should the team define promptly?"}

    assert _finding_diagnosis_matches(
        {"diagnosis": "Could entirely different words describe this decision?"},
        plant,
    )
    assert not _finding_diagnosis_matches(
        {"diagnosis": "This is a declarative sentence."},
        plant,
    )


def test_find_missing_authored_diagnosis_fails_closed() -> None:
    assert not _finding_diagnosis_matches(
        {"diagnosis": "Should this be clarified?"},
        {},
    )
