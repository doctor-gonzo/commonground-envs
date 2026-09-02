"""Focused regressions for Ask's public structured semantic contract."""

from __future__ import annotations

import copy
from typing import Any

import pytest
from commonground_elicit import environment

PRIMARY_QUOTE = (
    "The support desk may release urgent file requests during overnight coverage."
)
SUPPORT_SENTENCE = "The records coordinator logs the edition used for processing."
RELATED_QUOTE = "Only the records office may release urgent file requests during overnight coverage."


def _plant(*, contradiction: bool = False) -> dict[str, Any]:
    alternative_stances = {"group-a": "agree", "group-b": "disagree"}
    plant: dict[str, Any] = {
        "doc_id": "support-guide",
        "quote": PRIMARY_QUOTE,
        "type": "contradiction" if contradiction else "ambiguity",
        "decision": {
            "actor": "the support desk",
            "action": "release urgent file requests",
            "condition": "during overnight coverage",
            "anchor_outcome": "keep urgent file requests pending",
            "alternative_outcome": "release urgent file requests",
        },
        # Deliberately malicious aliases prove that Ask never consults this
        # prompt-hidden compatibility surface.
        "decision_aliases": {
            "actor": ["an unrelated actor"],
            "action": ["sell customer records"],
            "condition": ["whenever convenient"],
            "anchor_outcome": ["discard every request"],
            "alternative_outcome": ["publish every request"],
        },
        "yes_choice": "alternative",
        "target_stances": dict(alternative_stances),
        "alternative_stances": dict(alternative_stances),
        "decision_value": 1.0,
        "document_text": f"{PRIMARY_QUOTE} {SUPPORT_SENTENCE}",
        "related_evidence": None,
    }
    if contradiction:
        plant["related_evidence"] = {
            "doc_id": "records-rule",
            "quote": RELATED_QUOTE,
        }
        plant["related_document_text"] = (
            f"{RELATED_QUOTE} The archive owner records each revision."
        )
    return plant


def _candidate(plant: dict[str, Any]) -> dict[str, Any]:
    return {
        "doc_id": plant["doc_id"],
        "quote": plant["quote"],
        "type": plant["type"],
        "question": (
            "Should the support desk release urgent file requests during overnight "
            "coverage?"
        ),
        "decision": copy.deepcopy(plant["decision"]),
        "yes_choice": "alternative",
        "related_evidence": copy.deepcopy(plant["related_evidence"]),
        "target_stances": copy.deepcopy(plant["alternative_stances"]),
    }


def _score(candidate: dict[str, Any], plant: dict[str, Any]) -> float:
    return environment.question_utility_score(
        [candidate],
        [plant],
        panel_polarization=0.0,
        question_count=1,
    )


def test_ask_free_question_wording_does_not_change_structured_reward() -> None:
    plant = _plant()
    canonical = _candidate(plant)
    unrelated = copy.deepcopy(canonical)
    unrelated["question"] = "Could this deliberately unrelated wording be accepted?"

    assert canonical["question"] != unrelated["question"]
    assert _score(canonical, plant) == _score(unrelated, plant) == 1.0


def test_ask_question_must_still_have_yes_no_form() -> None:
    plant = _plant()
    declarative = _candidate(plant)
    declarative["question"] = "This is a declarative presentation sentence."

    assert _score(declarative, plant) == 0.0


@pytest.mark.parametrize("field", environment.DECISION_FRAME_FIELDS)
def test_ask_requires_every_exact_public_decision_slot(field: str) -> None:
    plant = _plant()
    exact = _candidate(plant)
    mutated = copy.deepcopy(exact)
    mutated["decision"][field] = f"{mutated['decision'][field]} without review"

    assert _score(exact, plant) == 1.0
    assert _score(mutated, plant) == 0.0


def test_ask_rejects_wrong_type_and_unexpected_related_evidence() -> None:
    plant = _plant()
    wrong_type = _candidate(plant)
    wrong_type["type"] = "gap"
    unexpected_relation = _candidate(plant)
    unexpected_relation["related_evidence"] = {
        "doc_id": "records-rule",
        "quote": RELATED_QUOTE,
    }

    assert _score(wrong_type, plant) == 0.0
    assert _score(unexpected_relation, plant) == 0.0


@pytest.mark.parametrize("key_change", ["missing", "extra"])
def test_ask_rejects_inexact_faction_key_sets(key_change: str) -> None:
    plant = _plant()
    candidate = _candidate(plant)
    if key_change == "missing":
        del candidate["target_stances"]["group-b"]
    else:
        candidate["target_stances"]["group-extra"] = "pass"

    assert _score(candidate, plant) == 0.0


def test_ask_requires_composed_alternative_stances() -> None:
    plant = _plant()
    candidate = _candidate(plant)
    del plant["alternative_stances"]

    assert _score(candidate, plant) == 0.0


def test_ask_never_reads_hidden_decision_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plant = _plant()
    exact = _candidate(plant)

    def forbidden_alias_access(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("Ask accessed hidden decision aliases")

    monkeypatch.setattr(
        environment, "_accepted_decision_aliases", forbidden_alias_access
    )

    assert _score(exact, plant) == 1.0
    alias_only = copy.deepcopy(exact)
    alias_only["decision"] = {
        field: aliases[0] for field, aliases in plant["decision_aliases"].items()
    }
    assert _score(alias_only, plant) == 0.0


def test_ask_contract_normalizes_only_unicode_case_and_whitespace() -> None:
    plant = _plant()
    normalized = _candidate(plant)
    normalized["quote"] = (
        PRIMARY_QUOTE.replace("file", "ﬁle").swapcase().replace(" ", "  ")
    )
    normalized["decision"] = {
        field: value.swapcase().replace(" ", "  ")
        for field, value in plant["decision"].items()
    }
    punctuation_changed = copy.deepcopy(normalized)
    punctuation_changed["quote"] = normalized["quote"].rstrip(".")

    assert _score(normalized, plant) == 1.0
    assert _score(punctuation_changed, plant) == 0.0


def test_ask_keeps_opaque_document_identifiers_exact() -> None:
    plant = _plant(contradiction=True)
    primary_id_changed = _candidate(plant)
    primary_id_changed["doc_id"] = "SUPPORT-GUIDE"
    related_id_changed = _candidate(plant)
    related_id_changed["related_evidence"]["doc_id"] = "RECORDS-RULE"

    assert _score(primary_id_changed, plant) == 0.0
    assert _score(related_id_changed, plant) == 0.0


@pytest.mark.parametrize(
    "quote",
    [
        "The support desk may release urgent file requests",
        f"{PRIMARY_QUOTE} Extra padding.",
        SUPPORT_SENTENCE,
        PRIMARY_QUOTE.rstrip("."),
    ],
)
def test_ask_rejects_fragments_padding_and_support_clauses(quote: str) -> None:
    plant = _plant()
    candidate = _candidate(plant)
    candidate["quote"] = quote

    assert _score(candidate, plant) == 0.0


@pytest.mark.parametrize(
    "related_quote",
    [
        "Only the records office may release urgent file requests",
        f"{RELATED_QUOTE} Extra padding.",
        "The archive owner records each revision.",
        RELATED_QUOTE.rstrip("."),
    ],
)
def test_ask_requires_exact_related_evidence(related_quote: str) -> None:
    plant = _plant(contradiction=True)
    exact = _candidate(plant)
    mutated = copy.deepcopy(exact)
    mutated["related_evidence"]["quote"] = related_quote

    assert _score(exact, plant) == 1.0
    assert _score(mutated, plant) == 0.0


def test_ask_orientation_is_explicit_and_stance_scoring_is_unchanged() -> None:
    plant = _plant()
    alternative = _candidate(plant)
    wrong_stances = copy.deepcopy(alternative)
    wrong_stances["yes_choice"] = "anchor"
    correctly_reoriented = copy.deepcopy(wrong_stances)
    correctly_reoriented["target_stances"] = {
        "group-a": "disagree",
        "group-b": "agree",
    }
    malformed = copy.deepcopy(alternative)
    malformed["yes_choice"] = "sideways"

    assert _score(alternative, plant) == 1.0
    assert _score(wrong_stances, plant) == 0.5
    assert _score(correctly_reoriented, plant) == 1.0
    assert _score(malformed, plant) == 0.0
