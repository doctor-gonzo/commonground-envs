from __future__ import annotations

import copy
import json

from jsonschema import Draft202012Validator, FormatChecker
import pytest

from commonground_scenarios import (
    HELDOUT_TEMPLATES,
    TRAIN_TEMPLATES,
    ScenarioValidationError,
    generate_scenario,
    load_scenario_schema,
    scenario_to_bytes,
    validate_scenario,
)


ALL_TEMPLATES = TRAIN_TEMPLATES + HELDOUT_TEMPLATES


def test_template_sets_meet_size_and_separation_contract() -> None:
    train_ids = {template.template_id for template in TRAIN_TEMPLATES}
    heldout_ids = {template.template_id for template in HELDOUT_TEMPLATES}
    train_sectors = {template.sector for template in TRAIN_TEMPLATES}
    heldout_sectors = {template.sector for template in HELDOUT_TEMPLATES}
    train_styles = {document["style"] for template in TRAIN_TEMPLATES for document in template.documents}
    heldout_styles = {
        document["style"] for template in HELDOUT_TEMPLATES for document in template.documents
    }
    train_patterns = {_planting_pattern(template) for template in TRAIN_TEMPLATES}
    heldout_patterns = {_planting_pattern(template) for template in HELDOUT_TEMPLATES}

    assert len(TRAIN_TEMPLATES) >= 4
    assert len(HELDOUT_TEMPLATES) >= 2
    assert train_ids.isdisjoint(heldout_ids)
    assert train_sectors.isdisjoint(heldout_sectors)
    assert train_styles.isdisjoint(heldout_styles)
    assert train_patterns.isdisjoint(heldout_patterns)
    assert len(heldout_patterns) == len(HELDOUT_TEMPLATES)


@pytest.mark.parametrize("template", ALL_TEMPLATES, ids=lambda template: template.template_id)
def test_same_seed_regenerates_byte_identical_scenario(template: object) -> None:
    first = scenario_to_bytes(generate_scenario(7352, template))
    second = scenario_to_bytes(generate_scenario(7352, template))

    assert first == second
    assert first.endswith(b"\n")
    assert json.loads(first) == json.loads(second)


@pytest.mark.parametrize("template", ALL_TEMPLATES, ids=lambda template: template.template_id)
def test_generated_scenario_contains_complete_planted_structure(template: object) -> None:
    scenario = generate_scenario(17, template, generated_at="2026-08-15")

    validate_scenario(scenario)
    assert 3 <= len(scenario["factions"]) <= 5
    assert 3 <= len(scenario["documents"]) <= 8
    assert {plant["type"] for plant in scenario["planted_items"]} == {
        "ambiguity",
        "contradiction",
        "gap",
    }
    assert all(
        {"agree", "disagree"}.issubset(set(plant["target_stances"].values()))
        for plant in scenario["planted_items"]
    )
    assert scenario["human_feedback"] is None
    assert scenario["provenance"]["synthetic"] is True
    assert scenario["provenance"]["template_set"] == template.template_set


def test_different_seed_changes_canonical_scenario() -> None:
    first = scenario_to_bytes(generate_scenario(11, TRAIN_TEMPLATES[0]))
    second = scenario_to_bytes(generate_scenario(12, TRAIN_TEMPLATES[0]))

    assert first != second


def test_prose_polish_is_operator_injected_and_off_by_default() -> None:
    calls: list[str] = []
    unpolished = generate_scenario(9, TRAIN_TEMPLATES[0])

    assert calls == []
    polished = generate_scenario(
        9,
        TRAIN_TEMPLATES[0],
        prose_polisher=lambda text: calls.append(text) or text,
    )

    assert len(calls) == len(polished["documents"])
    assert polished["provenance"]["generation_mode"] == "operator-polished"
    assert unpolished["provenance"]["generation_mode"] == "template"


def test_falsy_injected_polisher_is_still_recorded() -> None:
    class FalsyPolisher:
        def __bool__(self) -> bool:
            return False

        def __call__(self, text: str) -> str:
            return text

    scenario = generate_scenario(9, TRAIN_TEMPLATES[0], prose_polisher=FalsyPolisher())

    assert scenario["provenance"]["generation_mode"] == "operator-polished"


def test_packaged_json_schema_names_every_top_level_field() -> None:
    schema = load_scenario_schema()

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert set(schema["required"]) == {
        "scenario_id",
        "organization",
        "factions",
        "documents",
        "planted_items",
        "distractors",
        "persona_panel",
        "human_feedback",
        "provenance",
    }
    assert "contextEngineSnapshot" in schema["$defs"]
    state_machine = schema["allOf"][0]
    assert state_machine["then"]["properties"]["human_feedback"] == {"type": "null"}
    assert state_machine["else"]["properties"]["persona_panel"] == {"type": "null"}
    assert (
        schema["$defs"]["contextEngineSnapshot"]["properties"]["meta"]["properties"][
            "synthetic"
        ]
        == {"const": False}
    )


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (
            lambda scenario: scenario["planted_items"][0].__setitem__("anchor_quote", "missing"),
            "plant anchor is absent",
        ),
        (
            lambda scenario: scenario["planted_items"][0]["target_stances"].__setitem__(
                scenario["factions"][0]["faction_id"], "disagree"
            ),
            "target_stances do not match faction priors",
        ),
        (
            lambda scenario: scenario["provenance"].__setitem__("generation_mode", "human"),
            "synthetic scenarios require template generation provenance",
        ),
        (
            lambda scenario: scenario["provenance"].__setitem__("seed", 43),
            "scenario_id must match provenance template and seed",
        ),
    ],
)
def test_validator_rejects_broken_answer_key_or_provenance(mutation: object, error: str) -> None:
    scenario = copy.deepcopy(generate_scenario(42, TRAIN_TEMPLATES[0]))
    mutation(scenario)

    with pytest.raises(ScenarioValidationError, match=error):
        validate_scenario(scenario)


def test_valid_human_socket_replaces_panel_and_declares_real_provenance() -> None:
    scenario = as_human_scenario(generate_scenario(42, TRAIN_TEMPLATES[0]))

    validate_scenario(scenario)
    assert scenario["organization"]["fictional"] is False
    assert scenario["persona_panel"] is None
    assert scenario["human_feedback"]["meta"]["synthetic"] is False


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("masked_cells", "not-an-array", "masked_cells must be an array"),
        ("held_out", [], "held_out must be an object"),
        ("clusters", {}, "clusters must be an array"),
        ("stats", [], "stats must be an object"),
        ("meta", [], "meta must be an object"),
    ],
)
def test_human_socket_rejects_schema_invalid_ce_fields(field: str, value: object, error: str) -> None:
    scenario = as_human_scenario(generate_scenario(42, TRAIN_TEMPLATES[0]))
    scenario["human_feedback"][field] = value

    with pytest.raises(ScenarioValidationError, match=error):
        validate_scenario(scenario)


def test_human_socket_rejects_synthetic_embedded_snapshot() -> None:
    scenario = as_human_scenario(generate_scenario(42, TRAIN_TEMPLATES[0]))
    scenario["human_feedback"]["meta"]["synthetic"] = True

    with pytest.raises(ScenarioValidationError, match="meta.synthetic must be false"):
        validate_scenario(scenario)


def test_human_path_still_enforces_prior_derived_target_stances() -> None:
    scenario = as_human_scenario(generate_scenario(42, TRAIN_TEMPLATES[0]))
    first_faction = scenario["factions"][0]["faction_id"]
    scenario["planted_items"][0]["target_stances"][first_faction] = "disagree"

    with pytest.raises(ScenarioValidationError, match="target_stances do not match faction priors"):
        validate_scenario(scenario)


def test_schema_and_manual_validator_share_integral_number_semantics() -> None:
    schema_validator = Draft202012Validator(load_scenario_schema(), format_checker=FormatChecker())
    scenario = as_human_scenario(generate_scenario(42, TRAIN_TEMPLATES[0]))
    scenario["provenance"]["seed"] = 42.0
    scenario["human_feedback"]["statements"][0]["index"] = 0.0
    scenario["human_feedback"]["votes"] = [[None], [-1.0], [0.0]]
    scenario["human_feedback"]["masked_cells"] = [[0.0, 0.0]]
    scenario["human_feedback"]["held_out"] = {"0,0": 1.0}

    validate_scenario(scenario)
    assert list(schema_validator.iter_errors(scenario)) == []

    mutations = (
        lambda candidate: candidate["provenance"].__setitem__("seed", 42.5),
        lambda candidate: candidate["human_feedback"]["statements"][0].__setitem__(
            "index", 0.5
        ),
        lambda candidate: candidate["human_feedback"]["votes"][1].__setitem__(0, -0.5),
        lambda candidate: candidate["human_feedback"]["masked_cells"][0].__setitem__(0, 0.5),
        lambda candidate: candidate["human_feedback"]["held_out"].__setitem__("0,0", 0.5),
    )
    for mutation in mutations:
        invalid = copy.deepcopy(scenario)
        mutation(invalid)
        with pytest.raises(ScenarioValidationError):
            validate_scenario(invalid)
        assert list(schema_validator.iter_errors(invalid))

    invalid_values = (True, float("nan"), float("inf"), float("-inf"))
    numeric_field_mutations = (
        lambda candidate, value: candidate["provenance"].__setitem__("seed", value),
        lambda candidate, value: candidate["human_feedback"]["statements"][0].__setitem__(
            "index", value
        ),
        lambda candidate, value: candidate["human_feedback"]["votes"][1].__setitem__(0, value),
        lambda candidate, value: candidate["human_feedback"]["masked_cells"][0].__setitem__(
            0, value
        ),
        lambda candidate, value: candidate["human_feedback"]["held_out"].__setitem__(
            "0,0", value
        ),
    )
    for mutation in numeric_field_mutations:
        for invalid_value in invalid_values:
            invalid = copy.deepcopy(scenario)
            mutation(invalid, invalid_value)
            with pytest.raises(ScenarioValidationError):
                validate_scenario(invalid)
            assert list(schema_validator.iter_errors(invalid))


def test_generation_and_validation_require_canonical_full_date() -> None:
    with pytest.raises(ScenarioValidationError, match="YYYY-MM-DD"):
        generate_scenario(42, TRAIN_TEMPLATES[0], generated_at="20260815")

    scenario = generate_scenario(42, TRAIN_TEMPLATES[0])
    scenario["provenance"]["generated_at"] = "20260815"

    with pytest.raises(ScenarioValidationError, match="YYYY-MM-DD"):
        validate_scenario(scenario)
    schema_validator = Draft202012Validator(load_scenario_schema(), format_checker=FormatChecker())
    assert list(schema_validator.iter_errors(scenario))


def as_human_scenario(scenario: dict[str, object]) -> dict[str, object]:
    scenario["organization"]["fictional"] = False
    scenario["persona_panel"] = None
    scenario["provenance"]["synthetic"] = False
    scenario["provenance"]["generation_mode"] = "human"
    scenario["human_feedback"] = {
        "session_id": "verified-session",
        "statements": [{"index": 0, "text": "Should the proposed interpretation apply?"}],
        "participants": ["p000", "p001", "p002"],
        "votes": [[1], [-1], [0]],
        "masked_cells": [],
        "held_out": {},
        "clusters": [],
        "stats": {},
        "meta": {"synthetic": False},
    }
    return scenario


def _planting_pattern(template: object) -> tuple[object, ...]:
    document_positions = {
        document["doc_id"]: index for index, document in enumerate(template.documents)
    }

    def anchor_position(item: dict[str, str]) -> tuple[int, int]:
        document_index = document_positions[item["doc_id"]]
        document_text = template.documents[document_index]["text"]
        sentence_index = document_text[: document_text.index(item["anchor_quote"])].count(". ")
        return document_index, sentence_index

    planted = tuple(
        (item["type"], *anchor_position(item)) for item in template.planted_items
    )
    distractors = tuple(anchor_position(item) for item in template.distractors)
    return planted, distractors
