from __future__ import annotations

import copy
import json

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
from commonground_scenarios import generator as generator_module
from jsonschema import Draft202012Validator, FormatChecker

ALL_TEMPLATES = TRAIN_TEMPLATES + HELDOUT_TEMPLATES


def test_template_sets_meet_size_and_separation_contract() -> None:
    train_ids = {template.template_id for template in TRAIN_TEMPLATES}
    heldout_ids = {template.template_id for template in HELDOUT_TEMPLATES}
    train_sectors = {template.sector for template in TRAIN_TEMPLATES}
    heldout_sectors = {template.sector for template in HELDOUT_TEMPLATES}
    train_styles = {
        document["style"]
        for template in TRAIN_TEMPLATES
        for document in template.documents
    }
    heldout_styles = {
        document["style"]
        for template in HELDOUT_TEMPLATES
        for document in template.documents
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


@pytest.mark.parametrize(
    "template", ALL_TEMPLATES, ids=lambda template: template.template_id
)
def test_same_seed_regenerates_byte_identical_scenario(template: object) -> None:
    first = scenario_to_bytes(generate_scenario(7352, template))
    second = scenario_to_bytes(generate_scenario(7352, template))

    assert first == second
    assert first.endswith(b"\n")
    assert json.loads(first) == json.loads(second)


@pytest.mark.parametrize(
    "template", ALL_TEMPLATES, ids=lambda template: template.template_id
)
def test_generated_scenario_contains_complete_planted_structure(
    template: object,
) -> None:
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
    assert all(
        plant["canonical_question"].split(maxsplit=1)[0]
        in {"Can", "Does", "Is", "May", "Must", "Should"}
        for plant in scenario["planted_items"]
    )
    assert all(
        isinstance(plant["canonical_question_aliases"], list)
        for plant in scenario["planted_items"]
    )
    assert all(
        "For decisions involving" not in faction["summary"]
        and "leans toward yes" not in faction["summary"]
        and "leans toward no" not in faction["summary"]
        and all(
            ", ".join(plant["decision_terms"]) not in faction["summary"]
            for plant in scenario["planted_items"]
        )
        for faction in scenario["factions"]
    )
    assert all(
        any(
            phrase in faction["summary"]
            for phrase in generator_module._PRINCIPLE_EXAMPLES[
                (
                    plant["type"],
                    plant["target_stances"][faction["faction_id"]],
                )
            ]
        )
        for faction in scenario["factions"]
        for plant in scenario["planted_items"]
    )
    assert scenario["human_feedback"] is None
    assert scenario["provenance"]["synthetic"] is True
    assert scenario["provenance"]["template_set"] == template.template_set


def test_scope_terms_cannot_displace_the_actual_conflicting_rule() -> None:
    template = next(
        template
        for template in HELDOUT_TEMPLATES
        if template.template_id == "cooperative-housing-maintenance"
    )
    scenario = generate_scenario(8217, template)
    contradiction = next(
        plant for plant in scenario["planted_items"] if plant["type"] == "contradiction"
    )

    assert contradiction["related_evidence"]["quote"].startswith(
        "Emergency crews may enter an occupied unit"
    )


@pytest.mark.parametrize(
    "question",
    [
        "What conditions require a route pause?",
        "Should dispatchers decide which conditions require a route pause",
        "should dispatchers decide which conditions require a route pause?",
        "Should—dispatchers decide which conditions require a route pause?",
        " Should dispatchers decide which conditions require a route pause?",
        "Should dispatchers decide which conditions require a route pause? ",
        "Should dispatchers decide which conditions require a route pause?\n",
        "Should\ndispatchers decide which conditions require a route pause?",
        "Should\tdispatchers decide which conditions require a route pause?",
        "Should  dispatchers decide which conditions require a route pause?",
    ],
)
def test_validator_and_schema_reject_non_yes_no_canonical_questions(
    question: str,
) -> None:
    scenario = generate_scenario(17, HELDOUT_TEMPLATES[0])
    scenario["planted_items"][0]["canonical_question"] = question

    with pytest.raises(ScenarioValidationError, match="yes/no question"):
        validate_scenario(scenario)
    assert list(
        Draft202012Validator(
            load_scenario_schema(), format_checker=FormatChecker()
        ).iter_errors(scenario)
    )


@pytest.mark.parametrize(
    "question",
    [
        "Are dispatchers responsible for deciding which conditions require a route pause?",
        "Is it?",
    ],
)
def test_validator_and_schema_accept_supported_yes_no_auxiliary(
    question: str,
) -> None:
    scenario = generate_scenario(17, HELDOUT_TEMPLATES[0])
    scenario["planted_items"][0]["canonical_question"] = question

    validate_scenario(scenario)
    assert (
        list(
            Draft202012Validator(
                load_scenario_schema(), format_checker=FormatChecker()
            ).iter_errors(scenario)
        )
        == []
    )


def test_validator_preserves_punctuation_in_question_identity() -> None:
    scenario = generate_scenario(17, HELDOUT_TEMPLATES[0])
    first_question = next(
        plant["canonical_question"]
        for plant in scenario["planted_items"]
        if "dispatchers " in plant["canonical_question"]
    )
    scenario["planted_items"][1]["canonical_question"] = first_question.replace(
        "dispatchers ", "dispatchers, "
    )

    validate_scenario(scenario)


def test_validator_preserves_semantic_punctuation_in_question_fingerprints() -> None:
    scenario = generate_scenario(17, HELDOUT_TEMPLATES[0])
    scenario["planted_items"][0]["canonical_question"] = (
        "Should the balance threshold be -5?"
    )
    scenario["planted_items"][1]["canonical_question"] = (
        "Should the balance threshold be 5?"
    )

    validate_scenario(scenario)


def test_validator_allows_distinct_canonical_question_substrings() -> None:
    scenario = generate_scenario(17, HELDOUT_TEMPLATES[0])
    scenario["planted_items"][0]["canonical_question"] = "Is offline approval allowed?"
    scenario["planted_items"][1]["canonical_question"] = (
        "Should managers ask, Is offline approval allowed?"
    )

    validate_scenario(scenario)


def test_validator_allows_unlisted_semantic_paraphrases_as_distinct_identities() -> (
    None
):
    scenario = generate_scenario(17, HELDOUT_TEMPLATES[0])
    scenario["planted_items"][0]["canonical_question"] = (
        "Should dispatchers decide which observable conditions require a route pause?"
    )
    scenario["planted_items"][1]["canonical_question"] = (
        "Should dispatchers determine which observable conditions require pausing a route?"
    )

    validate_scenario(scenario)


def test_validator_rejects_alias_that_duplicates_its_canonical_question() -> None:
    scenario = generate_scenario(17, HELDOUT_TEMPLATES[0])
    plant = scenario["planted_items"][0]
    plant["canonical_question_aliases"] = [plant["canonical_question"]]

    with pytest.raises(
        ScenarioValidationError, match="duplicate canonical question or alias"
    ):
        validate_scenario(scenario)


def test_validator_rejects_alias_that_duplicates_another_plant_question() -> None:
    scenario = generate_scenario(17, HELDOUT_TEMPLATES[0])
    first, second = scenario["planted_items"][:2]
    first["canonical_question_aliases"] = [second["canonical_question"]]

    with pytest.raises(ScenarioValidationError, match="duplicate canonical_question"):
        validate_scenario(scenario)


def test_validator_and_schema_reject_non_yes_no_question_alias() -> None:
    scenario = generate_scenario(17, HELDOUT_TEMPLATES[0])
    scenario["planted_items"][0]["canonical_question_aliases"] = [
        "Which conditions require a route pause?"
    ]

    with pytest.raises(
        ScenarioValidationError, match="alias must be a yes/no question"
    ):
        validate_scenario(scenario)
    assert list(
        Draft202012Validator(
            load_scenario_schema(), format_checker=FormatChecker()
        ).iter_errors(scenario)
    )


def test_validator_rejects_compatibility_equivalent_distractor_anchor() -> None:
    scenario = generate_scenario(17, HELDOUT_TEMPLATES[0])
    plant = scenario["planted_items"][0]
    document = next(
        document
        for document in scenario["documents"]
        if document["doc_id"] == plant["doc_id"]
    )
    document["text"] = document["text"].replace(
        plant["anchor_quote"], "Set threshold to 25. Set threshold to 2⁵."
    )
    plant["anchor_quote"] = "Set threshold to 25."
    scenario["distractors"].append(
        {
            "doc_id": plant["doc_id"],
            "anchor_quote": "Set threshold to 2⁵.",
            "reason": "The exponent is explicit.",
        }
    )

    with pytest.raises(
        ScenarioValidationError, match="distractor cannot duplicate a planted anchor"
    ):
        validate_scenario(scenario)


def test_validator_rejects_duplicate_planted_anchor_in_same_document() -> None:
    scenario = generate_scenario(17, HELDOUT_TEMPLATES[0])
    first, second = scenario["planted_items"][:2]
    second["doc_id"] = first["doc_id"]
    second["anchor_quote"] = first["anchor_quote"]

    with pytest.raises(ScenarioValidationError, match="duplicate planted anchor"):
        validate_scenario(scenario)


@pytest.mark.parametrize("item_kind", ["plant", "distractor"])
def test_validator_rejects_noncanonical_anchor_whitespace(item_kind: str) -> None:
    scenario = generate_scenario(17, HELDOUT_TEMPLATES[0])
    items = (
        scenario["planted_items"] if item_kind == "plant" else scenario["distractors"]
    )
    item = items[0]
    document = next(
        document
        for document in scenario["documents"]
        if document["doc_id"] == item["doc_id"]
    )
    spaced_anchor = item["anchor_quote"].replace(" ", "  ", 1)
    document["text"] = document["text"].replace(item["anchor_quote"], spaced_anchor)
    item["anchor_quote"] = spaced_anchor

    with pytest.raises(ScenarioValidationError, match="canonical whitespace"):
        validate_scenario(scenario)


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
    assert schema["$id"] == "urn:commonground:schema:scenario:2"
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
    assert schema["$defs"]["contextEngineSnapshot"]["properties"]["meta"]["properties"][
        "synthetic"
    ] == {"const": False}


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (
            lambda scenario: scenario["planted_items"][0].__setitem__(
                "anchor_quote", "missing"
            ),
            "plant anchor is absent",
        ),
        (
            lambda scenario: scenario["planted_items"][0]["target_stances"].__setitem__(
                scenario["factions"][0]["faction_id"], "disagree"
            ),
            "target_stances do not match faction priors",
        ),
        (
            lambda scenario: scenario["provenance"].__setitem__(
                "generation_mode", "human"
            ),
            "synthetic scenarios require template generation provenance",
        ),
        (
            lambda scenario: scenario["provenance"].__setitem__("seed", 43),
            "scenario_id must match provenance template and seed",
        ),
    ],
)
def test_validator_rejects_broken_answer_key_or_provenance(
    mutation: object, error: str
) -> None:
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
        ("masked_cells", "not-an-array", "requires empty masked_cells"),
        ("held_out", [], "requires empty held_out"),
        ("clusters", {}, "clusters must be a non-empty list"),
        ("stats", [], "stats must be an object"),
        ("meta", [], "meta must be an object"),
    ],
)
def test_human_socket_rejects_schema_invalid_ce_fields(
    field: str, value: object, error: str
) -> None:
    scenario = as_human_scenario(generate_scenario(42, TRAIN_TEMPLATES[0]))
    scenario["human_feedback"][field] = value

    with pytest.raises(ScenarioValidationError, match=error):
        validate_scenario(scenario)


def test_human_socket_rejects_synthetic_embedded_snapshot() -> None:
    scenario = as_human_scenario(generate_scenario(42, TRAIN_TEMPLATES[0]))
    scenario["human_feedback"]["meta"]["synthetic"] = True

    with pytest.raises(ScenarioValidationError, match=r"meta\.synthetic must be false"):
        validate_scenario(scenario)


def test_human_path_still_enforces_prior_derived_target_stances() -> None:
    scenario = as_human_scenario(generate_scenario(42, TRAIN_TEMPLATES[0]))
    first_faction = scenario["factions"][0]["faction_id"]
    scenario["planted_items"][0]["target_stances"][first_faction] = "disagree"

    with pytest.raises(
        ScenarioValidationError, match="target_stances do not match faction priors"
    ):
        validate_scenario(scenario)


def test_schema_and_manual_validator_share_integral_number_semantics() -> None:
    schema_validator = Draft202012Validator(
        load_scenario_schema(), format_checker=FormatChecker()
    )
    scenario = as_human_scenario(generate_scenario(42, TRAIN_TEMPLATES[0]))
    scenario["provenance"]["seed"] = 42.0

    validate_scenario(scenario)
    assert list(schema_validator.iter_errors(scenario)) == []

    mutations = (
        lambda candidate: candidate["provenance"].__setitem__("seed", 42.5),
        lambda candidate: candidate["human_feedback"]["statements"][0].__setitem__(
            "index", 0.5
        ),
        lambda candidate: candidate["human_feedback"]["votes"][1].__setitem__(0, -0.5),
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
        lambda candidate, value: candidate["human_feedback"]["statements"][
            0
        ].__setitem__("index", value),
        lambda candidate, value: candidate["human_feedback"]["votes"][1].__setitem__(
            0, value
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
    schema_validator = Draft202012Validator(
        load_scenario_schema(), format_checker=FormatChecker()
    )
    assert list(schema_validator.iter_errors(scenario))


def as_human_scenario(scenario: dict[str, object]) -> dict[str, object]:
    scenario["organization"]["fictional"] = False
    scenario["persona_panel"] = None
    scenario["provenance"]["synthetic"] = False
    scenario["provenance"]["generation_mode"] = "human"
    participants = [f"p{index:03d}" for index in range(10)]
    votes = [[1] if index % 2 == 0 else [-1] for index in range(10)]
    scenario["human_feedback"] = {
        "session_id": "verified-session",
        "statements": [
            {"index": 0, "text": "Should the proposed interpretation apply?"}
        ],
        "participants": participants,
        "votes": votes,
        "masked_cells": [],
        "held_out": {},
        "clusters": [
            {
                "id": 0,
                "members": participants[:5],
                "member_indices": list(range(5)),
                "center": [],
            },
            {
                "id": 1,
                "members": participants[5:],
                "member_indices": list(range(5, 10)),
                "center": [],
            },
        ],
        "stats": {
            "comment": [
                {
                    "commentIndex": 0,
                    "agrees": 5,
                    "disagrees": 5,
                    "unsure": 0,
                    "total": 10,
                    "responded": 10,
                    "extremity": None,
                    "divisiveness": None,
                }
            ]
        },
        "meta": {
            "synthetic": False,
            "k_anonymity": 5,
            "source": "context-engine-session",
            "seed": 42,
            "consent_scope": "public-benchmark",
            "redistribution_rights_approved": True,
            "schema_version": "commonground-human-snapshot-v2",
            "exporter_version": "1.2.0",
            "source_commit": "a" * 40,
            "privacy_review": {
                "attested": True,
                "reviewed_at": "2026-08-26",
                "checks": [
                    "direct-identifiers",
                    "free-text",
                    "participant-pseudonyms",
                ],
            },
        },
    }
    return scenario


def _planting_pattern(template: object) -> tuple[object, ...]:
    document_positions = {
        document["doc_id"]: index for index, document in enumerate(template.documents)
    }

    def anchor_position(item: dict[str, str]) -> tuple[int, int]:
        document_index = document_positions[item["doc_id"]]
        document_text = template.documents[document_index]["text"]
        sentence_index = document_text[
            : document_text.index(item["anchor_quote"])
        ].count(". ")
        return document_index, sentence_index

    planted = tuple(
        (item["type"], *anchor_position(item)) for item in template.planted_items
    )
    distractors = tuple(anchor_position(item) for item in template.distractors)
    return planted, distractors
