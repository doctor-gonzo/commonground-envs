from __future__ import annotations

import copy
import json
import random
import re

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
from commonground_scenarios.templates import VALUE_DIMENSIONS, DomainTemplate
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
    assert any(
        template.balance_type_neutral_distractors for template in HELDOUT_TEMPLATES
    )


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
        set(plant["decision"])
        == {
            "actor",
            "action",
            "condition",
            "anchor_outcome",
            "alternative_outcome",
        }
        and all(value.strip() for value in plant["decision"].values())
        for plant in scenario["planted_items"]
    )
    assert all(
        isinstance(plant["canonical_question_aliases"], list)
        for plant in scenario["planted_items"]
    )
    assert all(
        set(plant["decision_aliases"]) == set(plant["decision"])
        and all(
            aliases[0] == plant["decision"][field]
            for field, aliases in plant["decision_aliases"].items()
        )
        for plant in scenario["planted_items"]
    )
    assert all(
        "For decisions involving" not in faction["summary"]
        and "leans toward yes" not in faction["summary"]
        and "leans toward no" not in faction["summary"]
        for faction in scenario["factions"]
    )
    assert all(
        any(
            phrase in faction["summary"]
            for phrase in generator_module._VALUE_PRINCIPLES[(dimension, direction)]
        )
        for faction in scenario["factions"]
        for dimension in VALUE_DIMENSIONS
        for direction in [
            (
                "positive"
                if faction["values"][dimension] >= 0.25
                else "negative"
                if faction["values"][dimension] <= -0.25
                else "balanced"
            )
        ]
    )
    assert all(
        all(
            f"{dimension}={float(faction['values'][dimension]):+.2f}"
            in faction["summary"]
            for dimension in VALUE_DIMENSIONS
        )
        for faction in scenario["factions"]
    )
    assert not hasattr(generator_module, "_PRINCIPLE_EXAMPLES")
    assert scenario["human_feedback"] is None
    assert scenario["provenance"]["synthetic"] is True
    assert scenario["provenance"]["template_set"] == template.template_set


@pytest.mark.parametrize(
    "template", ALL_TEMPLATES, ids=lambda template: template.template_id
)
def test_every_contradiction_uses_its_authored_opposing_rule(
    template: DomainTemplate,
) -> None:
    scenario = generate_scenario(8217, template)
    contradiction = next(
        plant for plant in scenario["planted_items"] if plant["type"] == "contradiction"
    )
    authored = next(
        plant for plant in template.planted_items if plant["type"] == "contradiction"
    )
    related = contradiction["related_evidence"]
    related_document = next(
        document
        for document in scenario["documents"]
        if document["doc_id"] == related["doc_id"]
    )
    other_plant_anchors = {
        plant["anchor_quote"]
        for plant in scenario["planted_items"]
        if plant["plant_id"] != contradiction["plant_id"]
    }
    distractor_anchors = {
        distractor["anchor_quote"] for distractor in scenario["distractors"]
    }

    assert related["quote"].startswith(
        authored["related_anchor_quote"].removesuffix(".")
    )
    assert related["quote"] in related_document["text"]
    assert related["doc_id"] != contradiction["doc_id"]
    assert related["quote"] not in other_plant_anchors
    assert related["quote"] not in distractor_anchors


def test_regional_archives_contradiction_uses_phone_photograph_rule() -> None:
    template = next(
        template
        for template in HELDOUT_TEMPLATES
        if template.template_id == "regional-archives-access"
    )
    scenario = generate_scenario(8217, template)
    contradiction = next(
        plant for plant in scenario["planted_items"] if plant["type"] == "contradiction"
    )

    assert contradiction["related_evidence"]["quote"].startswith(
        "The researcher guide permits phone photographs of cited pages"
    )


def test_faction_summaries_do_not_change_when_issue_tradeoffs_reverse() -> None:
    template = copy.deepcopy(HELDOUT_TEMPLATES[0])
    reversed_template = copy.deepcopy(template)
    for plant in reversed_template.planted_items:
        plant["value_weights"] = {
            dimension: -weight for dimension, weight in plant["value_weights"].items()
        }

    original = generate_scenario(7352, template)
    reversed_scenario = generate_scenario(7352, reversed_template)

    assert [faction["summary"] for faction in original["factions"]] == [
        faction["summary"] for faction in reversed_scenario["factions"]
    ]
    assert [plant["alternative_stances"] for plant in original["planted_items"]] != [
        plant["alternative_stances"] for plant in reversed_scenario["planted_items"]
    ]


def test_question_polarity_reversal_inverts_only_agree_and_disagree() -> None:
    scenario = generate_scenario(7352, HELDOUT_TEMPLATES[0])

    for plant in scenario["planted_items"]:
        alternative = plant["alternative_stances"]
        anchor = generator_module.orient_stances(alternative, yes_choice="anchor")
        restored = generator_module.orient_stances(
            alternative, yes_choice="alternative"
        )
        assert restored == alternative
        assert anchor == {
            faction_id: {
                "agree": "disagree",
                "disagree": "agree",
                "pass": "pass",
            }[stance]
            for faction_id, stance in alternative.items()
        }


def test_visible_value_renderer_never_receives_plants_or_stances() -> None:
    factions = copy.deepcopy(list(HELDOUT_TEMPLATES[0].factions))

    generator_module._add_visible_faction_values(random.Random(17), factions)

    assert all(
        "Value profile used for this panel:" in faction["summary"]
        for faction in factions
    )


def test_faction_value_change_is_visible_and_changes_composed_targets() -> None:
    template = copy.deepcopy(HELDOUT_TEMPLATES[0])
    changed_template = copy.deepcopy(template)
    changed_faction = changed_template.factions[2]
    changed_faction["values"]["access"] = -1.0

    original = generate_scenario(7352, template)
    changed = generate_scenario(7352, changed_template)
    original_by_name = {faction["name"]: faction for faction in original["factions"]}
    changed_by_name = {faction["name"]: faction for faction in changed["factions"]}
    faction_name = changed_faction["name"]

    assert (
        original_by_name[faction_name]["summary"]
        != changed_by_name[faction_name]["summary"]
    )
    assert any(
        original_plant["alternative_stances"] != changed_plant["alternative_stances"]
        for original_plant, changed_plant in zip(
            sorted(original["planted_items"], key=lambda plant: plant["type"]),
            sorted(changed["planted_items"], key=lambda plant: plant["type"]),
            strict=True,
        )
    )


def test_additional_heldout_distractors_are_compositional_and_role_neutral() -> None:
    balanced_templates = [
        template
        for template in HELDOUT_TEMPLATES
        if template.balance_type_neutral_distractors
    ]
    sentence_counts_by_role: dict[str, list[int]] = {
        role: []
        for role in (
            "ambiguity",
            "contradiction",
            "gap",
            "related-evidence",
            "neutral",
        )
    }
    count_profiles: list[tuple[int, ...]] = []
    composed_anchors: list[str] = []

    assert len(balanced_templates) >= 10
    assert not hasattr(generator_module, "TYPE_NEUTRAL_DISTRACTORS")
    assert not hasattr(generator_module, "BALANCED_SENTENCE_COUNT")
    for template_index, template in enumerate(balanced_templates):
        for repetition in range(5):
            scenario = generate_scenario(
                9200 + template_index * 5 + repetition,
                template,
            )
            documents = {
                document["doc_id"]: document for document in scenario["documents"]
            }
            sentences_by_doc = {
                doc_id: [
                    sentence
                    for sentence in re.split(r"(?<=[.!?])\s+", document["text"])
                    if sentence
                ]
                for doc_id, document in documents.items()
            }
            plants_by_type = {
                plant["type"]: plant for plant in scenario["planted_items"]
            }
            plant_doc_ids = {plant["doc_id"] for plant in scenario["planted_items"]}
            related_doc_id = plants_by_type["contradiction"]["related_evidence"][
                "doc_id"
            ]
            neutral_doc_id = next(
                iter(set(documents) - plant_doc_ids - {related_doc_id})
            )
            role_doc_ids = {
                **{
                    issue_type: plant["doc_id"]
                    for issue_type, plant in plants_by_type.items()
                },
                "related-evidence": related_doc_id,
                "neutral": neutral_doc_id,
            }
            for role, doc_id in role_doc_ids.items():
                sentence_counts_by_role[role].append(len(sentences_by_doc[doc_id]))
            count_profiles.append(
                tuple(
                    len(sentences_by_doc[role_doc_ids[role]])
                    for role in sorted(role_doc_ids)
                )
            )

            generated_context = [
                distractor
                for distractor in scenario["distractors"]
                if distractor["reason"]
                in {
                    generator_module._COMPOSED_DISTRACTOR_REASON,
                    generator_module.ACTOR_SUPPORT_REASON,
                }
            ]
            composed = [
                distractor
                for distractor in generated_context
                if distractor["reason"] == generator_module._COMPOSED_DISTRACTOR_REASON
            ]
            generated_doc_ids = {
                distractor["doc_id"] for distractor in generated_context
            }
            composed_anchors.extend(
                distractor["anchor_quote"] for distractor in composed
            )

            assert len(documents) == 5
            assert len(plant_doc_ids) == 3
            assert related_doc_id not in plant_doc_ids
            assert generated_doc_ids == set(documents)
            assert all(
                distractor["anchor_quote"] in sentences_by_doc[distractor["doc_id"]]
                for distractor in generated_context
            )
            assert not {
                plant["anchor_quote"] for plant in scenario["planted_items"]
            } & {distractor["anchor_quote"] for distractor in generated_context}
            assert all(
                "; the record " not in sentence
                for sentences in sentences_by_doc.values()
                for sentence in sentences
            )
            assert len({document["style"] for document in documents.values()}) == 5
            assert (
                len(
                    {
                        document["title"].rsplit(" ", maxsplit=1)[0]
                        for document in documents.values()
                    }
                )
                == 5
            )

    # Every semantic role sees the whole supported length range, rather than a
    # type-specific count. Seeded clause composition also avoids a finite list
    # of exact filler sentences across the release-sized matrix.
    for counts in sentence_counts_by_role.values():
        assert set(counts) == set(range(2, 8))
    assert len(set(count_profiles)) >= 80
    assert len(set(composed_anchors)) / len(composed_anchors) >= 0.99


def test_every_accepted_actor_alias_has_authored_or_classified_support() -> None:
    """Actor answer-key concepts must be public evidence, never hidden labels."""

    for template_index, template in enumerate(HELDOUT_TEMPLATES):
        for repetition in range(5):
            scenario = generate_scenario(
                8200 + template_index * 5 + repetition,
                template,
            )
            documents = {
                document["doc_id"]: document["text"].casefold()
                for document in scenario["documents"]
            }
            support = [
                distractor
                for distractor in scenario["distractors"]
                if distractor["reason"] == generator_module.ACTOR_SUPPORT_REASON
            ]
            for plant in scenario["planted_items"]:
                for alias in plant["decision_aliases"]["actor"]:
                    assert alias.casefold() in documents[plant["doc_id"]]
            assert all(
                any(
                    plant["doc_id"] == distractor["doc_id"]
                    and any(
                        alias.casefold() in distractor["anchor_quote"].casefold()
                        for alias in plant["decision_aliases"]["actor"]
                    )
                    for plant in scenario["planted_items"]
                )
                for distractor in support
            )


def test_eval_matrix_has_balanced_anchor_positions_and_no_decision_value_ties() -> None:
    positions_by_type = {
        issue_type: [] for issue_type in ("ambiguity", "contradiction", "gap")
    }
    top_k_ties = 0
    decision_value_gaps: list[float] = []
    for template_index, template in enumerate(HELDOUT_TEMPLATES):
        for repetition in range(5):
            scenario = generate_scenario(
                8200 + template_index * 5 + repetition,
                template,
            )
            documents = {
                document["doc_id"]: document for document in scenario["documents"]
            }
            for plant in scenario["planted_items"]:
                sentences = re.split(
                    r"(?<=[.!?])\s+", documents[plant["doc_id"]]["text"]
                )
                positions_by_type[plant["type"]].append(
                    sentences.index(plant["anchor_quote"])
                )
            utilities = sorted(
                (
                    float(plant["decision_value"])
                    * _panel_disagreement(plant["target_stances"])
                    for plant in scenario["planted_items"]
                ),
                reverse=True,
            )
            boundary_gap = utilities[1] - utilities[2]
            top_k_ties += boundary_gap == 0
            decision_value_gaps.append(boundary_gap)

    assert top_k_ties == 0
    assert min(decision_value_gaps) > 0
    for issue_type, positions in positions_by_type.items():
        counts = {position: positions.count(position) for position in set(positions)}
        assert len(counts) >= 5, issue_type
        assert max(counts.values()) / len(positions) < 0.4, issue_type


@pytest.mark.parametrize("stance_value", [None, True, 1, 1.0, [], {}])
@pytest.mark.parametrize("stance_field", ["alternative_stances", "target_stances"])
def test_validator_rejects_unhashable_or_non_string_stances(
    stance_field: str, stance_value: object
) -> None:
    scenario = generate_scenario(17, HELDOUT_TEMPLATES[0])
    faction_id = scenario["factions"][0]["faction_id"]
    scenario["planted_items"][0][stance_field][faction_id] = stance_value

    with pytest.raises(ScenarioValidationError, match=r"invalid .* stance"):
        validate_scenario(scenario)


def _panel_disagreement(stances: dict[str, str]) -> float:
    counts = {
        stance: list(stances.values()).count(stance) for stance in set(stances.values())
    }
    total = len(stances)
    return 1 - sum((count / total) ** 2 for count in counts.values())


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


def test_validator_rejects_decision_aliases_without_canonical_first() -> None:
    scenario = generate_scenario(17, HELDOUT_TEMPLATES[0])
    plant = scenario["planted_items"][0]
    plant["decision_aliases"]["actor"] = ["an unrelated role"]

    with pytest.raises(
        ScenarioValidationError, match="must begin with the canonical decision field"
    ):
        validate_scenario(scenario)


def test_validator_rejects_hidden_only_actor_aliases() -> None:
    scenario = generate_scenario(17, HELDOUT_TEMPLATES[0])
    plant = scenario["planted_items"][0]
    plant["decision"]["actor"] = "a hidden review council"
    plant["decision_aliases"]["actor"] = ["a hidden review council"]

    with pytest.raises(
        ScenarioValidationError, match="must all be source-observable roles"
    ):
        validate_scenario(scenario)


def test_validator_rejects_any_hidden_actor_alias() -> None:
    """Validation and runtime must enforce the same all-alias visibility rule."""

    scenario = generate_scenario(17, HELDOUT_TEMPLATES[0])
    plant = scenario["planted_items"][0]
    plant["decision_aliases"]["actor"].append("a hidden review council")

    with pytest.raises(
        ScenarioValidationError, match="must all be source-observable roles"
    ):
        validate_scenario(scenario)


def test_community_clinic_actor_alias_is_explicit_and_source_observable() -> None:
    template = next(
        candidate
        for candidate in HELDOUT_TEMPLATES
        if candidate.template_id == "community-clinic-scheduling"
    )
    scenario = generate_scenario(8214, template)
    plant = next(
        item for item in scenario["planted_items"] if item["type"] == "ambiguity"
    )
    all_document_text = " ".join(document["text"] for document in scenario["documents"])

    assert "community health scheduling team" in plant["decision_aliases"]["actor"]
    assert "community health scheduling team" in all_document_text


def _decision_support_tokens(text: str) -> set[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "be",
        "for",
        "in",
        "is",
        "of",
        "or",
        "the",
        "to",
        "when",
        "while",
        "with",
        "without",
    }
    return set(re.findall(r"[^\W_]+", text.casefold())) - stopwords


def test_every_decision_slot_has_a_prompt_supported_alias() -> None:
    """Keep hidden authoring labels from becoming required answer vocabulary."""

    for template_index, template in enumerate(ALL_TEMPLATES):
        for repetition in range(5):
            seed = 8200 + template_index * 5 + repetition
            scenario = generate_scenario(
                seed,
                template,
            )
            visible_prompt = " ".join(
                [
                    *(document["text"] for document in scenario["documents"]),
                    *(
                        f"{faction['name']} {faction['summary']}"
                        for faction in scenario["factions"]
                    ),
                ]
            )
            visible_tokens = _decision_support_tokens(visible_prompt)
            documents_by_id = {
                document["doc_id"]: document["text"]
                for document in scenario["documents"]
            }
            semantic_scope = generator_module.SEMANTIC_SCOPES[
                seed % len(generator_module.SEMANTIC_SCOPES)
            ]
            for plant in scenario["planted_items"]:
                # Actors are concrete roles: at least one accepted form must be
                # literally present in the exact evidence documents consulted
                # by the scorer. Other slots may express a missing-policy
                # alternative, but must still recover a material share of their
                # concept words from the prompt.
                actor_evidence = documents_by_id[plant["doc_id"]]
                if plant["related_evidence"] is not None:
                    actor_evidence = " ".join(
                        (
                            actor_evidence,
                            documents_by_id[plant["related_evidence"]["doc_id"]],
                        )
                    )
                assert any(
                    alias.casefold() in actor_evidence.casefold()
                    for alias in plant["decision_aliases"]["actor"]
                ), (template.template_id, plant["type"], "actor")
                if semantic_scope is not None:
                    assert all(
                        semantic_scope in alias
                        for alias in plant["decision_aliases"]["condition"]
                    ), (template.template_id, plant["type"], "condition-scope")
                for field, aliases in plant["decision_aliases"].items():
                    assert any(
                        (
                            len(_decision_support_tokens(alias) & visible_tokens)
                            / len(_decision_support_tokens(alias))
                        )
                        >= 0.25
                        for alias in aliases
                        if _decision_support_tokens(alias)
                    ), (template.template_id, plant["type"], field)


def test_visible_organization_actor_and_scoped_condition_alias_regressions() -> None:
    housing = generate_scenario(
        8215,
        next(
            template
            for template in HELDOUT_TEMPLATES
            if template.template_id == "cooperative-housing-maintenance"
        ),
    )
    assert housing["scenario_id"] == "cooperative-housing-maintenance-90b73d1b4b5b"
    housing_ambiguity = next(
        plant for plant in housing["planted_items"] if plant["type"] == "ambiguity"
    )
    assert (
        "the cooperative housing maintenance team"
        in housing_ambiguity["decision_aliases"]["actor"]
    )

    clinic = generate_scenario(
        8214,
        next(
            template
            for template in HELDOUT_TEMPLATES
            if template.template_id == "community-clinic-scheduling"
        ),
    )
    scope = "when the affected person cannot provide the usual record"
    assert clinic["scenario_id"] == "community-clinic-scheduling-6c51670a331a"
    assert all(
        scope in alias
        for plant in clinic["planted_items"]
        for alias in plant["decision_aliases"]["condition"]
    )


def test_shared_procedural_vocabulary_is_not_a_distractor_classifier() -> None:
    """Every procedural predicate must occur in both issue and neutral spans."""

    roles_by_predicate: dict[str, set[str]] = {
        predicate: set() for predicate in generator_module._SHARED_PROCEDURAL_PREDICATES
    }
    distractor_clause_counts: set[int] = set()
    for template_index, template in enumerate(HELDOUT_TEMPLATES):
        if not template.balance_type_neutral_distractors:
            continue
        for repetition in range(5):
            scenario = generate_scenario(
                8200 + template_index * 5 + repetition,
                template,
            )
            for role, spans in (
                (
                    "plant",
                    [plant["anchor_quote"] for plant in scenario["planted_items"]],
                ),
                (
                    "distractor",
                    [
                        distractor["anchor_quote"]
                        for distractor in scenario["distractors"]
                    ],
                ),
            ):
                for span in spans:
                    matching_predicates = [
                        predicate
                        for predicate in generator_module._SHARED_PROCEDURAL_PREDICATES
                        if predicate in span
                    ]
                    assert matching_predicates
                    for predicate in matching_predicates:
                        roles_by_predicate[predicate].add(role)
                    if role == "distractor":
                        distractor_clause_counts.add(
                            sum(
                                span.count(predicate)
                                for predicate in matching_predicates
                            )
                        )

    assert roles_by_predicate
    assert all(
        roles == {"plant", "distractor"} for roles in roles_by_predicate.values()
    )
    assert {1, 2} <= distractor_clause_counts


def test_sector_team_marker_cannot_recover_planted_spans() -> None:
    """Keep the broad visible actor from becoming an issue-location codebook."""

    total_plants = 0
    marked_plants = 0
    total_distractors = 0
    marked_distractors = 0
    localization_f1: list[float] = []
    for template_index, template in enumerate(HELDOUT_TEMPLATES):
        for repetition in range(5):
            scenario = generate_scenario(
                8200 + template_index * 5 + repetition,
                template,
            )
            marker = f"the {scenario['organization']['sector']} team"
            plants = scenario["planted_items"]
            distractors = scenario["distractors"]
            plant_hits = sum(
                marker in plant["anchor_quote"].casefold() for plant in plants
            )
            distractor_hits = sum(
                marker in distractor["anchor_quote"].casefold()
                for distractor in distractors
            )
            visible_hits = sum(
                marker in sentence.casefold()
                for document in scenario["documents"]
                for sentence in re.split(r"(?<=[.!?])\s+", document["text"])
                if sentence
            )

            total_plants += len(plants)
            marked_plants += plant_hits
            total_distractors += len(distractors)
            marked_distractors += distractor_hits
            localization_f1.append(2 * plant_hits / (visible_hits + len(plants)))
    plant_rate = marked_plants / total_plants
    distractor_rate = marked_distractors / total_distractors
    assert plant_rate < 0.05
    assert distractor_rate > 0.0
    assert abs(plant_rate - distractor_rate) < 0.10
    assert sum(localization_f1) / len(localization_f1) < 0.05


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


def test_validator_rejects_related_evidence_that_is_another_planted_anchor() -> None:
    scenario = generate_scenario(17, HELDOUT_TEMPLATES[0])
    contradiction = next(
        plant for plant in scenario["planted_items"] if plant["type"] == "contradiction"
    )
    other = next(
        plant
        for plant in scenario["planted_items"]
        if plant["plant_id"] != contradiction["plant_id"]
        and plant["doc_id"] != contradiction["doc_id"]
    )
    contradiction["related_evidence"] = {
        "doc_id": other["doc_id"],
        "quote": other["anchor_quote"],
    }

    with pytest.raises(
        ScenarioValidationError,
        match="related evidence cannot duplicate another planted anchor",
    ):
        validate_scenario(scenario)


def test_validator_rejects_distractor_that_is_related_evidence() -> None:
    scenario = generate_scenario(17, HELDOUT_TEMPLATES[0])
    contradiction = next(
        plant for plant in scenario["planted_items"] if plant["type"] == "contradiction"
    )
    related = contradiction["related_evidence"]
    scenario["distractors"].append(
        {
            "doc_id": related["doc_id"],
            "anchor_quote": related["quote"],
            "reason": "This must remain authored opposing evidence, not a distractor.",
        }
    )

    with pytest.raises(
        ScenarioValidationError,
        match="distractor cannot duplicate contradiction related evidence",
    ):
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
    assert schema["$id"] == "urn:commonground:schema:scenario:5"
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
    assert set(schema["$defs"]["decisionFrame"]["required"]) == {
        "actor",
        "action",
        "condition",
        "anchor_outcome",
        "alternative_outcome",
    }
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
            "target_stances do not match question polarity",
        ),
        (
            lambda scenario: scenario["planted_items"][0].__setitem__(
                "decision_value", 0.5
            ),
            "decision_value does not match preference trade-off",
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


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (
            lambda scenario, value: scenario["factions"][0]["values"].__setitem__(
                "access", value
            ),
            r"factions\[0\]\.values\.access must be finite",
        ),
        (
            lambda scenario, value: scenario["persona_panel"].__setitem__(
                "pass_threshold", value
            ),
            r"persona_panel\.pass_threshold must be finite",
        ),
        (
            lambda scenario, value: scenario["planted_items"][0].__setitem__(
                "decision_value", value
            ),
            r"planted_items\[0\]\.decision_value must be finite",
        ),
        (
            lambda scenario, value: scenario["planted_items"][0][
                "value_weights"
            ].__setitem__("access", value),
            r"planted_items\[0\]\.value_weights\.access must be finite",
        ),
    ],
)
def test_validator_contextualizes_huge_json_integer_errors(
    mutation: object, error: str
) -> None:
    scenario = generate_scenario(42, TRAIN_TEMPLATES[0])
    mutation(scenario, 10**500)
    parsed_scenario = json.loads(json.dumps(scenario))

    with pytest.raises(ScenarioValidationError, match=error):
        validate_scenario(parsed_scenario)


@pytest.mark.parametrize("field", ["template_set", "generation_mode"])
@pytest.mark.parametrize("value", [None, True, 1, 1.0, [], {}])
def test_provenance_enums_reject_non_text_without_raw_type_errors(
    field: str, value: object
) -> None:
    scenario = generate_scenario(42, TRAIN_TEMPLATES[0])
    scenario["provenance"][field] = value

    with pytest.raises(ScenarioValidationError, match=rf"provenance\.{field}"):
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


def test_human_path_still_enforces_value_composed_target_stances() -> None:
    scenario = as_human_scenario(generate_scenario(42, TRAIN_TEMPLATES[0]))
    first_faction = scenario["factions"][0]["faction_id"]
    scenario["planted_items"][0]["target_stances"][first_faction] = "disagree"

    with pytest.raises(
        ScenarioValidationError, match="target_stances do not match question polarity"
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
