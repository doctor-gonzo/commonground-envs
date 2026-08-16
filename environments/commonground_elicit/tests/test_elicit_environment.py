from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
import verifiers as vf
from commonground_scenarios import HELDOUT_TEMPLATES, generate_scenario

from commonground_elicit import ElicitJsonParser, finding_f1, load_environment
from commonground_elicit.environment import (
    BUNDLED_EVAL_PATH,
    _safe_truncation_index,
    build_document_view,
    match_findings,
    normalized_quote_overlap,
)


def test_load_environment_builds_default_heldout_rows() -> None:
    env = load_environment()

    assert isinstance(env, vf.SingleTurnEnv)
    assert env.env_id == "commonground-elicit"
    assert len(env.get_eval_dataset()) == 4
    assert {json.loads(row["info"])["template_set"] for row in env.get_eval_dataset()} == {
        "heldout"
    }


def test_default_loader_is_repeatable_before_split_is_bundled() -> None:
    first = [dict(row) for row in load_environment().get_eval_dataset()]
    second = [dict(row) for row in load_environment().get_eval_dataset()]

    assert first == second
    assert not BUNDLED_EVAL_PATH.exists()


def test_prebundle_env_args_round_trip_preserves_default_fallback() -> None:
    env = load_environment(docs_count=2, planted_density=0.5)

    reloaded = load_environment(**env.env_args)

    assert env.env_args["data_path"] is None
    assert [dict(row) for row in reloaded.get_eval_dataset()] == [
        dict(row) for row in env.get_eval_dataset()
    ]


def test_explicit_missing_data_path_is_an_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing.jsonl"

    with pytest.raises(FileNotFoundError, match="missing.jsonl"):
        load_environment(data_path=missing)


def test_prompt_does_not_leak_answer_key_or_faction_priors() -> None:
    scenario = generate_scenario(91, HELDOUT_TEMPLATES[0])
    env = load_environment()
    prompt = dict(env.get_eval_dataset()[0])["prompt"][0]["content"]

    assert "target_stances" not in prompt
    assert "canonical_question" not in prompt
    assert "faction_id" not in prompt
    for plant in scenario["planted_items"]:
        assert plant["canonical_question"] not in prompt


def test_rubric_scores_exact_answer_at_one() -> None:
    env = load_environment()
    row = dict(env.get_eval_dataset()[0])
    answer = json.loads(row["answer"])

    state = score_row(env, row, answer)

    assert state["reward"] == 1.0
    assert state["metrics"]["finding_f1"] == 1.0


def test_empty_findings_score_zero_on_a_nonempty_task() -> None:
    env = load_environment()
    row = dict(env.get_eval_dataset()[0])

    state = score_row(env, row, {"findings": []})

    assert state["reward"] == 0.0


def test_false_positive_strictly_reduces_precision_and_f1() -> None:
    env = load_environment()
    row = dict(env.get_eval_dataset()[0])
    answer = json.loads(row["answer"])
    answer["findings"].append(
        {"doc_id": "not-a-document", "quote": "This is a rule.", "type": "ambiguity"}
    )

    score = score_row(env, row, answer)["reward"]

    assert 0 < score < 1


def test_paraphrased_quote_matches_when_document_and_type_match() -> None:
    planted = [
        {
            "doc_id": "service-policy",
            "quote": "Agents may issue a small goodwill credit when a customer has experienced material inconvenience.",
            "type": "ambiguity",
        }
    ]
    candidates = [
        {
            "doc_id": "service-policy",
            "quote": "Agents can issue a goodwill credit after material customer inconvenience.",
            "type": "ambiguity",
        }
    ]

    result = match_findings(candidates, planted)

    assert result == {
        "true_positive": 1,
        "false_positive": 0,
        "false_negative": 0,
        "f1": 1.0,
    }


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    [("doc_id", "another-doc"), ("type", "gap")],
)
def test_wrong_document_or_type_cannot_match(field: str, wrong_value: str) -> None:
    planted = [{"doc_id": "policy", "quote": "respond promptly", "type": "ambiguity"}]
    candidate = dict(planted[0])
    candidate[field] = wrong_value

    result = match_findings([candidate], planted)

    assert result["true_positive"] == 0
    assert result["f1"] == 0


def test_overlapping_plants_use_one_to_one_matching() -> None:
    planted = [
        {
            "doc_id": "policy",
            "quote": "Requests should receive a prompt and complete response.",
            "type": "ambiguity",
        },
        {
            "doc_id": "policy",
            "quote": "A prompt and complete response should resolve each request.",
            "type": "ambiguity",
        },
    ]
    candidate = [
        {
            "doc_id": "policy",
            "quote": "Each request should receive a prompt and complete response.",
            "type": "ambiguity",
        }
    ]

    result = match_findings(candidate, planted)

    assert result["true_positive"] == 1
    assert result["false_negative"] == 1
    assert result["f1"] == pytest.approx(2 / 3)


def test_matching_finds_global_maximum_instead_of_greedy_local_choice() -> None:
    planted = [
        {
            "doc_id": "policy",
            "quote": "alpha beta gamma delta",
            "type": "ambiguity",
        },
        {
            "doc_id": "policy",
            "quote": "alpha beta gamma epsilon zeta eta theta",
            "type": "ambiguity",
        },
    ]
    candidates = [
        {
            "doc_id": "policy",
            "quote": "alpha beta gamma delta",
            "type": "ambiguity",
        },
        {
            "doc_id": "policy",
            "quote": "gamma delta",
            "type": "ambiguity",
        },
    ]

    result = match_findings(candidates, planted)

    assert result["true_positive"] == 2
    assert result["f1"] == 1.0


def test_duplicate_candidate_is_counted_as_false_positive() -> None:
    planted = [{"doc_id": "policy", "quote": "respond promptly", "type": "ambiguity"}]

    result = match_findings([planted[0], planted[0]], planted)

    assert result["true_positive"] == 1
    assert result["false_positive"] == 1
    assert result["f1"] == pytest.approx(2 / 3)


def test_quote_overlap_normalizes_unicode_case_and_punctuation() -> None:
    overlap = normalized_quote_overlap("CAFÉ—requests: promptly!", "café requests promptly")

    assert overlap == 1.0


def test_parser_handles_fenced_json() -> None:
    parser = ElicitJsonParser()

    parsed = parser.parse(
        '```json\n{"findings":[{"doc_id":"p","quote":"q","type":"gap"}]}\n```'
    )

    assert parsed["findings"][0]["type"] == "gap"


def test_parser_prefers_last_object_with_findings() -> None:
    parser = ElicitJsonParser()

    parsed = parser.parse(
        '{"findings":[]} then {"findings":[{"doc_id":"p","quote":"q","type":"gap"}]}'
    )

    assert len(parsed["findings"]) == 1


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        '{"findings":"not-a-list"}',
        '{"findings":[{"doc_id":"p","quote":"q"}]}',
        '{"findings":[{"doc_id":"p","quote":"q","type":"other"}]}',
        '{"findings":[],"extra":true}',
    ],
)
def test_parser_or_strict_schema_failures_score_zero(content: str) -> None:
    planted = [{"doc_id": "p", "quote": "q", "type": "gap"}]
    completion = [{"role": "assistant", "content": content}]

    score = asyncio.run(finding_f1(completion, planted, ElicitJsonParser()))

    assert score == 0


def test_strict_schema_failure_scores_zero_even_with_empty_answer_key() -> None:
    completion = [{"role": "assistant", "content": '{"findings":"invalid"}'}]

    score = asyncio.run(finding_f1(completion, [], ElicitJsonParser()))

    assert score == 0


def test_empty_findings_and_empty_answer_key_score_zero() -> None:
    completion = [{"role": "assistant", "content": '{"findings":[]}'}]

    score = asyncio.run(finding_f1(completion, [], ElicitJsonParser()))

    assert score == 0


def test_docs_count_and_planted_density_change_visible_answer() -> None:
    full_row = dict(load_environment().get_eval_dataset()[0])
    reduced_row = dict(
        load_environment(docs_count=2, planted_density=0.5).get_eval_dataset()[0]
    )

    full_info = json.loads(full_row["info"])
    reduced_info = json.loads(reduced_row["info"])
    assert full_info["document_count"] == 3
    assert full_info["plant_count"] == 3
    assert reduced_info["document_count"] == 2
    assert reduced_info["plant_count"] == 1


def test_reduced_planted_density_removes_omitted_issues_from_prompt() -> None:
    scenario = generate_scenario(42, HELDOUT_TEMPLATES[0])
    documents, visible_plants = build_document_view(
        scenario,
        docs_count=None,
        docs_length=None,
        planted_density=1 / 3,
        distractor_density=1.0,
    )
    visible_text = " ".join(document["text"] for document in documents)
    visible_plant_ids = {plant["plant_id"] for plant in visible_plants}

    assert len(visible_plant_ids) == 1
    for plant in scenario["planted_items"]:
        assert (plant["anchor_quote"] in visible_text) == (
            plant["plant_id"] in visible_plant_ids
        )


def test_docs_count_removes_contradiction_without_all_supporting_documents() -> None:
    scenario = generate_scenario(8200, HELDOUT_TEMPLATES[0])
    documents, visible_plants = build_document_view(
        scenario,
        docs_count=1,
        docs_length=None,
        planted_density=1.0,
        distractor_density=1.0,
    )
    prompt_text = " ".join(document["text"] for document in documents)
    contradiction = next(
        plant for plant in scenario["planted_items"] if plant["type"] == "contradiction"
    )

    assert {plant["type"] for plant in visible_plants} == {"gap"}
    assert contradiction["anchor_quote"] not in prompt_text


def test_docs_length_removes_contradiction_when_context_is_truncated() -> None:
    scenario = generate_scenario(8200, HELDOUT_TEMPLATES[0])
    documents, visible_plants = build_document_view(
        scenario,
        docs_count=None,
        docs_length=80,
        planted_density=1.0,
        distractor_density=1.0,
    )
    prompt_text = " ".join(document["text"] for document in documents)
    contradiction = next(
        plant for plant in scenario["planted_items"] if plant["type"] == "contradiction"
    )

    assert all(plant["type"] != "contradiction" for plant in visible_plants)
    assert contradiction["anchor_quote"] not in prompt_text


def test_docs_length_never_exposes_a_partial_unlabelled_plant_anchor() -> None:
    scenario = generate_scenario(8200, HELDOUT_TEMPLATES[0])
    documents, visible_plants = build_document_view(
        scenario,
        docs_count=None,
        docs_length=80,
        planted_density=1.0,
        distractor_density=1.0,
    )
    text_by_doc = {document["doc_id"]: document["text"] for document in documents}
    visible_plant_ids = {plant["plant_id"] for plant in visible_plants}

    for plant in scenario["planted_items"]:
        if plant["plant_id"] in visible_plant_ids:
            continue
        anchor_words = plant["anchor_quote"].split()
        for prefix_length in range(1, len(anchor_words)):
            anchor_prefix = " ".join(anchor_words[:prefix_length])
            assert anchor_prefix not in text_by_doc[plant["doc_id"]]


def test_safe_truncation_reaches_fixed_point_for_overlapping_anchors() -> None:
    text = "prefix AAA BBB CCC DDD EEE suffix"
    anchors = ["AAA BBB CCC DDD", "CCC DDD EEE"]

    truncation_index = _safe_truncation_index(text, 24, anchors)

    assert truncation_index == len("prefix ")
    truncated = text[:truncation_index].rstrip() + "…"
    assert "AAA" not in truncated
    assert "CCC" not in truncated


def test_loader_rejects_difficulty_view_with_no_planted_items() -> None:
    with pytest.raises(ValueError, match="remove all planted items"):
        load_environment(docs_length=20)


def test_loader_filters_individual_plant_free_rows() -> None:
    env = load_environment(docs_count=1)

    assert 0 < len(env.get_eval_dataset()) <= 4
    assert all(json.loads(row["planted_items"]) for row in env.get_eval_dataset())


def test_distractor_density_removes_hidden_near_miss_passages() -> None:
    scenario = generate_scenario(42, HELDOUT_TEMPLATES[0])

    all_documents, _ = build_document_view(
        scenario,
        docs_count=None,
        docs_length=None,
        planted_density=1.0,
        distractor_density=1.0,
    )
    no_distractor_documents, _ = build_document_view(
        scenario,
        docs_count=None,
        docs_length=None,
        planted_density=1.0,
        distractor_density=0.0,
    )
    all_text = " ".join(document["text"] for document in all_documents)
    no_distractor_text = " ".join(document["text"] for document in no_distractor_documents)

    for distractor in scenario["distractors"]:
        assert distractor["anchor_quote"] in all_text
        assert distractor["anchor_quote"] not in no_distractor_text


@pytest.mark.parametrize(
    "kwargs",
    [
        {"docs_count": 0},
        {"docs_length": True},
        {"planted_density": 0},
        {"planted_density": float("nan")},
        {"distractor_density": -0.1},
        {"distractor_density": 1.1},
    ],
)
def test_invalid_difficulty_args_are_rejected(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        load_environment(**kwargs)


def score_row(
    env: vf.SingleTurnEnv,
    row: dict[str, Any],
    response: dict[str, Any],
) -> dict[str, Any]:
    state = {
        "prompt": row["prompt"],
        "completion": [
            {"role": "assistant", "content": json.dumps(response, sort_keys=True)}
        ],
        "input": row,
    }
    asyncio.run(env.rubric.score_rollout(state))
    return state
