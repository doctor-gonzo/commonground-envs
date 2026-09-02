from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path
from typing import Any

import pytest
import tiktoken
import verifiers as legacy_vf
import verifiers.v1 as vf1
from commonground_elicit import (
    ElicitHarness,
    ElicitJsonParser,
    ElicitTaskset,
    finding_f1,
    panel_disagreement,
    question_utility,
    question_utility_score,
)
from commonground_elicit import (
    load_environment as load_legacy_environment,
)
from commonground_elicit import (
    load_taskset as load_environment,
)
from commonground_elicit.environment import (
    BUNDLED_EVAL_PATH,
    BUNDLED_TRAIN_PATH,
    _best_question_assignment,
    _maximum_weight_sum,
    _safe_truncation_index,
    build_document_view,
    match_findings,
    normalized_quote_overlap,
    scenario_to_row,
)
from commonground_scenarios import (
    HELDOUT_TEMPLATES,
    generate_scenario,
    validate_scenario,
)
from commonground_scenarios.generator import ACTOR_SUPPORT_REASON, orient_stances
from commonground_scenarios.templates import VALUE_DIMENSIONS
from commonground_scenarios.validation import (
    PASS_THRESHOLD,
    preference_tradeoff_value,
)
from verifiers.types import State
from verifiers.v1.harnesses.null import NullHarness
from verifiers.v1.utils.loaders import (
    default_harness_id,
    harness_class,
    taskset_class,
)

CANONICAL_TASK_COLUMNS = ("prompt", "answer", "info", "example_id")
QUESTION_RESPONSE_FIELDS = (
    "doc_id",
    "quote",
    "type",
    "question",
    "decision",
    "yes_choice",
    "related_evidence",
    "target_stances",
)
FINDING_RESPONSE_FIELDS = (
    "doc_id",
    "quote",
    "type",
    "diagnosis",
    "decision",
    "related_evidence",
)


@pytest.mark.parametrize("task", ["find", "elicit-ask"])
def test_legacy_hosted_eval_loader_returns_full_environment(task: str) -> None:
    env = legacy_vf.load_environment("commonground-elicit", task=task, split="eval")

    assert isinstance(env, legacy_vf.SingleTurnEnv)
    assert isinstance(load_legacy_environment(task=task), legacy_vf.SingleTurnEnv)
    assert env.env_id == "commonground-elicit"
    assert env.env_args["task"] == task
    assert env.env_args["split"] == "eval"
    assert len(env.get_dataset()) == 100
    assert len(env.get_eval_dataset()) == 100
    assert all(
        callable(getattr(env, method))
        for method in ("set_kwargs", "start_server", "evaluate", "stop_server")
    )

    row = dict(env.get_eval_dataset()[0])
    state = score_row(env, row, correct_response_from_row(row))
    assert 0.0 < state["reward"] <= 1.0


def test_native_plugin_resolution_preserves_taskset_and_pure_chat_harness() -> None:
    assert taskset_class("commonground-elicit") is ElicitTaskset
    assert default_harness_id("commonground-elicit") == "commonground-elicit"
    assert harness_class("commonground-elicit") is ElicitHarness
    assert issubclass(ElicitHarness, NullHarness)


def test_load_environment_builds_default_heldout_rows() -> None:
    env = load_environment()

    assert isinstance(env, ElicitTaskset)
    assert isinstance(env, vf1.Taskset)
    assert env.env_id == "commonground-elicit"
    assert len(env.get_dataset()) == 100
    assert len(env.get_eval_dataset()) == 100
    assert {json.loads(row["info"])["template_set"] for row in env.get_dataset()} == {
        "train"
    }
    assert {
        json.loads(row["info"])["template_set"] for row in env.get_eval_dataset()
    } == {"heldout"}


def test_native_v1_taskset_scores_a_trace_without_the_legacy_bridge() -> None:
    taskset = load_environment()
    task = next(iter(taskset))
    row = taskset.get_eval_dataset()[0]
    response = correct_response_from_row(row)
    trace = vf1.Trace(
        task=vf1.TraceTask(type=type(task).__name__, data=task.data),
        agent=vf1.AgentInfo(config=vf1.AgentConfig()),
        nodes=[
            vf1.MessageNode(
                message=vf1.AssistantMessage(
                    content=json.dumps(response, sort_keys=True)
                ),
                sampled=True,
            )
        ],
    )

    asyncio.run(task.score(trace))

    assert trace.reward == 1.0
    assert trace.rewards["finding_f1"].score == 1.0
    assert trace.metrics["question_utility"] > 0.0


def test_optional_shaped_find_reward_preserves_strict_f1_as_metric() -> None:
    strict_env = load_environment()
    shaped_env = load_environment(reward_mode="shaped")
    row = dict(shaped_env.get_eval_dataset()[0])
    exact = correct_response_from_row(row)
    partial_finding = dict(exact["findings"][0])
    partial_finding["type"] = "gap" if partial_finding["type"] != "gap" else "ambiguity"
    partial_finding["related_evidence"] = None
    response = {
        "findings": [partial_finding],
        "questions": exact["questions"],
    }

    strict_state = score_row(strict_env, row, response)
    shaped_state = score_row(shaped_env, row, response)
    exact_shaped_state = score_row(shaped_env, row, exact)

    assert strict_state["reward"] == 0.0
    assert shaped_state["reward"] == pytest.approx(1 / 8)
    assert shaped_state["metrics"]["finding_f1"] == 0.0
    assert exact_shaped_state["reward"] == 1.0
    assert shaped_env.env_args["reward_mode"] == "shaped"


def test_shaped_reward_strictly_penalizes_false_positive_spam() -> None:
    env = load_environment(reward_mode="shaped")
    row = dict(env.get_eval_dataset()[0])
    exact = correct_response_from_row(row)
    spammed = {
        "findings": [
            *exact["findings"],
            {
                "doc_id": exact["findings"][0]["doc_id"],
                "quote": "A distinct unsupported passage",
                "type": "ambiguity",
                "diagnosis": "Should this unsupported passage create another rule?",
                "related_evidence": None,
            },
        ]
    }

    exact_state = score_row(env, row, exact)
    spammed_state = score_row(env, row, spammed)

    assert exact_state["reward"] == 1.0
    assert spammed_state["reward"] < exact_state["reward"]
    assert spammed_state["metrics"]["finding_f1"] < 1.0


def test_unknown_reward_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown reward_mode"):
        load_environment(reward_mode="dense")


@pytest.mark.parametrize(
    ("split", "expected_path", "expected_rows", "expected_template_set"),
    [
        ("eval", BUNDLED_EVAL_PATH, 100, "heldout"),
        ("train", BUNDLED_TRAIN_PATH, 100, "train"),
    ],
)
def test_named_bundled_splits_resolve_to_packaged_rows(
    split: str,
    expected_path: Path,
    expected_rows: int,
    expected_template_set: str,
) -> None:
    env = load_environment(split=split)

    assert env.env_args["split"] == split
    assert env.env_args["data_path"] == str(expected_path)
    assert len(env.get_eval_dataset()) == expected_rows
    assert {
        json.loads(row["info"])["template_set"] for row in env.get_eval_dataset()
    } == {expected_template_set}


def test_named_eval_split_rows_are_byte_identical_to_default() -> None:
    default_env = load_environment()
    named_env = load_environment(split="eval")
    legacy_paths_env = load_environment(
        data_path=BUNDLED_EVAL_PATH,
        train_data_path=BUNDLED_TRAIN_PATH,
    )

    assert dataset_rows_bytes(named_env.get_dataset()) == dataset_rows_bytes(
        default_env.get_dataset()
    )
    assert dataset_rows_bytes(named_env.get_eval_dataset()) == dataset_rows_bytes(
        default_env.get_eval_dataset()
    )
    assert dataset_rows_bytes(named_env.get_dataset()) == dataset_rows_bytes(
        legacy_paths_env.get_dataset()
    )
    assert dataset_rows_bytes(named_env.get_eval_dataset()) == dataset_rows_bytes(
        legacy_paths_env.get_eval_dataset()
    )


def test_explicit_paths_take_precedence_over_split() -> None:
    env = load_environment(
        data_path=BUNDLED_EVAL_PATH,
        train_data_path=BUNDLED_EVAL_PATH,
        split="train",
    )

    assert env.env_args["data_path"] == str(BUNDLED_EVAL_PATH)
    assert env.env_args["train_data_path"] == str(BUNDLED_EVAL_PATH)
    assert {json.loads(row["info"])["template_set"] for row in env.get_dataset()} == {
        "heldout"
    }
    assert {
        json.loads(row["info"])["template_set"] for row in env.get_eval_dataset()
    } == {"heldout"}


def test_unknown_split_lists_valid_names() -> None:
    with pytest.raises(ValueError) as exc_info:
        load_environment(split="unknown")

    message = str(exc_info.value)
    assert "unknown split 'unknown'" in message
    assert "valid splits: eval, train" in message


@pytest.mark.parametrize("task", ["find", "elicit-ask"])
def test_server_state_path_scores_correct_and_incorrect(task: str) -> None:
    env = load_environment(task=task)
    row = dict(env.get_eval_dataset()[0])
    incorrect = {"questions": []} if task == "elicit-ask" else {"findings": []}

    correct_state = score_row(env, row, correct_response_from_row(row))
    incorrect_state = score_row(env, row, incorrect)

    assert correct_state["reward"] > 0.0
    assert incorrect_state["reward"] == 0.0
    assert set(correct_state["task"]) == set(CANONICAL_TASK_COLUMNS)
    assert "input" not in correct_state


@pytest.mark.parametrize("task", ["find", "elicit-ask"])
def test_all_built_rows_use_only_canonical_server_columns(task: str) -> None:
    env = load_environment(task=task)

    for dataset in (env.get_dataset(), env.get_eval_dataset()):
        for raw_row in dataset:
            row = dict(raw_row)
            info = json.loads(row["info"])

            assert set(row) == set(CANONICAL_TASK_COLUMNS)
            assert "task" not in info
            assert info["task_label"] == task
            assert info["panel_polarization"] == 1.0
            assert info["question_count"] == 1
            assert info["allow_combined_questions"] is (task == "find")


def test_loader_preserves_original_positional_docs_count_argument() -> None:
    env = load_environment(1)

    assert env.env_args["task"] == "find"
    assert env.env_args["docs_count"] == 1


def test_default_bundled_loader_is_repeatable() -> None:
    first_env = load_environment()
    second_env = load_environment()

    assert [dict(row) for row in first_env.get_dataset()] == [
        dict(row) for row in second_env.get_dataset()
    ]
    assert [dict(row) for row in first_env.get_eval_dataset()] == [
        dict(row) for row in second_env.get_eval_dataset()
    ]
    assert BUNDLED_TRAIN_PATH.is_file()
    assert BUNDLED_EVAL_PATH.is_file()


def test_bundled_env_args_round_trip_preserves_both_splits() -> None:
    env = load_environment(docs_count=2, planted_density=0.5)

    reloaded = load_environment(**env.env_args)

    assert env.env_args["data_path"] == str(BUNDLED_EVAL_PATH)
    assert env.env_args["train_data_path"] == str(BUNDLED_TRAIN_PATH)
    assert [dict(row) for row in reloaded.get_dataset()] == [
        dict(row) for row in env.get_dataset()
    ]
    assert [dict(row) for row in reloaded.get_eval_dataset()] == [
        dict(row) for row in env.get_eval_dataset()
    ]


def test_explicit_missing_data_path_is_an_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing.jsonl"

    with pytest.raises(FileNotFoundError, match=r"missing\.jsonl"):
        load_environment(data_path=missing)


def test_prompt_does_not_leak_answer_key_or_faction_priors() -> None:
    scenario = generate_scenario(91, HELDOUT_TEMPLATES[0])
    env = load_environment()
    row = dict(env.get_eval_dataset()[0])
    prompt = row["prompt"][0]["content"]
    answer = json.loads(row["answer"])

    assert "canonical_question" not in prompt
    assert "faction_id" not in prompt
    for plant in scenario["planted_items"]:
        assert plant["canonical_question"] not in prompt
        assert json.dumps(plant["target_stances"], sort_keys=True) not in prompt
    for question in answer["questions"]:
        assert question["question"] not in prompt
        assert json.dumps(question["target_stances"], sort_keys=True) not in prompt


def test_rubric_scores_exact_answer_at_one() -> None:
    env = load_environment()
    row = dict(env.get_eval_dataset()[0])

    state = score_row(env, row, correct_response_from_row(row))

    assert state["reward"] == 1.0
    assert state["metrics"]["finding_f1"] == 1.0
    assert state["metrics"]["finding_diagnosis_recall"] == 1.0
    assert state["metrics"]["finding_relation_recall"] == 1.0
    assert state["metrics"]["question_utility"] > 0.0


def test_find_primary_reward_is_independent_of_optional_companion_questions() -> None:
    env = load_environment()
    row = dict(env.get_eval_dataset()[0])
    exact = correct_response_from_row(row)
    without_questions = {"findings": exact["findings"]}

    combined_state = score_row(env, row, exact)
    findings_only_state = score_row(env, row, without_questions)

    assert combined_state["reward"] == 1.0
    assert findings_only_state["reward"] == 1.0
    assert findings_only_state["metrics"]["finding_f1"] == 1.0
    assert combined_state["metrics"]["question_utility"] > 0.0
    assert findings_only_state["metrics"]["question_utility"] == 0.0


@pytest.mark.parametrize("questions", ["invalid", []])
def test_invalid_or_wrong_k_companion_questions_do_not_erase_find_reward(
    questions: object,
) -> None:
    env = load_environment()
    row = dict(env.get_eval_dataset()[0])
    answer = json.loads(row["answer"])
    response = {
        "findings": [
            {field: finding[field] for field in FINDING_RESPONSE_FIELDS}
            for finding in answer["findings"]
        ],
        "questions": questions,
    }

    state = score_row(env, row, response)

    assert state["reward"] == 1.0
    assert state["metrics"]["finding_f1"] == 1.0
    assert state["metrics"]["question_utility"] == 0.0


def test_extra_well_formed_companion_question_only_fails_companion_metric() -> None:
    env = load_environment()
    row = dict(env.get_eval_dataset()[0])
    response = correct_response_from_row(row)
    response["questions"].append(dict(response["questions"][0]))

    state = score_row(env, row, response)

    assert state["reward"] == 1.0
    assert state["metrics"]["finding_f1"] == 1.0
    assert state["metrics"]["question_utility"] == 0.0


def test_ask_task_builds_same_env_id_with_task_specific_prompt() -> None:
    env = load_environment(task="elicit-ask")
    row = dict(env.get_eval_dataset()[0])
    prompt = row["prompt"][0]["content"]
    planted = planted_questions_from_row(row)

    assert env.env_id == "commonground-elicit"
    assert env.env_args["task"] == "elicit-ask"
    assert "Select and raise exactly 1 clarifying question" in prompt
    assert "agree means yes" in prompt
    assert "copy the exact supporting passage" in prompt.casefold()
    assert "question text is a presentation field" in prompt.casefold()
    assert "Stakeholder factions:" in prompt
    for plant in planted:
        assert plant["question"] not in prompt
        assert json.dumps(plant["target_stances"], sort_keys=True) not in prompt


def test_ask_task_exact_planted_response_is_positive_and_deterministic() -> None:
    env = load_environment(task="elicit-ask")
    row = dict(env.get_eval_dataset()[0])
    answer = correct_response_from_row(row)

    first = score_row(env, row, answer)
    second = score_row(env, row, answer)

    assert 0 < first["reward"] <= 1
    assert first["reward"] == second["reward"]
    assert first["metrics"]["question_utility"] == first["reward"]
    assert first["metrics"]["question_format_valid"] == 1.0
    assert first["metrics"]["question_top1_selection_accuracy"] == 1.0
    assert first["metrics"]["question_grounding_recall"] == 1.0
    assert first["metrics"]["question_grounded_stance_recall"] == 1.0
    assert first["metrics"]["question_evidence_match_recall"] == 1.0
    assert first["metrics"]["question_evidence_matched_stance_accuracy"] == 1.0


def test_ask_diagnostics_separate_format_grounding_and_stance_failures() -> None:
    env = load_environment(task="elicit-ask", question_count=1)
    row = dict(env.get_eval_dataset()[0])
    exact = correct_response_from_row(row)
    wrong_grounding = json.loads(json.dumps(exact))
    wrong_grounding["questions"][0]["doc_id"] = "not-a-document"
    wrong_stance = json.loads(json.dumps(exact))
    faction_id = next(iter(wrong_stance["questions"][0]["target_stances"]))
    original_stance = wrong_stance["questions"][0]["target_stances"][faction_id]
    wrong_stance["questions"][0]["target_stances"][faction_id] = (
        "disagree" if original_stance == "agree" else "agree"
    )
    semantic_error = json.loads(json.dumps(exact))
    semantic_error["questions"][0]["decision"]["actor"] = "an unrelated committee"
    declarative = json.loads(json.dumps(exact))
    declarative["questions"][0]["question"] = "This is not a yes-or-no question."

    malformed = score_row(env, row, {"questions": "invalid"})
    ungrounded = score_row(env, row, wrong_grounding)
    stance_error = score_row(env, row, wrong_stance)
    semantic_failure = score_row(env, row, semantic_error)
    declarative_failure = score_row(env, row, declarative)

    assert malformed["metrics"]["question_format_valid"] == 0.0
    assert malformed["metrics"]["question_top1_selection_accuracy"] == 0.0
    assert declarative_failure["metrics"]["question_format_valid"] == 0.0
    assert declarative_failure["reward"] == 0.0
    assert ungrounded["metrics"]["question_format_valid"] == 1.0
    assert ungrounded["metrics"]["question_grounding_recall"] == 0.0
    assert ungrounded["metrics"]["question_evidence_match_recall"] == 0.0
    assert stance_error["metrics"]["question_grounding_recall"] == 1.0
    assert 0.0 <= stance_error["metrics"]["question_grounded_stance_recall"] < 1.0
    assert (
        0.0
        <= stance_error["metrics"]["question_evidence_matched_stance_accuracy"]
        < 1.0
    )
    assert semantic_failure["metrics"]["question_grounding_recall"] == 0.0
    assert semantic_failure["metrics"]["question_grounded_stance_recall"] == 0.0
    assert semantic_failure["metrics"]["question_evidence_match_recall"] == 1.0
    assert (
        semantic_failure["metrics"]["question_evidence_matched_stance_accuracy"] == 1.0
    )


def test_ask_top1_selection_metric_separates_runner_up_from_component_quality() -> None:
    env = load_environment(task="elicit-ask", question_count=1)
    row = dict(env.get_eval_dataset()[0])
    plants = planted_questions_from_row(row)
    runner_up = {"questions": [candidate_for_plant(plants[1])]}

    state = score_row(env, row, runner_up)

    assert 0.0 < state["reward"] < 1.0
    assert state["metrics"]["question_top1_selection_accuracy"] == 0.0
    assert state["metrics"]["question_evidence_match_recall"] == 1.0
    assert state["metrics"]["question_evidence_matched_stance_accuracy"] == 1.0


@pytest.mark.parametrize("malformed_stance", [None, [], {}, True, 1, 1.0])
def test_malformed_stance_values_fail_closed_without_raising(
    malformed_stance: Any,
) -> None:
    env = load_environment(task="elicit-ask", question_count=1)
    row = dict(env.get_eval_dataset()[0])
    response = correct_response_from_row(row)
    faction_id = next(iter(response["questions"][0]["target_stances"]))
    response["questions"][0]["target_stances"][faction_id] = malformed_stance

    state = score_row(env, row, response)

    assert state["reward"] == 0.0
    assert state["metrics"]["question_format_valid"] == 0.0


@pytest.mark.parametrize("malformed_field", [None, [], {}, True, 1, 1.0])
@pytest.mark.parametrize("task", ["find", "elicit-ask"])
def test_malformed_decision_fields_fail_closed_without_raising(
    malformed_field: Any,
    task: str,
) -> None:
    env = load_environment(task=task, question_count=1)
    row = dict(env.get_eval_dataset()[0])
    response = correct_response_from_row(row)
    response_key = "findings" if task == "find" else "questions"
    response[response_key][0]["decision"]["actor"] = malformed_field

    state = score_row(env, row, response)

    assert state["reward"] == 0.0


def test_ask_task_env_args_round_trip_preserves_task_and_rows() -> None:
    env = load_environment(task="elicit-ask", question_count=1, panel_polarization=0.75)

    reloaded = load_environment(**env.env_args)

    assert reloaded.env_args == env.env_args
    assert [dict(row) for row in reloaded.get_eval_dataset()] == [
        dict(row) for row in env.get_eval_dataset()
    ]


def test_question_count_knob_changes_prompt_info_and_response_size() -> None:
    env = load_environment(task="elicit-ask", question_count=1)
    row = dict(env.get_eval_dataset()[0])
    info = json.loads(row["info"])
    response = correct_response_from_row(row)

    assert (
        "Select and raise exactly 1 clarifying question" in row["prompt"][0]["content"]
    )
    assert info["question_count"] == 1
    assert len(response["questions"]) == 1
    assert score_row(env, row, response)["reward"] > 0


def test_exact_extended_responses_fit_the_frozen_generation_budget() -> None:
    """Keep the expanded 0.6 schema well below the 2,048-token study cap."""

    encoding = tiktoken.get_encoding("cl100k_base")
    token_counts: list[int] = []
    for task in ("find", "elicit-ask"):
        env = load_environment(task=task, question_count=1)
        for row in env.get_eval_dataset():
            compact = json.dumps(
                correct_response_from_row(dict(row)),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            token_counts.append(len(encoding.encode(compact)))

    assert token_counts
    assert max(token_counts) <= 1_024


def test_ask_task_rejects_k_larger_than_available_plants() -> None:
    with pytest.raises(ValueError, match="remove all planted items"):
        load_environment(task="elicit-ask", question_count=4)


def test_panel_polarization_changes_selection_weights_without_changing_exact_maximum() -> (
    None
):
    full_env = load_environment(task="elicit-ask", question_count=1)
    half_env = load_environment(
        task="elicit-ask", question_count=1, panel_polarization=0.5
    )
    full_row = dict(full_env.get_eval_dataset()[0])
    half_row = dict(half_env.get_eval_dataset()[0])
    full_answer = correct_response_from_row(full_row)
    half_answer = correct_response_from_row(half_row)

    full_score = score_row(full_env, full_row, full_answer)["reward"]
    half_score = score_row(half_env, half_row, half_answer)["reward"]

    assert half_score == pytest.approx(full_score) == 1.0


def test_precise_distractor_quote_cannot_receive_planted_question_credit() -> None:
    scenario = generate_scenario(8200, HELDOUT_TEMPLATES[0])
    # Use a document with one plant so docs_length cannot first remove a
    # different, ineligible plant and shift the appended distractor before the
    # requested cut. This keeps the regression focused on partial-anchor
    # truncation rather than on multi-plant document filtering.
    plant = next(
        candidate
        for candidate in scenario["planted_items"]
        if sum(
            item["doc_id"] == candidate["doc_id"] for item in scenario["planted_items"]
        )
        == 1
    )
    document = next(
        document
        for document in scenario["documents"]
        if document["doc_id"] == plant["doc_id"]
    )
    distractor_quote = (
        "Pause a route when conditions become unsafe under the thresholds in "
        "weather matrix W-4."
    )
    document["text"] += f" {distractor_quote}"
    scenario["distractors"].append(
        {
            "doc_id": plant["doc_id"],
            "anchor_quote": distractor_quote,
            "reason": "The named matrix makes the threshold precise.",
        }
    )
    row = scenario_to_row(
        scenario,
        docs_count=None,
        docs_length=None,
        planted_density=1.0,
        distractor_density=1.0,
        panel_polarization=1.0,
        question_count=1,
        task="elicit-ask",
    )
    planted_question = next(
        question
        for question in planted_questions_from_row(row)
        if question["doc_id"] == plant["doc_id"]
        and question["quote"] == plant["anchor_quote"]
    )
    distractor_candidate = candidate_for_plant(planted_question, quote=distractor_quote)

    assert (
        question_utility_score(
            [distractor_candidate],
            [planted_question],
            panel_polarization=1.0,
            question_count=1,
        )
        == 0.0
    )
    assert (
        question_utility_score(
            [candidate_for_plant(planted_question)],
            [planted_question],
            panel_polarization=1.0,
            question_count=1,
        )
        > 0.0
    )

    visible_prefix = "Pause a route when conditions become unsafe under the thresholds"
    truncation_length = document["text"].index(distractor_quote) + len(visible_prefix)
    truncated_row = scenario_to_row(
        scenario,
        docs_count=None,
        docs_length=truncation_length,
        planted_density=1.0,
        distractor_density=1.0,
        panel_polarization=1.0,
        question_count=1,
        task="elicit-ask",
    )
    truncated_plant = next(
        question
        for question in planted_questions_from_row(truncated_row)
        if question["doc_id"] == plant["doc_id"]
        and question["quote"] == plant["anchor_quote"]
    )

    assert visible_prefix not in truncated_plant["document_text"]
    assert (
        question_utility_score(
            [candidate_for_plant(truncated_plant, quote=visible_prefix)],
            [truncated_plant],
            panel_polarization=1.0,
            question_count=1,
        )
        == 0.0
    )


def test_find_task_caps_companion_k_to_visible_plants() -> None:
    env = load_environment(task="find", planted_density=0.3, question_count=3)
    row = dict(env.get_eval_dataset()[0])
    answer = json.loads(row["answer"])
    info = json.loads(row["info"])

    assert info["question_count"] == 1
    assert (
        "may also include exactly 1 optional question object"
        in row["prompt"][0]["content"]
    )
    assert len(answer["questions"]) == 1
    state = score_row(env, row, correct_response_from_row(row))
    assert state["reward"] == 1.0
    assert state["metrics"]["question_utility"] > 0.0


@pytest.mark.parametrize(
    "question",
    [
        "route",
        "Dispatchers decide which observable conditions require a route pause.",
    ],
)
def test_question_must_be_yes_no(
    question: str,
) -> None:
    env = load_environment(task="elicit-ask", question_count=1)
    plant = planted_questions_from_row(dict(env.get_eval_dataset()[0]))[0]

    assert (
        question_utility_score(
            [candidate_for_plant(plant, question=question)],
            [plant],
            panel_polarization=1.0,
            question_count=1,
        )
        == 0.0
    )


def test_stance_accuracy_adds_monotonic_partial_credit() -> None:
    env = load_environment(task="elicit-ask", question_count=1)
    plant = planted_questions_from_row(dict(env.get_eval_dataset()[0]))[0]
    correct = dict(plant["target_stances"])
    one_wrong = dict(correct)
    changed_faction = next(iter(one_wrong))
    one_wrong[changed_faction] = (
        "disagree" if one_wrong[changed_faction] == "agree" else "agree"
    )
    all_wrong = {
        faction_id: {"agree": "disagree", "disagree": "agree", "pass": "agree"}[stance]
        for faction_id, stance in correct.items()
    }

    exact_score = question_utility_score(
        [candidate_for_plant(plant, target_stances=correct)],
        [plant],
        panel_polarization=1.0,
        question_count=1,
    )
    partial_score = question_utility_score(
        [candidate_for_plant(plant, target_stances=one_wrong)],
        [plant],
        panel_polarization=1.0,
        question_count=1,
    )
    grounding_only_score = question_utility_score(
        [candidate_for_plant(plant, target_stances=all_wrong)],
        [plant],
        panel_polarization=1.0,
        question_count=1,
    )

    assert exact_score > partial_score > grounding_only_score > 0.0


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    [
        ("doc_id", "another-document"),
        ("quote", "A fabricated passage not present in the document."),
    ],
)
def test_wrong_document_or_quote_cannot_claim_question_credit(
    field: str,
    wrong_value: str,
) -> None:
    env = load_environment(task="elicit-ask", question_count=1)
    plant = planted_questions_from_row(dict(env.get_eval_dataset()[0]))[0]
    candidate = candidate_for_plant(plant)
    candidate[field] = wrong_value

    assert (
        question_utility_score(
            [candidate], [plant], panel_polarization=1.0, question_count=1
        )
        == 0.0
    )


def test_exact_quote_preserves_semantic_operators() -> None:
    stances = {"operations": "agree", "risk": "disagree", "support": "pass"}
    inverse = {"agree": "disagree", "disagree": "agree", "pass": "pass"}
    plant = {
        "doc_id": "approval-policy",
        "quote": "Approval requires threshold >= 5.",
        "question": "Should approval require a threshold >= 5?",
        "yes_choice": "anchor",
        "target_stances": stances,
        "alternative_stances": {
            faction_id: inverse[stance] for faction_id, stance in stances.items()
        },
        "document_text": "Approval requires threshold >= 5.",
        "decision": {
            "actor": "approval",
            "action": "require threshold",
            "condition": "threshold 5",
            "anchor_outcome": "approval requires threshold 5",
            "alternative_outcome": "approval without threshold",
        },
    }

    assert (
        question_utility_score(
            [candidate_for_plant(plant, quote="Approval requires threshold <= 5.")],
            [plant],
            panel_polarization=1.0,
            question_count=1,
        )
        == 0.0
    )
    assert (
        question_utility_score(
            [candidate_for_plant(plant)],
            [plant],
            panel_polarization=1.0,
            question_count=1,
        )
        > 0.0
    )


def test_question_count_is_strict_for_missing_or_extra_outputs() -> None:
    env = load_environment(task="elicit-ask", question_count=1)
    plant = planted_questions_from_row(dict(env.get_eval_dataset()[0]))[0]
    candidate = candidate_for_plant(plant)

    assert (
        question_utility_score([], [plant], panel_polarization=1.0, question_count=1)
        == 0.0
    )
    assert (
        question_utility_score(
            [candidate, candidate],
            [plant],
            panel_polarization=1.0,
            question_count=1,
        )
        == 0.0
    )


def test_duplicate_grounding_is_penalized_by_one_to_one_assignment() -> None:
    env = load_environment(task="elicit-ask", question_count=2)
    plants = planted_questions_from_row(dict(env.get_eval_dataset()[0]))
    duplicate = candidate_for_plant(plants[0])
    single_score = question_utility_score(
        [duplicate],
        [plants[0]],
        panel_polarization=1.0,
        question_count=1,
    )

    duplicate_score = question_utility_score(
        [duplicate, duplicate],
        plants,
        panel_polarization=1.0,
        question_count=2,
    )

    assert 0.0 < duplicate_score < single_score


def test_distinct_grounded_questions_use_global_one_to_one_assignment() -> None:
    env = load_environment(task="elicit-ask", question_count=2)
    plants = planted_questions_from_row(dict(env.get_eval_dataset()[0]))
    candidates = [candidate_for_plant(plant) for plant in plants[:2]]

    score = question_utility_score(
        candidates,
        plants,
        panel_polarization=1.0,
        question_count=2,
    )

    assert score > 0.0


def test_ask_task_rejects_find_task_combined_root() -> None:
    env = load_environment(task="elicit-ask", question_count=1)
    row = dict(env.get_eval_dataset()[0])
    answer = correct_response_from_row(row)
    combined = {"findings": [], "questions": answer["questions"]}

    assert score_row(env, row, combined)["reward"] == 0.0


def test_ask_parser_rejects_nested_task_object_wrapper() -> None:
    env = load_environment(task="elicit-ask", question_count=1)
    row = dict(env.get_eval_dataset()[0])
    answer = correct_response_from_row(row)
    wrapped = {"findings": [], "wrapper": answer}

    assert score_row(env, row, wrapped)["reward"] == 0.0


def test_ask_parser_rejects_duplicate_root_keys_without_overwriting() -> None:
    env = load_environment(task="elicit-ask", question_count=1)
    row = dict(env.get_eval_dataset()[0])
    answer = correct_response_from_row(row)
    content = (
        '{"questions":[],"questions":'
        + json.dumps(answer["questions"], sort_keys=True)
        + "}"
    )
    completion = [{"role": "assistant", "content": content}]

    score = asyncio.run(
        question_utility(
            completion,
            row["answer"],
            row["info"],
            parser=ElicitJsonParser("questions"),
        )
    )

    assert score == 0.0


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_elicit_parser_rejects_nonstandard_json_constants(constant: str) -> None:
    parser = ElicitJsonParser("questions")

    assert parser.parse(f'{{"questions":{constant}}}') == {}


@pytest.mark.parametrize("wrapper", ["array", "unmatched-prose-brace"])
def test_ask_parser_rejects_wrapped_task_objects(wrapper: str) -> None:
    env = load_environment(task="elicit-ask", question_count=1)
    row = dict(env.get_eval_dataset()[0])
    answer = correct_response_from_row(row)
    content = (
        json.dumps([answer], sort_keys=True)
        if wrapper == "array"
        else "draft { not JSON; final " + json.dumps(answer, sort_keys=True)
    )
    completion = [{"role": "assistant", "content": content}]

    score = asyncio.run(
        question_utility(
            completion,
            row["answer"],
            row["info"],
            parser=ElicitJsonParser("questions"),
        )
    )

    assert score == 0.0


def test_configured_are_question_is_valid_and_scorable(tmp_path: Path) -> None:
    scenario = generate_scenario(8200, HELDOUT_TEMPLATES[0])
    scenario["planted_items"][0]["canonical_question"] = (
        "Are dispatchers responsible for deciding which observable conditions require a route pause?"
    )
    data_path = tmp_path / "configured.jsonl"
    data_path.write_text(json.dumps(scenario, sort_keys=True) + "\n", encoding="utf-8")
    env = load_environment(task="elicit-ask", question_count=1, data_path=data_path)
    row = dict(env.get_eval_dataset()[0])

    assert score_row(env, row, correct_response_from_row(row))["reward"] > 0.0


def test_configured_short_yes_no_question_and_anchor_are_scorable(
    tmp_path: Path,
) -> None:
    scenario = generate_scenario(8200, HELDOUT_TEMPLATES[0])
    plant = scenario["planted_items"][0]
    document = next(
        document
        for document in scenario["documents"]
        if document["doc_id"] == plant["doc_id"]
    )
    document["text"] = document["text"].replace(plant["anchor_quote"], "Pause now.")
    plant["anchor_quote"] = "Pause now."
    plant["canonical_question"] = "Should we pause now?"
    data_path = tmp_path / "short-configured.jsonl"
    data_path.write_text(json.dumps(scenario, sort_keys=True) + "\n", encoding="utf-8")
    env = load_environment(task="elicit-ask", question_count=1, data_path=data_path)
    row = dict(env.get_eval_dataset()[0])

    assert score_row(env, row, correct_response_from_row(row))["reward"] > 0.0


def test_maximum_weight_question_assignment_is_global() -> None:
    assert _maximum_weight_sum([[1.0, 0.9], [0.8, 0.0]]) == pytest.approx(1.7)
    assert _maximum_weight_sum([[0.2, 0.9, 0.1]]) == pytest.approx(0.9)
    assert _maximum_weight_sum([[0.5], [0.8]]) == pytest.approx(0.8)


def test_question_diagnostic_assignment_reuses_one_coverage_first_pairing() -> None:
    assert _best_question_assignment(
        [[True, True], [True, False]],
        [[1, 4], [4, 0]],
        [[4, 4], [4, 0]],
    ) == (2, 8, 8)

    # A conditional metric must first explain both localized questions rather
    # than silently switch to a one-item assignment with better stance credit.
    assert _best_question_assignment(
        [[True, False], [True, True]],
        [[0, 0], [4, 0]],
        [[4, 0], [4, 4]],
    ) == (2, 0, 8)


def test_panel_disagreement_uses_specific_planted_stance_vector() -> None:
    assert panel_disagreement({"a": "agree", "b": "agree", "c": "agree"}) == 0.0
    assert panel_disagreement(
        {"a": "agree", "b": "disagree", "c": "pass"}
    ) == pytest.approx(1.0)


def test_empty_findings_score_zero_on_a_nonempty_task() -> None:
    env = load_environment()
    row = dict(env.get_eval_dataset()[0])

    state = score_row(env, row, {"findings": []})

    assert state["reward"] == 0.0


def test_false_positive_strictly_reduces_precision_and_f1() -> None:
    env = load_environment()
    row = dict(env.get_eval_dataset()[0])
    answer = correct_response_from_row(row)
    answer["findings"].append(
        {
            "doc_id": "not-a-document",
            "quote": "This is a rule.",
            "type": "ambiguity",
            "diagnosis": "Should this rule apply?",
            "decision": answer["findings"][0]["decision"],
            "related_evidence": None,
        }
    )

    score = score_row(env, row, answer)["reward"]

    assert 0 < score < 1


def test_type_hedging_same_span_under_three_types_is_rejected() -> None:
    env = load_environment()
    row = dict(env.get_eval_dataset()[0])
    planted = json.loads(row["answer"])["findings"][0]
    findings = [
        {
            "doc_id": planted["doc_id"],
            "quote": planted["quote"],
            "type": finding_type,
            "diagnosis": planted["diagnosis"],
            "decision": planted["decision"],
            "related_evidence": (
                planted["related_evidence"] if finding_type == "contradiction" else None
            ),
        }
        for finding_type in ("ambiguity", "contradiction", "gap")
    ]

    assert score_row(env, row, {"findings": findings})["reward"] == 0.0


def test_contradiction_requires_second_conflicting_rule() -> None:
    env = load_environment()
    row = dict(env.get_eval_dataset()[0])
    response = correct_response_from_row(row)
    contradiction = next(
        finding
        for finding in response["findings"]
        if finding["type"] == "contradiction"
    )
    contradiction["related_evidence"] = None

    state = score_row(env, row, response)

    assert 0.0 < state["reward"] < 1.0
    assert state["metrics"]["finding_localization_recall"] == 1.0
    assert state["metrics"]["finding_type_accuracy"] == 1.0


def test_contradiction_rejects_broad_second_document_evidence() -> None:
    env = load_environment()
    row = dict(env.get_eval_dataset()[0])
    hidden = json.loads(row["answer"])
    hidden_contradiction = next(
        finding for finding in hidden["findings"] if finding["type"] == "contradiction"
    )
    response = correct_response_from_row(row)
    contradiction = next(
        finding
        for finding in response["findings"]
        if finding["type"] == "contradiction"
    )
    contradiction["related_evidence"]["quote"] = hidden_contradiction[
        "related_document_text"
    ]

    state = score_row(env, row, response)

    assert 0.0 < state["reward"] < 1.0
    assert state["metrics"]["finding_localization_recall"] == 1.0
    assert state["metrics"]["finding_type_accuracy"] == 1.0


def test_ask_rewards_top_k_selection_over_lower_value_issue() -> None:
    env = load_environment(task="elicit-ask", question_count=2)
    plants = planted_questions_from_row(dict(env.get_eval_dataset()[0]))
    top_k = [candidate_for_plant(plant) for plant in plants[:2]]
    lower_value = [candidate_for_plant(plants[1]), candidate_for_plant(plants[2])]

    top_score = question_utility_score(
        top_k, plants, panel_polarization=1.0, question_count=2
    )
    lower_score = question_utility_score(
        lower_value, plants, panel_polarization=1.0, question_count=2
    )

    assert top_score == 1.0
    assert 0.0 < lower_score < top_score


def test_default_k1_target_changes_when_visible_faction_values_change() -> None:
    """The single-question budget must rank the panel's actual value trade-off."""

    access_issue = semantic_contract_plant("ambiguity")
    safety_issue = semantic_contract_plant("gap")
    access_issue["value_weights"] = value_vector(access=1.0)
    safety_issue["value_weights"] = value_vector(safety=1.0)
    access_panel = visible_value_panel(access=1.0, safety=0.0)
    safety_panel = visible_value_panel(access=0.0, safety=1.0)

    first_panel_plants = apply_visible_value_panel(
        [access_issue, safety_issue], access_panel
    )
    second_panel_plants = apply_visible_value_panel(
        [access_issue, safety_issue], safety_panel
    )

    first_access_score = question_utility_score(
        [candidate_for_plant(first_panel_plants[0])],
        first_panel_plants,
        panel_polarization=1.0,
        question_count=1,
    )
    first_safety_score = question_utility_score(
        [candidate_for_plant(first_panel_plants[1])],
        first_panel_plants,
        panel_polarization=1.0,
        question_count=1,
    )
    second_access_score = question_utility_score(
        [candidate_for_plant(second_panel_plants[0])],
        second_panel_plants,
        panel_polarization=1.0,
        question_count=1,
    )
    second_safety_score = question_utility_score(
        [candidate_for_plant(second_panel_plants[1])],
        second_panel_plants,
        panel_polarization=1.0,
        question_count=1,
    )

    assert first_access_score == 1.0
    assert first_safety_score == 0.0
    assert second_access_score == 0.0
    assert second_safety_score == 1.0


def test_ask_renders_every_target_determining_weight_counterfactual() -> None:
    """Accepted stance changes must be visible in Ask but not leak into Find."""

    scenario = generate_scenario(8214, HELDOUT_TEMPLATES[0])
    original_ask = scenario_to_row(
        scenario,
        docs_count=None,
        docs_length=None,
        planted_density=1.0,
        distractor_density=1.0,
        panel_polarization=1.0,
        question_count=1,
        task="elicit-ask",
    )["prompt"][0]["content"]
    original_find = scenario_to_row(
        scenario,
        docs_count=None,
        docs_length=None,
        planted_density=1.0,
        distractor_density=1.0,
        panel_polarization=1.0,
        question_count=1,
        task="find",
    )["prompt"][0]["content"]

    assert (
        "alternative_preference = sum(faction_value[d] * "
        "alternative_tradeoff_weight[d]) / "
        "sum(abs(alternative_tradeoff_weight[d]))" in original_ask
    )
    assert "Scores >= +0.25 favor the alternative" in original_ask
    assert "Rank candidates by utility = decision_value *" in original_ask
    assert "panel_polarization=1 and pass_threshold=0.25" in original_ask

    section = original_ask.split(
        "Candidate decision profiles (unordered; one per possible issue):\n", 1
    )[1].split("\n\nDocuments:", 1)[0]
    profiles = json.loads(section)
    assert len(profiles) == 3
    assert all(
        set(profile) == {"decision", "alternative_tradeoff_weights"}
        and set(profile["decision"])
        == {
            "actor",
            "action",
            "condition",
            "anchor_outcome",
            "alternative_outcome",
        }
        and set(profile["alternative_tradeoff_weights"]) == set(VALUE_DIMENSIONS)
        for profile in profiles
    )

    for plant_index, original_plant in enumerate(scenario["planted_items"]):
        mutated = copy.deepcopy(scenario)
        plant = mutated["planted_items"][plant_index]
        weights = {
            dimension: -float(original_plant["value_weights"][dimension])
            for dimension in VALUE_DIMENSIONS
        }
        plant["value_weights"] = weights
        scale = sum(abs(weight) for weight in weights.values())
        alternative_stances: dict[str, str] = {}
        for faction in mutated["factions"]:
            preference = (
                sum(
                    float(faction["values"][dimension]) * weights[dimension]
                    for dimension in VALUE_DIMENSIONS
                )
                / scale
            )
            alternative_stances[str(faction["faction_id"])] = (
                "agree"
                if preference >= PASS_THRESHOLD
                else "disagree"
                if preference <= -PASS_THRESHOLD
                else "pass"
            )
        plant["alternative_stances"] = alternative_stances
        plant["target_stances"] = orient_stances(
            alternative_stances,
            yes_choice=str(plant["canonical_yes_choice"]),
        )
        plant["decision_value"] = preference_tradeoff_value(
            mutated["factions"], weights
        )
        assert plant["target_stances"] != original_plant["target_stances"]
        validate_scenario(mutated)

        mutated_ask = scenario_to_row(
            mutated,
            docs_count=None,
            docs_length=None,
            planted_density=1.0,
            distractor_density=1.0,
            panel_polarization=1.0,
            question_count=1,
            task="elicit-ask",
        )["prompt"][0]["content"]
        mutated_find = scenario_to_row(
            mutated,
            docs_count=None,
            docs_length=None,
            planted_density=1.0,
            distractor_density=1.0,
            panel_polarization=1.0,
            question_count=1,
            task="find",
        )["prompt"][0]["content"]

        assert mutated_ask != original_ask
        assert mutated_find == original_find


def test_prompt_visible_weight_swap_changes_the_default_top1_target() -> None:
    """Ask ranking must follow the displayed trade-off, not a hidden target."""

    scenario = generate_scenario(8214, HELDOUT_TEMPLATES[0])

    def rank(plants: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(
            plants,
            key=lambda plant: (
                -float(plant["decision_value"])
                * panel_disagreement(plant["target_stances"]),
                str(plant["doc_id"]),
                str(plant["anchor_quote"]),
            ),
        )

    original_rank = rank(scenario["planted_items"])
    original_top_decision = dict(original_rank[0]["decision"])
    mutated = copy.deepcopy(scenario)
    mutated_by_id = {plant["plant_id"]: plant for plant in mutated["planted_items"]}
    top = mutated_by_id[original_rank[0]["plant_id"]]
    bottom = mutated_by_id[original_rank[-1]["plant_id"]]
    top["value_weights"], bottom["value_weights"] = (
        bottom["value_weights"],
        top["value_weights"],
    )

    for plant in (top, bottom):
        weights = {
            dimension: float(plant["value_weights"][dimension])
            for dimension in VALUE_DIMENSIONS
        }
        scale = sum(abs(weight) for weight in weights.values())
        alternative_stances: dict[str, str] = {}
        for faction in mutated["factions"]:
            preference = (
                sum(
                    float(faction["values"][dimension]) * weights[dimension]
                    for dimension in VALUE_DIMENSIONS
                )
                / scale
            )
            alternative_stances[str(faction["faction_id"])] = (
                "agree"
                if preference >= PASS_THRESHOLD
                else "disagree"
                if preference <= -PASS_THRESHOLD
                else "pass"
            )
        plant["alternative_stances"] = alternative_stances
        plant["target_stances"] = orient_stances(
            alternative_stances,
            yes_choice=str(plant["canonical_yes_choice"]),
        )
        plant["decision_value"] = preference_tradeoff_value(
            mutated["factions"],
            weights,
        )

    validate_scenario(mutated)
    mutated_rank = rank(mutated["planted_items"])
    assert mutated_rank[0]["decision"] != original_top_decision

    original_ask = scenario_to_row(
        scenario,
        docs_count=None,
        docs_length=None,
        planted_density=1.0,
        distractor_density=1.0,
        panel_polarization=1.0,
        question_count=1,
        task="elicit-ask",
    )
    mutated_ask = scenario_to_row(
        mutated,
        docs_count=None,
        docs_length=None,
        planted_density=1.0,
        distractor_density=1.0,
        panel_polarization=1.0,
        question_count=1,
        task="elicit-ask",
    )
    original_find = scenario_to_row(
        scenario,
        docs_count=None,
        docs_length=None,
        planted_density=1.0,
        distractor_density=1.0,
        panel_polarization=1.0,
        question_count=1,
        task="find",
    )
    mutated_find = scenario_to_row(
        mutated,
        docs_count=None,
        docs_length=None,
        planted_density=1.0,
        distractor_density=1.0,
        panel_polarization=1.0,
        question_count=1,
        task="find",
    )

    assert original_ask["prompt"] != mutated_ask["prompt"]
    assert (
        json.loads(original_ask["answer"])["questions"][0]["decision"]
        != (json.loads(mutated_ask["answer"])["questions"][0]["decision"])
    )
    assert original_find["prompt"] == mutated_find["prompt"]


def test_paraphrased_quote_cannot_claim_verbatim_document_evidence() -> None:
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

    assert result["true_positive"] == 0
    assert result["f1"] == 0.0


@pytest.mark.parametrize(
    "meaning_changing_quote",
    [
        "¬The threshold is 25 units.",
        "!The threshold is 25 units.",
        "The threshold is ≠25 units.",
        "The threshold is ≤25 units.",
        "The threshold is +25 units.",
        "The threshold is -25 units.",
        "The threshold is \N{MINUS SIGN}25 units.",
        "The threshold is ~25 units.",
        "The threshold is $25 units.",
        "The threshold is 25% units.",
    ],
)
def test_finding_reward_rejects_symbolic_changes_to_exact_grounding(
    meaning_changing_quote: str,
) -> None:
    answer = {
        "findings": [
            {
                "doc_id": "policy",
                "quote": "The threshold is 25 units.",
                "type": "ambiguity",
                "document_text": "The threshold is 25 units.",
            }
        ]
    }
    completion = [
        {
            "role": "assistant",
            "content": json.dumps(
                {
                    "findings": [
                        {
                            "doc_id": "policy",
                            "quote": meaning_changing_quote,
                            "type": "ambiguity",
                        }
                    ]
                }
            ),
        }
    ]

    score = asyncio.run(finding_f1(completion, answer, ElicitJsonParser()))

    assert score == 0.0


def test_alphabetized_anchor_tokens_absent_from_document_score_zero() -> None:
    quote = (
        "Agents may issue a small goodwill credit when a customer has experienced "
        "material inconvenience."
    )
    planted = [
        {
            "doc_id": "service-policy",
            "quote": quote,
            "type": "ambiguity",
            "document_text": f"{quote} Credits are recorded before closure.",
        }
    ]
    word_salad = " ".join(sorted(quote.casefold().replace(".", "").split()))
    candidate = [
        {
            "doc_id": "service-policy",
            "quote": word_salad,
            "type": "ambiguity",
        }
    ]

    result = match_findings(candidate, planted)

    assert result["true_positive"] == 0
    assert result["false_positive"] == 1
    assert result["f1"] == 0.0


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
    document_text = (
        "Requests should receive a prompt and complete response. "
        "A prompt and complete response should resolve each request."
    )
    planted = [
        complete_find_contract(
            {
                "doc_id": "policy",
                "quote": "Requests should receive a prompt and complete response.",
                "type": "ambiguity",
                "document_text": document_text,
            }
        ),
        complete_find_contract(
            {
                "doc_id": "policy",
                "quote": "A prompt and complete response should resolve each request.",
                "type": "ambiguity",
                "document_text": document_text,
            }
        ),
    ]
    candidate = [
        complete_find_contract(
            {
                "doc_id": "policy",
                "quote": "Requests should receive a prompt and complete response.",
                "type": "ambiguity",
            }
        )
    ]

    result = match_findings(candidate, planted)

    assert result["true_positive"] == 1
    assert result["false_negative"] == 1
    assert result["f1"] == pytest.approx(2 / 3)


def test_matching_finds_global_maximum_instead_of_greedy_local_choice() -> None:
    document_text = "alpha beta gamma delta epsilon zeta"
    planted = [
        complete_find_contract(
            {
                "doc_id": "policy",
                "quote": "alpha beta gamma delta epsilon",
                "type": "ambiguity",
                "document_text": document_text,
            }
        ),
        complete_find_contract(
            {
                "doc_id": "policy",
                "quote": "beta gamma delta epsilon zeta",
                "type": "ambiguity",
                "document_text": document_text,
            }
        ),
    ]
    candidates = [
        complete_find_contract(
            {
                "doc_id": "policy",
                "quote": "alpha beta gamma delta epsilon zeta",
                "type": "ambiguity",
            }
        ),
        complete_find_contract(
            {
                "doc_id": "policy",
                "quote": "alpha beta gamma delta epsilon",
                "type": "ambiguity",
            }
        ),
    ]

    result = match_findings(candidates, planted)

    assert result["true_positive"] == 2
    assert result["f1"] == 1.0


@pytest.mark.parametrize(
    ("anchor", "candidate", "expected_true_positive"),
    [
        ("alpha beta", "alpha beta", 1),
        ("alpha beta", "alpha", 0),
        ("alpha beta gamma delta", "beta gamma", 0),
        ("alpha beta gamma delta", "alpha beta gamma", 0),
        ("alpha beta gamma delta epsilon", "beta gamma delta epsilon", 0),
        (
            "alpha beta gamma delta epsilon zeta eta theta iota kappa",
            "gamma delta",
            0,
        ),
        (
            "alpha beta gamma delta epsilon zeta eta theta iota kappa",
            "alpha beta gamma delta epsilon zeta eta theta",
            0,
        ),
        (
            "alpha beta gamma delta",
            "before alpha beta gamma delta after",
            0,
        ),
    ],
)
def test_quote_match_requires_substantial_contiguous_plant_coverage(
    anchor: str,
    candidate: str,
    expected_true_positive: int,
) -> None:
    document_text = f"before {anchor} after"
    planted = [
        complete_find_contract(
            {
                "doc_id": "policy",
                "quote": anchor,
                "type": "ambiguity",
                "document_text": document_text,
            }
        )
    ]

    result = match_findings(
        [
            complete_find_contract(
                {"doc_id": "policy", "quote": candidate, "type": "ambiguity"}
            )
        ],
        planted,
    )

    assert result["true_positive"] == expected_true_positive


def test_tiny_fragment_cannot_turn_partial_output_credit_into_a_second_match() -> None:
    document_text = "alpha beta gamma delta epsilon. lima mike november oscar papa."
    planted = [
        complete_find_contract(
            {
                "doc_id": "policy",
                "quote": "alpha beta gamma delta epsilon",
                "type": "ambiguity",
                "document_text": document_text,
            }
        ),
        complete_find_contract(
            {
                "doc_id": "policy",
                "quote": "lima mike november oscar papa",
                "type": "gap",
                "document_text": document_text,
            }
        ),
    ]
    candidates = [
        complete_find_contract(
            {
                "doc_id": "policy",
                "quote": "alpha beta gamma delta epsilon",
                "type": "ambiguity",
            }
        ),
        complete_find_contract(
            {"doc_id": "policy", "quote": "lima mike", "type": "gap"}
        ),
    ]

    result = match_findings(candidates, planted)

    assert result == {
        "true_positive": 1,
        "false_positive": 1,
        "false_negative": 1,
        "f1": 0.5,
        "localization_recall": 0.5,
        "type_accuracy": 1.0,
        "type_recall": 0.5,
        "diagnosis_recall": 0.5,
        "relation_recall": 0.5,
        "localization_f1": 0.5,
        "type_f1": 0.5,
        "diagnosis_f1": 0.5,
        "relation_f1": 0.5,
    }


def test_duplicate_candidate_is_counted_as_false_positive() -> None:
    planted = [
        complete_find_contract(
            {"doc_id": "policy", "quote": "respond promptly", "type": "ambiguity"}
        )
    ]

    result = match_findings([planted[0], planted[0]], planted)

    assert result["true_positive"] == 1
    assert result["false_positive"] == 1
    assert result["f1"] == pytest.approx(2 / 3)
    assert result["localization_f1"] == pytest.approx(2 / 3)
    assert result["relation_f1"] == pytest.approx(2 / 3)


def test_overlapping_span_hedge_cannot_improve_any_shaped_stage() -> None:
    document_text = "Teams respond promptly when a documented safety risk is active."
    planted = [
        complete_find_contract(
            {
                "doc_id": "policy",
                "quote": "respond promptly when a documented safety risk is active",
                "type": "ambiguity",
                "document_text": document_text,
            }
        )
    ]
    exact = [dict(planted[0])]
    hedged = [
        dict(planted[0]),
        {
            **planted[0],
            "quote": "Teams respond promptly when a documented safety risk is active",
        },
    ]

    exact_result = match_findings(exact, planted)
    hedged_result = match_findings(hedged, planted)

    for component in (
        "localization_f1",
        "type_f1",
        "diagnosis_f1",
        "relation_f1",
    ):
        assert hedged_result[component] < exact_result[component]


def test_quote_overlap_normalizes_unicode_case_and_punctuation() -> None:
    overlap = normalized_quote_overlap(
        "CAFÉ—requests: promptly!", "café requests promptly"
    )

    assert overlap == 1.0


def test_parser_rejects_fenced_json() -> None:
    parser = ElicitJsonParser()

    parsed = parser.parse(
        '```json\n{"findings":[{"doc_id":"p","quote":"q","type":"gap"}]}\n```'
    )

    assert parsed == {}


def test_parser_rejects_multiple_objects_with_findings() -> None:
    parser = ElicitJsonParser()

    parsed = parser.parse(
        '{"findings":[]} then {"findings":[{"doc_id":"p","quote":"q","type":"gap"}]}'
    )

    assert parsed == {}


@pytest.mark.parametrize(
    "content",
    [
        "x" * 32_769,
        "{" + '"findings":' + "[" * 1_500 + "]" * 1_500 + "}",
        "{" * 65,
    ],
)
def test_parser_fails_closed_on_oversized_or_adversarial_nesting(
    content: str,
) -> None:
    parser = ElicitJsonParser()

    assert parser.parse(content) == {}


def test_large_trusted_answer_key_does_not_inherit_completion_size_limit() -> None:
    env = load_environment()
    row = dict(env.get_eval_dataset()[0])
    response = correct_response_from_row(row)
    answer = json.loads(row["answer"])
    answer["trusted_audit_note"] = "x" * 40_000
    row["answer"] = json.dumps(answer, sort_keys=True)

    assert len(row["answer"]) > 32_768
    assert len(json.dumps(response, sort_keys=True)) < 32_768
    assert score_row(env, row, response)["reward"] == 1.0


def test_ask_parser_rejects_multiple_objects_with_questions() -> None:
    parser = ElicitJsonParser("questions")

    parsed = parser.parse(
        '{"questions":[]} then {"questions":[{"doc_id":"p","quote":"q","question":"why?","target_stances":{"a":"agree"}}]}'
    )

    assert parsed == {}


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        '{"questions":"not-a-list"}',
        '{"questions":[{"doc_id":"p","quote":"q","question":"why?"}]}',
        '{"questions":[{"doc_id":"p","quote":"q","question":"why?","target_stances":{}}]}',
        '{"questions":[{"doc_id":"p","quote":"q","question":"why?","target_stances":{"a":"maybe"}}]}',
        '{"questions":[],"extra":true}',
    ],
)
def test_ask_parser_or_strict_schema_failures_score_zero(content: str) -> None:
    planted = [
        {
            "plant_id": "plant",
            "doc_id": "p",
            "quote": "q",
            "question": "why this threshold?",
            "target_stances": {"a": "agree", "b": "disagree"},
        }
    ]
    completion = [{"role": "assistant", "content": content}]

    score = asyncio.run(
        question_utility(
            completion,
            {"questions": planted},
            {
                "panel_polarization": 1.0,
                "question_count": 1,
                "allow_combined_questions": False,
            },
            parser=ElicitJsonParser("questions"),
        )
    )

    assert score == 0.0


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

    score = asyncio.run(
        finding_f1(completion, {"findings": planted}, ElicitJsonParser())
    )

    assert score == 0


def test_strict_schema_failure_scores_zero_even_with_empty_answer_key() -> None:
    completion = [{"role": "assistant", "content": '{"findings":"invalid"}'}]

    score = asyncio.run(finding_f1(completion, {"findings": []}, ElicitJsonParser()))

    assert score == 0


def test_empty_findings_and_empty_answer_key_score_zero() -> None:
    completion = [{"role": "assistant", "content": '{"findings":[]}'}]

    score = asyncio.run(finding_f1(completion, {"findings": []}, ElicitJsonParser()))

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

    assert all(plant["type"] != "contradiction" for plant in visible_plants)
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

    assert 0 < len(env.get_eval_dataset()) <= 100
    assert all(json.loads(row["answer"])["findings"] for row in env.get_eval_dataset())


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
    no_distractor_text = " ".join(
        document["text"] for document in no_distractor_documents
    )

    for distractor in scenario["distractors"]:
        assert distractor["anchor_quote"] in all_text
        if distractor["reason"] == ACTOR_SUPPORT_REASON:
            assert distractor["anchor_quote"] in no_distractor_text
        else:
            assert distractor["anchor_quote"] not in no_distractor_text


@pytest.mark.parametrize("distractor_density", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("docs_length", [None, 400, 800, 1_200])
def test_every_retained_plant_actor_alias_is_visible_across_release_views(
    distractor_density: float,
    docs_length: int | None,
) -> None:
    """Density/truncation controls cannot create hidden semantic answer keys."""

    retained = 0
    for template_index, template in enumerate(HELDOUT_TEMPLATES):
        for repetition in range(5):
            scenario = generate_scenario(
                8200 + template_index * 5 + repetition,
                template,
            )
            documents, visible_plants = build_document_view(
                scenario,
                docs_count=None,
                docs_length=docs_length,
                planted_density=1.0,
                distractor_density=distractor_density,
            )
            text_by_doc = {
                document["doc_id"]: document["text"].casefold()
                for document in documents
            }
            visible_ids = {plant["plant_id"] for plant in visible_plants}
            retained += len(visible_plants)
            for plant in visible_plants:
                assert all(
                    alias.casefold() in text_by_doc[plant["doc_id"]]
                    for alias in plant["decision_aliases"]["actor"]
                )
            for plant in scenario["planted_items"]:
                if (
                    plant["doc_id"] in text_by_doc
                    and plant["plant_id"] not in visible_ids
                ):
                    assert (
                        plant["anchor_quote"].casefold()
                        not in text_by_doc[plant["doc_id"]]
                    )

    assert retained > 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"docs_count": 0},
        {"docs_length": True},
        {"planted_density": 0},
        {"planted_density": float("nan")},
        {"distractor_density": -0.1},
        {"distractor_density": 1.1},
        {"panel_polarization": -0.1},
        {"panel_polarization": 0},
        {"panel_polarization": float("inf")},
        {"question_count": 0},
        {"question_count": True},
        {"task": "other"},
    ],
)
def test_invalid_difficulty_args_are_rejected(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        load_environment(**kwargs)


def dataset_rows_bytes(dataset: Any) -> bytes:
    """Serialize dataset rows canonically for byte-level comparisons."""

    lines = [
        json.dumps(dict(row), sort_keys=True, separators=(",", ":")) for row in dataset
    ]
    return ("\n".join(lines) + "\n").encode()


def score_row(
    env: Any,
    row: dict[str, Any],
    response: dict[str, Any],
) -> dict[str, Any]:
    task = {key: row[key] for key in CANONICAL_TASK_COLUMNS if key in row}
    state = State.for_task(task)
    state["completion"] = [
        {"role": "assistant", "content": json.dumps(response, sort_keys=True)}
    ]
    asyncio.run(env.rubric.score_rollout(state))
    return state


def planted_questions_from_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    return json.loads(row["answer"])["questions"]


def correct_response_from_row(row: dict[str, Any]) -> dict[str, Any]:
    answer = json.loads(row["answer"])
    info = json.loads(row["info"])
    questions = [
        {field: question[field] for field in QUESTION_RESPONSE_FIELDS}
        for question in answer.get("questions", [])[: info["question_count"]]
    ]
    response: dict[str, Any] = {"questions": questions}
    if info["allow_combined_questions"]:
        response["findings"] = [
            {field: finding[field] for field in FINDING_RESPONSE_FIELDS}
            for finding in answer["findings"]
        ]
    return response


def candidate_for_plant(
    plant: dict[str, Any],
    *,
    question: str | None = None,
    quote: str | None = None,
    decision: dict[str, str] | None = None,
    yes_choice: str | None = None,
    target_stances: dict[str, str] | None = None,
) -> dict[str, Any]:
    default_decision = plant.get(
        "decision",
        {
            "actor": plant["question"],
            "action": plant["question"],
            "condition": plant["question"],
            "anchor_outcome": plant.get("quote", "preserve the rule"),
            "alternative_outcome": (
                plant.get("related_evidence", {}).get("quote", "")
                if isinstance(plant.get("related_evidence"), dict)
                else ""
            )
            or plant["question"],
        },
    )
    return {
        "doc_id": plant["doc_id"],
        "quote": plant["quote"] if quote is None else quote,
        "type": plant.get("type", "ambiguity"),
        "question": plant["question"] if question is None else question,
        "decision": dict(default_decision if decision is None else decision),
        "yes_choice": plant.get("yes_choice", "alternative")
        if yes_choice is None
        else yes_choice,
        "related_evidence": plant.get("related_evidence"),
        "target_stances": (
            plant["target_stances"] if target_stances is None else target_stances
        ),
    }


def complete_find_contract(item: dict[str, Any]) -> dict[str, Any]:
    """Attach the smallest complete authored Find contract to a span fixture."""

    decision = {
        "actor": "the policy owner",
        "action": "clarify the documented decision",
        "condition": "the documented condition applies",
        "anchor_outcome": "retain the documented outcome",
        "alternative_outcome": "adopt the clarified alternative",
    }
    return {
        **item,
        "diagnosis": "Should the policy owner clarify the documented decision?",
        "decision": decision,
        "decision_aliases": {field: [value] for field, value in decision.items()},
    }


def semantic_contract_plant(issue_type: str) -> dict[str, Any]:
    """Return a compact authored oracle for decision-semantics regressions."""

    stances = {
        "operations": "agree",
        "safety": "agree",
        "access": "disagree",
    }
    inverse = {"agree": "disagree", "disagree": "agree", "pass": "pass"}
    alternative_stances = {
        faction_id: inverse[stance] for faction_id, stance in stances.items()
    }
    if issue_type == "ambiguity":
        quote = "Dispatchers pause a route whenever conditions may be unsafe."
        question = (
            "Should dispatchers pause a route when conditions may be unsafe "
            "(yes selects the anchor outcome: pause the route when conditions "
            "are unsafe)?"
        )
        return {
            "doc_id": "route-policy",
            "quote": quote,
            "type": issue_type,
            "question": question,
            "diagnosis": question,
            "decision": {
                "actor": "dispatchers",
                "action": "pause a route",
                "condition": "conditions may be unsafe",
                "anchor_outcome": "pause the route when conditions are unsafe",
                "alternative_outcome": (
                    "define observable conditions before pausing the route"
                ),
            },
            "decision_aliases": {
                "actor": ["dispatchers", "route dispatchers"],
                "action": ["pause a route", "pause unsafe routes"],
                "condition": [
                    "conditions may be unsafe",
                    "travel conditions may be unsafe",
                ],
                "anchor_outcome": [
                    "pause the route when conditions are unsafe",
                    "pause unsafe routes",
                ],
                "alternative_outcome": [
                    "define observable conditions before pausing the route",
                    "define observable unsafe conditions before a route pause",
                ],
            },
            "yes_choice": "anchor",
            "related_evidence": None,
            "target_stances": stances,
            "alternative_stances": alternative_stances,
            "decision_value": 1.0,
            "document_text": (
                f"{quote} The guide does not define observable unsafe conditions "
                "before a route pause."
            ),
        }
    if issue_type == "contradiction":
        quote = (
            "The duty coordinator may transfer a delayed load to any available "
            "route after hours."
        )
        related_quote = (
            "Only the assigned dispatcher may transfer the delayed load after hours."
        )
        question = (
            "Should the duty coordinator transfer a delayed load after hours "
            "(yes selects the anchor outcome: the duty coordinator transfers the "
            "load to an available route)?"
        )
        return {
            "doc_id": "after-hours-guide",
            "quote": quote,
            "type": issue_type,
            "question": question,
            "diagnosis": question,
            "decision": {
                "actor": "duty coordinator",
                "action": "transfer a delayed load",
                "condition": "after hours when the assigned dispatcher is unavailable",
                "anchor_outcome": (
                    "the duty coordinator transfers the delayed load to an "
                    "available route"
                ),
                "alternative_outcome": (
                    "only the assigned dispatcher transfers the delayed load"
                ),
            },
            "decision_aliases": {
                "actor": ["duty coordinator", "on-duty coordinator"],
                "action": ["transfer a delayed load", "transfer delayed loads"],
                "condition": [
                    "after hours when the assigned dispatcher is unavailable",
                    "after hours while the assigned dispatcher is unavailable",
                ],
                "anchor_outcome": [
                    "the duty coordinator transfers the delayed load to an available route",
                    "the coordinator transfers the delayed load to an available route",
                ],
                "alternative_outcome": [
                    "only the assigned dispatcher transfers the delayed load",
                    "the assigned dispatcher alone transfers the delayed load",
                ],
            },
            "yes_choice": "anchor",
            "related_evidence": {
                "doc_id": "dispatcher-rule",
                "quote": related_quote,
            },
            "target_stances": stances,
            "alternative_stances": alternative_stances,
            "decision_value": 1.0,
            "document_text": quote,
            "related_document_text": related_quote,
        }
    if issue_type == "gap":
        quote = "Staff mail every resident notice to the verified address on file."
        question = (
            "Should staff use a non-digital delivery channel when a resident has "
            "no verified address (yes selects the alternative outcome: use a "
            "non-digital delivery channel without a verified address)?"
        )
        return {
            "doc_id": "notice-policy",
            "quote": quote,
            "type": issue_type,
            "question": question,
            "diagnosis": question,
            "decision": {
                "actor": "staff",
                "action": "deliver resident notices",
                "condition": "a resident has no verified address",
                "anchor_outcome": "mail notices to the verified address on file",
                "alternative_outcome": (
                    "deliver a notice through a non-digital channel without a "
                    "verified address"
                ),
            },
            "decision_aliases": {
                "actor": ["staff", "notice staff"],
                "action": ["deliver resident notices"],
                "condition": [
                    "a resident has no verified address",
                    "a resident lacks a verified address",
                ],
                "anchor_outcome": ["mail notices to the verified address on file"],
                "alternative_outcome": [
                    "deliver a notice through a non-digital channel without a verified address",
                    "use a non-digital delivery channel without a verified address",
                ],
            },
            "yes_choice": "alternative",
            "related_evidence": None,
            "target_stances": stances,
            "alternative_stances": stances,
            "decision_value": 1.0,
            "document_text": (
                f"{quote} Staff deliver resident notices through the documented "
                "channel. No procedure states whether staff may use a non-digital "
                "delivery channel without a verified address."
            ),
        }
    raise ValueError(f"unsupported issue type: {issue_type}")


def semantic_contract_candidate(
    issue_type: str,
    plant: dict[str, Any],
) -> dict[str, Any]:
    """Author a natural answer independently from ``plant['decision']``."""

    if issue_type == "ambiguity":
        decision = {
            "actor": "the route dispatchers",
            "action": "pause unsafe routes",
            "condition": "whenever travel conditions may be unsafe",
            "anchor_outcome": "pause unsafe routes",
            "alternative_outcome": (
                "define observable unsafe conditions before a route pause"
            ),
        }
        question = (
            "Should route dispatchers pause unsafe routes unless observable unsafe "
            "conditions are defined (yes selects the anchor outcome: pause unsafe "
            "routes)?"
        )
    elif issue_type == "contradiction":
        decision = {
            "actor": "the on-duty coordinator",
            "action": "transfer delayed loads",
            "condition": "after hours while the assigned dispatcher is unavailable",
            "anchor_outcome": (
                "the coordinator transfers the delayed load to an available route"
            ),
            "alternative_outcome": (
                "the assigned dispatcher alone transfers the delayed load"
            ),
        }
        question = (
            "Should the on-duty coordinator transfer delayed loads after hours "
            "rather than wait for the assigned dispatcher (yes selects the anchor "
            "outcome: the coordinator transfers the delayed load to an available "
            "route)?"
        )
    elif issue_type == "gap":
        decision = {
            "actor": "notice staff",
            "action": "deliver resident notices",
            "condition": "when a resident lacks a verified address",
            "anchor_outcome": "mail notices to the verified address on file",
            "alternative_outcome": (
                "use a non-digital delivery channel without a verified address"
            ),
        }
        question = (
            "Should notice staff use a non-digital delivery channel for a resident "
            "who lacks a verified address (yes selects the alternative outcome: "
            "use a non-digital delivery channel without a verified address)?"
        )
    else:
        raise ValueError(f"unsupported issue type: {issue_type}")
    return candidate_for_plant(plant, question=question, decision=decision)


def value_vector(**overrides: float) -> dict[str, float]:
    """Build a complete panel value vector for a ranking regression."""

    unknown = set(overrides) - set(VALUE_DIMENSIONS)
    if unknown:
        raise ValueError(f"unknown value dimensions: {sorted(unknown)}")
    return {
        dimension: float(overrides.get(dimension, 0.0))
        for dimension in VALUE_DIMENSIONS
    }


def visible_value_panel(*, access: float, safety: float) -> list[dict[str, Any]]:
    """Render one symmetric panel whose public values emphasize one dimension."""

    factions: list[dict[str, Any]] = []
    for index, sign in enumerate((1.0, -1.0, 0.8, -0.8)):
        values = value_vector(access=sign * access, safety=sign * safety)
        exact_profile = ", ".join(
            f"{dimension}={float(values[dimension])!r}"
            for dimension in VALUE_DIMENSIONS
        )
        factions.append(
            {
                "faction_id": f"panel-{index}",
                "values": values,
                "summary": f"Value profile used for this panel: {exact_profile}.",
            }
        )
    return factions


def apply_visible_value_panel(
    plants: list[dict[str, Any]],
    factions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Recompute the hidden labels that deterministically follow public values."""

    updated = json.loads(json.dumps(plants))
    for plant in updated:
        weights = plant["value_weights"]
        scale = sum(abs(float(weights[dimension])) for dimension in VALUE_DIMENSIONS)
        alternative_stances: dict[str, str] = {}
        for faction in factions:
            preference = (
                sum(
                    float(faction["values"][dimension]) * float(weights[dimension])
                    for dimension in VALUE_DIMENSIONS
                )
                / scale
            )
            stance = (
                "agree"
                if preference >= PASS_THRESHOLD
                else "disagree"
                if preference <= -PASS_THRESHOLD
                else "pass"
            )
            alternative_stances[faction["faction_id"]] = stance
        plant["alternative_stances"] = alternative_stances
        plant["target_stances"] = orient_stances(
            alternative_stances,
            yes_choice=plant["yes_choice"],
        )
        plant["decision_value"] = preference_tradeoff_value(factions, weights)
    return updated
