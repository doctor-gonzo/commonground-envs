from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
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
    _maximum_weight_sum,
    _safe_truncation_index,
    build_document_view,
    match_findings,
    normalized_quote_overlap,
    scenario_to_row,
)
from commonground_scenarios import HELDOUT_TEMPLATES, generate_scenario
from verifiers.types import State
from verifiers.v1.harnesses.null import NullHarness
from verifiers.v1.utils.loaders import (
    default_harness_id,
    harness_class,
    taskset_class,
)

CANONICAL_TASK_COLUMNS = ("prompt", "answer", "info", "example_id")
QUESTION_RESPONSE_FIELDS = ("doc_id", "quote", "question", "target_stances")


@pytest.mark.parametrize("task", ["find", "elicit-ask"])
def test_legacy_hosted_eval_loader_returns_full_environment(task: str) -> None:
    env = legacy_vf.load_environment("commonground-elicit", task=task, split="eval")

    assert isinstance(env, legacy_vf.SingleTurnEnv)
    assert isinstance(load_legacy_environment(task=task), legacy_vf.SingleTurnEnv)
    assert env.env_id == "commonground-elicit"
    assert env.env_args["task"] == task
    assert env.env_args["split"] == "eval"
    assert len(env.get_dataset()) == 40
    assert len(env.get_eval_dataset()) == 20
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
    assert len(env.get_dataset()) == 40
    assert len(env.get_eval_dataset()) == 20
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


@pytest.mark.parametrize(
    ("split", "expected_path", "expected_rows", "expected_template_set"),
    [
        ("eval", BUNDLED_EVAL_PATH, 20, "heldout"),
        ("train", BUNDLED_TRAIN_PATH, 40, "train"),
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
            assert info["question_count"] == 3
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
        assert all(alias not in prompt for alias in question["question_aliases"])
        assert json.dumps(question["target_stances"], sort_keys=True) not in prompt


def test_rubric_scores_exact_answer_at_one() -> None:
    env = load_environment()
    row = dict(env.get_eval_dataset()[0])

    state = score_row(env, row, correct_response_from_row(row))

    assert state["reward"] == 1.0
    assert state["metrics"]["finding_f1"] == 1.0
    assert state["metrics"]["question_utility"] > 0.0


def test_find_task_companion_question_metric_is_reachable_without_affecting_reward() -> (
    None
):
    env = load_environment()
    row = dict(env.get_eval_dataset()[0])
    exact = correct_response_from_row(row)
    without_questions = {"findings": exact["findings"]}

    combined_state = score_row(env, row, exact)
    findings_only_state = score_row(env, row, without_questions)

    assert combined_state["metrics"]["question_utility"] > 0.0
    assert findings_only_state["metrics"]["question_utility"] == 0.0
    assert combined_state["reward"] == findings_only_state["reward"] == 1.0


@pytest.mark.parametrize("questions", ["invalid", []])
def test_invalid_or_wrong_k_companion_questions_do_not_zero_t1(
    questions: object,
) -> None:
    env = load_environment()
    row = dict(env.get_eval_dataset()[0])
    answer = json.loads(row["answer"])
    response = {
        "findings": [
            {field: finding[field] for field in ("doc_id", "quote", "type")}
            for finding in answer["findings"]
        ],
        "questions": questions,
    }

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
    assert "Raise exactly 3 clarifying questions" in prompt
    assert "agree means yes" in prompt
    assert "copy the exact supporting passage" in prompt.casefold()
    assert "reuse at least one informative word" in prompt.casefold()
    assert "Stakeholder factions:" in prompt
    for plant in planted:
        assert plant["question"] not in prompt
        assert all(alias not in prompt for alias in plant["question_aliases"])
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

    assert "Raise exactly 1 clarifying questions" in row["prompt"][0]["content"]
    assert info["question_count"] == 1
    assert len(response["questions"]) == 1
    assert score_row(env, row, response)["reward"] > 0


def test_ask_task_rejects_k_larger_than_available_plants() -> None:
    with pytest.raises(ValueError, match="remove all planted items"):
        load_environment(task="elicit-ask", question_count=4)


def test_panel_polarization_scales_question_reward() -> None:
    full_env = load_environment(task="elicit-ask", question_count=1)
    half_env = load_environment(
        task="elicit-ask", question_count=1, panel_polarization=0.5
    )
    full_row = dict(full_env.get_eval_dataset()[0])
    half_row = dict(half_env.get_eval_dataset()[0])
    answer = correct_response_from_row(full_row)

    full_score = score_row(full_env, full_row, answer)["reward"]
    half_score = score_row(half_env, half_row, answer)["reward"]

    assert half_score == pytest.approx(full_score / 2)


def test_planted_specific_question_strictly_beats_generic_divisiveness() -> None:
    env = load_environment(task="elicit-ask", question_count=1)
    row = dict(env.get_eval_dataset()[0])
    plant = planted_questions_from_row(row)[0]
    targeted = {
        "doc_id": plant["doc_id"],
        "quote": plant["quote"],
        "question": plant["question"],
        "target_stances": plant["target_stances"],
    }
    generic = {
        **targeted,
        "question": "Should we have rules at all?",
    }
    targeted_paraphrase = {
        **targeted,
        "question": plant["question_aliases"][0],
    }

    targeted_score = question_utility_score(
        [targeted], [plant], panel_polarization=1.0, question_count=1
    )
    generic_score = question_utility_score(
        [generic], [plant], panel_polarization=1.0, question_count=1
    )
    paraphrase_score = question_utility_score(
        [targeted_paraphrase],
        [plant],
        panel_polarization=1.0,
        question_count=1,
    )

    assert targeted_score >= paraphrase_score > generic_score
    assert generic_score == 0.0


def test_precise_distractor_quote_cannot_receive_planted_question_credit() -> None:
    scenario = generate_scenario(8200, HELDOUT_TEMPLATES[0])
    plant = scenario["planted_items"][0]
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
    planted_question = planted_questions_from_row(row)[0]
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
    truncated_plant = planted_questions_from_row(truncated_row)[0]

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
    assert "Return exactly 1 question objects" in row["prompt"][0]["content"]
    assert len(answer["questions"]) == 1
    state = score_row(env, row, correct_response_from_row(row))
    assert state["reward"] == 1.0
    assert state["metrics"]["question_utility"] > 0.0


@pytest.mark.parametrize(
    "question",
    [
        "route",
        "Dispatchers decide which observable conditions require a route pause.",
        "Should we have rules at all?",
        "Should this policy apply?",
    ],
)
def test_question_must_be_yes_no_and_reference_grounding(
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


@pytest.mark.parametrize(
    "question",
    [
        "Should dispatchers choose the observable conditions under which a route must pause?",
        "Should route conditions determine when operations pause?",
    ],
)
def test_unlisted_grounded_paraphrases_receive_credit(question: str) -> None:
    env = load_environment(task="elicit-ask", question_count=1)
    plant = planted_questions_from_row(dict(env.get_eval_dataset()[0]))[0]
    assert question not in {plant["question"], *plant["question_aliases"]}

    score = question_utility_score(
        [candidate_for_plant(plant, question=question)],
        [plant],
        panel_polarization=1.0,
        question_count=1,
    )

    assert score > 0.0


def test_hidden_canonical_wording_does_not_affect_generated_question_score() -> None:
    env = load_environment(task="elicit-ask", question_count=1)
    plant = planted_questions_from_row(dict(env.get_eval_dataset()[0]))[0]
    candidate = candidate_for_plant(
        plant,
        question="Should route conditions determine when operations pause?",
    )
    rewritten_oracle = {
        **plant,
        "question": "Is unrelated hidden wording allowed?",
        "question_aliases": ["Does unrelated hidden wording apply?"],
    }

    original = question_utility_score(
        [candidate], [plant], panel_polarization=1.0, question_count=1
    )
    rewritten = question_utility_score(
        [candidate], [rewritten_oracle], panel_polarization=1.0, question_count=1
    )

    assert rewritten == pytest.approx(original)
    assert rewritten > 0.0


def test_baseline_style_paraphrase_regression_is_scorable() -> None:
    plant = {
        "doc_id": "listing-rules",
        "quote": "Trust reviewers may remove a misleading listing immediately.",
        "question": "Should immediate trust removal override review?",
        "question_aliases": [],
        "target_stances": {
            "buyers": "agree",
            "creators": "disagree",
            "trust": "agree",
        },
        "document_text": (
            "Removed listing records retain the cited rule. "
            "Trust reviewers may remove a misleading listing immediately."
        ),
    }
    candidate = candidate_for_plant(
        plant,
        question=(
            "Is immediate removal by trust reviewers allowed before a creator "
            "can appeal?"
        ),
        target_stances={
            "buyers": "pass",
            "creators": "disagree",
            "trust": "agree",
        },
    )

    score = question_utility_score(
        [candidate], [plant], panel_polarization=1.0, question_count=1
    )

    assert (
        0.0
        < score
        < question_utility_score(
            [candidate_for_plant(plant)],
            [plant],
            panel_polarization=1.0,
            question_count=1,
        )
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
    plant = {
        "doc_id": "approval-policy",
        "quote": "Approval requires threshold >= 5.",
        "question": "Should approval require a threshold >= 5?",
        "question_aliases": [],
        "target_stances": stances,
        "document_text": "Approval requires threshold >= 5.",
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

    assert duplicate_score == pytest.approx(single_score / 2)


def test_distinct_grounded_questions_use_global_one_to_one_assignment() -> None:
    env = load_environment(task="elicit-ask", question_count=2)
    plants = planted_questions_from_row(dict(env.get_eval_dataset()[0]))
    candidates = [
        candidate_for_plant(
            plant,
            question=(
                f"Should {' '.join(plant['quote'].rstrip('.').split()[:3])} apply?"
            ),
        )
        for plant in plants[:2]
    ]

    score = question_utility_score(
        candidates,
        plants,
        panel_polarization=1.0,
        question_count=2,
    )

    assert score > 0.0


def test_question_needs_an_informative_quote_token() -> None:
    env = load_environment(task="elicit-ask", question_count=1)
    plant = planted_questions_from_row(dict(env.get_eval_dataset()[0]))[0]
    no_token = candidate_for_plant(plant, question="Should this policy apply?")
    grounded = candidate_for_plant(
        plant,
        question="Should this route apply?",
    )

    assert (
        question_utility_score(
            [no_token], [plant], panel_polarization=1.0, question_count=1
        )
        == 0.0
    )
    assert (
        question_utility_score(
            [grounded], [plant], panel_polarization=1.0, question_count=1
        )
        > 0.0
    )


def test_short_anchor_without_informative_tokens_is_not_scorable() -> None:
    plant = {
        "doc_id": "policy",
        "quote": "Is it?",
        "question": "Is it?",
        "question_aliases": [],
        "target_stances": {"operations": "agree", "risk": "disagree"},
        "document_text": "Is it?",
    }

    assert (
        question_utility_score(
            [candidate_for_plant(plant)],
            [plant],
            panel_polarization=1.0,
            question_count=1,
        )
        == 0.0
    )


def test_ask_task_rejects_find_task_combined_root() -> None:
    env = load_environment(task="elicit-ask", question_count=1)
    row = dict(env.get_eval_dataset()[0])
    answer = correct_response_from_row(row)
    combined = {"findings": [], "questions": answer["questions"]}

    assert score_row(env, row, combined)["reward"] == 0.0


def test_ask_parser_extracts_nested_task_object_like_predict() -> None:
    env = load_environment(task="elicit-ask", question_count=1)
    row = dict(env.get_eval_dataset()[0])
    answer = correct_response_from_row(row)
    wrapped = {"findings": [], "wrapper": answer}

    assert score_row(env, row, wrapped)["reward"] > 0.0


@pytest.mark.parametrize("wrapper", ["array", "unmatched-prose-brace"])
def test_ask_parser_recovers_wrapped_task_objects_like_predict(wrapper: str) -> None:
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

    assert score > 0.0


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
        {"doc_id": "not-a-document", "quote": "This is a rule.", "type": "ambiguity"}
    )

    score = score_row(env, row, answer)["reward"]

    assert 0 < score < 1


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
        {
            "doc_id": "policy",
            "quote": "Requests should receive a prompt and complete response.",
            "type": "ambiguity",
            "document_text": document_text,
        },
        {
            "doc_id": "policy",
            "quote": "A prompt and complete response should resolve each request.",
            "type": "ambiguity",
            "document_text": document_text,
        },
    ]
    candidate = [
        {
            "doc_id": "policy",
            "quote": "Requests should receive a prompt and complete response.",
            "type": "ambiguity",
        }
    ]

    result = match_findings(candidate, planted)

    assert result["true_positive"] == 1
    assert result["false_negative"] == 1
    assert result["f1"] == pytest.approx(2 / 3)


def test_matching_finds_global_maximum_instead_of_greedy_local_choice() -> None:
    document_text = "alpha beta gamma delta epsilon zeta"
    planted = [
        {
            "doc_id": "policy",
            "quote": "alpha beta gamma delta epsilon",
            "type": "ambiguity",
            "document_text": document_text,
        },
        {
            "doc_id": "policy",
            "quote": "beta gamma delta epsilon zeta",
            "type": "ambiguity",
            "document_text": document_text,
        },
    ]
    candidates = [
        {
            "doc_id": "policy",
            "quote": "alpha beta gamma delta epsilon zeta",
            "type": "ambiguity",
        },
        {
            "doc_id": "policy",
            "quote": "alpha beta gamma delta",
            "type": "ambiguity",
        },
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
        ("alpha beta gamma delta epsilon", "beta gamma delta epsilon", 1),
        (
            "alpha beta gamma delta epsilon zeta eta theta iota kappa",
            "gamma delta",
            0,
        ),
        (
            "alpha beta gamma delta epsilon zeta eta theta iota kappa",
            "alpha beta gamma delta epsilon zeta eta theta",
            1,
        ),
        (
            "alpha beta gamma delta",
            "before alpha beta gamma delta after",
            1,
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
        {
            "doc_id": "policy",
            "quote": anchor,
            "type": "ambiguity",
            "document_text": document_text,
        }
    ]

    result = match_findings(
        [{"doc_id": "policy", "quote": candidate, "type": "ambiguity"}],
        planted,
    )

    assert result["true_positive"] == expected_true_positive


def test_tiny_fragment_cannot_turn_partial_output_credit_into_a_second_match() -> None:
    document_text = "alpha beta gamma delta epsilon. lima mike november oscar papa."
    planted = [
        {
            "doc_id": "policy",
            "quote": "alpha beta gamma delta epsilon",
            "type": "ambiguity",
            "document_text": document_text,
        },
        {
            "doc_id": "policy",
            "quote": "lima mike november oscar papa",
            "type": "gap",
            "document_text": document_text,
        },
    ]
    candidates = [
        {
            "doc_id": "policy",
            "quote": "alpha beta gamma delta epsilon",
            "type": "ambiguity",
        },
        {"doc_id": "policy", "quote": "lima mike", "type": "gap"},
    ]

    result = match_findings(candidates, planted)

    assert result == {
        "true_positive": 1,
        "false_positive": 1,
        "false_negative": 1,
        "f1": 0.5,
    }


def test_duplicate_candidate_is_counted_as_false_positive() -> None:
    planted = [{"doc_id": "policy", "quote": "respond promptly", "type": "ambiguity"}]

    result = match_findings([planted[0], planted[0]], planted)

    assert result["true_positive"] == 1
    assert result["false_positive"] == 1
    assert result["f1"] == pytest.approx(2 / 3)


def test_quote_overlap_normalizes_unicode_case_and_punctuation() -> None:
    overlap = normalized_quote_overlap(
        "CAFÉ—requests: promptly!", "café requests promptly"
    )

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


def test_ask_parser_prefers_last_object_with_questions() -> None:
    parser = ElicitJsonParser("questions")

    parsed = parser.parse(
        '{"questions":[]} then {"questions":[{"doc_id":"p","quote":"q","question":"why?","target_stances":{"a":"agree"}}]}'
    )

    assert len(parsed["questions"]) == 1


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

    assert 0 < len(env.get_eval_dataset()) <= 20
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
            {field: finding[field] for field in ("doc_id", "quote", "type")}
            for finding in answer["findings"]
        ]
    return response


def candidate_for_plant(
    plant: dict[str, Any],
    *,
    question: str | None = None,
    quote: str | None = None,
    target_stances: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "doc_id": plant["doc_id"],
        "quote": plant["quote"] if quote is None else quote,
        "question": plant["question"] if question is None else question,
        "target_stances": (
            plant["target_stances"] if target_stances is None else target_stances
        ),
    }
