from __future__ import annotations

import asyncio
import json
from math import isclose
from pathlib import Path
from typing import Any

import pytest
import verifiers.v1 as vf
from commonground_predict import PredictionJsonParser, load_environment
from commonground_predict.environment import (
    BUNDLED_CE_DEMO_PATH,
    BUNDLED_EVAL_PATH,
    BUNDLED_TRAIN_PATH,
    DATA_ENV_VAR,
    CommonGroundPredictTaskset,
    PredictionTask,
    apply_masked_vote_count,
    brier,
    vote_accuracy,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "commonground_predict" / "data"


def test_load_environment_builds_bundled_split() -> None:
    env = load_environment()

    assert isinstance(env, CommonGroundPredictTaskset)
    assert isinstance(env, vf.Taskset)
    assert env.config.id == "commonground-predict"
    assert len(list(env)) == 20
    assert all(isinstance(task, PredictionTask) for task in env)


@pytest.mark.parametrize(
    ("split", "expected_path", "expected_rows", "masked_vote_count"),
    [
        ("eval", BUNDLED_EVAL_PATH, 20, None),
        ("train", BUNDLED_TRAIN_PATH, 150, None),
        ("ce-demo", BUNDLED_CE_DEMO_PATH, 1, 3),
    ],
)
def test_named_bundled_splits_resolve_to_packaged_rows(
    split: str,
    expected_path: Path,
    expected_rows: int,
    masked_vote_count: int | None,
) -> None:
    env = load_environment(split=split, masked_vote_count=masked_vote_count)

    assert env.config.split == split
    assert env.config.data_path is None
    assert expected_path.is_file()
    assert len(list(env)) == expected_rows


def test_named_eval_split_rows_are_byte_identical_to_default() -> None:
    default_env = load_environment()
    named_env = load_environment(split="eval")
    legacy_path_env = load_environment(data_path=BUNDLED_EVAL_PATH)

    assert dataset_rows_bytes(task_rows(named_env)) == dataset_rows_bytes(
        task_rows(default_env)
    )
    assert dataset_rows_bytes(task_rows(named_env)) == dataset_rows_bytes(
        task_rows(legacy_path_env)
    )


def test_explicit_data_path_takes_precedence_over_split() -> None:
    env = load_environment(data_path=BUNDLED_EVAL_PATH, split="train")

    assert env.config.data_path == BUNDLED_EVAL_PATH
    assert len(list(env)) == 20


def test_unknown_split_lists_valid_names() -> None:
    with pytest.raises(ValueError) as exc_info:
        list(load_environment(split="unknown"))

    message = str(exc_info.value)
    assert "unknown split 'unknown'" in message
    assert "valid splits: eval, train, ce-demo" in message


def test_server_state_path_binds_answer_for_correct_and_incorrect() -> None:
    env = load_environment()
    row = task_rows(env)[0]
    answer = row["answer"]
    incorrect = {cell_id: wrong_vote(vote) for cell_id, vote in answer.items()}

    correct_state = score_row(env, row, answer)
    incorrect_state = score_row(env, row, incorrect)

    assert correct_state["reward"] == 1.0
    assert correct_state["rewards"]["vote_accuracy"] == 1.0
    assert incorrect_state["reward"] == 0.0
    assert incorrect_state["rewards"]["vote_accuracy"] == 0.0
    assert correct_state["task"]["answer"] == answer
    assert "held_out" not in correct_state["task"]["snapshot"]


def test_all_built_rows_use_standard_answer_contract() -> None:
    env = load_environment()

    for task in env:
        row = task.data.model_dump(mode="json")
        answer = row["answer"]
        info = row["info"]
        snapshot = row["snapshot"]

        assert "held_out" not in snapshot
        assert info["masked_vote_count"] == len(answer)
        assert all(
            snapshot["votes"][participant_index][statement_index] is None
            for participant_index, statement_index in snapshot["masked_cells"]
        )


def test_bundled_eval_path_is_inside_import_package() -> None:
    import commonground_predict.environment as environment_module

    expected = (
        Path(environment_module.__file__).resolve().parent
        / "data"
        / "eval_synthetic.jsonl"
    )

    assert expected == BUNDLED_EVAL_PATH
    assert BUNDLED_EVAL_PATH.is_file()


def test_load_environment_builds_ce_demo_split_from_env(monkeypatch: Any) -> None:
    data_path = DATA_DIR / "eval_ce_demo.jsonl"
    monkeypatch.setenv(DATA_ENV_VAR, str(data_path))

    env = load_environment(masked_vote_count=3, split="train")
    row = task_rows(env)[0]
    held_out = row["answer"]
    info = row["info"]
    snapshot = row["snapshot"]

    assert len(list(env)) == 1
    assert info["synthetic"] is True
    assert snapshot["meta"]["source"] == "ce-demo-authored"
    assert info["cluster_count"] == 2
    assert len(snapshot["participants"]) == 62
    assert len(snapshot["statements"]) == 30
    assert len(snapshot["votes"]) == 62
    assert {len(row) for row in snapshot["votes"]} == {30}
    assert len(held_out) == 3
    assert env.config.data_path is None
    state = score_row(env, row, held_out)
    assert state["reward"] == 1.0
    assert state["rewards"]["vote_accuracy"] == 1.0


def test_load_environment_rejects_unmasked_ce_demo_by_default() -> None:
    data_path = DATA_DIR / "eval_ce_demo.jsonl"

    with pytest.raises(ValueError) as exc_info:
        list(load_environment(data_path=data_path))

    message = str(exc_info.value)
    assert str(data_path) in message
    assert "masked_vote_count=N" in message
    assert "pre-masked data" in message


def test_load_environment_rejects_transposed_votes(tmp_path: Path) -> None:
    snapshot = orientation_snapshot()
    snapshot["votes"] = [
        list(statement_votes)
        for statement_votes in zip(*snapshot["votes"], strict=True)
    ]
    data_path = write_snapshot_jsonl(tmp_path, snapshot)

    with pytest.raises(ValueError) as exc_info:
        list(load_environment(data_path=data_path))

    message = str(exc_info.value)
    assert "invalid snapshot dimensions" in message
    assert "session_id=orientation-test" in message
    assert "votes rows=3 participants=2" in message
    assert "votes row_lengths=0:2,1:2,2:2 statements=3" in message


def test_load_environment_rejects_nonpositional_statement_indices(
    tmp_path: Path,
) -> None:
    snapshot = orientation_snapshot()
    snapshot["statements"][1]["index"] = 2
    snapshot["statements"][2]["index"] = 1
    data_path = write_snapshot_jsonl(tmp_path, snapshot)

    with pytest.raises(ValueError) as exc_info:
        list(load_environment(data_path=data_path))

    message = str(exc_info.value)
    assert f"{data_path}:1" in message
    assert "session_id=orientation-test" in message
    assert "statements indices=1:2,2:1 expected positional" in message


def test_load_environment_rejects_cluster_assignment_length_mismatch(
    tmp_path: Path,
) -> None:
    snapshot = orientation_snapshot()
    snapshot["clusters"] = [0]
    data_path = write_snapshot_jsonl(tmp_path, snapshot)

    with pytest.raises(ValueError) as exc_info:
        list(load_environment(data_path=data_path))

    assert "clusters=1 participants=2" in str(exc_info.value)


@pytest.mark.parametrize(
    ("mutate", "expected_message"),
    [
        (
            lambda snapshot: snapshot["masked_cells"].__setitem__(0, [99, 1]),
            "out-of-bounds masked cell",
        ),
        (
            lambda snapshot: snapshot["masked_cells"].__setitem__(0, [0.0, 1]),
            "masked cell indices must be integers",
        ),
        (
            lambda snapshot: snapshot["held_out"].__setitem__("0,1", 1.9),
            "invalid held-out vote",
        ),
        (
            lambda snapshot: snapshot["votes"][0].__setitem__(0, True),
            "invalid vote at 0,0",
        ),
        (
            lambda snapshot: snapshot["votes"][0].__setitem__(1, 0),
            "masked vote 0,1 must be null",
        ),
        (
            lambda snapshot: snapshot["clusters"].__setitem__(0, 0.5),
            "cluster assignments must be integer or non-empty string IDs",
        ),
    ],
)
def test_load_environment_rejects_semantically_invalid_snapshots(
    tmp_path: Path,
    mutate: Any,
    expected_message: str,
) -> None:
    snapshot = orientation_snapshot()
    mutate(snapshot)
    data_path = write_snapshot_jsonl(tmp_path, snapshot)

    with pytest.raises(ValueError, match=expected_message):
        list(load_environment(data_path=data_path))


def test_load_environment_rejects_mask_label_disagreement(tmp_path: Path) -> None:
    snapshot = orientation_snapshot()
    snapshot["held_out"] = {"1,2": 0}
    data_path = write_snapshot_jsonl(tmp_path, snapshot)

    with pytest.raises(ValueError, match="held_out must match masked_cells"):
        list(load_environment(data_path=data_path))


def test_custom_human_path_validates_then_applies_deterministic_masking(
    tmp_path: Path,
) -> None:
    data_path = write_snapshot_jsonl(tmp_path, human_snapshot())

    rows = task_rows(load_environment(data_path=data_path, masked_vote_count=2))

    assert len(rows) == 1
    assert rows[0]["info"]["synthetic"] is False
    assert len(rows[0]["answer"]) == 2
    assert len(rows[0]["snapshot"]["masked_cells"]) == 2


def test_custom_human_path_enforces_shared_governance_metadata(
    tmp_path: Path,
) -> None:
    snapshot = human_snapshot()
    snapshot["meta"]["redistribution_rights_approved"] = False
    data_path = write_snapshot_jsonl(tmp_path, snapshot)

    with pytest.raises(ValueError) as exc_info:
        list(load_environment(data_path=data_path, masked_vote_count=1))

    message = str(exc_info.value)
    assert f"{data_path}:1" in message
    assert "human snapshot governance validation failed" in message
    assert "redistribution_rights_approved must be true" in message


def test_parser_handles_fenced_json() -> None:
    parser = PredictionJsonParser()

    parsed = parser.parse('```json\n{"predictions":{"0,1":1,"2,3":0}}\n```\n')

    assert parsed == {"predictions": {"0,1": 1, "2,3": 0}}


def test_parser_prefers_last_object_with_predictions() -> None:
    parser = PredictionJsonParser()

    parsed = parser.parse(
        'I considered {} first. Final answer: {"predictions":{"0,1":1}}'
    )

    assert parsed == {"predictions": {"0,1": 1}}


def test_parser_falls_back_to_last_decodable_object() -> None:
    parser = PredictionJsonParser()

    parsed = parser.parse('{"draft":true} then {"final":true}')

    assert parsed == {"final": True}


def test_parser_rejects_excessive_json_nesting_without_crashing() -> None:
    parser = PredictionJsonParser()
    completion = '{"predictions":' + "[" * 10_000 + "]" * 10_000 + "}"

    assert parser.parse(completion) == {}


def test_parser_rejects_oversized_completion() -> None:
    parser = PredictionJsonParser()
    completion = "x" * 70_000 + '{"predictions":{"0,1":1}}'

    assert parser.parse(completion) == {}


def test_parser_caps_json_candidate_work() -> None:
    parser = PredictionJsonParser()
    completion = " ".join('{"draft":true}' for _ in range(129))
    completion += ' {"predictions":{"0,1":1}}'

    assert parser.parse(completion) == {}


def test_cell_keys_ignore_whitespace_for_point_and_brier_scores() -> None:
    parser = PredictionJsonParser()
    completion = [
        {
            "role": "assistant",
            "content": json.dumps(
                {"predictions": {"0, 5": {"agree": 1.0, "disagree": 0.0, "pass": 0.0}}}
            ),
        }
    ]

    accuracy = asyncio.run(vote_accuracy(completion, {"0,5": 1}, parser))
    brier_score = asyncio.run(brier(completion, {"0,5": 1}, parser))

    assert accuracy == 1.0
    assert brier_score == 0.0


def test_rubric_scores_perfect_completion_at_one() -> None:
    env = load_environment()
    row = task_rows(env)[0]
    held_out = row["answer"]

    state = score_row(env, row, held_out)

    assert state["reward"] == 1.0
    assert state["rewards"]["vote_accuracy"] == 1.0
    assert state["metrics"]["brier"] == 0.0


def test_rubric_scores_all_wrong_completion_at_zero() -> None:
    env = load_environment()
    row = task_rows(env)[0]
    held_out = row["answer"]
    wrong_predictions = {
        cell_id: wrong_vote(vote) for cell_id, vote in held_out.items()
    }

    state = score_row(env, row, wrong_predictions)

    assert state["reward"] == 0.0
    assert state["rewards"]["vote_accuracy"] == 0.0


def test_brier_scores_correct_probability_vector() -> None:
    score = score_brier_prediction({"1": 0.8, "-1": 0.1, "0": 0.1})

    assert isclose(score, 0.06, abs_tol=1e-12)


def test_brier_scores_perfect_one_hot_probability_vector() -> None:
    score = score_brier_prediction({"agree": 1.0, "disagree": 0.0, "pass": 0.0})

    assert score == 0.0


def test_brier_scores_uniform_probability_vector() -> None:
    score = score_brier_prediction({"agree": 1 / 3, "disagree": 1 / 3, "pass": 1 / 3})

    # (1/3 - 1)^2 + (1/3 - 0)^2 + (1/3 - 0)^2 = 2/3
    assert isclose(score, 2 / 3, abs_tol=1e-12)


def test_brier_invalid_probability_mapping_scores_as_uniform() -> None:
    score = score_brier_prediction({"agree": 0.8, "disagree": -0.1, "pass": 0.3})

    assert isclose(score, 2 / 3, abs_tol=1e-12)


def test_masked_vote_count_knob_changes_held_out_count() -> None:
    env = load_environment(masked_vote_count=3, min_cluster_count=2)
    row = task_rows(env)[0]

    assert len(row["answer"]) == 3


def test_masked_vote_selection_is_repeatable() -> None:
    data_path = DATA_DIR / "eval_ce_demo.jsonl"

    first = load_environment(data_path=data_path, masked_vote_count=8)
    second = load_environment(data_path=data_path, masked_vote_count=8)

    first_snapshot = task_rows(first)[0]["snapshot"]
    second_snapshot = task_rows(second)[0]["snapshot"]
    assert first_snapshot["masked_cells"] == second_snapshot["masked_cells"]


def test_masked_vote_selection_spreads_across_participants() -> None:
    data_path = DATA_DIR / "eval_ce_demo.jsonl"
    source = json.loads(data_path.read_text(encoding="utf-8").splitlines()[0])
    first_eight_sorted = [
        [participant_index, statement_index]
        for participant_index, row in enumerate(source["votes"])
        for statement_index, vote in enumerate(row)
        if vote in {-1, 0, 1}
    ][:8]

    env = load_environment(data_path=data_path, masked_vote_count=8)
    snapshot = task_rows(env)[0]["snapshot"]
    selected = snapshot["masked_cells"]

    assert len({participant_index for participant_index, _ in selected}) > 1
    assert selected != first_eight_sorted


def test_masked_vote_count_zero_masks_no_cells_and_scores() -> None:
    env = load_environment(masked_vote_count=0, min_cluster_count=2)
    assert_env_rows_have_no_masks(env)

    row = task_rows(env)[0]
    state = score_row(env, row, {})

    assert state["reward"] == 0.0
    assert state["rewards"]["vote_accuracy"] == 0.0
    assert state["metrics"]["brier"] == 0.0


def test_masked_vote_count_negative_masks_no_cells() -> None:
    env = load_environment(masked_vote_count=-10, min_cluster_count=2)

    assert_env_rows_have_no_masks(env)


def test_masked_vote_count_huge_caps_at_candidate_pool() -> None:
    masked = apply_masked_vote_count(mask_count_snapshot(), 1_000)

    assert len(masked["held_out"]) == 4
    assert set(masked["held_out"]) == {"0,0", "0,1", "1,0", "1,1"}
    assert {tuple(cell) for cell in masked["masked_cells"]} == {
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    }
    assert all(vote is None for row in masked["votes"] for vote in row)


def test_file_masked_cells_keep_selection_precedence() -> None:
    masked = apply_masked_vote_count(mask_count_snapshot(), 1)

    assert masked["masked_cells"] == [[0, 1]]
    assert masked["held_out"] == {"0,1": -1}


def score_row(
    env: CommonGroundPredictTaskset,
    row: dict[str, Any],
    predictions: dict[str, int],
) -> dict[str, Any]:
    task = next(task for task in env if task.data.idx == row["idx"])
    trace = vf.Trace(
        task=vf.TraceTask(type=type(task).__name__, data=task.data),
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        nodes=[
            vf.MessageNode(
                message=vf.AssistantMessage(
                    content=json.dumps({"predictions": predictions}, sort_keys=True)
                ),
                sampled=True,
            )
        ],
    )
    asyncio.run(task.score(trace))
    return {
        "reward": trace.reward,
        "rewards": {
            name: reward.score
            for name, reward in trace.rewards.items()
            if reward is not None
        },
        "metrics": trace.metrics,
        "task": trace.task.data.model_dump(mode="json"),
    }


def score_brier_prediction(prediction: dict[Any, float]) -> float:
    completion = [
        {
            "role": "assistant",
            "content": json.dumps({"predictions": {"0,1": prediction}}, sort_keys=True),
        }
    ]
    return asyncio.run(brier(completion, {"0,1": 1}, PredictionJsonParser()))


def assert_env_rows_have_no_masks(env: CommonGroundPredictTaskset) -> None:
    for row in task_rows(env):
        snapshot = row["snapshot"]
        info = row["info"]

        assert row["answer"] == {}
        assert "held_out" not in snapshot
        assert snapshot["masked_cells"] == []
        assert info["masked_vote_count"] == 0


def mask_count_snapshot() -> dict[str, Any]:
    return {
        "session_id": "mask-count-test",
        "statements": [
            {"index": 0, "text": "Statement A"},
            {"index": 1, "text": "Statement B"},
        ],
        "votes": [
            [1, None],
            [0, -1],
        ],
        "masked_cells": [[0, 1]],
        "held_out": {"0,1": -1},
        "clusters": [0, 1],
        "meta": {"synthetic": True},
    }


def orientation_snapshot() -> dict[str, Any]:
    return {
        "session_id": "orientation-test",
        "statements": [
            {"index": 0, "text": "Statement A"},
            {"index": 1, "text": "Statement B"},
            {"index": 2, "text": "Statement C"},
        ],
        "participants": ["p0", "p1"],
        "votes": [
            [1, None, -1],
            [-1, 1, 0],
        ],
        "masked_cells": [[0, 1]],
        "held_out": {"0,1": 0},
        "clusters": [0, 1],
        "meta": {"synthetic": True},
    }


def human_snapshot() -> dict[str, Any]:
    participants = [f"p{index:03d}" for index in range(10)]
    votes = [[1, -1] if index % 2 == 0 else [-1, 1] for index in range(10)]
    return {
        "session_id": "reviewed-session",
        "statements": [
            {"index": 0, "text": "Fund the pilot."},
            {"index": 1, "text": "Publish the aggregate results."},
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
                    "commentIndex": index,
                    "agrees": 5,
                    "disagrees": 5,
                    "unsure": 0,
                    "total": 10,
                    "responded": 10,
                    "extremity": None,
                    "divisiveness": None,
                }
                for index in range(2)
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


def write_snapshot_jsonl(tmp_path: Path, snapshot: dict[str, Any]) -> Path:
    data_path = tmp_path / "snapshot.jsonl"
    data_path.write_text(json.dumps(snapshot, sort_keys=True) + "\n", encoding="utf-8")
    return data_path


def dataset_rows_bytes(dataset: Any) -> bytes:
    """Serialize dataset rows canonically for byte-level comparisons."""

    lines = [
        json.dumps(dict(row), sort_keys=True, separators=(",", ":")) for row in dataset
    ]
    return ("\n".join(lines) + "\n").encode()


def task_rows(env: CommonGroundPredictTaskset) -> list[dict[str, Any]]:
    return [task.data.model_dump(mode="json") for task in env]


def wrong_vote(vote: int) -> int:
    if vote == 1:
        return -1
    return 1
