from __future__ import annotations

import asyncio
import json
from math import isclose
from pathlib import Path
from typing import Any

import pytest
import verifiers as vf

from commonground_predict import PredictionJsonParser, load_environment
from commonground_predict.environment import DATA_ENV_VAR, apply_masked_vote_count, brier


def test_load_environment_builds_bundled_split() -> None:
    env = load_environment()

    assert isinstance(env, vf.SingleTurnEnv)
    assert env.env_id == "commonground-predict"
    assert len(env.get_eval_dataset()) == 20


def test_load_environment_builds_ce_demo_split_from_env(monkeypatch: Any) -> None:
    data_path = Path(__file__).resolve().parents[1] / "data" / "eval_ce_demo.jsonl"
    monkeypatch.setenv(DATA_ENV_VAR, str(data_path))

    env = load_environment(masked_vote_count=3)
    row = dict(env.get_eval_dataset()[0])
    held_out = json.loads(row["held_out"])
    info = json.loads(row["info"])
    snapshot = json.loads(row["snapshot"])

    assert len(env.get_eval_dataset()) == 1
    assert info["synthetic"] is False
    assert info["cluster_count"] == 2
    assert len(snapshot["participants"]) == 62
    assert len(snapshot["statements"]) == 30
    assert len(snapshot["votes"]) == 62
    assert {len(row) for row in snapshot["votes"]} == {30}
    assert len(held_out) == 3
    state = score_row(env, row, held_out)
    assert state["reward"] == 1.0
    assert state["metrics"]["vote_accuracy"] == 1.0


def test_load_environment_rejects_transposed_votes(tmp_path: Path) -> None:
    snapshot = orientation_snapshot()
    snapshot["votes"] = [
        list(statement_votes)
        for statement_votes in zip(*snapshot["votes"], strict=True)
    ]
    data_path = write_snapshot_jsonl(tmp_path, snapshot)

    with pytest.raises(ValueError) as exc_info:
        load_environment(data_path=data_path)

    message = str(exc_info.value)
    assert "invalid snapshot dimensions" in message
    assert "session_id=orientation-test" in message
    assert "votes rows=3 participants=2" in message
    assert "votes row_lengths=0:2,1:2,2:2 statements=3" in message


def test_load_environment_rejects_cluster_assignment_length_mismatch(
    tmp_path: Path,
) -> None:
    snapshot = orientation_snapshot()
    snapshot["clusters"] = [0]
    data_path = write_snapshot_jsonl(tmp_path, snapshot)

    with pytest.raises(ValueError) as exc_info:
        load_environment(data_path=data_path)

    assert "clusters=1 participants=2" in str(exc_info.value)


def test_parser_handles_fenced_json() -> None:
    parser = PredictionJsonParser()

    parsed = parser.parse(
        '```json\n{"predictions":{"0,1":1,"2,3":0}}\n```\n'
    )

    assert parsed == {"predictions": {"0,1": 1, "2,3": 0}}


def test_rubric_scores_perfect_completion_at_one() -> None:
    env = load_environment()
    row = dict(env.get_eval_dataset()[0])
    held_out = json.loads(row["held_out"])

    state = score_row(env, row, held_out)

    assert state["reward"] == 1.0
    assert state["metrics"]["vote_accuracy"] == 1.0
    assert state["metrics"]["brier"] == 0.0


def test_rubric_scores_all_wrong_completion_at_zero() -> None:
    env = load_environment()
    row = dict(env.get_eval_dataset()[0])
    held_out = json.loads(row["held_out"])
    wrong_predictions = {
        cell_id: wrong_vote(vote)
        for cell_id, vote in held_out.items()
    }

    state = score_row(env, row, wrong_predictions)

    assert state["reward"] == 0.0
    assert state["metrics"]["vote_accuracy"] == 0.0


def test_brier_scores_correct_probability_vector() -> None:
    score = score_brier_prediction({"1": 0.8, "-1": 0.1, "0": 0.1})

    assert isclose(score, 0.06, abs_tol=1e-12)


def test_brier_scores_perfect_one_hot_probability_vector() -> None:
    score = score_brier_prediction({"agree": 1.0, "disagree": 0.0, "pass": 0.0})

    assert score == 0.0


def test_brier_scores_uniform_probability_vector() -> None:
    score = score_brier_prediction(
        {"agree": 1 / 3, "disagree": 1 / 3, "pass": 1 / 3}
    )

    # (1/3 - 1)^2 + (1/3 - 0)^2 + (1/3 - 0)^2 = 2/3
    assert isclose(score, 2 / 3, abs_tol=1e-12)


def test_brier_invalid_probability_mapping_scores_as_uniform() -> None:
    score = score_brier_prediction({"agree": 0.8, "disagree": -0.1, "pass": 0.3})

    assert isclose(score, 2 / 3, abs_tol=1e-12)


def test_masked_vote_count_knob_changes_held_out_count() -> None:
    env = load_environment(masked_vote_count=3, min_cluster_count=2)
    row = dict(env.get_eval_dataset()[0])

    assert len(json.loads(row["held_out"])) == 3


def test_masked_vote_count_zero_masks_no_cells_and_scores() -> None:
    env = load_environment(masked_vote_count=0, min_cluster_count=2)
    assert_env_rows_have_no_masks(env)

    row = dict(env.get_eval_dataset()[0])
    state = score_row(env, row, {})

    assert state["reward"] == 0.0
    assert state["metrics"]["vote_accuracy"] == 0.0
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


def score_row(
    env: vf.SingleTurnEnv,
    row: dict[str, Any],
    predictions: dict[str, int],
) -> dict[str, Any]:
    completion = [
        {
            "role": "assistant",
            "content": json.dumps({"predictions": predictions}, sort_keys=True),
        }
    ]
    state = {
        "prompt": row["prompt"],
        "completion": completion,
        "input": row,
    }
    asyncio.run(env.rubric.score_rollout(state))
    return state


def score_brier_prediction(prediction: dict[Any, float]) -> float:
    completion = [
        {
            "role": "assistant",
            "content": json.dumps({"predictions": {"0,1": prediction}}, sort_keys=True),
        }
    ]
    return asyncio.run(brier(completion, {"0,1": 1}, PredictionJsonParser()))


def assert_env_rows_have_no_masks(env: vf.SingleTurnEnv) -> None:
    for row in env.get_eval_dataset():
        row = dict(row)
        snapshot = json.loads(row["snapshot"])
        info = json.loads(row["info"])

        assert json.loads(row["held_out"]) == {}
        assert snapshot["held_out"] == {}
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
            [1, 0, -1],
            [-1, 1, 0],
        ],
        "masked_cells": [[0, 1]],
        "held_out": {"0,1": 0},
        "clusters": [0, 1],
        "meta": {"synthetic": True},
    }


def write_snapshot_jsonl(tmp_path: Path, snapshot: dict[str, Any]) -> Path:
    data_path = tmp_path / "snapshot.jsonl"
    data_path.write_text(json.dumps(snapshot, sort_keys=True) + "\n", encoding="utf-8")
    return data_path


def wrong_vote(vote: int) -> int:
    if vote == 1:
        return -1
    return 1
