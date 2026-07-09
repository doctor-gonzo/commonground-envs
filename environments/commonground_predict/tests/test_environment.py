from __future__ import annotations

import asyncio
import json
from math import isclose
from typing import Any

import verifiers as vf

from commonground_predict import PredictionJsonParser, load_environment
from commonground_predict.environment import brier


def test_load_environment_builds_bundled_split() -> None:
    env = load_environment()

    assert isinstance(env, vf.SingleTurnEnv)
    assert env.env_id == "commonground-predict"
    assert len(env.get_eval_dataset()) == 20


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


def test_brier_invalid_probability_mapping_falls_back_to_one_hot_argmax() -> None:
    score = score_brier_prediction({"agree": 0.8, "disagree": -0.1, "pass": 0.3})

    assert score == 0.0


def test_masked_vote_count_knob_changes_held_out_count() -> None:
    env = load_environment(masked_vote_count=3, min_cluster_count=2)
    row = dict(env.get_eval_dataset()[0])

    assert len(json.loads(row["held_out"])) == 3


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


def wrong_vote(vote: int) -> int:
    if vote == 1:
        return -1
    return 1
