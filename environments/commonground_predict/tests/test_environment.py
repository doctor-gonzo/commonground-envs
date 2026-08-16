from __future__ import annotations

import asyncio
import json
from math import isclose
from pathlib import Path
from typing import Any

import pytest
import verifiers as vf

from commonground_predict import PredictionJsonParser, load_environment
from commonground_predict.environment import (
    BUNDLED_EVAL_PATH,
    DATA_ENV_VAR,
    apply_masked_vote_count,
    brier,
    vote_accuracy,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "commonground_predict" / "data"


def test_load_environment_builds_bundled_split() -> None:
    env = load_environment()

    assert isinstance(env, vf.SingleTurnEnv)
    assert env.env_id == "commonground-predict"
    assert len(env.get_eval_dataset()) == 20


def test_bundled_eval_path_is_inside_import_package() -> None:
    import commonground_predict.environment as environment_module

    expected = (
        Path(environment_module.__file__).resolve().parent
        / "data"
        / "eval_synthetic.jsonl"
    )

    assert BUNDLED_EVAL_PATH == expected
    assert BUNDLED_EVAL_PATH.is_file()


def test_load_environment_builds_ce_demo_split_from_env(monkeypatch: Any) -> None:
    data_path = DATA_DIR / "eval_ce_demo.jsonl"
    monkeypatch.setenv(DATA_ENV_VAR, str(data_path))

    env = load_environment(masked_vote_count=3)
    row = dict(env.get_eval_dataset()[0])
    held_out = json.loads(row["answer"])
    info = json.loads(row["info"])
    snapshot = json.loads(row["snapshot"])

    assert len(env.get_eval_dataset()) == 1
    assert info["synthetic"] is True
    assert snapshot["meta"]["source"] == "ce-demo-authored"
    assert info["cluster_count"] == 2
    assert len(snapshot["participants"]) == 62
    assert len(snapshot["statements"]) == 30
    assert len(snapshot["votes"]) == 62
    assert {len(row) for row in snapshot["votes"]} == {30}
    assert len(held_out) == 3
    state = score_row(env, row, held_out)
    assert state["reward"] == 1.0
    assert state["metrics"]["vote_accuracy"] == 1.0


def test_load_environment_rejects_unmasked_ce_demo_by_default() -> None:
    data_path = DATA_DIR / "eval_ce_demo.jsonl"

    with pytest.raises(ValueError) as exc_info:
        load_environment(data_path=data_path)

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
        load_environment(data_path=data_path)

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
        load_environment(data_path=data_path)

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
        load_environment(data_path=data_path)

    assert "clusters=1 participants=2" in str(exc_info.value)


def test_parser_handles_fenced_json() -> None:
    parser = PredictionJsonParser()

    parsed = parser.parse(
        '```json\n{"predictions":{"0,1":1,"2,3":0}}\n```\n'
    )

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


def test_cell_keys_ignore_whitespace_for_point_and_brier_scores() -> None:
    parser = PredictionJsonParser()
    completion = [
        {
            "role": "assistant",
            "content": json.dumps(
                {
                    "predictions": {
                        "0, 5": {"agree": 1.0, "disagree": 0.0, "pass": 0.0}
                    }
                }
            ),
        }
    ]

    accuracy = asyncio.run(vote_accuracy(completion, {"0,5": 1}, parser))
    brier_score = asyncio.run(brier(completion, {"0,5": 1}, parser))

    assert accuracy == 1.0
    assert brier_score == 0.0


def test_rubric_scores_perfect_completion_at_one() -> None:
    env = load_environment()
    row = dict(env.get_eval_dataset()[0])
    held_out = json.loads(row["answer"])

    state = score_row(env, row, held_out)

    assert state["reward"] == 1.0
    assert state["metrics"]["vote_accuracy"] == 1.0
    assert state["metrics"]["brier"] == 0.0


def test_rubric_scores_all_wrong_completion_at_zero() -> None:
    env = load_environment()
    row = dict(env.get_eval_dataset()[0])
    held_out = json.loads(row["answer"])
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

    assert len(json.loads(row["answer"])) == 3


def test_masked_vote_selection_is_repeatable() -> None:
    data_path = DATA_DIR / "eval_ce_demo.jsonl"

    first = load_environment(data_path=data_path, masked_vote_count=8)
    second = load_environment(data_path=data_path, masked_vote_count=8)

    first_snapshot = json.loads(dict(first.get_eval_dataset()[0])["snapshot"])
    second_snapshot = json.loads(dict(second.get_eval_dataset()[0])["snapshot"])
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
    snapshot = json.loads(dict(env.get_eval_dataset()[0])["snapshot"])
    selected = snapshot["masked_cells"]

    assert len({participant_index for participant_index, _ in selected}) > 1
    assert selected != first_eight_sorted


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


def test_file_masked_cells_keep_selection_precedence() -> None:
    masked = apply_masked_vote_count(mask_count_snapshot(), 1)

    assert masked["masked_cells"] == [[0, 1]]
    assert masked["held_out"] == {"0,1": -1}


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
        "answer": row["answer"],
        "info": row["info"],
        "input": row,
        "trajectory": [],
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

        assert json.loads(row["answer"]) == {}
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
