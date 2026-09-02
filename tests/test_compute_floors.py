from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_script() -> Any:
    script_path = ROOT / "scripts" / "compute_floors.py"
    spec = importlib.util.spec_from_file_location("compute_floors", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


floors_module = _load_script()
compute_floors = floors_module.compute_floors
nearest_participant_probabilities = floors_module.nearest_participant_probabilities
probability_distribution = floors_module.probability_distribution
render_markdown = floors_module.render_markdown


def _snapshot(
    session_id: str,
    votes: list[list[int | None]],
    *,
    masked_cell: tuple[int, int],
    held_out_vote: int,
) -> dict[str, Any]:
    participant_index, statement_index = masked_cell
    return {
        "session_id": session_id,
        "statements": [
            {"index": index, "text": f"Policy statement {index}."}
            for index in range(len(votes[0]))
        ],
        "participants": [f"p{index:03d}" for index in range(len(votes))],
        "votes": votes,
        "masked_cells": [[participant_index, statement_index]],
        "held_out": {f"{participant_index},{statement_index}": held_out_vote},
        "clusters": [index % 2 for index in range(len(votes))],
        "meta": {"synthetic": True},
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(f"{json.dumps(row, sort_keys=True)}\n" for row in rows),
        encoding="utf-8",
    )


def test_probability_distribution_is_normalized_smoothed_and_fail_closed() -> None:
    assert probability_distribution(()) == {
        "disagree": pytest.approx(1 / 3),
        "pass": pytest.approx(1 / 3),
        "agree": pytest.approx(1 / 3),
    }
    assert probability_distribution([1, 1, -1], smoothing=1.0) == {
        "disagree": pytest.approx(2 / 6),
        "pass": pytest.approx(1 / 6),
        "agree": pytest.approx(3 / 6),
    }

    # Booleans compare equal to integers in Python but are not valid vote labels.
    assert probability_distribution([True, False, 1]) == {
        "disagree": 0.0,
        "pass": 0.0,
        "agree": 1.0,
    }
    with pytest.raises(ValueError, match="finite non-negative"):
        probability_distribution([1], smoothing=-0.1)
    with pytest.raises(ValueError, match="finite non-negative"):
        probability_distribution([1], smoothing=float("nan"))
    with pytest.raises(TypeError, match="finite non-negative"):
        probability_distribution([1], smoothing=True)


def test_neighbor_probability_forecasts_are_deterministic_and_smoothed() -> None:
    votes = [
        [1, None, 1, -1],
        [1, 1, 1, -1],
        [1, -1, 0, -1],
        [-1, 0, 1, 1],
    ]

    unweighted = nearest_participant_probabilities(votes, 0, 1, neighbor_count=2)
    weighted = nearest_participant_probabilities(
        votes,
        0,
        1,
        neighbor_count=2,
        weighted=True,
        smoothing=0.5,
    )

    assert unweighted == {"disagree": 0.5, "pass": 0.0, "agree": 0.5}
    assert sum(weighted.values()) == pytest.approx(1.0)
    assert all(probability > 0 for probability in weighted.values())
    assert weighted["agree"] > weighted["disagree"] > weighted["pass"]

    with pytest.raises(ValueError, match="positive integer"):
        nearest_participant_probabilities(votes, 0, 1, neighbor_count=0)
    with pytest.raises(ValueError, match="finite non-negative"):
        nearest_participant_probabilities(
            votes, 0, 1, neighbor_count=1, smoothing=float("inf")
        )


def test_compute_floors_reports_native_probability_ladder_and_named_skill(
    tmp_path: Path,
) -> None:
    split = tmp_path / "custom-eval.jsonl"
    _write_jsonl(
        split,
        [
            _snapshot(
                "fixture-a",
                [
                    [None, 1],
                    [1, 1],
                    [-1, 1],
                    [0, 1],
                ],
                masked_cell=(0, 0),
                held_out_vote=1,
            ),
            _snapshot(
                "fixture-b",
                [
                    [1, -1],
                    [-1, None],
                    [-1, 0],
                    [-1, -1],
                ],
                masked_cell=(1, 1),
                held_out_vote=-1,
            ),
        ],
    )

    floors = compute_floors(split, masked_vote_count=1)

    assert {
        "uniform-probability",
        "snapshot-visible-prior",
        "global-visible-prior",
        "statement-visible-frequency",
        "five-neighbor-frequency",
        "distance-weighted-five-neighbor",
    } <= floors.keys()
    assert "train-global-prior" not in floors
    assert "train-text-naive-bayes" not in floors
    uniform = floors["uniform-probability"]
    assert uniform["vote_accuracy"] == 0.5
    assert uniform["probability_reward"] == pytest.approx(2 / 3)
    assert uniform["brier"] == pytest.approx(1 / 3)
    assert uniform["brier_skill_vs_uniform"] == pytest.approx(0.0)
    assert uniform["brier_skill_vs_original_snapshot_visible_prior"] == pytest.approx(
        1.0 - uniform["brier"] / floors["snapshot-visible-prior"]["brier"]
    )
    assert (
        floors["snapshot-visible-prior"][
            "brier_skill_vs_original_snapshot_visible_prior"
        ]
        == 0.0
    )
    assert floors["snapshot-visible-prior"] != floors["global-visible-prior"]

    rendered = render_markdown(floors)
    assert "Per-snapshot visible class prior" in rendered
    assert "Global visible class prior" in rendered
    assert "Evaluation-corpus visible (transductive)" in rendered
    assert "Brier skill vs uniform" in rendered
    assert "Brier skill vs snapshot prior" in rendered


def test_compute_floors_includes_train_only_probability_comparators(
    tmp_path: Path,
) -> None:
    split = tmp_path / "eval.jsonl"
    train = tmp_path / "train.jsonl"
    row = _snapshot(
        "fixture-eval",
        [[None, 1], [1, -1], [-1, 0], [0, 1]],
        masked_cell=(0, 0),
        held_out_vote=1,
    )
    train_row = _snapshot(
        "fixture-train",
        [[1, None], [1, -1], [0, -1], [-1, 0]],
        masked_cell=(0, 1),
        held_out_vote=1,
    )
    _write_jsonl(split, [row])
    _write_jsonl(train, [train_row])

    floors = compute_floors(split, masked_vote_count=1, train_path=train)

    assert "train-global-prior" in floors
    assert "train-text-naive-bayes" in floors
    assert set(floors["train-global-prior"]) == {
        "vote_accuracy",
        "probability_reward",
        "brier",
        "brier_skill_vs_uniform",
        "brier_skill_vs_original_snapshot_visible_prior",
    }
