from __future__ import annotations

import hashlib
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

BASELINES_WITHOUT_TRAIN = {
    "uniform-probability",
    "statement-visible-frequency",
    "distance-weighted-five-neighbor",
}
BASE_METRICS = {
    "vote_accuracy",
    "probability_reward",
    "brier",
    "brier_skill_vs_uniform",
}


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


def test_neighbor_probability_forecast_is_deterministic_weighted_and_smoothed() -> None:
    votes = [
        [1, None, 1, -1],
        [1, 1, 1, -1],
        [1, -1, 0, -1],
        [-1, 0, 1, 1],
    ]

    forecast = nearest_participant_probabilities(
        votes,
        0,
        1,
        neighbor_count=5,
        smoothing=0.5,
    )

    assert sum(forecast.values()) == pytest.approx(1.0)
    assert all(probability > 0 for probability in forecast.values())
    assert forecast["agree"] > forecast["disagree"] > forecast["pass"]
    assert forecast == nearest_participant_probabilities(
        votes,
        0,
        1,
        neighbor_count=5,
        smoothing=0.5,
    )

    with pytest.raises(ValueError, match="positive integer"):
        nearest_participant_probabilities(votes, 0, 1, neighbor_count=0)
    with pytest.raises(ValueError, match="finite non-negative"):
        nearest_participant_probabilities(
            votes, 0, 1, neighbor_count=1, smoothing=float("inf")
        )


def test_compute_floors_has_only_required_probability_baselines_without_train(
    tmp_path: Path,
) -> None:
    split = tmp_path / "custom-eval.jsonl"
    _write_jsonl(
        split,
        [
            _snapshot(
                "fixture-a",
                [[None, 1], [1, 1], [-1, 1], [0, 1]],
                masked_cell=(0, 0),
                held_out_vote=1,
            ),
            _snapshot(
                "fixture-b",
                [[1, -1], [-1, None], [-1, 0], [-1, -1]],
                masked_cell=(1, 1),
                held_out_vote=-1,
            ),
        ],
    )

    floors = compute_floors(split, masked_vote_count=1)

    assert set(floors) == BASELINES_WITHOUT_TRAIN
    assert all(set(metrics) == BASE_METRICS for metrics in floors.values())
    uniform = floors["uniform-probability"]
    assert uniform["vote_accuracy"] == 0.5
    assert uniform["probability_reward"] == pytest.approx(2 / 3)
    assert uniform["brier"] == pytest.approx(1 / 3)
    assert uniform["brier_skill_vs_uniform"] == 0.0

    rendered = render_markdown(floors)
    assert "Uniform probability" in rendered
    assert "Per-statement visible class frequencies" in rendered
    assert "Smoothed distance-weighted 5-neighbor frequencies" in rendered
    assert "Brier skill vs empirical prior" in rendered
    assert "| — |" in rendered
    assert "Naive Bayes" not in rendered
    assert "oracle" not in rendered.casefold()


def test_compute_floors_adds_labeled_train_prior_and_empirical_skill(
    tmp_path: Path,
) -> None:
    split = tmp_path / "eval.jsonl"
    train = tmp_path / "train.jsonl"
    _write_jsonl(
        split,
        [
            _snapshot(
                "fixture-eval",
                [[None, 1], [1, -1], [-1, 0], [0, 1]],
                masked_cell=(0, 0),
                held_out_vote=1,
            )
        ],
    )
    _write_jsonl(
        train,
        [
            _snapshot(
                "fixture-train",
                [[1, None], [1, -1], [0, -1], [-1, 0]],
                masked_cell=(0, 1),
                held_out_vote=1,
            )
        ],
    )

    floors = compute_floors(split, masked_vote_count=1, train_path=train)

    assert list(floors) == [
        "uniform-probability",
        "train-global-prior",
        "statement-visible-frequency",
        "distance-weighted-five-neighbor",
    ]
    expected_metrics = BASE_METRICS | {"brier_skill_vs_empirical_prior"}
    assert all(set(metrics) == expected_metrics for metrics in floors.values())
    # The labeled train counts are disagree=3, pass=2, agree=3; the held-out
    # train label must be included rather than silently treated as missing.
    assert floors["train-global-prior"]["probability_reward"] == pytest.approx(0.703125)
    assert floors["train-global-prior"]["brier_skill_vs_empirical_prior"] == 0.0


def test_predict_floor_cli_infers_train_and_writes_deterministic_digests(
    tmp_path: Path,
) -> None:
    split = tmp_path / "eval_synthetic.jsonl"
    train = tmp_path / "train_synthetic.jsonl"
    first_output = tmp_path / "floors-a.json"
    second_output = tmp_path / "floors-b.json"
    _write_jsonl(
        split,
        [
            _snapshot(
                "fixture-json-eval",
                [[None, 1], [1, -1], [-1, 0], [0, 1]],
                masked_cell=(0, 0),
                held_out_vote=1,
            )
        ],
    )
    _write_jsonl(
        train,
        [
            _snapshot(
                "fixture-json-train",
                [[1, None], [1, -1], [0, -1], [-1, 0]],
                masked_cell=(0, 1),
                held_out_vote=1,
            )
        ],
    )

    for output in (first_output, second_output):
        assert (
            floors_module.main(
                [
                    str(split),
                    "--masked-vote-count",
                    "1",
                    "--seed",
                    "fixed-seed",
                    "--output-json",
                    str(output),
                ]
            )
            == 0
        )

    assert first_output.read_bytes() == second_output.read_bytes()
    report = json.loads(first_output.read_text())
    assert report["schema_version"] == 1
    assert report["seed"] == "fixed-seed"
    assert report["split"] == {
        "path": str(split),
        "sha256": hashlib.sha256(split.read_bytes()).hexdigest(),
    }
    assert report["train_split"] == {
        "path": str(train),
        "sha256": hashlib.sha256(train.read_bytes()).hexdigest(),
    }
    assert set(report["floors"]) == BASELINES_WITHOUT_TRAIN | {"train-global-prior"}


def test_predict_floor_cli_omits_missing_train_cleanly(tmp_path: Path) -> None:
    split = tmp_path / "custom-eval.jsonl"
    output = tmp_path / "floors.json"
    _write_jsonl(
        split,
        [
            _snapshot(
                "fixture-no-train",
                [[None, 1], [1, -1], [-1, 0], [0, 1]],
                masked_cell=(0, 0),
                held_out_vote=1,
            )
        ],
    )

    assert (
        floors_module.main(
            [
                str(split),
                "--masked-vote-count",
                "1",
                "--output-json",
                str(output),
            ]
        )
        == 0
    )

    report = json.loads(output.read_text())
    assert "train_split" not in report
    assert set(report["floors"]) == BASELINES_WITHOUT_TRAIN
    assert all(
        "brier_skill_vs_empirical_prior" not in metrics
        for metrics in report["floors"].values()
    )
