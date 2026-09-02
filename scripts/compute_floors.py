"""Compute the preregistered probability-native Predict baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from math import isfinite
from pathlib import Path
from typing import Any, cast

from commonground_predict.environment import (
    apply_masked_vote_count,
    validate_snapshot_dimensions,
)
from commonground_score import (
    brier_score,
    brier_skill_score,
    probability_reward,
    vote_accuracy,
)

VALID_VOTES = (-1, 0, 1)
VOTE_TO_LABEL = {-1: "disagree", 0: "pass", 1: "agree"}
Forecast = dict[str, float]
MetricRow = dict[str, float]


def compute_floors(
    path: Path,
    masked_vote_count: int,
    seed: str | None = None,
    *,
    train_path: Path | None = None,
) -> dict[str, MetricRow]:
    """Score the four preregistered probability-native comparators."""

    snapshots = load_snapshot_rows(path)
    prepared_snapshots: list[tuple[dict[str, Any], dict[str, int]]] = []
    for snapshot in snapshots:
        prepared = apply_masked_vote_count(snapshot, masked_vote_count, seed=seed)
        held_out = {str(cell): int(vote) for cell, vote in prepared["held_out"].items()}
        if held_out:
            prepared_snapshots.append((prepared, held_out))

    resolved_train_path = resolve_train_split(path, train_path)
    train_prior = (
        empirical_train_distribution(load_snapshot_rows(resolved_train_path))
        if resolved_train_path is not None
        else None
    )
    baseline_names = ["uniform-probability"]
    if train_prior is not None:
        baseline_names.append("train-global-prior")
    baseline_names.extend(
        [
            "statement-visible-frequency",
            "distance-weighted-five-neighbor",
        ]
    )
    totals = {
        name: {
            "vote_accuracy": 0.0,
            "probability_reward": 0.0,
            "brier": 0.0,
            "brier_skill_vs_uniform": 0.0,
        }
        for name in baseline_names
    }

    target_count = 0
    uniform = probability_distribution(())
    for prepared, held_out in prepared_snapshots:
        forecasts: dict[str, dict[str, Forecast]] = {
            name: {} for name in baseline_names
        }
        for cell_id in held_out:
            participant_text, statement_text = cell_id.split(",", maxsplit=1)
            participant_index = int(participant_text)
            statement_index = int(statement_text)
            visible_column = [
                row[statement_index]
                for row in prepared["votes"]
                if row[statement_index] in VALID_VOTES
            ]
            forecasts["uniform-probability"][cell_id] = uniform
            if train_prior is not None:
                forecasts["train-global-prior"][cell_id] = train_prior
            forecasts["statement-visible-frequency"][cell_id] = (
                probability_distribution(visible_column)
            )
            forecasts["distance-weighted-five-neighbor"][cell_id] = (
                nearest_participant_probabilities(
                    prepared["votes"],
                    participant_index,
                    statement_index,
                    neighbor_count=5,
                    smoothing=0.5,
                )
            )

        snapshot_target_count = len(held_out)
        target_count += snapshot_target_count
        for name, predictions in forecasts.items():
            typed_predictions = cast(
                Mapping[str, int | Mapping[object, object]], predictions
            )
            point_predictions = {
                cell_id: forecast_vote(prediction)
                for cell_id, prediction in predictions.items()
            }
            totals[name]["vote_accuracy"] += (
                vote_accuracy(point_predictions, held_out) * snapshot_target_count
            )
            totals[name]["probability_reward"] += (
                probability_reward(typed_predictions, held_out) * snapshot_target_count
            )
            totals[name]["brier"] += (
                brier_score(typed_predictions, held_out) * snapshot_target_count
            )
            totals[name]["brier_skill_vs_uniform"] += (
                brier_skill_score(typed_predictions, held_out) * snapshot_target_count
            )

    if target_count == 0:
        raise ValueError(f"no held-out votes available in {path}")
    averaged = {
        name: {
            metric: _clean_float(total / target_count)
            for metric, total in metric_totals.items()
        }
        for name, metric_totals in totals.items()
    }
    if train_prior is not None:
        reference_brier = averaged["train-global-prior"]["brier"]
        for metrics in averaged.values():
            metrics["brier_skill_vs_empirical_prior"] = skill_against_reference(
                metrics["brier"], reference_brier
            )
    return averaged


def load_snapshot_rows(path: Path) -> list[dict[str, Any]]:
    """Load and dimension-check snapshot rows."""

    snapshots: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        snapshot = json.loads(line)
        validate_snapshot_dimensions(snapshot, path, line_number)
        snapshots.append(snapshot)
    return snapshots


def inferred_train_split(eval_path: Path) -> Path | None:
    """Find the bundled train peer without imposing it on arbitrary inputs."""

    if eval_path.name != "eval_synthetic.jsonl":
        return None
    return eval_path.with_name("train_synthetic.jsonl")


def resolve_train_split(eval_path: Path, train_path: Path | None) -> Path | None:
    """Return an explicit or conventional train split only when it exists."""

    candidate = (
        train_path if train_path is not None else inferred_train_split(eval_path)
    )
    return candidate if candidate is not None and candidate.is_file() else None


def empirical_train_distribution(
    snapshots: Sequence[Mapping[str, Any]],
) -> dict[str, float] | None:
    """Compute the empirical class prior from all labeled train cells."""

    votes: list[int] = []
    for snapshot in snapshots:
        votes.extend(
            int(vote)
            for row in snapshot["votes"]
            for vote in row
            if type(vote) is int and vote in VALID_VOTES
        )
        # Train artifacts store designated masked labels separately from the
        # matrix; omitting them would make the "train prior" only a visible prior.
        votes.extend(
            int(vote)
            for vote in snapshot["held_out"].values()
            if type(vote) is int and vote in VALID_VOTES
        )
    return probability_distribution(votes) if votes else None


def probability_distribution(
    votes: Iterable[int],
    *,
    smoothing: float = 0.0,
) -> dict[str, float]:
    """Normalize vote counts into an agree/disagree/pass forecast."""

    if not isinstance(smoothing, (int, float)) or isinstance(smoothing, bool):
        raise TypeError("smoothing must be a finite non-negative number")
    if not isfinite(float(smoothing)) or smoothing < 0:
        raise ValueError("smoothing must be a finite non-negative number")
    counts: Counter[int] = Counter(
        vote for vote in votes if type(vote) is int and vote in VALID_VOTES
    )
    weighted = {vote: float(counts[vote]) + smoothing for vote in VALID_VOTES}
    total = sum(weighted.values())
    if total <= 0:
        return {label: 1 / len(VALID_VOTES) for label in VOTE_TO_LABEL.values()}
    return {VOTE_TO_LABEL[vote]: weighted[vote] / total for vote in VALID_VOTES}


def forecast_vote(prediction: Mapping[str, float]) -> int:
    """Convert one distribution to its deterministic maximum-probability class."""

    return max(
        VALID_VOTES,
        key=lambda vote: (float(prediction[VOTE_TO_LABEL[vote]]), vote),
    )


def _clean_float(value: float) -> float:
    return 0.0 if abs(value) < 1e-15 else value


def skill_against_reference(brier: float, reference_brier: float) -> float:
    """Return Brier skill relative to a supplied empirical reference."""

    if reference_brier <= 0:
        return 0.0
    return _clean_float(1.0 - brier / reference_brier)


def ranked_neighbor_votes(
    votes: Sequence[Sequence[int | None]],
    participant_index: int,
    statement_index: int,
) -> list[tuple[float, int, int, int]]:
    """Rank prompt-visible neighbor votes by row agreement and stable index."""

    target_row = votes[participant_index]
    ranked: list[tuple[float, int, int, int]] = []
    for other_index, other_row in enumerate(votes):
        if other_index == participant_index:
            continue
        target_vote = other_row[statement_index]
        if target_vote not in VALID_VOTES:
            continue
        jointly_visible = [
            index
            for index, (left, right) in enumerate(
                zip(target_row, other_row, strict=True)
            )
            if index != statement_index and left in VALID_VOTES and right in VALID_VOTES
        ]
        if not jointly_visible:
            continue
        matches = sum(
            target_row[index] == other_row[index] for index in jointly_visible
        )
        ranked.append(
            (matches / len(jointly_visible), matches, -other_index, int(target_vote))
        )
    return sorted(ranked, reverse=True)


def nearest_participant_probabilities(
    votes: Sequence[Sequence[int | None]],
    participant_index: int,
    statement_index: int,
    *,
    neighbor_count: int,
    smoothing: float = 0.0,
) -> dict[str, float]:
    """Return a smoothed, distance-weighted prompt-visible k-NN forecast."""

    if type(neighbor_count) is not int or neighbor_count <= 0:
        raise ValueError("neighbor_count must be a positive integer")
    if not isinstance(smoothing, (int, float)) or isinstance(smoothing, bool):
        raise TypeError("smoothing must be a finite non-negative number")
    if not isfinite(float(smoothing)) or smoothing < 0:
        raise ValueError("smoothing must be a finite non-negative number")
    selected = ranked_neighbor_votes(votes, participant_index, statement_index)[
        :neighbor_count
    ]
    if not selected:
        column = [
            cast(int, row[statement_index])
            for row in votes
            if row[statement_index] in VALID_VOTES
        ]
        return probability_distribution(column, smoothing=smoothing)
    counts = {vote: smoothing for vote in VALID_VOTES}
    for similarity, _, _, vote in selected:
        counts[vote] += max(similarity, 1e-6)
    total = sum(counts.values())
    return {VOTE_TO_LABEL[vote]: counts[vote] / total for vote in VALID_VOTES}


def render_markdown(floors: Mapping[str, Mapping[str, float]]) -> str:
    """Render the fixed comparator suite without expanding the evidence claim."""

    labels = {
        "uniform-probability": ("No-input", "Uniform probability"),
        "train-global-prior": (
            "Train-split no-text",
            "Global empirical class prior",
        ),
        "statement-visible-frequency": (
            "Prompt-observable matrix-only",
            "Per-statement visible class frequencies",
        ),
        "distance-weighted-five-neighbor": (
            "Prompt-observable matrix-only",
            "Smoothed distance-weighted 5-neighbor frequencies",
        ),
    }
    lines = [
        "| Comparator class | Comparator | probability reward | vote_accuracy | normalized Brier | Brier skill vs uniform | Brier skill vs empirical prior |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in floors.items():
        empirical_skill = metrics.get("brier_skill_vs_empirical_prior")
        empirical_cell = "—" if empirical_skill is None else f"{empirical_skill:.3f}"
        lines.append(
            f"| {labels[name][0]} | {labels[name][1]} | "
            f"{metrics['probability_reward']:.3f} | "
            f"{metrics['vote_accuracy']:.3f} | "
            f"{metrics['brier']:.3f} | "
            f"{metrics['brier_skill_vs_uniform']:.3f} | "
            f"{empirical_cell} |"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", type=Path)
    parser.add_argument("--masked-vote-count", type=int, required=True)
    parser.add_argument("--seed")
    parser.add_argument("--train-split", type=Path)
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Write comparator metrics and exact split digests as JSON.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    resolved_train_path = resolve_train_split(args.split, args.train_split)
    floors = compute_floors(
        args.split,
        args.masked_vote_count,
        seed=args.seed,
        train_path=args.train_split,
    )
    print(render_markdown(floors))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "schema_version": 1,
            "split": {
                "path": str(args.split),
                "sha256": hashlib.sha256(args.split.read_bytes()).hexdigest(),
            },
            "masked_vote_count": args.masked_vote_count,
            "seed": args.seed,
            "floors": floors,
        }
        if resolved_train_path is not None:
            payload["train_split"] = {
                "path": str(resolved_train_path),
                "sha256": hashlib.sha256(resolved_train_path.read_bytes()).hexdigest(),
            }
        args.output_json.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
