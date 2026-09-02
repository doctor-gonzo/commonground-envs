"""Compute deterministic point and probability floors for a snapshot split."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from math import exp, isfinite, log
from pathlib import Path
from typing import Any, Protocol, cast

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
Forecast = int | dict[str, float]
MetricRow = dict[str, float]
TextVoteModel = tuple[
    Counter[int],
    dict[str, Counter[int]],
    Counter[int],
    int,
]
ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_GENERATOR_PATH = (
    ROOT
    / "environments"
    / "commonground_predict"
    / "scripts"
    / "generate_synthetic_eval.py"
)
_GENERATOR_SOURCE = re.compile(
    r"seeded-semantic-generator:heldout-archetype-threshold-v2:(\d+)"
)
_SYNTHETIC_SESSION_ID = re.compile(r"synthetic-session-(\d+)")


class SyntheticGenerator(Protocol):
    def make_snapshot_with_cluster_patterns(
        self, rng: random.Random, session_index: int
    ) -> tuple[dict[str, Any], list[list[int]]]: ...


def compute_floors(
    path: Path,
    masked_vote_count: int,
    seed: str | None = None,
    *,
    train_path: Path | None = None,
) -> dict[str, MetricRow]:
    """Score probability-native prompt baselines and labeled diagnostics."""

    baseline_names = (
        "uniform-probability",
        "snapshot-visible-prior",
        "global-visible-prior",
        "train-global-prior",
        "train-text-naive-bayes",
        "always-agree",
        "visible-majority",
        "statement-visible-frequency",
        "nearest-participant",
        "five-neighbor",
        "five-neighbor-frequency",
        "distance-weighted-five-neighbor",
        "best-constant-oracle",
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
    snapshots: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        snapshot = json.loads(line)
        validate_snapshot_dimensions(snapshot, path, line_number)
        snapshots.append(snapshot)

    prepared_snapshots: list[tuple[dict[str, Any], dict[str, Any], dict[str, int]]] = []
    for snapshot in snapshots:
        prepared = apply_masked_vote_count(snapshot, masked_vote_count, seed=seed)
        held_out = {str(cell): int(vote) for cell, vote in prepared["held_out"].items()}
        if held_out:
            prepared_snapshots.append((snapshot, prepared, held_out))

    global_visible_prior = probability_distribution(
        vote
        for _, prepared, _ in prepared_snapshots
        for row in prepared["votes"]
        for vote in row
        if vote in VALID_VOTES
    )
    resolved_train_path = train_path or inferred_train_split(path)
    training_snapshots = (
        load_snapshot_rows(resolved_train_path)
        if resolved_train_path is not None and resolved_train_path.is_file()
        else None
    )
    text_vote_model = (
        train_text_vote_model(training_snapshots)
        if training_snapshots is not None
        else None
    )
    train_global_prior = (
        probability_distribution(
            vote
            for snapshot in training_snapshots
            for row in snapshot["votes"]
            for vote in row
            if vote in VALID_VOTES
        )
        if training_snapshots is not None
        else None
    )
    if text_vote_model is None or train_global_prior is None:
        del totals["train-global-prior"]
        del totals["train-text-naive-bayes"]
    cluster_patterns = replay_cluster_patterns(snapshots)
    if cluster_patterns is not None:
        totals["cluster-pattern-oracle"] = {
            "vote_accuracy": 0.0,
            "probability_reward": 0.0,
            "brier": 0.0,
            "brier_skill_vs_uniform": 0.0,
        }

    for snapshot, prepared, held_out in prepared_snapshots:
        snapshot_truth = list(held_out.values())
        truth_counts = Counter(snapshot_truth)
        best_constant = max(VALID_VOTES, key=lambda vote: (truth_counts[vote], vote))
        snapshot_visible_prior = probability_distribution(
            vote
            for row in prepared["votes"]
            for vote in row
            if type(vote) is int and vote in VALID_VOTES
        )
        forecasts: dict[str, dict[str, Forecast]] = {name: {} for name in totals}
        for cell_id in held_out:
            participant_index_text, statement_index_text = cell_id.split(
                ",", maxsplit=1
            )
            participant_index = int(participant_index_text)
            statement_index = int(statement_index_text)
            statement_text = str(prepared["statements"][statement_index]["text"])
            visible_votes = [
                row[statement_index]
                for row in prepared["votes"]
                if row[statement_index] in VALID_VOTES
            ]
            counts = Counter(visible_votes)
            visible_majority = max(VALID_VOTES, key=lambda vote: (counts[vote], vote))
            statement_distribution = probability_distribution(visible_votes)
            five_neighbor_distribution = nearest_participant_probabilities(
                prepared["votes"],
                participant_index,
                statement_index,
                neighbor_count=5,
            )
            weighted_neighbor_distribution = nearest_participant_probabilities(
                prepared["votes"],
                participant_index,
                statement_index,
                neighbor_count=5,
                weighted=True,
                smoothing=0.5,
            )
            forecasts["uniform-probability"][cell_id] = probability_distribution(())
            forecasts["snapshot-visible-prior"][cell_id] = snapshot_visible_prior
            forecasts["global-visible-prior"][cell_id] = global_visible_prior
            if text_vote_model is not None and train_global_prior is not None:
                forecasts["train-global-prior"][cell_id] = train_global_prior
                forecasts["train-text-naive-bayes"][cell_id] = text_vote_probabilities(
                    statement_text, text_vote_model
                )
            forecasts["always-agree"][cell_id] = 1
            forecasts["visible-majority"][cell_id] = visible_majority
            forecasts["statement-visible-frequency"][cell_id] = statement_distribution
            forecasts["nearest-participant"][cell_id] = nearest_participant_vote(
                prepared["votes"], participant_index, statement_index
            )
            forecasts["five-neighbor"][cell_id] = nearest_participant_vote(
                prepared["votes"],
                participant_index,
                statement_index,
                neighbor_count=5,
            )
            forecasts["five-neighbor-frequency"][cell_id] = five_neighbor_distribution
            forecasts["distance-weighted-five-neighbor"][cell_id] = (
                weighted_neighbor_distribution
            )
            forecasts["best-constant-oracle"][cell_id] = best_constant
            if cluster_patterns is not None:
                snapshot_patterns = cluster_patterns[str(snapshot["session_id"])]
                participant_cluster = int(prepared["clusters"][participant_index])
                forecasts["cluster-pattern-oracle"][cell_id] = snapshot_patterns[
                    participant_cluster
                ][statement_index]

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
    # A uniform reference is distribution-free but easy on three classes
    # (reward 2/3). The current snapshot's visible class prior is a stronger,
    # prompt-observable climatology. Report both so model reward is not framed
    # only against the weaker reference.
    snapshot_reference_brier = averaged["snapshot-visible-prior"]["brier"]
    for metrics in averaged.values():
        metrics["brier_skill_vs_original_snapshot_visible_prior"] = _clean_float(
            0.0
            if snapshot_reference_brier <= 0
            else 1.0 - metrics["brier"] / snapshot_reference_brier
        )
    return averaged


def load_snapshot_rows(path: Path) -> list[dict[str, Any]]:
    """Load and dimension-check snapshot rows for a model-free comparator."""

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


def train_text_vote_model(snapshots: Sequence[Mapping[str, Any]]) -> TextVoteModel:
    """Fit a deterministic train-split-only bag-of-words vote model."""

    class_counts: Counter[int] = Counter()
    token_counts: dict[str, Counter[int]] = {}
    token_totals: Counter[int] = Counter()
    vocabulary: set[str] = set()
    for snapshot in snapshots:
        statements = snapshot["statements"]
        votes = snapshot["votes"]
        for statement_index, statement in enumerate(statements):
            column_counts = Counter(
                row[statement_index]
                for row in votes
                if row[statement_index] in VALID_VOTES
            )
            class_counts.update(column_counts)
            tokens = statement_tokens(str(statement["text"]))
            vocabulary.update(tokens)
            for token in tokens:
                counts = token_counts.setdefault(token, Counter())
                counts.update(column_counts)
                token_totals.update(column_counts)
    return class_counts, token_counts, token_totals, max(1, len(vocabulary))


def text_vote_probabilities(text: str, model: TextVoteModel) -> dict[str, float]:
    """Return Laplace-smoothed multinomial probabilities from train text only."""

    class_counts, token_counts, token_totals, vocabulary_size = model
    class_total = sum(class_counts.values())
    log_scores: dict[int, float] = {}
    for vote in VALID_VOTES:
        prior = (class_counts[vote] + 1.0) / (class_total + len(VALID_VOTES))
        score = log(prior)
        denominator = token_totals[vote] + vocabulary_size
        for token in statement_tokens(text):
            score += log((token_counts.get(token, Counter())[vote] + 1.0) / denominator)
        log_scores[vote] = score
    maximum = max(log_scores.values())
    weights = {vote: exp(score - maximum) for vote, score in log_scores.items()}
    total = sum(weights.values())
    return {VOTE_TO_LABEL[vote]: weights[vote] / total for vote in VALID_VOTES}


def statement_tokens(text: str) -> tuple[str, ...]:
    """Extract unique lexical features without reading evaluation labels."""

    return tuple(
        dict.fromkeys(
            token
            for token in re.findall(r"[^\W_]+", text.casefold())
            if len(token) >= 3
        )
    )


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


def forecast_vote(prediction: Forecast) -> int:
    """Convert one forecast to its deterministic maximum-probability class."""

    if isinstance(prediction, int):
        return prediction
    return max(
        VALID_VOTES,
        key=lambda vote: (float(prediction[VOTE_TO_LABEL[vote]]), vote),
    )


def _clean_float(value: float) -> float:
    return 0.0 if abs(value) < 1e-15 else value


def nearest_participant_vote(
    votes: Sequence[Sequence[int | None]],
    participant_index: int,
    statement_index: int,
    *,
    neighbor_count: int = 1,
) -> int:
    """Copy/aggregate the most similar prompt-visible participant rows."""

    ranked = ranked_neighbor_votes(votes, participant_index, statement_index)
    if not ranked:
        column = [
            row[statement_index] for row in votes if row[statement_index] in VALID_VOTES
        ]
        counts = Counter(column)
        return max(VALID_VOTES, key=lambda vote: (counts[vote], vote))
    selected = ranked[:neighbor_count]
    counts = Counter(item[3] for item in selected)
    return max(VALID_VOTES, key=lambda vote: (counts[vote], vote))


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
    weighted: bool = False,
    smoothing: float = 0.0,
) -> dict[str, float]:
    """Return a prompt-visible k-NN vote distribution."""

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
        counts[vote] += max(similarity, 1e-6) if weighted else 1.0
    total = sum(counts.values())
    return {VOTE_TO_LABEL[vote]: counts[vote] / total for vote in VALID_VOTES}


def replay_cluster_patterns(
    snapshots: Sequence[dict[str, Any]],
) -> dict[str, list[list[int]]] | None:
    """Replay a generator-backed split and recover its hidden planted patterns."""

    if not snapshots:
        return None
    generator = load_synthetic_generator()
    rngs: dict[int, random.Random] = {}
    recovered: dict[str, list[list[int]]] = {}
    for snapshot in snapshots:
        meta = snapshot.get("meta")
        source = meta.get("source") if isinstance(meta, dict) else None
        session_id = snapshot.get("session_id")
        if not isinstance(source, str) or not isinstance(session_id, str):
            return None
        source_match = _GENERATOR_SOURCE.fullmatch(source)
        session_match = _SYNTHETIC_SESSION_ID.fullmatch(session_id)
        if source_match is None or session_match is None:
            return None
        generator_seed = int(source_match.group(1))
        rng = rngs.setdefault(generator_seed, random.Random(generator_seed))
        replayed, patterns = generator.make_snapshot_with_cluster_patterns(
            rng, int(session_match.group(1))
        )
        if json.loads(json.dumps(replayed)) != snapshot:
            return None
        recovered[session_id] = patterns
    return recovered


def load_synthetic_generator() -> SyntheticGenerator:
    script_dir = str(SYNTHETIC_GENERATOR_PATH.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    spec = importlib.util.spec_from_file_location(
        "commonground_predict_synthetic_generator", SYNTHETIC_GENERATOR_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {SYNTHETIC_GENERATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return cast(SyntheticGenerator, module)


def render_markdown(floors: Mapping[str, Mapping[str, float]]) -> str:
    labels = {
        "uniform-probability": ("No-input", "Uniform probability"),
        "snapshot-visible-prior": (
            "Prompt-observable matrix-only",
            "Per-snapshot visible class prior",
        ),
        "global-visible-prior": (
            "Evaluation-corpus visible (transductive)",
            "Global visible class prior",
        ),
        "train-global-prior": (
            "Train-split no-text",
            "Global empirical class prior",
        ),
        "train-text-naive-bayes": (
            "Train-split text-only",
            "Bag-of-words vote probabilities",
        ),
        "always-agree": ("No-input", "Always agree"),
        "visible-majority": (
            "Prompt-observable matrix-only",
            "Per-statement visible majority",
        ),
        "statement-visible-frequency": (
            "Prompt-observable matrix-only",
            "Per-statement visible class frequencies",
        ),
        "nearest-participant": (
            "Prompt-observable matrix-only",
            "Nearest participant (1-NN)",
        ),
        "five-neighbor": (
            "Prompt-observable matrix-only",
            "Five-neighbor vote",
        ),
        "five-neighbor-frequency": (
            "Prompt-observable matrix-only",
            "Five-neighbor vote frequencies",
        ),
        "distance-weighted-five-neighbor": (
            "Prompt-observable matrix-only",
            "Distance-weighted 5-NN with smoothing",
        ),
        "best-constant-oracle": (
            "Held-out-label diagnostic",
            "Per-snapshot best constant",
        ),
        "cluster-pattern-oracle": (
            "Generator diagnostic",
            "Latent cluster-pattern replay",
        ),
    }
    lines = [
        "| Comparator class | Comparator | probability reward | vote_accuracy | normalized Brier | Brier skill vs uniform | Brier skill vs snapshot prior |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(
        f"| {labels[name][0]} | {labels[name][1]} | "
        f"{metrics['probability_reward']:.3f} | "
        f"{metrics['vote_accuracy']:.3f} | "
        f"{metrics['brier']:.3f} | "
        f"{metrics['brier_skill_vs_uniform']:.3f} | "
        f"{metrics['brier_skill_vs_original_snapshot_visible_prior']:.3f} |"
        for name, metrics in floors.items()
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", type=Path)
    parser.add_argument("--masked-vote-count", type=int, required=True)
    parser.add_argument("--seed")
    parser.add_argument("--train-split", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    floors = compute_floors(
        args.split,
        args.masked_vote_count,
        seed=args.seed,
        train_path=args.train_split,
    )
    print(render_markdown(floors))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
