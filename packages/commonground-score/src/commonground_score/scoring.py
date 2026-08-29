"""Polis-compatible vote statistics and reward helpers."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite, log, sqrt
from typing import Any, cast

Vote = int | None
PointPrediction = int
ProbPrediction = Mapping[object, object]
Prediction = PointPrediction | ProbPrediction

_LABEL_TO_VOTE = {"agree": 1, "disagree": -1, "pass": 0}
_VOTE_TO_LABEL = {vote: label for label, vote in _LABEL_TO_VOTE.items()}
_LABELS = ("agree", "disagree", "pass")
RATING_MIN = 0
RATING_MAX = 10
VALID_VOTES = (-1, 0, 1)


def prop_test(successes: int, trials: int) -> float:
    """Return the one-proportion test statistic used by Polis math."""

    return 2 * sqrt(trials + 1) * ((successes + 1) / (trials + 1) - 0.5)


def two_prop_test(s_in: int, s_out: int, p_in: int, p_out: int) -> float:
    """Return the smoothed two-proportion test statistic used by Polis math."""

    s_in += 1
    s_out += 1
    p_in += 1
    p_out += 1
    pi1 = s_in / p_in
    pi2 = s_out / p_out
    pi_hat = (s_in + s_out) / (p_in + p_out)
    if pi_hat == 1:
        return 0
    return (pi1 - pi2) / sqrt(pi_hat * (1 - pi_hat) * ((1 / p_in) + (1 / p_out)))


def comment_stats(votes: list[Vote]) -> dict[str, float | int]:
    """Summarize agree/disagree/pass counts and smoothed proportions for a statement."""

    agree = sum(vote == 1 for vote in votes)
    disagree = sum(vote == -1 for vote in votes)
    passed = sum(vote == 0 for vote in votes)
    seen = sum(vote is not None for vote in votes)
    return {
        "agree": agree,
        "disagree": disagree,
        "pass": passed,
        "seen": seen,
        "pa": (agree + 1) / (seen + 2),
        "pd": (disagree + 1) / (seen + 2),
        "pat": prop_test(agree, seen),
        "pdt": prop_test(disagree, seen),
    }


def vote_entropy(votes: list[Vote]) -> float:
    """Return normalized agree/disagree/pass entropy over seen valid votes."""

    seen = [vote for vote in votes if vote in VALID_VOTES]
    if not seen:
        return 0.0
    entropy = 0.0
    for vote in VALID_VOTES:
        count = seen.count(vote)
        if count:
            probability = count / len(seen)
            entropy -= probability * log(probability)
    return entropy / log(len(VALID_VOTES))


def cluster_separation(votes: list[Vote]) -> float:
    """Return the fraction of seen faction pairs taking different stances."""

    seen = [vote for vote in votes if vote in VALID_VOTES]
    pair_count = len(seen) * (len(seen) - 1) // 2
    if pair_count == 0:
        return 0.0
    separated_pairs = sum(
        left_vote != right_vote
        for left_index, left_vote in enumerate(seen)
        for right_vote in seen[left_index + 1 :]
    )
    return separated_pairs / pair_count


def rating_to_vote(value: float) -> float:
    """Map a 0-10 rating to the canonical signed vote score.

    Non-finite values and values outside ``[0, 10]`` return ``0`` (no stance).
    In-range values return ``(2 * (value - 5)) / 10``: positive values lean
    agree, negative values lean disagree, and exactly ``5`` is neutral/pass.
    """

    if not isfinite(value) or value < RATING_MIN or value > RATING_MAX:
        return 0
    return (2 * (value - 5)) / 10


def vote_accuracy(predictions: Mapping[str, int], held_out: Mapping[str, int]) -> float:
    """Return the exact-match fraction over held-out cells.

    Missing predictions are counted as wrong.
    """

    if not held_out:
        return 0.0
    correct = sum(
        predictions.get(cell_id) == vote for cell_id, vote in held_out.items()
    )
    return correct / len(held_out)


def brier_score(
    predictions: Mapping[str, Prediction],
    held_out: Mapping[str, int],
) -> float:
    """Return normalized multiclass Brier score for agree/disagree/pass.

    Probabilistic predictions are dictionaries keyed by ``agree``, ``disagree``,
    and ``pass``. Valid non-negative finite mappings are normalized to sum to
    one. Invalid or non-normalizable mappings score as the uniform
    distribution, representing no information. Point predictions using ``1``,
    ``-1``, or ``0`` are converted to one-hot probabilities. Missing
    predictions use the uniform distribution. The conventional three-class
    squared-error sum is divided by two, giving a documented ``[0, 1]`` range.
    """

    if not held_out:
        return 0.0
    total = 0.0
    for cell_id, actual_vote in held_out.items():
        pred_probs = _prediction_probs(predictions.get(cell_id))
        actual_label = _VOTE_TO_LABEL[actual_vote]
        total += 0.5 * sum(
            (pred_probs[label] - float(label == actual_label)) ** 2 for label in _LABELS
        )
    return total / len(held_out)


def probability_reward(
    predictions: Mapping[str, Prediction],
    held_out: Mapping[str, int],
) -> float:
    """Return a proper probability reward derived from normalized Brier loss.

    A perfect forecast receives ``1`` and a confidently wrong forecast receives
    ``0``. An empty target set receives ``0`` because there is no prediction
    task to reward. Response-shape validation belongs to the calling
    environment; this helper scores an already accepted prediction mapping.
    """

    if not held_out:
        return 0.0
    return 1.0 - brier_score(predictions, held_out)


def _prediction_probs(prediction: Prediction | None) -> dict[str, float]:
    if isinstance(prediction, Mapping):
        return _mapping_prediction_probs(prediction)
    if prediction in _VOTE_TO_LABEL:
        label = _VOTE_TO_LABEL[prediction]
        return {candidate: float(candidate == label) for candidate in _LABELS}
    return _uniform_probs()


def _mapping_prediction_probs(prediction: Mapping[object, object]) -> dict[str, float]:
    scores: dict[str, float] = {}
    invalid = False
    for key, value in prediction.items():
        label = _coerce_class_label(key)
        if label is None:
            continue
        try:
            score = float(cast(Any, value))
        except (TypeError, ValueError):
            invalid = True
            continue
        if not isfinite(score):
            invalid = True
            continue
        scores[label] = score
        if score < 0:
            invalid = True

    total = sum(scores.get(label, 0.0) for label in _LABELS)
    if scores and not invalid and total > 0:
        return {label: scores.get(label, 0.0) / total for label in _LABELS}
    return _uniform_probs()


def _uniform_probs() -> dict[str, float]:
    return {label: 1 / len(_LABELS) for label in _LABELS}


def _coerce_class_label(key: object) -> str | None:
    if isinstance(key, bool):
        return None
    if isinstance(key, int) and key in _VOTE_TO_LABEL:
        return _VOTE_TO_LABEL[key]
    if isinstance(key, str):
        if key in _LABEL_TO_VOTE:
            return key
        if key in {"-1", "0", "1"}:
            return _VOTE_TO_LABEL[int(key)]
    return None
