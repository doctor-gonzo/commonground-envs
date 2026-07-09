"""Polis-compatible vote statistics and reward helpers."""

from __future__ import annotations

from math import sqrt
from typing import Mapping

Vote = int | None
PointPrediction = int
ProbPrediction = Mapping[str, float]
Prediction = PointPrediction | ProbPrediction

_LABEL_TO_VOTE = {"agree": 1, "disagree": -1, "pass": 0}
_VOTE_TO_LABEL = {vote: label for label, vote in _LABEL_TO_VOTE.items()}
_LABELS = ("agree", "disagree", "pass")


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


def rating_to_vote(value: float) -> int:
    """Map a 0-10 rating to a canonical vote.

    The signed score is ``clamp((2 * (value - 5)) / 10, -1, 1)``. Its sign is
    the vote: positive ratings become agree (``1``), negative ratings become
    disagree (``-1``), and exactly ``5`` is neutral/pass (``0``).
    """

    signed = max(-1.0, min(1.0, (2 * (value - 5)) / 10))
    if signed > 0:
        return 1
    if signed < 0:
        return -1
    return 0


def vote_accuracy(predictions: Mapping[str, int], held_out: Mapping[str, int]) -> float:
    """Return the exact-match fraction over held-out cells.

    Missing predictions are counted as wrong.
    """

    if not held_out:
        return 0.0
    correct = sum(predictions.get(cell_id) == vote for cell_id, vote in held_out.items())
    return correct / len(held_out)


def brier_score(
    predictions: Mapping[str, Prediction],
    held_out: Mapping[str, int],
) -> float:
    """Return mean multiclass Brier score for agree/disagree/pass predictions.

    Probabilistic predictions are dictionaries keyed by ``agree``, ``disagree``,
    and ``pass``. Point predictions using ``1``, ``-1``, or ``0`` are converted
    to one-hot probabilities. Missing predictions use an all-zero vector.
    """

    if not held_out:
        return 0.0
    total = 0.0
    for cell_id, actual_vote in held_out.items():
        pred_probs = _prediction_probs(predictions.get(cell_id))
        actual_label = _VOTE_TO_LABEL[actual_vote]
        total += sum((pred_probs[label] - float(label == actual_label)) ** 2 for label in _LABELS)
    return total / len(held_out)


def _prediction_probs(prediction: Prediction | None) -> dict[str, float]:
    if isinstance(prediction, Mapping):
        return {label: float(prediction.get(label, 0.0)) for label in _LABELS}
    if prediction in _VOTE_TO_LABEL:
        label = _VOTE_TO_LABEL[prediction]
        return {candidate: float(candidate == label) for candidate in _LABELS}
    return {label: 0.0 for label in _LABELS}
