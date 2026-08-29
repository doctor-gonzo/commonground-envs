"""Scoring utilities for Common Ground deliberation evaluations."""

from importlib.metadata import PackageNotFoundError, version

from commonground_score.scoring import (
    brier_score,
    cluster_separation,
    comment_stats,
    probability_reward,
    prop_test,
    rating_to_vote,
    two_prop_test,
    vote_accuracy,
    vote_entropy,
)

try:
    __version__ = version("commonground-score")
except PackageNotFoundError:  # pragma: no cover - source tree without installation
    __version__ = "0+unknown"

__all__ = [
    "brier_score",
    "cluster_separation",
    "comment_stats",
    "probability_reward",
    "prop_test",
    "rating_to_vote",
    "two_prop_test",
    "vote_accuracy",
    "vote_entropy",
]
