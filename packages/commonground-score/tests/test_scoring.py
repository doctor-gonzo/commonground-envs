from __future__ import annotations

from math import isclose, log, nan

from commonground_score import (
    brier_score,
    cluster_separation,
    comment_stats,
    prop_test,
    rating_to_vote,
    two_prop_test,
    vote_entropy,
    vote_accuracy,
)


def test_prop_test_hand_computed_value() -> None:
    assert isclose(prop_test(2, 4), 0.4472135954999579, abs_tol=1e-12)


def test_two_prop_test_hand_computed_value() -> None:
    assert isclose(two_prop_test(2, 1, 4, 4), 0.6324555320336758, abs_tol=1e-12)


def test_comment_stats_hand_computed_values() -> None:
    stats = comment_stats([1, 1, -1, 0, None])

    assert stats["seen"] == 4
    assert stats["agree"] == 2
    assert stats["disagree"] == 1
    assert stats["pass"] == 1
    assert stats["pa"] == 0.5
    assert isclose(stats["pd"], 1 / 3, abs_tol=1e-12)
    assert isclose(stats["pat"], prop_test(2, 4), abs_tol=1e-12)
    assert isclose(stats["pdt"], prop_test(1, 4), abs_tol=1e-12)


def test_vote_entropy_and_cluster_separation_hand_computed_values() -> None:
    votes = [1, 1, -1]
    expected_entropy = -(
        (2 / 3) * log(2 / 3)
        + (1 / 3) * log(1 / 3)
    ) / log(3)

    assert isclose(vote_entropy(votes), expected_entropy, abs_tol=1e-12)
    assert cluster_separation(votes) == 2 / 3


def test_panel_disagreement_math_handles_unanimity_passes_and_missing_votes() -> None:
    assert vote_entropy([1, 1, 1]) == 0.0
    assert cluster_separation([1, 1, 1]) == 0.0
    assert isclose(vote_entropy([1, -1, 0]), 1.0, abs_tol=1e-12)
    assert cluster_separation([1, -1, 0]) == 1.0
    assert vote_entropy([1, -1, None]) == vote_entropy([1, -1])
    assert cluster_separation([1, -1, None]) == 1.0


def test_rating_to_vote_uses_canonical_signed_mapping() -> None:
    assert rating_to_vote(-1) == 0
    assert rating_to_vote(0) == -1
    assert isclose(rating_to_vote(4.999), -0.00019999999999988916, abs_tol=1e-12)
    assert rating_to_vote(5) == 0
    assert isclose(rating_to_vote(5.001), 0.00019999999999988916, abs_tol=1e-12)
    assert rating_to_vote(10) == 1
    assert rating_to_vote(11) == 0
    assert rating_to_vote(nan) == 0


def test_vote_accuracy_counts_missing_predictions_wrong() -> None:
    held_out = {"0,1": 1, "1,2": -1, "2,3": 0}

    assert vote_accuracy({"0,1": 1, "1,2": -1, "2,3": 0}, held_out) == 1.0
    assert vote_accuracy({"0,1": 1, "1,2": -1}, held_out) == 2 / 3
    assert vote_accuracy({"0,1": -1, "1,2": 1, "2,3": 1}, held_out) == 0.0


def test_brier_score_accepts_point_and_probabilistic_predictions() -> None:
    assert brier_score({"0,1": 1}, {"0,1": 1}) == 0.0
    assert brier_score({"0,1": -1}, {"0,1": 1}) == 2.0
    assert isclose(
        brier_score({"0,1": {"agree": 0.8, "disagree": 0.1, "pass": 0.1}}, {"0,1": 1}),
        0.06,
        abs_tol=1e-12,
    )


def test_brier_score_normalizes_probabilistic_mapping_predictions() -> None:
    assert isclose(
        brier_score({"0,1": {"agree": 8, "disagree": 1, "pass": 1}}, {"0,1": 1}),
        0.06,
        abs_tol=1e-12,
    )


def test_brier_score_all_zero_probability_mapping_scores_as_uniform() -> None:
    assert isclose(
        brier_score({"0,1": {"agree": 0, "disagree": 0, "pass": 0}}, {"0,1": 1}),
        2 / 3,
        abs_tol=1e-12,
    )


def test_brier_score_negative_probability_mapping_scores_as_uniform() -> None:
    assert isclose(
        brier_score(
            {"0,1": {"agree": 0.4, "disagree": 0.9, "pass": -0.1}},
            {"0,1": -1},
        ),
        2 / 3,
        abs_tol=1e-12,
    )


def test_brier_score_nan_probability_mapping_scores_as_uniform() -> None:
    assert isclose(
        brier_score(
            {"0,1": {"agree": nan, "disagree": 0.2, "pass": 0.1}},
            {"0,1": 1},
        ),
        2 / 3,
        abs_tol=1e-12,
    )
