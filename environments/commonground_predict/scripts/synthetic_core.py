"""Shared semantic vote-generation primitives for Common Ground Predict.

Statement indices are presentation-only. Votes are generated from an explicit
statement dimension and a latent cluster preference over that dimension.
Train and evaluation scripts supply different profile families.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

K_ANONYMITY = 5
MASKED_VOTE_COUNT = 8
PASS_THRESHOLD = 0.28
PARTICIPANT_NOISE = 0.12


@dataclass(frozen=True)
class StatementSpec:
    """One model-visible policy statement and its hidden semantic attributes."""

    text: str
    dimension: str
    polarity: int = 1
    bias: float = 0.0


class ProfileBuilder(Protocol):
    def __call__(
        self,
        rng: random.Random,
        cluster_count: int,
        statements: list[StatementSpec],
    ) -> list[dict[str, float]]: ...


def make_snapshot(
    rng: random.Random,
    session_index: int,
    *,
    seed: int,
    family: str,
    statement_bank: Sequence[StatementSpec],
    profile_builder: ProfileBuilder,
) -> dict[str, Any]:
    """Build one masked snapshot from semantic statements and latent profiles."""

    snapshot, _ = make_snapshot_with_latent_patterns(
        rng,
        session_index,
        seed=seed,
        family=family,
        statement_bank=statement_bank,
        profile_builder=profile_builder,
    )
    return snapshot


def make_snapshot_with_latent_patterns(
    rng: random.Random,
    session_index: int,
    *,
    seed: int,
    family: str,
    statement_bank: Sequence[StatementSpec],
    profile_builder: ProfileBuilder,
) -> tuple[dict[str, Any], list[list[int]]]:
    """Return a snapshot plus the hidden cluster-by-statement vote patterns."""

    cluster_count = rng.choice([2, 3])
    participant_count = rng.randint(cluster_count * K_ANONYMITY, 20)
    statement_count = rng.randint(8, 16)
    clusters = balanced_clusters(rng, participant_count, cluster_count)
    participants = [
        f"synthetic-{session_index:04d}-p{participant_index:02d}"
        for participant_index in range(participant_count)
    ]
    selected_specs = select_statements(rng, statement_bank, statement_count)
    statements = [
        {"index": index, "text": spec.text} for index, spec in enumerate(selected_specs)
    ]
    profiles = profile_builder(rng, cluster_count, selected_specs)
    latent_patterns = make_latent_patterns(rng, profiles, selected_specs)
    full_votes = make_votes(rng, clusters, latent_patterns)
    votes = cast(list[list[int | None]], [row[:] for row in full_votes])
    masked_cells = sorted(mask_cells(rng, participant_count, statement_count))
    held_out: dict[str, int] = {}
    for participant_index, statement_index in masked_cells:
        held_out[f"{participant_index},{statement_index}"] = full_votes[
            participant_index
        ][statement_index]
        votes[participant_index][statement_index] = None

    return (
        {
            "session_id": f"synthetic-session-{session_index:04d}",
            "statements": statements,
            "participants": participants,
            "votes": votes,
            "masked_cells": masked_cells,
            "held_out": held_out,
            "clusters": clusters,
            "meta": {
                "k_anonymity": K_ANONYMITY,
                "source": f"seeded-semantic-generator:{family}:{seed}",
                "generator_family": family,
                "synthetic": True,
            },
        },
        latent_patterns,
    )


def select_statements(
    rng: random.Random,
    statement_bank: Sequence[StatementSpec],
    statement_count: int,
) -> list[StatementSpec]:
    """Select a semantically mixed set without relying on presentation indices."""

    selected = rng.sample(list(statement_bank), statement_count)
    rng.shuffle(selected)
    return selected


def make_latent_patterns(
    rng: random.Random,
    profiles: Sequence[Mapping[str, float]],
    statements: Sequence[StatementSpec],
) -> list[list[int]]:
    """Map cluster preferences and statement attributes to discrete votes."""

    return [
        [semantic_vote(rng, profile, statement) for statement in statements]
        for profile in profiles
    ]


def semantic_vote(
    rng: random.Random,
    profile: Mapping[str, float],
    statement: StatementSpec,
) -> int:
    """Return the vote caused by one semantic dimension and statement polarity."""

    score = (
        float(profile[statement.dimension]) * statement.polarity
        + statement.bias
        + rng.uniform(-0.12, 0.12)
    )
    if abs(score) < PASS_THRESHOLD:
        return 0
    return 1 if score > 0 else -1


def balanced_clusters(
    rng: random.Random,
    participant_count: int,
    cluster_count: int,
) -> list[int]:
    clusters = [cluster for cluster in range(cluster_count) for _ in range(K_ANONYMITY)]
    clusters.extend(
        rng.randrange(cluster_count) for _ in range(participant_count - len(clusters))
    )
    rng.shuffle(clusters)
    return clusters


def make_votes(
    rng: random.Random,
    clusters: Sequence[int],
    cluster_patterns: Sequence[Sequence[int]],
) -> list[list[int]]:
    votes: list[list[int]] = []
    for cluster in clusters:
        row: list[int] = []
        for base_vote in cluster_patterns[cluster]:
            row.append(
                rng.choice([-1, 0, 1])
                if rng.random() < PARTICIPANT_NOISE
                else base_vote
            )
        votes.append(row)
    return votes


def mask_cells(
    rng: random.Random,
    participant_count: int,
    statement_count: int,
) -> list[tuple[int, int]]:
    candidates = [
        (participant_index, statement_index)
        for participant_index in range(participant_count)
        for statement_index in range(statement_count)
    ]
    return rng.sample(candidates, MASKED_VOTE_COUNT)
