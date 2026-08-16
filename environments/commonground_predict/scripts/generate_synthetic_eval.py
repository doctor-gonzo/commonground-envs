"""Generate the bundled synthetic eval split for commonground-predict."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import cast

SEED = 20260709
SNAPSHOT_COUNT = 20
MASKED_VOTE_COUNT = 8
K_ANONYMITY = 5

STATEMENT_BANK = [
    "The support assistant should retrieve current prices and rates from the approved pricing system before quoting them to a customer.",
    "The support assistant should obtain human approval before offering a credit outside the published service-recovery limits.",
    "The coding assistant should reference credentials through the managed secrets service and never place secret values in generated code.",
    "The coding assistant should run available checks and clearly label any generated change it could not verify.",
    "The communications assistant should omit competitor names from outbound drafts unless an approved comparison brief is provided.",
    "The HR assistant should cite the controlling policy section and effective date in every employee-policy answer.",
    "The HR assistant should route disciplinary and performance recommendations to an authorized HR reviewer before delivery.",
    "The sales assistant should ground capability claims in approved product documentation before adding them to a proposal.",
    "The finance assistant may draft payment or journal-entry instructions but should not submit them without authorized review.",
    "The contract assistant should separate quoted terms from its interpretations and flag every interpretation for legal review.",
    "The operations assistant should be allowed to escalate a safety or service risk directly to the on-call human.",
    "The assistant should verify a data classification before sending work content to any external model or tool.",
    "The assistant should include only the minimum customer fields needed to complete the requested task.",
    "The analytics assistant should label its data sources, reporting window, and known gaps in every generated analysis.",
    "The customer-facing assistant should preserve approved accessibility and localization requirements when rewriting content.",
    "Every deployed AI assistant should have a named business owner, a documented purpose, and a tested rollback path.",
    "A material change to an assistant's model, tools, or data access should trigger a fresh risk review before release.",
    "People affected by a high-impact assistant recommendation should be able to contest it and request human review.",
    "An AI assistant should pass documented evaluations across relevant user groups before its deployment is expanded.",
    "Assistant interaction logs should be retained only for a defined audit period and deleted on a published schedule.",
]


def main() -> None:
    rng = random.Random(SEED)
    output_path = (
        Path(__file__).parents[1]
        / "commonground_predict"
        / "data"
        / "eval_synthetic.jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    snapshots = [make_snapshot(rng, index) for index in range(SNAPSHOT_COUNT)]
    output_path.write_text(
        "".join(
            json.dumps(snapshot, separators=(",", ":")) + "\n" for snapshot in snapshots
        ),
        encoding="utf-8",
    )


def make_snapshot(rng: random.Random, session_index: int) -> dict:
    snapshot, _ = make_snapshot_with_cluster_patterns(rng, session_index)
    return snapshot


def make_snapshot_with_cluster_patterns(
    rng: random.Random, session_index: int
) -> tuple[dict, list[list[int]]]:
    """Build a snapshot and return its hidden planted cluster patterns."""

    cluster_count = rng.choice([2, 3])
    participant_count = rng.randint(cluster_count * K_ANONYMITY, 20)
    statement_count = rng.randint(6, 15)

    clusters = balanced_clusters(rng, participant_count, cluster_count)
    participants = [
        f"synthetic-{session_index:03d}-p{participant_index:02d}"
        for participant_index in range(participant_count)
    ]
    statements = make_statements(rng, session_index, statement_count)
    cluster_patterns = make_cluster_patterns(rng, cluster_count, statement_count)
    full_votes = make_votes(rng, clusters, cluster_patterns)
    votes = cast(list[list[int | None]], [row[:] for row in full_votes])
    masked_cells = sorted(mask_cells(rng, participant_count, statement_count))
    held_out = {}
    for participant_index, statement_index in masked_cells:
        held_out[f"{participant_index},{statement_index}"] = full_votes[
            participant_index
        ][statement_index]
        votes[participant_index][statement_index] = None

    snapshot = {
        "session_id": f"synthetic-session-{session_index:03d}",
        "statements": statements,
        "participants": participants,
        "votes": votes,
        "masked_cells": masked_cells,
        "held_out": held_out,
        "clusters": clusters,
        "meta": {
            "k_anonymity": K_ANONYMITY,
            "source": f"seeded_synthetic_generator:{SEED}",
            "synthetic": True,
        },
    }
    return snapshot, cluster_patterns


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


def make_statements(
    rng: random.Random,
    session_index: int,
    statement_count: int,
) -> list[dict[str, int | str]]:
    start = rng.randrange(len(STATEMENT_BANK))
    return [
        {
            "index": statement_index,
            "text": STATEMENT_BANK[
                (start + session_index + statement_index) % len(STATEMENT_BANK)
            ],
        }
        for statement_index in range(statement_count)
    ]


def make_cluster_patterns(
    rng: random.Random,
    cluster_count: int,
    statement_count: int,
) -> list[list[int]]:
    patterns: list[list[int]] = []
    for cluster in range(cluster_count):
        pattern = []
        for statement_index in range(statement_count):
            signal = 1 if (statement_index + cluster) % 3 else -1
            if rng.random() < 0.2:
                signal = 0
            if rng.random() < 0.15:
                signal *= -1
            pattern.append(signal)
        patterns.append(pattern)
    return patterns


def make_votes(
    rng: random.Random,
    clusters: list[int],
    cluster_patterns: list[list[int]],
) -> list[list[int]]:
    votes = []
    for cluster in clusters:
        row = []
        for base_vote in cluster_patterns[cluster]:
            if rng.random() < 0.15:
                row.append(rng.choice([-1, 0, 1]))
            else:
                row.append(base_vote)
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


if __name__ == "__main__":
    main()
