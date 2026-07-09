"""Generate the bundled synthetic eval split for commonground-predict."""

from __future__ import annotations

import json
import random
from pathlib import Path

SEED = 20260709
SNAPSHOT_COUNT = 20
MASKED_VOTE_COUNT = 8
K_ANONYMITY = 5

STATEMENT_BANK = [
    "Public services should publish plain-language summaries for major decisions.",
    "Budgets should reserve more funding for preventative community programs.",
    "Digital participation should be offered alongside in-person meetings.",
    "Local projects should prioritize long-term maintenance over rapid launches.",
    "Decision makers should publish tradeoffs before choosing a policy option.",
    "Community feedback should be collected earlier in planning processes.",
    "Pilot programs should be evaluated before they are expanded.",
    "Public dashboards should track whether commitments are completed.",
    "Residents most affected by a policy should have extra consultation channels.",
    "Procurement should favor transparent criteria over lowest upfront cost alone.",
    "Small working groups should draft options before full-community votes.",
    "Independent facilitators should moderate contentious deliberations.",
    "Programs should be retired when evidence shows limited impact.",
    "Consensus statements should include minority concerns when disagreement remains.",
    "Public data releases should minimize collection of personal information.",
    "Meeting materials should be available before live discussions begin.",
    "Decision timelines should slow down when participation is unusually low.",
    "Community grants should support both established groups and new organizers.",
    "Quantitative surveys should be paired with open-ended comments.",
    "Policy experiments should define success measures before launch.",
]


def main() -> None:
    rng = random.Random(SEED)
    output_path = Path(__file__).parents[1] / "data" / "eval_synthetic.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    snapshots = [make_snapshot(rng, index) for index in range(SNAPSHOT_COUNT)]
    output_path.write_text(
        "".join(json.dumps(snapshot, separators=(",", ":")) + "\n" for snapshot in snapshots),
        encoding="utf-8",
    )


def make_snapshot(rng: random.Random, session_index: int) -> dict:
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
    votes = [row[:] for row in full_votes]
    masked_cells = sorted(mask_cells(rng, participant_count, statement_count))
    held_out = {}
    for participant_index, statement_index in masked_cells:
        held_out[f"{participant_index},{statement_index}"] = full_votes[participant_index][statement_index]
        votes[participant_index][statement_index] = None

    return {
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


def balanced_clusters(
    rng: random.Random,
    participant_count: int,
    cluster_count: int,
) -> list[int]:
    clusters = [cluster for cluster in range(cluster_count) for _ in range(K_ANONYMITY)]
    clusters.extend(rng.randrange(cluster_count) for _ in range(participant_count - len(clusters)))
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
            "text": STATEMENT_BANK[(start + session_index + statement_index) % len(STATEMENT_BANK)],
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
