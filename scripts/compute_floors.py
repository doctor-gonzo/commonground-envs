"""Compute deterministic naive vote-accuracy floors for a snapshot split."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Sequence

from commonground_predict.environment import apply_masked_vote_count, validate_snapshot_dimensions


VALID_VOTES = (-1, 0, 1)


def compute_floors(path: Path, masked_vote_count: int, seed: str | None = None) -> dict[str, float]:
    always_agree_correct = 0
    visible_majority_correct = 0
    best_constant_correct = 0
    target_count = 0

    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        snapshot = json.loads(line)
        validate_snapshot_dimensions(snapshot, path, line_number)
        prepared = apply_masked_vote_count(snapshot, masked_vote_count, seed=seed)
        held_out = {str(cell): int(vote) for cell, vote in prepared["held_out"].items()}
        if not held_out:
            continue

        snapshot_truth = list(held_out.values())
        best_constant_correct += max(Counter(snapshot_truth).values())
        target_count += len(snapshot_truth)
        for cell_id, actual_vote in held_out.items():
            _, statement_index_text = cell_id.split(",", maxsplit=1)
            statement_index = int(statement_index_text)
            visible_votes = [
                row[statement_index]
                for row in prepared["votes"]
                if row[statement_index] in VALID_VOTES
            ]
            counts = Counter(visible_votes)
            visible_majority = max(VALID_VOTES, key=lambda vote: (counts[vote], vote))
            always_agree_correct += int(actual_vote == 1)
            visible_majority_correct += int(actual_vote == visible_majority)

    if target_count == 0:
        raise ValueError(f"no held-out votes available in {path}")
    return {
        "always-agree": always_agree_correct / target_count,
        "visible-majority": visible_majority_correct / target_count,
        "best-constant-oracle": best_constant_correct / target_count,
    }


def render_markdown(floors: dict[str, float]) -> str:
    labels = {
        "always-agree": "Always agree",
        "visible-majority": "Per-statement visible majority",
        "best-constant-oracle": "Per-snapshot best constant oracle",
    }
    lines = ["| Baseline | vote_accuracy |", "| --- | ---: |"]
    lines.extend(f"| {labels[name]} | {score:.3f} |" for name, score in floors.items())
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", type=Path)
    parser.add_argument("--masked-vote-count", type=int, required=True)
    parser.add_argument("--seed")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    floors = compute_floors(args.split, args.masked_vote_count, seed=args.seed)
    print(render_markdown(floors))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
