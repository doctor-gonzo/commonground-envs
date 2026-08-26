"""Compute deterministic naive vote-accuracy floors for a snapshot split."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, cast

from commonground_predict.environment import (
    apply_masked_vote_count,
    validate_snapshot_dimensions,
)

VALID_VOTES = (-1, 0, 1)
ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_GENERATOR_PATH = (
    ROOT
    / "environments"
    / "commonground_predict"
    / "scripts"
    / "generate_synthetic_eval.py"
)
_GENERATOR_SOURCE = re.compile(r"seeded_synthetic_generator:(\d+)")
_SYNTHETIC_SESSION_ID = re.compile(r"synthetic-session-(\d+)")


class SyntheticGenerator(Protocol):
    def make_snapshot_with_cluster_patterns(
        self, rng: random.Random, session_index: int
    ) -> tuple[dict[str, Any], list[list[int]]]: ...


def compute_floors(
    path: Path, masked_vote_count: int, seed: str | None = None
) -> dict[str, float]:
    always_agree_correct = 0
    visible_majority_correct = 0
    best_constant_correct = 0
    cluster_pattern_correct = 0
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

    cluster_patterns = replay_cluster_patterns(snapshots)
    for snapshot in snapshots:
        prepared = apply_masked_vote_count(snapshot, masked_vote_count, seed=seed)
        held_out = {str(cell): int(vote) for cell, vote in prepared["held_out"].items()}
        if not held_out:
            continue

        snapshot_truth = list(held_out.values())
        best_constant_correct += max(Counter(snapshot_truth).values())
        target_count += len(snapshot_truth)
        for cell_id, actual_vote in held_out.items():
            participant_index_text, statement_index_text = cell_id.split(
                ",", maxsplit=1
            )
            participant_index = int(participant_index_text)
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
            if cluster_patterns is not None:
                snapshot_patterns = cluster_patterns[str(snapshot["session_id"])]
                participant_cluster = int(prepared["clusters"][participant_index])
                planted_vote = snapshot_patterns[participant_cluster][statement_index]
                cluster_pattern_correct += int(actual_vote == planted_vote)

    if target_count == 0:
        raise ValueError(f"no held-out votes available in {path}")
    floors = {
        "always-agree": always_agree_correct / target_count,
        "visible-majority": visible_majority_correct / target_count,
        "best-constant-oracle": best_constant_correct / target_count,
    }
    if cluster_patterns is not None:
        floors["cluster-pattern-oracle"] = cluster_pattern_correct / target_count
    return floors


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
    spec = importlib.util.spec_from_file_location(
        "commonground_predict_synthetic_generator", SYNTHETIC_GENERATOR_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {SYNTHETIC_GENERATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return cast(SyntheticGenerator, module)


def render_markdown(floors: dict[str, float]) -> str:
    labels = {
        "always-agree": "Always agree",
        "visible-majority": "Per-statement visible majority",
        "best-constant-oracle": "Per-snapshot best constant oracle",
        "cluster-pattern-oracle": "Planted cluster-pattern oracle (ceiling)",
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
