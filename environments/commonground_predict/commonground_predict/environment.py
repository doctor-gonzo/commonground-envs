"""Verifiers environment for Common Ground masked-vote prediction."""

from __future__ import annotations

import copy
import json
import os
from collections.abc import Mapping
from math import isfinite
from pathlib import Path
from typing import Any

import verifiers as vf
from commonground_score import brier_score as score_brier_score
from commonground_score import vote_accuracy as score_vote_accuracy
from datasets import Dataset

ENV_ID = "commonground-predict"
DATA_ENV_VAR = "COMMONGROUND_DATA_PATH"
BUNDLED_EVAL_PATH = Path(__file__).resolve().parents[1] / "data" / "eval_synthetic.jsonl"
VALID_VOTES = {-1, 0, 1}
LABEL_TO_VOTE = {"agree": 1, "disagree": -1, "pass": 0}
VOTE_TO_LABEL = {vote: label for label, vote in LABEL_TO_VOTE.items()}


class PredictionJsonParser(vf.Parser):
    """Extract the first JSON object from plain text or fenced completions."""

    def parse(self, text: str) -> dict[str, Any]:
        try:
            parsed = extract_json_object(text)
        except ValueError:
            return {}
        if isinstance(parsed, dict):
            return parsed
        return {}


def load_environment(
    masked_vote_count: int | None = None,
    min_cluster_count: int | None = None,
    data_path: str | os.PathLike[str] | None = None,
    **kwargs: Any,
) -> vf.SingleTurnEnv:
    """Build the deterministic single-turn masked-vote prediction environment."""

    resolved_path = Path(data_path or os.environ.get(DATA_ENV_VAR) or BUNDLED_EVAL_PATH)
    snapshots = load_snapshots(
        resolved_path,
        masked_vote_count=masked_vote_count,
        min_cluster_count=min_cluster_count,
    )
    dataset = Dataset.from_list([snapshot_to_row(snapshot) for snapshot in snapshots])
    parser = PredictionJsonParser()
    rubric = vf.Rubric(funcs=[vote_accuracy, brier], weights=[1.0, 0.0], parser=parser)
    env_args = {
        "masked_vote_count": masked_vote_count,
        "min_cluster_count": min_cluster_count,
        "data_path": str(resolved_path),
    }
    return vf.SingleTurnEnv(
        dataset=dataset,
        eval_dataset=dataset,
        parser=parser,
        rubric=rubric,
        env_id=ENV_ID,
        env_args=env_args,
        **kwargs,
    )


def load_snapshots(
    path: Path,
    masked_vote_count: int | None,
    min_cluster_count: int | None,
) -> list[dict[str, Any]]:
    """Load JSONL snapshots and apply deterministic difficulty filters."""

    snapshots = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        snapshot = json.loads(line)
        if min_cluster_count is not None:
            cluster_count = snapshot_cluster_count(snapshot)
            if cluster_count < min_cluster_count:
                continue
        snapshots.append(apply_masked_vote_count(snapshot, masked_vote_count))
    if not snapshots:
        raise ValueError(f"no snapshots loaded from {path}")
    return snapshots


def apply_masked_vote_count(
    snapshot: dict[str, Any],
    masked_vote_count: int | None,
) -> dict[str, Any]:
    """Return a copy of a snapshot with capped ``masked_vote_count`` masked cells."""

    prepared = copy.deepcopy(snapshot)
    if masked_vote_count is None:
        return prepared

    full_votes = reconstruct_known_votes(prepared)
    candidates = [
        (participant_index, statement_index)
        for participant_index, row in enumerate(full_votes)
        for statement_index, vote in enumerate(row)
        if vote in VALID_VOTES
    ]
    target_count = min(max(masked_vote_count, 0), len(candidates))

    selected: list[tuple[int, int]] = []
    original_masked = [
        (int(participant_index), int(statement_index))
        for participant_index, statement_index in prepared.get("masked_cells", [])
    ]
    for cell in original_masked:
        if len(selected) == target_count:
            break
        if cell in candidates and cell not in selected:
            selected.append(cell)
    for cell in sorted(candidates):
        if len(selected) == target_count:
            break
        if cell not in selected:
            selected.append(cell)

    votes = [row[:] for row in full_votes]
    held_out: dict[str, int] = {}
    for participant_index, statement_index in selected:
        vote = votes[participant_index][statement_index]
        if vote not in VALID_VOTES:
            continue
        held_out[f"{participant_index},{statement_index}"] = int(vote)
        votes[participant_index][statement_index] = None

    prepared["votes"] = votes
    prepared["masked_cells"] = [[participant_index, statement_index] for participant_index, statement_index in selected]
    prepared["held_out"] = held_out
    return prepared


def reconstruct_known_votes(snapshot: Mapping[str, Any]) -> list[list[int | None]]:
    """Reconstruct true values for held-out cells while preserving unknown cells."""

    votes = [list(row) for row in snapshot["votes"]]
    for cell_id, vote in snapshot.get("held_out", {}).items():
        participant_index_text, statement_index_text = str(cell_id).split(",", maxsplit=1)
        votes[int(participant_index_text)][int(statement_index_text)] = int(vote)
    return votes


def snapshot_cluster_count(snapshot: Mapping[str, Any]) -> int:
    """Return cluster count for synthetic ID lists or exported cluster objects."""

    cluster_ids: set[Any] = set()
    for index, cluster in enumerate(snapshot.get("clusters", [])):
        if isinstance(cluster, Mapping):
            cluster = cluster.get("id", index)
        try:
            cluster_ids.add(cluster)
        except TypeError:
            cluster_ids.add(json.dumps(cluster, sort_keys=True))
    return len(cluster_ids)


def snapshot_to_row(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    held_out = {str(cell_id): int(vote) for cell_id, vote in snapshot["held_out"].items()}
    info = {
        "session_id": snapshot["session_id"],
        "masked_vote_count": len(held_out),
        "cluster_count": snapshot_cluster_count(snapshot),
        "synthetic": bool(snapshot["meta"].get("synthetic")),
    }
    return {
        "prompt": [{"role": "user", "content": render_prompt(snapshot)}],
        "answer": json.dumps(held_out, sort_keys=True),
        "held_out": json.dumps(held_out, sort_keys=True),
        "info": json.dumps(info, sort_keys=True),
        "snapshot": json.dumps(snapshot, sort_keys=True),
    }


def render_prompt(snapshot: Mapping[str, Any]) -> str:
    """Render the compact masked-vote prediction prompt."""

    statements = snapshot["statements"]
    masked_cells = [f"{participant_index},{statement_index}" for participant_index, statement_index in snapshot["masked_cells"]]
    lines = [
        "Predict the held-out votes in this deliberation snapshot.",
        "Votes use 1=agree, -1=disagree, 0=pass/unsure, ?=not seen or masked.",
        "",
        "Statements:",
    ]
    lines.extend(f"{statement['index']}: {statement['text']}" for statement in statements)
    lines.extend(
        [
            "",
            "Visible vote matrix:",
            "columns: " + " ".join(str(statement["index"]) for statement in statements),
        ]
    )
    for participant_index, row in enumerate(snapshot["votes"]):
        lines.append(
            f"p{participant_index:02d}: "
            + " ".join(_vote_symbol(vote) for vote in row)
        )
    lines.extend(
        [
            "",
            "Masked cells: " + ", ".join(masked_cells),
            "",
            "Return STRICT JSON only, with this shape:",
            '{"predictions":{"<participant_idx>,<statement_idx>":1|-1|0}}',
        ]
    )
    return "\n".join(lines)


async def vote_accuracy(
    completion: list[dict[str, Any]],
    held_out: Mapping[str, int] | str,
    parser: PredictionJsonParser,
) -> float:
    """Reward: exact point-prediction accuracy over masked cells."""

    parsed = parse_completion_predictions(completion, parser)
    point_predictions = coerce_point_predictions(parsed)
    return score_vote_accuracy(point_predictions, parse_held_out(held_out))


async def brier(
    completion: list[dict[str, Any]],
    held_out: Mapping[str, int] | str,
    parser: PredictionJsonParser,
) -> float:
    """Metric: delegated Brier score; invalid probability mappings score uniform."""

    parsed = parse_completion_predictions(completion, parser)
    brier_predictions = coerce_brier_predictions(parsed)
    return score_brier_score(brier_predictions, parse_held_out(held_out))


def parse_completion_predictions(
    completion: list[dict[str, Any]],
    parser: PredictionJsonParser,
) -> Mapping[str, Any]:
    parsed = parser.parse_answer(completion)
    if not isinstance(parsed, Mapping):
        return {}
    predictions = parsed.get("predictions", {})
    if not isinstance(predictions, Mapping):
        return {}
    return predictions


def parse_held_out(held_out: Mapping[str, int] | str) -> dict[str, int]:
    if isinstance(held_out, str):
        loaded = json.loads(held_out)
    else:
        loaded = held_out
    return {str(cell_id): int(vote) for cell_id, vote in loaded.items() if int(vote) in VALID_VOTES}


def coerce_point_predictions(predictions: Mapping[str, Any]) -> dict[str, int]:
    coerced = {}
    for cell_id, prediction in predictions.items():
        vote = coerce_vote(prediction)
        if vote is not None:
            coerced[str(cell_id)] = vote
    return coerced


def coerce_brier_predictions(predictions: Mapping[str, Any]) -> dict[str, int | dict[Any, Any]]:
    coerced: dict[str, int | dict[Any, Any]] = {}
    for cell_id, prediction in predictions.items():
        if isinstance(prediction, Mapping):
            coerced[str(cell_id)] = dict(prediction)
            continue
        vote = coerce_vote(prediction)
        if vote is not None:
            coerced[str(cell_id)] = vote
    return coerced


def coerce_vote(prediction: Any) -> int | None:
    if isinstance(prediction, bool):
        return None
    if isinstance(prediction, int) and prediction in VALID_VOTES:
        return prediction
    if isinstance(prediction, float) and prediction.is_integer() and int(prediction) in VALID_VOTES:
        return int(prediction)
    if isinstance(prediction, str) and prediction in {"-1", "0", "1"}:
        return int(prediction)
    if isinstance(prediction, Mapping):
        vote_scores = coerce_vote_scores(prediction)
        if vote_scores:
            return max(vote_scores.items(), key=lambda item: item[1])[0]
    return None


def coerce_vote_scores(prediction: Mapping[str, Any]) -> dict[int, float]:
    vote_scores: dict[int, float] = {}
    for key, value in prediction.items():
        label = coerce_class_label(key)
        if label is None:
            continue
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        if isfinite(score):
            vote_scores[LABEL_TO_VOTE[label]] = score
    return vote_scores


def coerce_class_label(key: Any) -> str | None:
    if isinstance(key, bool):
        return None
    if isinstance(key, int) and key in VALID_VOTES:
        return VOTE_TO_LABEL[key]
    if isinstance(key, str):
        if key in LABEL_TO_VOTE:
            return key
        if key in {"-1", "0", "1"}:
            return VOTE_TO_LABEL[int(key)]
    return None


def extract_json_object(text: str) -> Any:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        return parsed
    raise ValueError("no JSON object found")


def _vote_symbol(vote: Any) -> str:
    if vote == 1:
        return "1"
    if vote == -1:
        return "-1"
    if vote == 0:
        return "0"
    return "?"
