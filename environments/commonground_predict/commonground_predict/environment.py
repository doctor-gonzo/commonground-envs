"""Verifiers environment for Common Ground masked-vote prediction."""

from __future__ import annotations

import copy
import hashlib
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
DATA_DIR = Path(__file__).resolve().parent / "data"
BUNDLED_EVAL_PATH = DATA_DIR / "eval_synthetic.jsonl"
BUNDLED_TRAIN_PATH = DATA_DIR / "train_synthetic.jsonl"
BUNDLED_CE_DEMO_PATH = DATA_DIR / "eval_ce_demo.jsonl"
BUNDLED_SPLIT_PATHS = {
    "eval": BUNDLED_EVAL_PATH,
    "train": BUNDLED_TRAIN_PATH,
    "ce-demo": BUNDLED_CE_DEMO_PATH,
}
VALID_VOTES = {-1, 0, 1}
LABEL_TO_VOTE = {"agree": 1, "disagree": -1, "pass": 0}
VOTE_TO_LABEL = {vote: label for label, vote in LABEL_TO_VOTE.items()}


class PredictionJsonParser(vf.Parser):
    """Extract the last predictions JSON object from a completion."""

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
    split: str = "eval",
    **kwargs: Any,
) -> vf.SingleTurnEnv:
    """Build the deterministic single-turn masked-vote prediction environment."""

    bundled_path = _bundled_data_path(split)
    configured_path = data_path or os.environ.get(DATA_ENV_VAR)
    resolved_path = Path(configured_path) if configured_path else bundled_path
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
        "split": split,
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


def _bundled_data_path(split: str) -> Path:
    """Resolve a named bundled split to its packaged JSONL path."""

    try:
        return BUNDLED_SPLIT_PATHS[split]
    except KeyError:
        valid_names = ", ".join(BUNDLED_SPLIT_PATHS)
        raise ValueError(
            f"unknown split {split!r}; valid splits: {valid_names}"
        ) from None


def load_snapshots(
    path: Path,
    masked_vote_count: int | None,
    min_cluster_count: int | None,
) -> list[dict[str, Any]]:
    """Load JSONL snapshots and apply deterministic difficulty filters."""

    snapshots = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        snapshot = json.loads(line)
        validate_snapshot_dimensions(snapshot, path, line_number)
        if min_cluster_count is not None:
            cluster_count = snapshot_cluster_count(snapshot)
            if cluster_count < min_cluster_count:
                continue
        snapshots.append(apply_masked_vote_count(snapshot, masked_vote_count))
    if not snapshots:
        raise ValueError(f"no snapshots loaded from {path}")
    if masked_vote_count is None and all(
        not snapshot.get("held_out") for snapshot in snapshots
    ):
        raise ValueError(
            f"unmasked snapshots loaded from {path}; pass masked_vote_count=N "
            "or provide pre-masked data"
        )
    return snapshots


def validate_snapshot_dimensions(
    snapshot: Mapping[str, Any],
    path: Path,
    line_number: int,
) -> None:
    """Reject snapshots whose participant-major dimensions are inconsistent."""

    participants = snapshot["participants"]
    statements = snapshot["statements"]
    votes = snapshot["votes"]
    clusters = snapshot.get("clusters", [])

    participant_count = len(participants)
    statement_count = len(statements)
    errors = []

    if len(votes) != participant_count:
        errors.append(f"votes rows={len(votes)} participants={participant_count}")

    bad_vote_rows = [
        f"{row_index}:{len(row)}"
        for row_index, row in enumerate(votes)
        if len(row) != statement_count
    ]
    if bad_vote_rows:
        errors.append(
            "votes row_lengths="
            f"{','.join(bad_vote_rows[:5])} statements={statement_count}"
        )

    bad_statement_indices = []
    for statement_position, statement in enumerate(statements):
        actual_index = (
            statement.get("index", "<missing>")
            if isinstance(statement, Mapping)
            else "<invalid>"
        )
        if actual_index != statement_position:
            bad_statement_indices.append(f"{statement_position}:{actual_index}")
    if bad_statement_indices:
        errors.append(
            "statements indices="
            f"{','.join(bad_statement_indices[:5])} expected positional"
        )

    cluster_error = snapshot_cluster_dimension_error(clusters, participants)
    if cluster_error is not None:
        errors.append(cluster_error)

    if errors:
        session_id = snapshot.get("session_id", "<unknown>")
        joined_errors = "; ".join(errors)
        raise ValueError(
            f"invalid snapshot dimensions at {path}:{line_number} "
            f"session_id={session_id}: {joined_errors}"
        )


def snapshot_cluster_dimension_error(
    clusters: Any,
    participants: Any,
) -> str | None:
    """Return a dimension error for supported cluster encodings, if any."""

    participant_count = len(participants)
    if (
        isinstance(clusters, list)
        and clusters
        and all(isinstance(cluster, Mapping) for cluster in clusters)
    ):
        member_indices = [
            int(member_index)
            for cluster in clusters
            for member_index in cluster.get("member_indices", [])
        ]
        if member_indices:
            expected_indices = set(range(participant_count))
            unique_indices = set(member_indices)
            if unique_indices != expected_indices or len(member_indices) != len(
                unique_indices
            ):
                return (
                    f"clusters member_indices={len(unique_indices)} unique/{len(member_indices)} total "
                    f"participants={participant_count}"
                )
            return None

        member_ids = [
            str(member_id)
            for cluster in clusters
            for member_id in cluster.get("members", [])
        ]
        if member_ids:
            expected_ids = {str(participant_id) for participant_id in participants}
            unique_ids = set(member_ids)
            if unique_ids != expected_ids or len(member_ids) != len(unique_ids):
                return (
                    f"clusters members={len(unique_ids)} unique/{len(member_ids)} total "
                    f"participants={participant_count}"
                )
            return None

    if len(clusters) != participant_count:
        return f"clusters={len(clusters)} participants={participant_count}"
    return None


def apply_masked_vote_count(
    snapshot: dict[str, Any],
    masked_vote_count: int | None,
    seed: str | None = None,
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
    session_id = prepared["session_id"]
    ordered_candidates = sorted(
        candidates,
        key=lambda cell: hashlib.sha256(
            (
                f"{session_id}:{cell[0]},{cell[1]}"
                if seed is None
                else f"{seed}:{session_id}:{cell[0]},{cell[1]}"
            ).encode()
        ).hexdigest(),
    )
    for cell in ordered_candidates:
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
    prepared["masked_cells"] = [
        [participant_index, statement_index]
        for participant_index, statement_index in selected
    ]
    prepared["held_out"] = held_out
    return prepared


def reconstruct_known_votes(snapshot: Mapping[str, Any]) -> list[list[int | None]]:
    """Reconstruct true values for held-out cells while preserving unknown cells."""

    votes = [list(row) for row in snapshot["votes"]]
    for cell_id, vote in snapshot.get("held_out", {}).items():
        participant_index_text, statement_index_text = str(cell_id).split(
            ",", maxsplit=1
        )
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
    held_out = {
        str(cell_id): int(vote) for cell_id, vote in snapshot["held_out"].items()
    }
    info = {
        "session_id": snapshot["session_id"],
        "masked_vote_count": len(held_out),
        "cluster_count": snapshot_cluster_count(snapshot),
        "synthetic": bool(snapshot["meta"].get("synthetic")),
    }
    return {
        "prompt": [{"role": "user", "content": render_prompt(snapshot)}],
        "answer": json.dumps(held_out, sort_keys=True),
        "info": json.dumps(info, sort_keys=True),
        "snapshot": json.dumps(snapshot, sort_keys=True),
    }


def render_prompt(snapshot: Mapping[str, Any]) -> str:
    """Render the compact masked-vote prediction prompt."""

    statements = snapshot["statements"]
    masked_cells = [
        f"{participant_index},{statement_index}"
        for participant_index, statement_index in snapshot["masked_cells"]
    ]
    lines = [
        "Predict the held-out votes in this deliberation snapshot.",
        "Votes use 1=agree, -1=disagree, 0=pass/unsure, ?=not seen or masked.",
        "",
        "Statements:",
    ]
    lines.extend(
        f"{statement['index']}: {statement['text']}" for statement in statements
    )
    lines.extend(
        [
            "",
            "Visible vote matrix:",
            "columns: " + " ".join(str(statement["index"]) for statement in statements),
        ]
    )
    for participant_index, row in enumerate(snapshot["votes"]):
        lines.append(
            f"p{participant_index:02d}: " + " ".join(_vote_symbol(vote) for vote in row)
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
    answer: Mapping[str, int] | str,
    parser: PredictionJsonParser,
) -> float:
    """Reward: exact point-prediction accuracy over masked cells."""

    parsed = parse_completion_predictions(completion, parser)
    point_predictions = coerce_point_predictions(parsed)
    return score_vote_accuracy(point_predictions, parse_held_out(answer))


async def brier(
    completion: list[dict[str, Any]],
    answer: Mapping[str, int] | str,
    parser: PredictionJsonParser,
) -> float:
    """Metric: delegated Brier score; invalid probability mappings score uniform."""

    parsed = parse_completion_predictions(completion, parser)
    brier_predictions = coerce_brier_predictions(parsed)
    return score_brier_score(brier_predictions, parse_held_out(answer))


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
    loaded = json.loads(held_out) if isinstance(held_out, str) else held_out
    return {
        str(cell_id): int(vote)
        for cell_id, vote in loaded.items()
        if int(vote) in VALID_VOTES
    }


def coerce_point_predictions(predictions: Mapping[str, Any]) -> dict[str, int]:
    coerced = {}
    for cell_id, prediction in predictions.items():
        vote = coerce_vote(prediction)
        if vote is not None:
            coerced["".join(str(cell_id).split())] = vote
    return coerced


def coerce_brier_predictions(
    predictions: Mapping[str, Any],
) -> dict[str, int | dict[Any, Any]]:
    coerced: dict[str, int | dict[Any, Any]] = {}
    for cell_id, prediction in predictions.items():
        normalized_cell_id = "".join(str(cell_id).split())
        if isinstance(prediction, Mapping):
            coerced[normalized_cell_id] = dict(prediction)
            continue
        vote = coerce_vote(prediction)
        if vote is not None:
            coerced[normalized_cell_id] = vote
    return coerced


def coerce_vote(prediction: Any) -> int | None:
    if isinstance(prediction, bool):
        return None
    if isinstance(prediction, int) and prediction in VALID_VOTES:
        return prediction
    if (
        isinstance(prediction, float)
        and prediction.is_integer()
        and int(prediction) in VALID_VOTES
    ):
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
    last_decodable: Any = None
    last_with_predictions: dict[str, Any] | None = None
    found_decodable = False
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        found_decodable = True
        last_decodable = parsed
        if isinstance(parsed, dict) and isinstance(parsed.get("predictions"), dict):
            last_with_predictions = parsed
    if last_with_predictions is not None:
        return last_with_predictions
    if found_decodable:
        return last_decodable
    raise ValueError("no JSON object found")


def _vote_symbol(vote: Any) -> str:
    if vote == 1:
        return "1"
    if vote == -1:
        return "-1"
    if vote == 0:
        return "0"
    return "?"
