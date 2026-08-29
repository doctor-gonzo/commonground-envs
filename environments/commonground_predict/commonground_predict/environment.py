"""Verifiers environment for Common Ground masked-vote prediction."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from collections.abc import Mapping
from math import isfinite
from pathlib import Path
from typing import Any, Never

import verifiers as legacy_vf
import verifiers.v1 as vf
from commonground_scenarios import (
    HumanSnapshotValidationError,
    validate_human_snapshot,
)
from commonground_score import brier_score as score_brier_score
from commonground_score import probability_reward as score_probability_reward
from commonground_score import vote_accuracy as score_vote_accuracy
from datasets import Dataset
from verifiers.v1.harnesses.null import NullHarness

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
MAX_COMPLETION_CHARS = 65_536
MAX_JSON_NESTING = 64
MAX_JSON_CANDIDATES = 128
CELL_ID_PATTERN = re.compile(r"(0|[1-9]\d*),(0|[1-9]\d*)")


class PredictionJsonParser(legacy_vf.Parser):
    """Extract the last predictions JSON object from a completion."""

    def __init__(self) -> None:
        super().__init__()

    def parse(self, text: str) -> dict[str, Any]:
        try:
            parsed = extract_json_object(text)
        except (RecursionError, ValueError):
            return {}
        if isinstance(parsed, dict):
            return parsed
        return {}

    def parse_answer(self, completion: Any) -> dict[str, Any]:
        """Parse the last assistant message for compatibility with local helpers."""

        if isinstance(completion, str):
            return self.parse(completion)
        if not isinstance(completion, list):
            return {}
        for message in reversed(completion):
            if isinstance(message, Mapping):
                role = message.get("role")
                content = message.get("content")
            else:
                role = getattr(message, "role", None)
                content = getattr(message, "content", None)
            if role == "assistant" and isinstance(content, str):
                return self.parse(content)
        return {}


class PredictionTaskData(vf.TaskData):
    """One immutable masked-vote task and its scoring-side reference data."""

    answer: dict[str, int]
    info: dict[str, Any]
    snapshot: dict[str, Any]


class PredictionTask(vf.Task[PredictionTaskData]):
    """Native Verifiers v1 prediction task."""

    @vf.reward
    async def probability_reward(self, trace: vf.Trace) -> float:
        predictions = exact_probability_predictions(
            parse_prediction_text(trace.last_reply), self.data.answer
        )
        if predictions is None:
            return 0.0
        return float(score_probability_reward(predictions, self.data.answer))

    @vf.metric
    async def vote_accuracy(self, trace: vf.Trace) -> float:
        predictions = exact_probability_predictions(
            parse_prediction_text(trace.last_reply), self.data.answer
        )
        if predictions is None:
            return 0.0
        return float(
            score_vote_accuracy(coerce_point_predictions(predictions), self.data.answer)
        )

    @vf.metric
    async def brier(self, trace: vf.Trace) -> float:
        predictions = exact_probability_predictions(
            parse_prediction_text(trace.last_reply), self.data.answer
        )
        if predictions is None:
            return 1.0 if self.data.answer else 0.0
        return float(score_brier_score(predictions, self.data.answer))


class CommonGroundPredictConfig(vf.TasksetConfig):
    """Public load-time controls for the prediction taskset."""

    masked_vote_count: int | None = None
    min_cluster_count: int | None = None
    data_path: Path | None = None
    split: str = "eval"


class PredictionHarness(NullHarness):
    """Pure-chat harness bundled with the prediction taskset."""


class CommonGroundPredictTaskset(vf.Taskset[PredictionTask, CommonGroundPredictConfig]):
    """Load deterministic masked-vote tasks through the Verifiers v1 API."""

    def load(self) -> list[PredictionTask]:
        bundled_path = _bundled_data_path(self.config.split)
        configured_path = self.config.data_path or os.environ.get(DATA_ENV_VAR)
        resolved_path = Path(configured_path) if configured_path else bundled_path
        snapshots = load_snapshots(
            resolved_path,
            masked_vote_count=self.config.masked_vote_count,
            min_cluster_count=self.config.min_cluster_count,
        )
        return [
            PredictionTask(snapshot_to_task_data(snapshot, index), self.config.task)
            for index, snapshot in enumerate(snapshots)
        ]


def load_taskset(
    masked_vote_count: int | None = None,
    min_cluster_count: int | None = None,
    data_path: str | os.PathLike[str] | None = None,
    split: str = "eval",
    **config_kwargs: Any,
) -> CommonGroundPredictTaskset:
    """Build the native v1 taskset with the public load controls."""

    return CommonGroundPredictTaskset(
        CommonGroundPredictConfig(
            id=ENV_ID,
            masked_vote_count=masked_vote_count,
            min_cluster_count=min_cluster_count,
            data_path=Path(data_path) if data_path is not None else None,
            split=split,
            **config_kwargs,
        )
    )


def load_environment(
    masked_vote_count: int | None = None,
    min_cluster_count: int | None = None,
    data_path: str | os.PathLike[str] | None = None,
    split: str = "eval",
    **kwargs: Any,
) -> legacy_vf.SingleTurnEnv:
    """Build the legacy adapter required by Prime Hosted Evaluations."""

    # Hosted Evaluations still call the v0 factory, while native v1 discovers
    # CommonGroundPredictTaskset through __all__. Keep the two entry points real
    # and separate so neither runner receives a partial compatibility object.
    taskset = load_taskset(
        masked_vote_count=masked_vote_count,
        min_cluster_count=min_cluster_count,
        data_path=data_path,
        split=split,
    )
    rows = [_prediction_task_to_legacy_row(task) for task in taskset]
    dataset = Dataset.from_list(rows)
    parser = PredictionJsonParser()
    rubric = legacy_vf.Rubric(
        funcs=[probability_reward, vote_accuracy, brier],
        weights=[1.0, 0.0, 0.0],
        parser=parser,
    )
    configured_path = data_path or os.environ.get(DATA_ENV_VAR)
    resolved_path = (
        Path(configured_path) if configured_path else _bundled_data_path(split)
    )
    env_args = {
        "masked_vote_count": masked_vote_count,
        "min_cluster_count": min_cluster_count,
        "data_path": str(resolved_path),
        "split": split,
    }
    return legacy_vf.SingleTurnEnv(
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

    if masked_vote_count is not None and type(masked_vote_count) is not int:
        raise ValueError("masked_vote_count must be an integer or null")
    if min_cluster_count is not None and type(min_cluster_count) is not int:
        raise ValueError("min_cluster_count must be an integer or null")

    snapshots = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        snapshot = json.loads(line)
        if (
            isinstance(snapshot, Mapping)
            and isinstance(snapshot.get("meta"), Mapping)
            and snapshot["meta"].get("synthetic") is False
        ):
            try:
                snapshot = validate_human_snapshot(snapshot)
            except HumanSnapshotValidationError as error:
                _raise_snapshot_error(
                    snapshot,
                    path,
                    line_number,
                    f"human snapshot governance validation failed: {error}",
                )
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
    snapshot: Any,
    path: Path,
    line_number: int,
) -> None:
    """Reject structurally or semantically inconsistent prediction snapshots."""

    if not isinstance(snapshot, Mapping):
        _raise_snapshot_error(snapshot, path, line_number, "snapshot must be an object")
    required_fields = {
        "session_id",
        "statements",
        "participants",
        "votes",
        "masked_cells",
        "held_out",
        "clusters",
        "meta",
    }
    missing_fields = sorted(required_fields - set(snapshot))
    if missing_fields:
        _raise_snapshot_error(
            snapshot,
            path,
            line_number,
            f"missing fields={','.join(missing_fields)}",
        )

    session_id = snapshot["session_id"]
    if not isinstance(session_id, str) or not session_id.strip():
        _raise_snapshot_error(
            snapshot, path, line_number, "session_id must be a non-empty string"
        )

    participants = snapshot["participants"]
    if (
        not isinstance(participants, list)
        or not participants
        or not all(
            isinstance(participant, str) and participant.strip()
            for participant in participants
        )
    ):
        _raise_snapshot_error(
            snapshot,
            path,
            line_number,
            "participants must be a non-empty list of strings",
        )
    if len(set(participants)) != len(participants):
        _raise_snapshot_error(
            snapshot, path, line_number, "participants must be unique"
        )

    statements = snapshot["statements"]
    if not isinstance(statements, list) or not statements:
        _raise_snapshot_error(
            snapshot, path, line_number, "statements must be a non-empty list"
        )
    bad_statement_indices = []
    for statement_position, statement in enumerate(statements):
        if not isinstance(statement, Mapping):
            _raise_snapshot_error(
                snapshot,
                path,
                line_number,
                f"statement {statement_position} must be an object",
            )
        actual_index = statement.get("index", "<missing>")
        if type(actual_index) is not int or actual_index != statement_position:
            bad_statement_indices.append(f"{statement_position}:{actual_index}")
        text = statement.get("text")
        if not isinstance(text, str) or not text.strip():
            _raise_snapshot_error(
                snapshot,
                path,
                line_number,
                f"statement {statement_position} must have non-empty text",
            )
    if bad_statement_indices:
        _raise_snapshot_error(
            snapshot,
            path,
            line_number,
            "statements indices="
            f"{','.join(bad_statement_indices[:5])} expected positional",
        )

    votes = snapshot["votes"]
    if not isinstance(votes, list):
        _raise_snapshot_error(
            snapshot, path, line_number, "votes must be a participant-major list"
        )
    participant_count = len(participants)
    statement_count = len(statements)
    dimension_errors = []
    if len(votes) != participant_count:
        dimension_errors.append(
            f"votes rows={len(votes)} participants={participant_count}"
        )
    bad_vote_rows = [
        f"{row_index}:{len(row) if isinstance(row, list) else '<invalid>'}"
        for row_index, row in enumerate(votes)
        if not isinstance(row, list) or len(row) != statement_count
    ]
    if bad_vote_rows:
        dimension_errors.append(
            "votes row_lengths="
            f"{','.join(bad_vote_rows[:5])} statements={statement_count}"
        )
    if dimension_errors:
        _raise_snapshot_error(snapshot, path, line_number, "; ".join(dimension_errors))
    for participant_index, row in enumerate(votes):
        for statement_index, vote in enumerate(row):
            if vote is not None and (type(vote) is not int or vote not in VALID_VOTES):
                _raise_snapshot_error(
                    snapshot,
                    path,
                    line_number,
                    f"invalid vote at {participant_index},{statement_index}: {vote!r}",
                )

    masked_cells = snapshot["masked_cells"]
    if not isinstance(masked_cells, list):
        _raise_snapshot_error(
            snapshot, path, line_number, "masked_cells must be an array"
        )
    normalized_masked_cells: set[str] = set()
    for cell in masked_cells:
        if not isinstance(cell, list) or len(cell) != 2:
            _raise_snapshot_error(
                snapshot, path, line_number, "invalid masked cell shape"
            )
        participant_index, statement_index = cell
        if type(participant_index) is not int or type(statement_index) is not int:
            _raise_snapshot_error(
                snapshot,
                path,
                line_number,
                "masked cell indices must be integers",
            )
        if (
            not 0 <= participant_index < participant_count
            or not 0 <= statement_index < statement_count
        ):
            _raise_snapshot_error(
                snapshot,
                path,
                line_number,
                f"out-of-bounds masked cell {participant_index},{statement_index}",
            )
        cell_id = f"{participant_index},{statement_index}"
        if cell_id in normalized_masked_cells:
            _raise_snapshot_error(
                snapshot, path, line_number, f"duplicate masked cell {cell_id}"
            )
        normalized_masked_cells.add(cell_id)
        if votes[participant_index][statement_index] is not None:
            _raise_snapshot_error(
                snapshot,
                path,
                line_number,
                f"masked vote {cell_id} must be null",
            )

    held_out = snapshot["held_out"]
    if not isinstance(held_out, Mapping):
        _raise_snapshot_error(snapshot, path, line_number, "held_out must be an object")
    for cell_id, vote in held_out.items():
        if not isinstance(cell_id, str) or CELL_ID_PATTERN.fullmatch(cell_id) is None:
            _raise_snapshot_error(
                snapshot, path, line_number, f"invalid held-out cell {cell_id!r}"
            )
        participant_index, statement_index = map(int, cell_id.split(","))
        if (
            not 0 <= participant_index < participant_count
            or not 0 <= statement_index < statement_count
        ):
            _raise_snapshot_error(
                snapshot,
                path,
                line_number,
                f"out-of-bounds held-out cell {cell_id}",
            )
        if type(vote) is not int or vote not in VALID_VOTES:
            _raise_snapshot_error(
                snapshot,
                path,
                line_number,
                f"invalid held-out vote at {cell_id}: {vote!r}",
            )
    if set(held_out) != normalized_masked_cells:
        _raise_snapshot_error(
            snapshot, path, line_number, "held_out must match masked_cells"
        )

    cluster_error = snapshot_cluster_dimension_error(snapshot["clusters"], participants)
    if cluster_error is not None:
        _raise_snapshot_error(snapshot, path, line_number, cluster_error)

    meta = snapshot["meta"]
    if not isinstance(meta, Mapping):
        _raise_snapshot_error(snapshot, path, line_number, "meta must be an object")
    if type(meta.get("synthetic")) is not bool:
        _raise_snapshot_error(
            snapshot, path, line_number, "meta.synthetic must be boolean"
        )


def _raise_snapshot_error(
    snapshot: Any,
    path: Path,
    line_number: int,
    detail: str,
) -> Never:
    session_id = (
        snapshot.get("session_id", "<unknown>")
        if isinstance(snapshot, Mapping)
        else "<unknown>"
    )
    raise ValueError(
        f"invalid snapshot dimensions or values at {path}:{line_number} "
        f"session_id={session_id}: {detail}"
    )


def snapshot_cluster_dimension_error(
    clusters: Any,
    participants: Any,
) -> str | None:
    """Return a dimension error for supported cluster encodings, if any."""

    participant_count = len(participants)
    if not isinstance(clusters, list) or not clusters:
        cluster_count = len(clusters) if isinstance(clusters, list) else "<invalid>"
        return f"clusters={cluster_count} participants={participant_count}"
    object_encoding = [isinstance(cluster, Mapping) for cluster in clusters]
    if any(object_encoding) and not all(object_encoding):
        return "clusters must not mix assignment IDs and cluster objects"
    if all(object_encoding):
        member_indices: list[int] = []
        member_ids: list[str] = []
        for cluster_index, cluster in enumerate(clusters):
            indices = cluster.get("member_indices", [])
            members = cluster.get("members", [])
            if not isinstance(indices, list) or not isinstance(members, list):
                return f"cluster {cluster_index} members must be arrays"
            if any(type(member_index) is not int for member_index in indices):
                return f"cluster {cluster_index} member_indices must be integers"
            if any(
                member_index < 0 or member_index >= participant_count
                for member_index in indices
            ):
                return f"cluster {cluster_index} contains out-of-bounds member_indices"
            if any(not isinstance(member_id, str) for member_id in members):
                return f"cluster {cluster_index} members must be strings"
            if (
                indices
                and members
                and {participants[member_index] for member_index in indices}
                != set(members)
            ):
                return f"cluster {cluster_index} members disagree with member_indices"
            member_indices.extend(indices)
            member_ids.extend(members)
        expected_indices = set(range(participant_count))
        if member_indices:
            unique_indices = set(member_indices)
            if unique_indices != expected_indices or len(member_indices) != len(
                unique_indices
            ):
                return (
                    f"clusters member_indices={len(unique_indices)} unique/"
                    f"{len(member_indices)} total participants={participant_count}"
                )
        if member_ids:
            expected_ids = set(participants)
            unique_ids = set(member_ids)
            if unique_ids != expected_ids or len(member_ids) != len(unique_ids):
                return (
                    f"clusters members={len(unique_ids)} unique/{len(member_ids)} "
                    f"total participants={participant_count}"
                )
        if not member_indices and not member_ids:
            return "cluster objects must contain members or member_indices"
        return None

    if len(clusters) != participant_count:
        return f"clusters={len(clusters)} participants={participant_count}"
    if any(
        isinstance(cluster_id, bool)
        or not isinstance(cluster_id, (int, str))
        or (isinstance(cluster_id, str) and not cluster_id.strip())
        for cluster_id in clusters
    ):
        return "cluster assignments must be integer or non-empty string IDs"
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


def snapshot_to_task_data(
    snapshot: Mapping[str, Any],
    index: int,
) -> PredictionTaskData:
    """Build typed task data without duplicating hidden labels in snapshot state."""

    held_out = {
        str(cell_id): int(vote) for cell_id, vote in snapshot["held_out"].items()
    }
    info = {
        "session_id": snapshot["session_id"],
        "masked_vote_count": len(held_out),
        "cluster_count": snapshot_cluster_count(snapshot),
        "synthetic": bool(snapshot["meta"].get("synthetic")),
    }
    public_snapshot = copy.deepcopy(dict(snapshot))
    public_snapshot.pop("held_out", None)
    return PredictionTaskData(
        idx=index,
        name=str(snapshot["session_id"]),
        prompt=render_prompt(snapshot),
        answer=held_out,
        info=info,
        snapshot=public_snapshot,
    )


def _prediction_task_to_legacy_row(task: PredictionTask) -> dict[str, Any]:
    """Serialize one native task into the canonical v0 dataset row."""

    data = task.data
    return {
        "prompt": [{"role": "user", "content": data.prompt_text}],
        "answer": json.dumps(data.answer, sort_keys=True),
        "info": json.dumps(data.info, sort_keys=True),
        "snapshot": json.dumps(data.snapshot, sort_keys=True),
        "example_id": data.name or str(data.idx),
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
            "Return a probability distribution for every masked cell.",
            "Each distribution must contain exactly agree, disagree, and pass, use finite non-negative numbers, and have a positive total.",
            "Return STRICT JSON only, with this shape:",
            '{"predictions":{"<participant_idx>,<statement_idx>":{"agree":0.0,"disagree":0.0,"pass":0.0}}}',
        ]
    )
    return "\n".join(lines)


async def vote_accuracy(
    completion: list[dict[str, Any]],
    answer: Mapping[str, int] | str,
    parser: PredictionJsonParser,
) -> float:
    """Reward: exact point-prediction accuracy over masked cells."""

    held_out = parse_held_out(answer)
    parsed = exact_probability_predictions(
        parse_completion_predictions(completion, parser), held_out
    )
    if parsed is None:
        return 0.0
    point_predictions = coerce_point_predictions(parsed)
    return float(score_vote_accuracy(point_predictions, held_out))


async def probability_reward(
    completion: list[dict[str, Any]],
    answer: Mapping[str, int] | str,
    parser: PredictionJsonParser,
) -> float:
    """Reward: one minus normalized Brier under the exact response contract."""

    held_out = parse_held_out(answer)
    parsed = exact_probability_predictions(
        parse_completion_predictions(completion, parser), held_out
    )
    if parsed is None:
        return 0.0
    return float(score_probability_reward(parsed, held_out))


async def brier(
    completion: list[dict[str, Any]],
    answer: Mapping[str, int] | str,
    parser: PredictionJsonParser,
) -> float:
    """Metric: normalized 0-1 Brier; invalid forecasts score as uniform."""

    held_out = parse_held_out(answer)
    parsed = exact_probability_predictions(
        parse_completion_predictions(completion, parser), held_out
    )
    if parsed is None:
        return 1.0 if held_out else 0.0
    return float(score_brier_score(parsed, held_out))


def parse_completion_predictions(
    completion: list[dict[str, Any]],
    parser: PredictionJsonParser,
) -> Mapping[str, Any]:
    parsed = parser.parse_answer(completion)
    predictions = parsed.get("predictions", {})
    if not isinstance(predictions, Mapping):
        return {}
    return predictions


def parse_prediction_text(text: str) -> Mapping[str, Any]:
    """Return the predictions mapping from one bounded completion string."""

    parsed = PredictionJsonParser().parse(text)
    predictions = parsed.get("predictions", {})
    return predictions if isinstance(predictions, Mapping) else {}


def parse_held_out(held_out: Mapping[str, int] | str) -> dict[str, int]:
    loaded = json.loads(held_out) if isinstance(held_out, str) else held_out
    return {
        str(cell_id): vote
        for cell_id, vote in loaded.items()
        if type(vote) is int and vote in VALID_VOTES
    }


def coerce_point_predictions(predictions: Mapping[str, Any]) -> dict[str, int]:
    coerced = {}
    for cell_id, prediction in predictions.items():
        vote = coerce_probability_vote(prediction)
        if vote is not None:
            coerced["".join(str(cell_id).split())] = vote
    return coerced


def coerce_brier_predictions(
    predictions: Mapping[str, Any],
) -> dict[str, int | dict[Any, Any]]:
    coerced: dict[str, int | dict[Any, Any]] = {}
    for cell_id, prediction in predictions.items():
        normalized_cell_id = "".join(str(cell_id).split())
        if valid_probability_mapping(prediction):
            coerced[normalized_cell_id] = dict(prediction)
    return coerced


def exact_probability_predictions(
    predictions: Mapping[str, Any], held_out: Mapping[str, int]
) -> dict[str, dict[Any, Any]] | None:
    """Validate exactly one complete forecast for every held-out cell."""

    coerced: dict[str, dict[Any, Any]] = {}
    for cell_id, prediction in predictions.items():
        normalized_cell_id = "".join(str(cell_id).split())
        if normalized_cell_id in coerced or not valid_probability_mapping(prediction):
            return None
        coerced[normalized_cell_id] = dict(prediction)
    if set(coerced) != set(held_out):
        return None
    return coerced


def coerce_probability_vote(prediction: Any) -> int | None:
    """Return an argmax vote only for a complete valid probability mapping."""

    if not valid_probability_mapping(prediction):
        return None
    vote_scores = coerce_vote_scores(prediction)
    return max(vote_scores.items(), key=lambda item: item[1])[0]


def valid_probability_mapping(prediction: Any) -> bool:
    """Enforce the public probability-forecast response contract."""

    if not isinstance(prediction, Mapping) or set(prediction) != set(LABEL_TO_VOTE):
        return False
    values: list[float] = []
    for value in prediction.values():
        if isinstance(value, bool):
            return False
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return False
        if not isfinite(numeric) or numeric < 0:
            return False
        values.append(numeric)
    return sum(values) > 0


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
    """Decode bounded JSON candidates without suffix copies or parser crashes."""

    if not isinstance(text, str):
        raise ValueError("completion must be text")
    if len(text) > MAX_COMPLETION_CHARS:
        raise ValueError(
            f"completion exceeds {MAX_COMPLETION_CHARS} character JSON limit"
        )

    decoder = json.JSONDecoder()
    last_decodable: Any = None
    last_with_predictions: dict[str, Any] | None = None
    found_decodable = False
    for index in _json_candidate_indices(text):
        try:
            parsed, _ = decoder.raw_decode(text, index)
        except (json.JSONDecodeError, RecursionError):
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


def _json_candidate_indices(text: str) -> list[int]:
    """Find object starts in one pass while enforcing depth and work bounds."""

    candidates: list[int] = []
    stack: list[str] = []
    in_string = False
    escaped = False
    pairs = {"}": "{", "]": "["}
    for index, character in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
            continue
        if character in "{[":
            stack.append(character)
            if len(stack) > MAX_JSON_NESTING:
                raise ValueError(
                    f"JSON nesting exceeds maximum depth {MAX_JSON_NESTING}"
                )
            if character == "{":
                candidates.append(index)
                if len(candidates) > MAX_JSON_CANDIDATES:
                    raise ValueError(
                        f"completion exceeds {MAX_JSON_CANDIDATES} JSON candidates"
                    )
            continue
        if character in "}]" and stack:
            if stack[-1] == pairs[character]:
                stack.pop()
            else:
                stack.clear()
    return candidates


def _vote_symbol(vote: Any) -> str:
    if vote == 1:
        return "1"
    if vote == -1:
        return "-1"
    if vote == 0:
        return "0"
    return "?"
