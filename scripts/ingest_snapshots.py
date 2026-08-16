"""Validate Context Engine snapshot exports and build the real-data split."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from math import isfinite
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import unicodedata
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "environments" / "commonground_predict" / "commonground_predict" / "data"
DEFAULT_OUTPUT = DATA_DIR / "eval_real.jsonl"
DEFAULT_MANIFEST = DATA_DIR / "eval_real.manifest.json"
PROTECTED_SPLITS = {"eval_synthetic.jsonl", "eval_ce_demo.jsonl"}
VALID_VOTES = {-1, 0, 1}
MIN_K_ANONYMITY = 5
TOP_LEVEL_FIELDS = {
    "session_id",
    "statements",
    "participants",
    "votes",
    "masked_cells",
    "held_out",
    "clusters",
    "stats",
    "meta",
}
STATEMENT_FIELDS = {"index", "text"}
CLUSTER_FIELDS = {"id", "members", "member_indices", "center"}
META_FIELDS = {"k_anonymity", "source", "synthetic", "seed"}
COMMENT_STAT_FIELDS = {
    "commentIndex",
    "agrees",
    "disagrees",
    "unsure",
    "total",
    "responded",
    "extremity",
    "divisiveness",
}

ADDRESS_PATTERN = re.compile(r"0x[0-9a-f]{40}", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)
ENS_PATTERN = re.compile(r"[^\s@.]+(?:\.[^\s@.]+)*\.eth\b", re.IGNORECASE)


class SnapshotValidationError(ValueError):
    """Raised when an exported snapshot is unsafe or malformed."""


def contains_identifier(text: str) -> bool:
    normalized = unicodedata.normalize("NFKC", text)
    return bool(
        ADDRESS_PATTERN.search(normalized)
        or EMAIL_PATTERN.search(normalized)
        or ENS_PATTERN.search(normalized)
    )


def validate_snapshot(snapshot: Any) -> dict[str, Any]:
    if not isinstance(snapshot, dict):
        raise SnapshotValidationError("snapshot must be a JSON object")
    _require_exact_fields(snapshot, TOP_LEVEL_FIELDS, "snapshot")

    session_id = snapshot.get("session_id")
    if (
        not isinstance(session_id, str)
        or not session_id.strip()
        or session_id != session_id.strip()
    ):
        raise SnapshotValidationError("session_id must be a non-empty string")
    if contains_identifier(session_id):
        raise SnapshotValidationError("session_id contains a redacted identifier pattern")

    statements = snapshot.get("statements")
    participants = snapshot.get("participants")
    votes = snapshot.get("votes")
    if not isinstance(statements, list) or not statements:
        raise SnapshotValidationError("statements must be a non-empty list")
    if not isinstance(participants, list) or not participants:
        raise SnapshotValidationError("participants must be a non-empty list")
    if not isinstance(votes, list):
        raise SnapshotValidationError("votes must be a participant-major list")

    expected_participants = [f"p{participant_index:03d}" for participant_index in range(len(participants))]
    if participants != expected_participants:
        raise SnapshotValidationError(
            "participants must use positional pseudonyms p000, p001, ..."
        )

    for statement_index, statement in enumerate(statements):
        if not isinstance(statement, Mapping):
            raise SnapshotValidationError(f"statement {statement_index} must be an object")
        _require_exact_fields(statement, STATEMENT_FIELDS, f"statement {statement_index}")
        actual_index = statement.get("index")
        if type(actual_index) is not int or actual_index != statement_index:
            raise SnapshotValidationError(
                f"statement {statement_index} must use positional index {statement_index}"
            )
        text = statement.get("text")
        if not isinstance(text, str) or not text.strip():
            raise SnapshotValidationError(f"statement {statement_index} must have text")
        if contains_identifier(text):
            raise SnapshotValidationError(
                f"statement {statement_index} contains a redacted identifier pattern"
            )

    if len(votes) != len(participants):
        raise SnapshotValidationError(
            "votes must be participant-major: "
            f"rows={len(votes)} participants={len(participants)}"
        )
    for participant_index, row in enumerate(votes):
        if not isinstance(row, list) or len(row) != len(statements):
            row_length = len(row) if isinstance(row, list) else "not-a-list"
            raise SnapshotValidationError(
                f"votes row {participant_index} is ragged: "
                f"length={row_length} statements={len(statements)}"
            )
        for statement_index, vote in enumerate(row):
            if vote is not None and (type(vote) is not int or vote not in VALID_VOTES):
                raise SnapshotValidationError(
                    f"invalid vote at {participant_index},{statement_index}: {vote!r}"
                )

    masked_cells = snapshot.get("masked_cells")
    if masked_cells != []:
        raise SnapshotValidationError("real intake requires empty masked_cells")
    held_out = snapshot.get("held_out")
    if held_out != {}:
        raise SnapshotValidationError("real intake requires empty held_out labels")

    meta = snapshot.get("meta")
    if not isinstance(meta, Mapping):
        raise SnapshotValidationError("meta must be an object")
    _require_exact_fields(meta, META_FIELDS, "meta")
    k_anonymity = meta.get("k_anonymity")
    if (
        isinstance(k_anonymity, bool)
        or not isinstance(k_anonymity, int)
        or k_anonymity < MIN_K_ANONYMITY
    ):
        raise SnapshotValidationError(
            f"meta.k_anonymity must be at least {MIN_K_ANONYMITY}"
        )
    if meta.get("synthetic") is not False:
        raise SnapshotValidationError("real intake requires meta.synthetic=false")
    if meta.get("source") != "context-engine-session":
        raise SnapshotValidationError(
            'real intake requires meta.source="context-engine-session"'
        )
    seed = meta.get("seed")
    if type(seed) is not int:
        raise SnapshotValidationError("meta.seed must be an integer")

    cluster_sizes = validate_clusters(snapshot.get("clusters"), participants)
    small_clusters = sorted(size for size in cluster_sizes if size < k_anonymity)
    if small_clusters:
        raise SnapshotValidationError(
            f"cluster size below k={k_anonymity}: {small_clusters}"
        )
    stats = validate_stats(snapshot.get("stats"), len(statements))
    return {
        "session_id": session_id,
        "statements": [
            {"index": statement["index"], "text": statement["text"]}
            for statement in statements
        ],
        "participants": list(expected_participants),
        "votes": [list(row) for row in votes],
        "masked_cells": [],
        "held_out": {},
        "clusters": [
            {
                "id": cluster["id"],
                "members": list(cluster["members"]),
                "member_indices": list(cluster["member_indices"]),
                "center": list(cluster["center"]),
            }
            for cluster in snapshot["clusters"]
        ],
        "stats": stats,
        "meta": {
            "k_anonymity": k_anonymity,
            "source": "context-engine-session",
            "synthetic": False,
            "seed": seed,
        },
    }


def validate_clusters(clusters: Any, participants: Sequence[str]) -> list[int]:
    if not isinstance(clusters, list) or not clusters:
        raise SnapshotValidationError("clusters must be a non-empty list")
    if not all(isinstance(cluster, Mapping) for cluster in clusters):
        raise SnapshotValidationError("clusters must use exporter cluster objects")

    member_indices: list[int] = []
    sizes: list[int] = []
    cluster_ids: set[int] = set()
    for cluster_index, cluster in enumerate(clusters):
        _require_exact_fields(cluster, CLUSTER_FIELDS, f"cluster {cluster_index}")
        cluster_id = cluster.get("id")
        if type(cluster_id) is not int or cluster_id in cluster_ids:
            raise SnapshotValidationError(
                f"cluster {cluster_index} requires a unique integer id"
            )
        cluster_ids.add(cluster_id)
        raw_indices = cluster.get("member_indices")
        if not isinstance(raw_indices, list):
            raise SnapshotValidationError(
                f"cluster {cluster_index} requires member_indices"
            )
        if any(type(index) is not int for index in raw_indices):
            raise SnapshotValidationError(
                f"cluster {cluster_index} has invalid member_indices"
            )
        raw_members = cluster.get("members")
        if not isinstance(raw_members, list) or len(raw_members) != len(raw_indices):
            raise SnapshotValidationError(
                f"cluster {cluster_index} members must match member_indices"
            )
        expected_members = [participants[index] for index in raw_indices if 0 <= index < len(participants)]
        if len(expected_members) != len(raw_indices) or raw_members != expected_members:
            raise SnapshotValidationError(
                f"cluster {cluster_index} members do not match participant indices"
            )
        center = cluster.get("center")
        if not isinstance(center, list) or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
            for value in center
        ):
            raise SnapshotValidationError(
                f"cluster {cluster_index} center must contain finite numbers"
            )
        member_indices.extend(raw_indices)
        sizes.append(len(raw_indices))

    if sorted(member_indices) != list(range(len(participants))):
        raise SnapshotValidationError(
            "cluster member_indices must partition all participants exactly once"
        )
    return sizes


def validate_stats(stats: Any, statement_count: int) -> dict[str, Any]:
    if not isinstance(stats, Mapping):
        raise SnapshotValidationError("stats must be an object")
    _require_exact_fields(stats, {"comment"}, "stats")
    comments = stats.get("comment")
    if not isinstance(comments, list) or len(comments) != statement_count:
        raise SnapshotValidationError(
            f"stats.comment must contain {statement_count} entries"
        )

    validated_comments: list[dict[str, int | float]] = []
    integer_fields = {"commentIndex", "agrees", "disagrees", "unsure", "total", "responded"}
    for comment_index, comment in enumerate(comments):
        if not isinstance(comment, Mapping):
            raise SnapshotValidationError(f"stats.comment {comment_index} must be an object")
        _require_exact_fields(comment, COMMENT_STAT_FIELDS, f"stats.comment {comment_index}")
        if type(comment.get("commentIndex")) is not int or comment["commentIndex"] != comment_index:
            raise SnapshotValidationError(
                f"stats.comment {comment_index} must use positional commentIndex"
            )
        for field in integer_fields - {"commentIndex"}:
            if type(comment.get(field)) is not int or comment[field] < 0:
                raise SnapshotValidationError(
                    f"stats.comment {comment_index}.{field} must be a non-negative integer"
                )
        for field in {"extremity", "divisiveness"}:
            value = comment.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
            ):
                raise SnapshotValidationError(
                    f"stats.comment {comment_index}.{field} must be finite"
                )
        validated_comments.append({field: comment[field] for field in COMMENT_STAT_FIELDS})
    return {"comment": validated_comments}


def _require_exact_fields(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise SnapshotValidationError(
            f"{label} fields mismatch: missing={missing} unexpected={unexpected}"
        )


def _reject_json_constant(value: str) -> None:
    raise SnapshotValidationError(f"non-standard JSON constant is not allowed: {value}")


def ingest_files(input_paths: Sequence[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    accepted: list[dict[str, Any]] = []
    manifest_entries: list[dict[str, Any]] = []
    errors: list[str] = []
    seen_session_ids: set[str] = set()

    for source_index, input_path in enumerate(input_paths):
        try:
            input_bytes = input_path.read_bytes()
            lines = input_bytes.decode("utf-8").splitlines()
        except (OSError, UnicodeError) as error:
            errors.append(f"{input_path}: unable to read input: {error}")
            continue
        source_sha256 = hashlib.sha256(input_bytes).hexdigest()
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                snapshot = json.loads(line, parse_constant=_reject_json_constant)
                if not isinstance(snapshot, dict):
                    raise SnapshotValidationError("snapshot must be a JSON object")
                session_id = snapshot.get("session_id")
                if isinstance(session_id, str) and session_id in seen_session_ids:
                    raise SnapshotValidationError(f"duplicate session_id: {session_id}")
                if isinstance(session_id, str):
                    seen_session_ids.add(session_id)
                validated = validate_snapshot(snapshot)
            except (json.JSONDecodeError, SnapshotValidationError) as error:
                errors.append(f"{input_path}:{line_number}: {error}")
                continue

            accepted.append(validated)
            manifest_entries.append(
                {
                    "source_index": source_index,
                    "source_sha256": source_sha256,
                    "source_line": line_number,
                    "session_id": validated["session_id"],
                    "participant_count": len(validated["participants"]),
                    "statement_count": len(validated["statements"]),
                    "cluster_count": len(validate_clusters(validated["clusters"], validated["participants"])),
                    "k_anonymity": validated["meta"]["k_anonymity"],
                }
            )
    return accepted, manifest_entries, errors


def write_outputs(
    snapshots: Sequence[Mapping[str, Any]],
    manifest_entries: Sequence[Mapping[str, Any]],
    output_path: Path,
    manifest_path: Path,
) -> None:
    protected_names = {name.casefold() for name in PROTECTED_SPLITS}
    for destination in (output_path, manifest_path):
        if destination.name.casefold() in protected_names:
            raise ValueError(f"refusing to overwrite protected split: {destination.name}")
    if str(output_path.resolve()).casefold() == str(manifest_path.resolve()).casefold():
        raise ValueError("output and manifest paths must be different")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl = "".join(
        json.dumps(snapshot, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        for snapshot in snapshots
    )
    manifest = {
        "version": 1,
        "output_file": output_path.name,
        "snapshot_count": len(snapshots),
        "snapshots": list(manifest_entries),
    }
    manifest_json = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"

    output_temporary: Path | None = None
    manifest_temporary: Path | None = None
    output_backup: Path | None = None
    manifest_backup: Path | None = None
    try:
        output_temporary = _write_temporary(output_path, jsonl)
        manifest_temporary = _write_temporary(manifest_path, manifest_json)
        output_backup = _backup_existing(output_path)
        manifest_backup = _backup_existing(manifest_path)
        try:
            os.replace(output_temporary, output_path)
            os.replace(manifest_temporary, manifest_path)
        except OSError:
            _restore_destination(output_path, output_backup)
            _restore_destination(manifest_path, manifest_backup)
            raise
    finally:
        for temporary_path in (
            output_temporary,
            manifest_temporary,
            output_backup,
            manifest_backup,
        ):
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)


def _write_temporary(destination: Path, content: str) -> Path:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        temporary_file.write(content)
        return Path(temporary_file.name)


def _backup_existing(destination: Path) -> Path | None:
    if not destination.exists():
        return None
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".backup",
        delete=False,
    ) as backup_file:
        backup_path = Path(backup_file.name)
    shutil.copy2(destination, backup_path)
    return backup_path


def _restore_destination(destination: Path, backup: Path | None) -> None:
    if backup is None:
        destination.unlink(missing_ok=True)
        return
    os.replace(backup, destination)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Context Engine exporter JSONL files")
    parser.add_argument("--skip-invalid", action="store_true", help="write valid rows despite rejected inputs")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    snapshots, manifest_entries, errors = ingest_files(args.inputs)
    for error in errors:
        print(f"rejected: {error}", file=sys.stderr)
    if errors and not args.skip_invalid:
        return 1
    try:
        write_outputs(snapshots, manifest_entries, args.output, args.manifest)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"accepted {len(snapshots)} snapshots into {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
