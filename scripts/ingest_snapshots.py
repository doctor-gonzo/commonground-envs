"""Validate Context Engine snapshot exports and build the real-data split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from commonground_scenarios.snapshot_validation import (
    HUMAN_SNAPSHOT_SCHEMA_VERSION,
    HumanSnapshotValidationError,
    contains_direct_identifier,
    validate_human_snapshot,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = (
    ROOT / "environments" / "commonground_predict" / "commonground_predict" / "data"
)
DEFAULT_OUTPUT = DATA_DIR / "eval_real.jsonl"
DEFAULT_MANIFEST = DATA_DIR / "eval_real.manifest.json"
PROTECTED_SPLITS = {"eval_synthetic.jsonl", "eval_ce_demo.jsonl"}
MAX_INPUT_LINE_BYTES = 4 * 1024 * 1024
MANIFEST_VERSION = 2

# Compatibility aliases for callers of the original intake module. The
# implementation is shared with the elicit human-data socket to prevent drift.
SnapshotValidationError = HumanSnapshotValidationError


def contains_identifier(text: str) -> bool:
    return contains_direct_identifier(text)


def validate_snapshot(snapshot: Any) -> dict[str, Any]:
    """Validate one human snapshot through the canonical shared contract."""

    return validate_human_snapshot(snapshot)


def _reject_json_constant(value: str) -> None:
    raise SnapshotValidationError(f"non-standard JSON constant is not allowed: {value}")


def ingest_files(
    input_paths: Sequence[Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
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
            if len(line.encode("utf-8")) > MAX_INPUT_LINE_BYTES:
                errors.append(
                    f"{input_path}:{line_number}: input row exceeds "
                    f"{MAX_INPUT_LINE_BYTES} bytes"
                )
                continue
            try:
                snapshot = json.loads(line, parse_constant=_reject_json_constant)
                validated = validate_snapshot(snapshot)
                session_id = validated["session_id"]
                if session_id in seen_session_ids:
                    raise SnapshotValidationError(f"duplicate session_id: {session_id}")
            except (
                json.JSONDecodeError,
                RecursionError,
                SnapshotValidationError,
            ) as error:
                errors.append(f"{input_path}:{line_number}: {error}")
                continue

            seen_session_ids.add(session_id)
            accepted.append(validated)
            meta = validated["meta"]
            manifest_entries.append(
                {
                    "source_index": source_index,
                    "source_sha256": source_sha256,
                    "source_line": line_number,
                    "session_id": validated["session_id"],
                    "participant_count": len(validated["participants"]),
                    "statement_count": len(validated["statements"]),
                    "cluster_count": len(validated["clusters"]),
                    "k_anonymity": meta["k_anonymity"],
                    "consent_scope": meta["consent_scope"],
                    "redistribution_rights_approved": meta[
                        "redistribution_rights_approved"
                    ],
                    "schema_version": meta["schema_version"],
                    "exporter_version": meta["exporter_version"],
                    "source_commit": meta["source_commit"],
                    "privacy_review": meta["privacy_review"],
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
            raise ValueError(
                f"refusing to overwrite protected split: {destination.name}"
            )
    if str(output_path.resolve()).casefold() == str(manifest_path.resolve()).casefold():
        raise ValueError("output and manifest paths must be different")
    if not snapshots:
        raise ValueError("refusing to publish an empty real-data split")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl = "".join(
        json.dumps(snapshot, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
        for snapshot in snapshots
    )
    manifest = {
        "version": MANIFEST_VERSION,
        "contract": HUMAN_SNAPSHOT_SCHEMA_VERSION,
        "output_file": output_path.name,
        "output_sha256": hashlib.sha256(jsonl.encode("utf-8")).hexdigest(),
        "snapshot_count": len(snapshots),
        "snapshots": list(manifest_entries),
    }
    manifest_json = (
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )

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
    parser.add_argument(
        "inputs", nargs="+", type=Path, help="Context Engine exporter JSONL files"
    )
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="write valid rows despite rejected inputs",
    )
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
