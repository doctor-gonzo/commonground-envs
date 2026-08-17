"""Aggregate complete local Common Ground evaluation runs.

The saved-result contract follows ``verifiers==0.1.14``: each run directory
contains ``metadata.json`` and one JSON object per rollout in ``results.jsonl``.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIRS_GLOB = "environments/*/outputs/evals/*/*"
METRIC_NAMES = ("vote_accuracy", "brier", "finding_f1", "question_utility")


class InvalidRunError(ValueError):
    """Raised when a saved run cannot support a complete baseline row."""


@dataclass(frozen=True)
class Summary:
    model: str
    environment: str
    run_id: str
    rollout_count: int
    reward_mean: float
    reward_std: float
    metrics: Mapping[str, tuple[float, float]]


@dataclass(frozen=True)
class CompleteRun:
    model: str
    environment: str
    run_id: str
    metadata_timestamp_ns: int
    rewards: tuple[float, ...]
    metrics: Mapping[str, tuple[float, ...]]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate complete Common Ground baseline eval outputs."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Repository root containing environments/*/outputs/evals (default: repo root).",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Also write machine-readable summary rows to this CSV path.",
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Include every complete run instead of only the newest env/model pair.",
    )
    return parser.parse_args(argv)


def discover_result_paths(root: Path) -> list[Path]:
    """Find saved results and diagnose vf-eval's log-only default."""

    run_dirs = sorted(
        (path for path in root.glob(DEFAULT_RUN_DIRS_GLOB) if path.is_dir()),
        key=lambda path: path.as_posix(),
    )
    result_paths: list[Path] = []
    partial_run_dirs: list[Path] = []
    log_only_run_dirs: list[Path] = []
    for run_dir in run_dirs:
        metadata_path = run_dir / "metadata.json"
        result_path = run_dir / "results.jsonl"
        has_metadata = metadata_path.is_file()
        has_results = result_path.is_file()
        if has_metadata and has_results:
            result_paths.append(result_path)
        elif has_metadata or has_results:
            partial_run_dirs.append(run_dir)
        elif any(run_dir.glob("*.log")):
            log_only_run_dirs.append(run_dir)

    if partial_run_dirs:
        rendered = ", ".join(path.as_posix() for path in partial_run_dirs)
        raise InvalidRunError(f"partially saved eval run(s): {rendered}")
    if not result_paths:
        message = f"no saved eval results found under {root / 'environments'}"
        if log_only_run_dirs:
            message += (
                f"; found {len(log_only_run_dirs)} log-only run dir(s). "
                "vf-eval requires --save-results to write metadata.json and "
                "results.jsonl"
            )
        raise InvalidRunError(message)
    return result_paths


def load_summaries(root: Path, *, all_runs: bool = False) -> list[Summary]:
    result_paths = discover_result_paths(root)
    runs = [load_complete_run(result_path) for result_path in result_paths]
    selected_runs = runs if all_runs else newest_runs(runs)

    return [
        summarize_run(run)
        for run in sorted(
            selected_runs,
            key=lambda run: (
                run.model.casefold(),
                run.environment,
                run.metadata_timestamp_ns,
                run.run_id,
            ),
        )
    ]


def newest_runs(runs: Sequence[CompleteRun]) -> list[CompleteRun]:
    """Select the latest metadata file per environment/model pair."""

    newest: dict[tuple[str, str], CompleteRun] = {}
    for run in runs:
        key = (run.environment, run.model)
        incumbent = newest.get(key)
        if incumbent is None or (
            run.metadata_timestamp_ns,
            run.run_id,
        ) > (
            incumbent.metadata_timestamp_ns,
            incumbent.run_id,
        ):
            newest[key] = run
    return list(newest.values())


def summarize_run(run: CompleteRun) -> Summary:
    return Summary(
        model=run.model,
        environment=run.environment,
        run_id=run.run_id,
        rollout_count=len(run.rewards),
        reward_mean=statistics.fmean(run.rewards),
        reward_std=statistics.pstdev(run.rewards),
        metrics={
            metric_name: summarize(run.metrics[metric_name])
            for metric_name in METRIC_NAMES
            if metric_name in run.metrics
        },
    )


def load_complete_run(result_path: Path) -> CompleteRun:
    metadata_path = result_path.with_name("metadata.json")
    metadata = load_json_object(metadata_path)
    environment = require_nonempty_string(metadata, "env_id", metadata_path)
    model = require_nonempty_string(metadata, "model", metadata_path)
    num_examples = require_positive_int(metadata, "num_examples", metadata_path)
    rollouts_per_example = require_positive_int(
        metadata, "rollouts_per_example", metadata_path
    )
    expected_count = num_examples * rollouts_per_example

    outputs = load_json_lines(result_path)
    if len(outputs) != expected_count:
        raise InvalidRunError(
            f"{result_path}: expected {expected_count} rollouts "
            f"({num_examples} examples x {rollouts_per_example}), found {len(outputs)}"
        )

    counts_by_example: dict[int, int] = {}
    rewards: list[float] = []
    metric_values: dict[str, list[float]] = {}
    expected_metric_keys: tuple[str, ...] | None = None
    for line_number, output in enumerate(outputs, start=1):
        example_id = output.get("example_id")
        if isinstance(example_id, bool) or not isinstance(example_id, int):
            raise InvalidRunError(
                f"{result_path}:{line_number}: example_id must be an integer"
            )
        counts_by_example[example_id] = counts_by_example.get(example_id, 0) + 1
        rewards.append(require_number(output, "reward", result_path, line_number))

        raw_metrics = output.get("metrics")
        if not isinstance(raw_metrics, Mapping):
            raise InvalidRunError(
                f"{result_path}:{line_number}: metrics must be an object"
            )
        known_metric_keys = tuple(name for name in METRIC_NAMES if name in raw_metrics)
        if expected_metric_keys is None:
            expected_metric_keys = known_metric_keys
        elif known_metric_keys != expected_metric_keys:
            raise InvalidRunError(
                f"{result_path}:{line_number}: inconsistent logged metrics"
            )
        for metric_name in known_metric_keys:
            metric_values.setdefault(metric_name, []).append(
                require_number(raw_metrics, metric_name, result_path, line_number)
            )

    if len(counts_by_example) != num_examples:
        raise InvalidRunError(
            f"{result_path}: expected {num_examples} distinct examples, "
            f"found {len(counts_by_example)}"
        )
    incorrect_counts = {
        example_id: count
        for example_id, count in counts_by_example.items()
        if count != rollouts_per_example
    }
    if incorrect_counts:
        raise InvalidRunError(
            f"{result_path}: expected {rollouts_per_example} rollouts per example; "
            f"found {incorrect_counts}"
        )

    return CompleteRun(
        model=model,
        environment=environment,
        run_id=result_path.parent.name,
        metadata_timestamp_ns=metadata_path.stat().st_mtime_ns,
        rewards=tuple(rewards),
        metrics={name: tuple(values) for name, values in metric_values.items()},
    )


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise InvalidRunError(
            f"missing metadata beside saved results: {path}"
        ) from error
    except json.JSONDecodeError as error:
        raise InvalidRunError(f"invalid JSON in {path}: {error}") from error
    if not isinstance(value, dict):
        raise InvalidRunError(f"{path}: expected a JSON object")
    return value


def load_json_lines(path: Path) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise InvalidRunError(
                f"invalid JSON in {path}:{line_number}: {error}"
            ) from error
        if not isinstance(value, dict):
            raise InvalidRunError(f"{path}:{line_number}: expected a JSON object")
        outputs.append(value)
    return outputs


def require_nonempty_string(value: Mapping[str, Any], key: str, path: Path) -> str:
    field = value.get(key)
    if not isinstance(field, str) or not field.strip():
        raise InvalidRunError(f"{path}: {key} must be a non-empty string")
    return field


def require_positive_int(value: Mapping[str, Any], key: str, path: Path) -> int:
    field = value.get(key)
    if isinstance(field, bool) or not isinstance(field, int) or field <= 0:
        raise InvalidRunError(f"{path}: {key} must be a positive integer")
    return field


def require_number(
    value: Mapping[str, Any], key: str, path: Path, line_number: int
) -> float:
    field = value.get(key)
    if isinstance(field, bool) or not isinstance(field, int | float):
        raise InvalidRunError(f"{path}:{line_number}: {key} must be numeric")
    return float(field)


def summarize(values: Iterable[float]) -> tuple[float, float]:
    collected = tuple(values)
    if not collected:
        raise InvalidRunError("cannot summarize an empty metric")
    return statistics.fmean(collected), statistics.pstdev(collected)


def format_mean_std(mean: float, std: float) -> str:
    return f"{mean:.3f} ± {std:.3f}"


def render_markdown(summaries: Sequence[Summary]) -> str:
    headers = [
        "Model",
        "Environment",
        "Run ID",
        "Rollouts",
        "Reward (mean ± std)",
    ]
    headers.extend(f"{name} (mean ± std)" for name in METRIC_NAMES)
    alignments = ["---", "---", "---", "---:", "---:"]
    alignments.extend("---:" for _ in METRIC_NAMES)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(alignments) + " |",
    ]
    for summary in summaries:
        row = [
            summary.model,
            summary.environment,
            summary.run_id,
            str(summary.rollout_count),
            format_mean_std(summary.reward_mean, summary.reward_std),
        ]
        row.extend(
            format_mean_std(*summary.metrics[name]) if name in summary.metrics else "—"
            for name in METRIC_NAMES
        )
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def write_csv(path: Path, summaries: Sequence[Summary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "environment",
        "run_id",
        "rollouts",
        "reward_mean",
        "reward_std",
    ]
    for metric_name in METRIC_NAMES:
        fieldnames.extend((f"{metric_name}_mean", f"{metric_name}_std"))

    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for summary in summaries:
            row: dict[str, str | int | float] = {
                "model": summary.model,
                "environment": summary.environment,
                "run_id": summary.run_id,
                "rollouts": summary.rollout_count,
                "reward_mean": summary.reward_mean,
                "reward_std": summary.reward_std,
            }
            for metric_name in METRIC_NAMES:
                if metric_name in summary.metrics:
                    row[f"{metric_name}_mean"], row[f"{metric_name}_std"] = (
                        summary.metrics[metric_name]
                    )
                else:
                    row[f"{metric_name}_mean"] = ""
                    row[f"{metric_name}_std"] = ""
            writer.writerow(row)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summaries = load_summaries(args.root.resolve(), all_runs=args.all_runs)
        markdown = render_markdown(summaries)
        if args.csv is not None:
            write_csv(args.csv.resolve(), summaries)
    except (InvalidRunError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
