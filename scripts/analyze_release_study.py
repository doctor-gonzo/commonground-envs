"""Analyze complete Common Ground studies at the independent-task level.

Repeated rollouts for one task are correlated observations. This script first
averages reward within each task. It then resamples tasks, or generator
templates followed by variants within each template, to produce percentile
bootstrap intervals. The same resamples are shared across models so pairwise
intervals remain paired.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import aggregate_baselines as aggregate

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RunEstimate:
    environment: str
    model: str
    run_id: str
    task_count: int
    rollouts_per_task: int
    rollout_count: int
    recovered_rollout_count: int
    reward_mean: float
    rollout_std: float
    task_mean_std: float
    ci_low: float
    ci_high: float
    zero_reward_tasks: int
    cluster_count: int
    resampling_unit: str


@dataclass(frozen=True)
class PairwiseEstimate:
    environment: str
    model_a: str
    model_b: str
    mean_difference: float
    ci_low: float
    ci_high: float
    interpretation: str


@dataclass(frozen=True)
class StudyAnalysis:
    bootstrap_samples: int
    confidence_level: float
    seed: int
    summaries: tuple[RunEstimate, ...]
    pairwise: tuple[PairwiseEstimate, ...]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute task-level or template-hierarchical bootstrap evidence "
            "for saved Common Ground runs."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Study root containing outputs/*/* native run directories.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20_260_828)
    parser.add_argument("--expected-model-count", type=int)
    parser.add_argument("--expected-task-count", type=int)
    parser.add_argument("--expected-rollouts-per-task", type=int)
    parser.add_argument(
        "--require-no-recoveries",
        action="store_true",
        help="Reject any run with recovered retry/error history.",
    )
    parser.add_argument(
        "--elicit-eval-split",
        type=Path,
        help=(
            "Elicit JSONL whose row-order template IDs define hierarchical "
            "template/variant bootstrap clusters."
        ),
    )
    parser.add_argument("--csv", type=Path, help="Write run estimates as CSV.")
    parser.add_argument(
        "--pairwise-csv", type=Path, help="Write paired model differences as CSV."
    )
    parser.add_argument(
        "--json", type=Path, help="Write the complete analysis as JSON."
    )
    return parser.parse_args(argv)


def stable_seed(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}\0{label}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise aggregate.InvalidRunError("cannot take a percentile of no values")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("percentile probability must be within [0, 1]")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    return ordered[lower_index] * (1.0 - fraction) + ordered[upper_index] * fraction


def task_means(run: aggregate.CompleteRun) -> dict[int, float]:
    if len(run.task_ids) != len(run.rewards):
        raise aggregate.InvalidRunError(
            f"{run.run_id}: task identifiers and rewards are not aligned"
        )
    grouped: dict[int, list[float]] = defaultdict(list)
    for task_id, reward in zip(run.task_ids, run.rewards, strict=True):
        grouped[task_id].append(reward)
    if not grouped:
        raise aggregate.InvalidRunError(f"{run.run_id}: no task rewards")
    return {task_id: statistics.fmean(values) for task_id, values in grouped.items()}


def rollouts_per_task(run: aggregate.CompleteRun) -> int:
    counts: dict[int, int] = defaultdict(int)
    for task_id in run.task_ids:
        counts[task_id] += 1
    unique_counts = set(counts.values())
    if len(unique_counts) != 1:
        raise aggregate.InvalidRunError(
            f"{run.run_id}: inconsistent rollout counts by task: {dict(counts)}"
        )
    return next(iter(unique_counts))


def analyze_runs(
    runs: Sequence[aggregate.CompleteRun],
    *,
    bootstrap_samples: int,
    seed: int,
    expected_model_count: int | None = None,
    expected_task_count: int | None = None,
    expected_rollouts_per_task: int | None = None,
    require_no_recoveries: bool = False,
    task_cluster_labels: Mapping[str, Mapping[int, str]] | None = None,
) -> StudyAnalysis:
    if bootstrap_samples < 1_000:
        raise aggregate.InvalidRunError("bootstrap_samples must be at least 1000")
    if not runs:
        raise aggregate.InvalidRunError("no complete runs supplied")

    duplicate_keys: set[tuple[str, str]] = set()
    seen_keys: set[tuple[str, str]] = set()
    for run in runs:
        key = (run.environment, run.model)
        if key in seen_keys:
            duplicate_keys.add(key)
        seen_keys.add(key)
    if duplicate_keys:
        raise aggregate.InvalidRunError(
            f"study contains duplicate environment/model runs: {sorted(duplicate_keys)}"
        )

    runs_by_environment: dict[str, list[aggregate.CompleteRun]] = defaultdict(list)
    for run in runs:
        if require_no_recoveries and run.recovered_rollout_count:
            raise aggregate.InvalidRunError(
                f"{run.run_id}: recovered rollouts are not allowed for this study"
            )
        runs_by_environment[run.environment].append(run)

    summaries: list[RunEstimate] = []
    pairwise: list[PairwiseEstimate] = []
    lower_probability = 0.025
    upper_probability = 0.975

    for environment in sorted(runs_by_environment):
        environment_runs = runs_by_environment[environment]
        if (
            expected_model_count is not None
            and len(environment_runs) != expected_model_count
        ):
            raise aggregate.InvalidRunError(
                f"{environment}: expected {expected_model_count} models, "
                f"found {len(environment_runs)}"
            )

        means_by_model: dict[str, dict[int, float]] = {}
        run_by_model = {run.model: run for run in environment_runs}
        rollouts_by_model: dict[str, int] = {}
        for run in environment_runs:
            means_by_model[run.model] = task_means(run)
            rollouts_by_model[run.model] = rollouts_per_task(run)
            if (
                expected_task_count is not None
                and len(means_by_model[run.model]) != expected_task_count
            ):
                raise aggregate.InvalidRunError(
                    f"{run.run_id}: expected {expected_task_count} tasks, "
                    f"found {len(means_by_model[run.model])}"
                )
            if (
                expected_rollouts_per_task is not None
                and rollouts_by_model[run.model] != expected_rollouts_per_task
            ):
                raise aggregate.InvalidRunError(
                    f"{run.run_id}: expected {expected_rollouts_per_task} rollouts "
                    f"per task, found {rollouts_by_model[run.model]}"
                )

        task_id_sets = {frozenset(values) for values in means_by_model.values()}
        if len(task_id_sets) != 1:
            raise aggregate.InvalidRunError(
                f"{environment}: models do not cover the same task identifiers"
            )
        task_ids = sorted(next(iter(task_id_sets)))
        if not task_ids:
            raise aggregate.InvalidRunError(f"{environment}: no paired tasks")

        model_order = sorted(means_by_model, key=str.casefold)
        values_by_model = {
            model: [means_by_model[model][task_id] for task_id in task_ids]
            for model in model_order
        }
        bootstrap_by_model: dict[str, list[float]] = {
            model: [] for model in model_order
        }
        generator = random.Random(stable_seed(seed, environment))
        task_positions = list(range(len(task_ids)))
        cluster_labels = (
            task_cluster_labels.get(environment, {}) if task_cluster_labels else {}
        )
        if cluster_labels:
            missing = sorted(set(task_ids) - set(cluster_labels))
            extra = sorted(set(cluster_labels) - set(task_ids))
            if missing or extra:
                raise aggregate.InvalidRunError(
                    f"{environment}: task cluster labels do not match tasks "
                    f"(missing={missing[:5]}, extra={extra[:5]})"
                )
        else:
            cluster_labels = {task_id: f"task:{task_id}" for task_id in task_ids}
        positions_by_cluster: dict[str, list[int]] = defaultdict(list)
        for position, task_id in enumerate(task_ids):
            positions_by_cluster[str(cluster_labels[task_id])].append(position)
        cluster_order = sorted(positions_by_cluster)
        hierarchical = any(
            len(positions) > 1 for positions in positions_by_cluster.values()
        )
        for _ in range(bootstrap_samples):
            if hierarchical:
                selected = []
                for cluster in generator.choices(cluster_order, k=len(cluster_order)):
                    positions = positions_by_cluster[cluster]
                    selected.extend(generator.choices(positions, k=len(positions)))
            else:
                selected = generator.choices(task_positions, k=len(task_positions))
            for model in model_order:
                values = values_by_model[model]
                bootstrap_by_model[model].append(
                    sum(values[position] for position in selected) / len(selected)
                )

        for model in model_order:
            run = run_by_model[model]
            per_task = values_by_model[model]
            bootstrapped = bootstrap_by_model[model]
            summaries.append(
                RunEstimate(
                    environment=environment,
                    model=model,
                    run_id=run.run_id,
                    task_count=len(per_task),
                    rollouts_per_task=rollouts_by_model[model],
                    rollout_count=len(run.rewards),
                    recovered_rollout_count=run.recovered_rollout_count,
                    reward_mean=statistics.fmean(run.rewards),
                    rollout_std=statistics.pstdev(run.rewards),
                    task_mean_std=statistics.pstdev(per_task),
                    ci_low=percentile(bootstrapped, lower_probability),
                    ci_high=percentile(bootstrapped, upper_probability),
                    zero_reward_tasks=sum(value == 0.0 for value in per_task),
                    cluster_count=len(cluster_order),
                    resampling_unit=(
                        "template then variant" if hierarchical else "task"
                    ),
                )
            )

        ranked_models = sorted(
            model_order,
            key=lambda model: (
                -statistics.fmean(values_by_model[model]),
                model.casefold(),
            ),
        )
        for model_a, model_b in combinations(ranked_models, 2):
            observed = statistics.fmean(values_by_model[model_a]) - statistics.fmean(
                values_by_model[model_b]
            )
            differences = [
                value_a - value_b
                for value_a, value_b in zip(
                    bootstrap_by_model[model_a],
                    bootstrap_by_model[model_b],
                    strict=True,
                )
            ]
            ci_low = percentile(differences, lower_probability)
            ci_high = percentile(differences, upper_probability)
            interpretation = (
                f"{model_a} higher"
                if ci_low > 0.0
                else f"{model_b} higher"
                if ci_high < 0.0
                else "interval includes 0"
            )
            pairwise.append(
                PairwiseEstimate(
                    environment=environment,
                    model_a=model_a,
                    model_b=model_b,
                    mean_difference=observed,
                    ci_low=ci_low,
                    ci_high=ci_high,
                    interpretation=interpretation,
                )
            )

    return StudyAnalysis(
        bootstrap_samples=bootstrap_samples,
        confidence_level=0.95,
        seed=seed,
        summaries=tuple(summaries),
        pairwise=tuple(pairwise),
    )


def render_markdown(analysis: StudyAnalysis) -> str:
    lines = [
        "## Task-cluster bootstrap summary",
        "",
        "| Environment | Model | Run ID | Tasks x rollouts | Resampling | Reward mean | Rollout std | 95% cluster CI | Task-mean std | Zero-reward tasks | Recovered |",
        "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in analysis.summaries:
        lines.append(
            "| "
            + " | ".join(
                [
                    summary.environment,
                    summary.model,
                    summary.run_id,
                    f"{summary.task_count} x {summary.rollouts_per_task}",
                    f"{summary.resampling_unit} ({summary.cluster_count})",
                    f"{summary.reward_mean:.3f}",
                    f"{summary.rollout_std:.3f}",
                    f"[{summary.ci_low:.3f}, {summary.ci_high:.3f}]",
                    f"{summary.task_mean_std:.3f}",
                    str(summary.zero_reward_tasks),
                    str(summary.recovered_rollout_count),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Paired task-level model differences",
            "",
            "| Environment | Model A | Model B | Mean A - B | 95% paired task-bootstrap CI | Interpretation |",
            "| --- | --- | --- | ---: | ---: | --- |",
        ]
    )
    for difference in analysis.pairwise:
        lines.append(
            "| "
            + " | ".join(
                [
                    difference.environment,
                    difference.model_a,
                    difference.model_b,
                    f"{difference.mean_difference:+.3f}",
                    f"[{difference.ci_low:+.3f}, {difference.ci_high:+.3f}]",
                    difference.interpretation,
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            (
                f"Method: {analysis.bootstrap_samples:,} deterministic percentile "
                f"bootstrap resamples at the {analysis.confidence_level:.0%} "
                "level. Elicit runs resample base templates, then variants within "
                "each sampled template; other runs resample tasks. Each task "
                "contributes its mean across repeated rollouts, and paired "
                "comparisons reuse the same resample. "
                f"Seed: {analysis.seed}."
            ),
        ]
    )
    return "\n".join(lines)


def write_csv(path: Path, rows: Sequence[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise aggregate.InvalidRunError(f"cannot write empty CSV: {path}")
    data = [asdict(row) for row in rows]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(data)


def write_json(path: Path, analysis: StudyAnalysis) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Mapping[str, Any] = {
        "method": {
            "unit": "task mean across repeated rollouts",
            "interval": "paired hierarchical cluster percentile bootstrap",
            "bootstrap_samples": analysis.bootstrap_samples,
            "confidence_level": analysis.confidence_level,
            "seed": analysis.seed,
        },
        "summaries": [asdict(item) for item in analysis.summaries],
        "pairwise": [asdict(item) for item in analysis.pairwise],
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def load_template_clusters(path: Path) -> dict[int, str]:
    """Map JSONL row indices to declared base-template identifiers."""

    labels: dict[int, str] = {}
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        scenario = json.loads(line)
        provenance = scenario.get("provenance", {})
        template_id = provenance.get("template_id")
        if not isinstance(template_id, str) or not template_id.strip():
            raise aggregate.InvalidRunError(
                f"{path}:{index + 1}: missing provenance.template_id"
            )
        labels[len(labels)] = template_id
    if not labels:
        raise aggregate.InvalidRunError(f"{path}: no scenarios")
    return labels


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result_paths = aggregate.discover_result_paths(args.root.resolve())
        runs = [aggregate.load_complete_run(path) for path in result_paths]
        newest = aggregate.newest_runs(runs)
        cluster_labels: dict[str, Mapping[int, str]] = {}
        if args.elicit_eval_split is not None:
            elicit_clusters = load_template_clusters(args.elicit_eval_split.resolve())
            cluster_labels = {
                run.environment: elicit_clusters
                for run in newest
                if "commonground-elicit" in run.environment
            }
        analysis = analyze_runs(
            newest,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
            expected_model_count=args.expected_model_count,
            expected_task_count=args.expected_task_count,
            expected_rollouts_per_task=args.expected_rollouts_per_task,
            require_no_recoveries=args.require_no_recoveries,
            task_cluster_labels=cluster_labels,
        )
        if args.csv is not None:
            write_csv(args.csv.resolve(), analysis.summaries)
        if args.pairwise_csv is not None:
            write_csv(args.pairwise_csv.resolve(), analysis.pairwise)
        if args.json is not None:
            write_json(args.json.resolve(), analysis)
    except (aggregate.InvalidRunError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(render_markdown(analysis))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
