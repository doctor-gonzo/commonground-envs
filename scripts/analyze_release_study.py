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
import math
import random
import statistics
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from itertools import combinations
from pathlib import Path
from typing import Any

import aggregate_baselines as aggregate

ROOT = Path(__file__).resolve().parents[1]
UNIFORM_THREE_CLASS_BRIER = 1.0 / 3.0


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
    prompt_mode: str | None = None


@dataclass(frozen=True)
class DiagnosticEstimate:
    environment: str
    model: str
    run_id: str
    metric: str
    task_count: int
    rollout_count: int
    rollout_mean: float
    rollout_std: float
    task_mean: float
    task_mean_std: float
    ci_low: float
    ci_high: float
    cluster_count: int
    resampling_unit: str
    prompt_mode: str | None = None


@dataclass(frozen=True)
class PairwiseEstimate:
    environment: str
    model_a: str
    model_b: str
    mean_difference: float
    ci_low: float
    ci_high: float
    interpretation: str
    raw_p_value: float = 1.0
    holm_adjusted_p_value: float = 1.0
    multiplicity_family_size: int = 0
    exploratory: bool = True
    holm_interpretation: str = "not evaluated"
    prompt_mode: str | None = None


@dataclass(frozen=True)
class PromptAblationEstimate:
    environment: str
    model: str
    metric: str
    reference_mode: str
    prompt_mode: str
    task_count: int
    reference_mean: float
    prompt_mean: float
    mean_difference: float
    ci_low: float
    ci_high: float
    interpretation: str
    exploratory: bool = True
    comparison_signature: str | None = None


@dataclass(frozen=True)
class StudyAnalysis:
    bootstrap_samples: int
    confidence_level: float
    seed: int
    summaries: tuple[RunEstimate, ...]
    pairwise: tuple[PairwiseEstimate, ...]
    diagnostics: tuple[DiagnosticEstimate, ...] = ()
    prompt_ablations: tuple[PromptAblationEstimate, ...] = ()


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
    parser.add_argument(
        "--expected-model-count",
        type=int,
        help=(
            "Required model count for each primary environment; non-full Predict "
            "ablation modes may cover a declared subset of those models."
        ),
    )
    parser.add_argument("--expected-task-count", type=int)
    parser.add_argument("--expected-rollouts-per-task", type=int)
    parser.add_argument(
        "--expected-pairwise-family-size",
        type=int,
        help=(
            "Predeclared number of primary pairwise model comparisons in the "
            "single global Holm family (18 for four models across three tasks)."
        ),
    )
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
        "--diagnostics-csv",
        type=Path,
        help="Write clustered diagnostic metric estimates as CSV.",
    )
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


def diagnostic_task_means(
    run: aggregate.CompleteRun,
    metric: str,
    values: Sequence[float] | None = None,
) -> dict[int, float]:
    values = run.metrics[metric] if values is None else values
    if len(run.task_ids) != len(values):
        raise aggregate.InvalidRunError(
            f"{run.run_id}: {metric} values and task identifiers are not aligned"
        )
    grouped: dict[int, list[float]] = defaultdict(list)
    for task_id, value in zip(run.task_ids, values, strict=True):
        grouped[task_id].append(value)
    if not grouped:
        raise aggregate.InvalidRunError(
            f"{run.run_id}: no task values for diagnostic metric {metric}"
        )
    return {task_id: statistics.fmean(items) for task_id, items in grouped.items()}


def effective_prompt_mode(run: aggregate.CompleteRun) -> str | None:
    """Normalize historical Predict runs to the pre-ablation ``full`` view."""

    is_predict = (
        aggregate.environment_package_name(run.environment) == "commonground-predict"
    )
    if not is_predict:
        if run.prompt_mode is not None:
            raise aggregate.InvalidRunError(
                f"{run.run_id}: prompt_mode is only valid for Predict runs"
            )
        return None
    prompt_mode = run.prompt_mode or "full"
    if prompt_mode not in aggregate.PREDICT_PROMPT_MODES:
        raise aggregate.InvalidRunError(
            f"{run.run_id}: unknown Predict prompt mode {prompt_mode!r}"
        )
    return prompt_mode


def diagnostic_rollout_values(
    run: aggregate.CompleteRun,
) -> dict[str, tuple[float, ...]]:
    values = {name: tuple(items) for name, items in run.metrics.items()}
    if (
        aggregate.environment_package_name(run.environment) == "commonground-predict"
        and "brier" in values
    ):
        # The uniform three-class forecast has normalized Brier 1/3 exactly.
        values["brier_skill_vs_uniform"] = tuple(
            1.0 - item / UNIFORM_THREE_CLASS_BRIER for item in values["brier"]
        )
    return values


def pooled_brier_skill_contributions(
    brier_values: Sequence[float], reference_values: Sequence[float]
) -> tuple[float, ...]:
    """Represent pooled skill as rollout contributions with the same mean."""

    aggregate.pooled_brier_skill(brier_values, reference_values)
    reference_mean = statistics.fmean(reference_values)
    return tuple(1.0 - value / reference_mean for value in brier_values)


def holm_adjusted_p_values(p_values: Sequence[float]) -> tuple[float, ...]:
    """Return Holm step-down family-wise-error adjusted p-values."""

    if not p_values:
        return ()
    for p_value in p_values:
        if not 0.0 <= p_value <= 1.0:
            raise ValueError("p-values must be within [0, 1]")
    ordered = sorted(enumerate(p_values), key=lambda item: (item[1], item[0]))
    adjusted = [0.0] * len(ordered)
    running_maximum = 0.0
    family_size = len(ordered)
    for rank, (original_index, p_value) in enumerate(ordered):
        candidate = min(1.0, (family_size - rank) * p_value)
        running_maximum = max(running_maximum, candidate)
        adjusted[original_index] = running_maximum
    return tuple(adjusted)


def paired_cluster_sign_flip_p_value(
    differences: Sequence[float],
    *,
    positions_by_cluster: Mapping[str, Sequence[int]],
    samples: int,
    seed: int,
) -> float:
    """Estimate a two-sided paired cluster sign-flip p-value.

    One sign is drawn per independent cluster, preserving the dependence among
    variants from the same template. The add-one correction prevents reported
    Monte Carlo p-values of exactly zero.
    """

    if not differences:
        raise aggregate.InvalidRunError(
            "cannot test a paired difference with no task values"
        )
    if samples <= 0:
        raise ValueError("sign-flip samples must be positive")
    observed = abs(statistics.fmean(differences))
    if observed == 0.0:
        return 1.0
    cluster_order = sorted(positions_by_cluster)
    generator = random.Random(seed)
    exceedances = 0
    tolerance = 1e-15
    for _ in range(samples):
        signed_total = 0.0
        for cluster in cluster_order:
            sign = 1.0 if generator.getrandbits(1) else -1.0
            signed_total += sign * sum(
                differences[position] for position in positions_by_cluster[cluster]
            )
        null_statistic = abs(signed_total / len(differences))
        if null_statistic + tolerance >= observed:
            exceedances += 1
    return (exceedances + 1) / (samples + 1)


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


def prompt_ablation_probability_reward_task_means(
    run: aggregate.CompleteRun,
) -> dict[int, float]:
    """Return raw named probability reward after validating its saved contract."""

    scores = run.raw_reward_scores.get("probability_reward")
    weights = run.raw_reward_weights.get("probability_reward")
    if scores is None or weights is None:
        raise aggregate.InvalidRunError(
            f"{run.run_id}: declared Predict ablation is missing raw named "
            "probability_reward score/weight provenance"
        )
    if len(scores) != len(run.task_ids) or len(weights) != len(run.task_ids):
        raise aggregate.InvalidRunError(
            f"{run.run_id}: raw probability_reward values and task identifiers "
            "are not aligned"
        )
    if any(not math.isfinite(score) or not 0.0 <= score <= 1.0 for score in scores):
        raise aggregate.InvalidRunError(
            f"{run.run_id}: raw probability_reward scores must be within [0, 1]"
        )
    if any(weight != 1.0 for weight in weights):
        raise aggregate.InvalidRunError(
            f"{run.run_id}: declared Predict ablations require "
            "probability_reward weight 1.0"
        )
    for total, score, weight in zip(run.rewards, scores, weights, strict=True):
        if not math.isclose(total, score * weight, rel_tol=1e-12, abs_tol=1e-12):
            raise aggregate.InvalidRunError(
                f"{run.run_id}: weighted reward total does not match the raw "
                "probability_reward score/weight"
            )
    return diagnostic_task_means(run, "probability_reward", scores)


def require_prompt_ablation_provenance(
    run_by_mode: Mapping[str, aggregate.CompleteRun],
) -> tuple[str, dict[str, dict[int, float]]]:
    """Require one complete comparison identity across the four prompt views."""

    signatures: set[str] = set()
    answer_rosters: set[tuple[tuple[int, str], ...]] = set()
    reward_means_by_mode: dict[str, dict[int, float]] = {}
    for prompt_mode in aggregate.PREDICT_PROMPT_MODES:
        run = run_by_mode[prompt_mode]
        signature = run.comparison_signature
        if (
            signature is None
            or len(signature) != 64
            or any(character not in "0123456789abcdef" for character in signature)
        ):
            raise aggregate.InvalidRunError(
                f"{run.run_id}: declared Predict ablation is missing a stable "
                "comparison signature"
            )
        signatures.add(signature)

        answer_roster = run.task_answer_digests
        answer_ids = [task_id for task_id, _ in answer_roster]
        if (
            not answer_roster
            or len(answer_ids) != len(set(answer_ids))
            or set(answer_ids) != set(run.task_ids)
            or any(
                len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                for _, digest in answer_roster
            )
        ):
            raise aggregate.InvalidRunError(
                f"{run.run_id}: declared Predict ablation is missing complete "
                "per-task answer digests"
            )
        answer_rosters.add(tuple(sorted(answer_roster)))
        reward_means_by_mode[prompt_mode] = (
            prompt_ablation_probability_reward_task_means(run)
        )

    # Regression guard: task IDs alone can silently pair different corpora, and
    # weighted totals can silently turn reward configuration into a prompt effect.
    if len(answer_rosters) != 1:
        raise aggregate.InvalidRunError(
            "Predict prompt modes do not use the same per-task answers"
        )
    if len(signatures) != 1:
        raise aggregate.InvalidRunError(
            "Predict prompt modes differ in model/client/sampling/concurrency, "
            "task-answer, or reward-contract provenance"
        )
    return next(iter(signatures)), reward_means_by_mode


def analyze_prompt_ablations(
    runs: Sequence[aggregate.CompleteRun],
    *,
    bootstrap_samples: int,
    seed: int,
    lower_probability: float,
    upper_probability: float,
) -> tuple[PromptAblationEstimate, ...]:
    """Compare complete same-model Predict prompt-mode sets against ``full``."""

    runs_by_environment_model: dict[
        tuple[str, str], dict[str, aggregate.CompleteRun]
    ] = defaultdict(dict)
    for run in runs:
        if (
            aggregate.environment_package_name(run.environment)
            != "commonground-predict"
        ):
            continue
        prompt_mode = effective_prompt_mode(run)
        if prompt_mode is None:  # pragma: no cover - guarded by package identity
            continue
        runs_by_environment_model[(run.environment, run.model)][prompt_mode] = run

    estimates: list[PromptAblationEstimate] = []
    expected_modes = set(aggregate.PREDICT_PROMPT_MODES)
    metric_order = (
        "probability_reward",
        "brier",
        "vote_accuracy",
        "brier_skill_vs_uniform",
        "brier_skill_vs_original_snapshot_visible_prior",
    )
    for environment_model in sorted(
        runs_by_environment_model, key=lambda item: (item[0], item[1].casefold())
    ):
        environment, model = environment_model
        run_by_mode = runs_by_environment_model[environment_model]
        if set(run_by_mode) == {"full"}:
            continue
        if set(run_by_mode) != expected_modes:
            missing = sorted(
                expected_modes - set(run_by_mode), key=aggregate.prompt_mode_sort_key
            )
            extra = sorted(set(run_by_mode) - expected_modes)
            raise aggregate.InvalidRunError(
                f"{environment}/{model}: prompt ablation requires a complete "
                f"prompt-mode set (missing={missing}, extra={extra})"
            )

        reference_run = run_by_mode["full"]
        reference_task_ids = set(reference_run.task_ids)
        reference_rollouts = rollouts_per_task(reference_run)
        for prompt_mode in aggregate.PREDICT_PROMPT_MODES:
            if prompt_mode == "full":
                continue
            prompt_run = run_by_mode[prompt_mode]
            if set(prompt_run.task_ids) != reference_task_ids:
                raise aggregate.InvalidRunError(
                    f"{environment}/{model}: prompt modes do not cover the same "
                    "task identifiers"
                )
            if rollouts_per_task(prompt_run) != reference_rollouts:
                raise aggregate.InvalidRunError(
                    f"{environment}/{model}: prompt modes do not use the same "
                    "rollout count per task"
                )

        comparison_signature, reward_means_by_mode = require_prompt_ablation_provenance(
            run_by_mode
        )
        reference_rewards = reward_means_by_mode["full"]
        reference_diagnostics = diagnostic_rollout_values(reference_run)
        missing_reference_metrics = {
            "brier",
            "original_snapshot_visible_prior_brier",
            "vote_accuracy",
            "brier_skill_vs_uniform",
        } - set(reference_diagnostics)
        if missing_reference_metrics:
            raise aggregate.InvalidRunError(
                f"{reference_run.run_id}: Predict prompt ablation is missing metrics "
                f"{sorted(missing_reference_metrics)}"
            )

        for prompt_mode in aggregate.PREDICT_PROMPT_MODES:
            if prompt_mode == "full":
                continue
            prompt_run = run_by_mode[prompt_mode]
            prompt_rewards = reward_means_by_mode[prompt_mode]
            if set(prompt_rewards) != set(reference_rewards):
                raise aggregate.InvalidRunError(
                    f"{environment}/{model}: prompt modes do not cover the same "
                    "task identifiers"
                )
            prompt_rollouts = rollouts_per_task(prompt_run)
            if prompt_rollouts != reference_rollouts:
                raise aggregate.InvalidRunError(
                    f"{environment}/{model}: prompt modes do not use the same "
                    "rollout count per task"
                )

            prompt_diagnostics = diagnostic_rollout_values(prompt_run)
            missing_prompt_metrics = {
                "brier",
                "original_snapshot_visible_prior_brier",
                "vote_accuracy",
                "brier_skill_vs_uniform",
            } - set(prompt_diagnostics)
            if missing_prompt_metrics:
                raise aggregate.InvalidRunError(
                    f"{prompt_run.run_id}: Predict prompt ablation is missing metrics "
                    f"{sorted(missing_prompt_metrics)}"
                )

            task_ids = sorted(reference_rewards)
            reference_by_metric: dict[str, dict[int, float]] = {
                "probability_reward": reference_rewards,
            }
            prompt_by_metric: dict[str, dict[int, float]] = {
                "probability_reward": prompt_rewards,
            }
            for metric in (
                "brier",
                "original_snapshot_visible_prior_brier",
                "vote_accuracy",
                "brier_skill_vs_uniform",
            ):
                reference_by_metric[metric] = diagnostic_task_means(
                    reference_run, metric, reference_diagnostics[metric]
                )
                prompt_by_metric[metric] = diagnostic_task_means(
                    prompt_run, metric, prompt_diagnostics[metric]
                )
            reference_prior_values = [
                reference_by_metric["original_snapshot_visible_prior_brier"][task_id]
                for task_id in task_ids
            ]
            prompt_prior_values = [
                prompt_by_metric["original_snapshot_visible_prior_brier"][task_id]
                for task_id in task_ids
            ]
            if any(
                not math.isclose(reference, prompt, rel_tol=0.0, abs_tol=1e-12)
                for reference, prompt in zip(
                    reference_prior_values, prompt_prior_values, strict=True
                )
            ):
                raise aggregate.InvalidRunError(
                    f"{environment}/{model}: prompt modes changed the "
                    "snapshot-prior reference loss"
                )
            reference_brier_values = [
                reference_by_metric["brier"][task_id] for task_id in task_ids
            ]
            prompt_brier_values = [
                prompt_by_metric["brier"][task_id] for task_id in task_ids
            ]
            reference_by_metric["brier_skill_vs_original_snapshot_visible_prior"] = (
                dict(
                    zip(
                        task_ids,
                        pooled_brier_skill_contributions(
                            reference_brier_values, reference_prior_values
                        ),
                        strict=True,
                    )
                )
            )
            prompt_by_metric["brier_skill_vs_original_snapshot_visible_prior"] = dict(
                zip(
                    task_ids,
                    pooled_brier_skill_contributions(
                        prompt_brier_values, prompt_prior_values
                    ),
                    strict=True,
                )
            )

            for metric in metric_order:
                reference_values = [
                    reference_by_metric[metric][task_id] for task_id in task_ids
                ]
                prompt_values = [
                    prompt_by_metric[metric][task_id] for task_id in task_ids
                ]
                differences = [
                    reference - prompt
                    for reference, prompt in zip(
                        reference_values, prompt_values, strict=True
                    )
                ]
                generator = random.Random(
                    stable_seed(
                        seed,
                        f"{environment}\0{model}\0full\0{prompt_mode}\0{metric}",
                    )
                )
                positions = list(range(len(task_ids)))
                bootstrapped = []
                for _ in range(bootstrap_samples):
                    selected = generator.choices(positions, k=len(positions))
                    if metric == "brier_skill_vs_original_snapshot_visible_prior":
                        bootstrapped.append(
                            aggregate.pooled_brier_skill(
                                [
                                    reference_brier_values[position]
                                    for position in selected
                                ],
                                [
                                    reference_prior_values[position]
                                    for position in selected
                                ],
                            )
                            - aggregate.pooled_brier_skill(
                                [
                                    prompt_brier_values[position]
                                    for position in selected
                                ],
                                [
                                    prompt_prior_values[position]
                                    for position in selected
                                ],
                            )
                        )
                    else:
                        bootstrapped.append(
                            statistics.fmean(
                                differences[position] for position in selected
                            )
                        )
                ci_low = percentile(bootstrapped, lower_probability)
                ci_high = percentile(bootstrapped, upper_probability)
                if ci_low <= 0.0 <= ci_high:
                    interpretation = "unadjusted interval includes 0"
                elif metric == "brier":
                    interpretation = (
                        "full lower loss (unadjusted interval)"
                        if ci_high < 0.0
                        else f"{prompt_mode} lower loss (unadjusted interval)"
                    )
                else:
                    interpretation = (
                        "full higher (unadjusted interval)"
                        if ci_low > 0.0
                        else f"{prompt_mode} higher (unadjusted interval)"
                    )
                estimates.append(
                    PromptAblationEstimate(
                        environment=environment,
                        model=model,
                        metric=metric,
                        reference_mode="full",
                        prompt_mode=prompt_mode,
                        task_count=len(task_ids),
                        reference_mean=statistics.fmean(reference_values),
                        prompt_mean=statistics.fmean(prompt_values),
                        mean_difference=statistics.fmean(differences),
                        ci_low=ci_low,
                        ci_high=ci_high,
                        interpretation=interpretation,
                        comparison_signature=comparison_signature,
                    )
                )
    return tuple(estimates)


def analyze_runs(
    runs: Sequence[aggregate.CompleteRun],
    *,
    bootstrap_samples: int,
    seed: int,
    expected_model_count: int | None = None,
    expected_task_count: int | None = None,
    expected_rollouts_per_task: int | None = None,
    expected_pairwise_family_size: int | None = None,
    require_no_recoveries: bool = False,
    task_cluster_labels: Mapping[str, Mapping[int, str]] | None = None,
) -> StudyAnalysis:
    if bootstrap_samples < 1_000:
        raise aggregate.InvalidRunError("bootstrap_samples must be at least 1000")
    if not runs:
        raise aggregate.InvalidRunError("no complete runs supplied")

    duplicate_keys: set[tuple[str, str, str | None]] = set()
    seen_keys: set[tuple[str, str, str | None]] = set()
    for run in runs:
        key = (run.environment, run.model, effective_prompt_mode(run))
        if key in seen_keys:
            duplicate_keys.add(key)
        seen_keys.add(key)
    if duplicate_keys:
        raise aggregate.InvalidRunError(
            "study contains duplicate environment/model/prompt-mode runs: "
            f"{sorted(duplicate_keys, key=str)}"
        )

    runs_by_stratum: dict[tuple[str, str | None], list[aggregate.CompleteRun]] = (
        defaultdict(list)
    )
    for run in runs:
        if require_no_recoveries and run.recovered_rollout_count:
            raise aggregate.InvalidRunError(
                f"{run.run_id}: recovered rollouts are not allowed for this study"
            )
        runs_by_stratum[(run.environment, effective_prompt_mode(run))].append(run)

    summaries: list[RunEstimate] = []
    diagnostics: list[DiagnosticEstimate] = []
    pairwise: list[PairwiseEstimate] = []
    lower_probability = 0.025
    upper_probability = 0.975

    stratum_order = sorted(
        runs_by_stratum,
        key=lambda item: (item[0], aggregate.prompt_mode_sort_key(item[1])),
    )
    for environment, prompt_mode in stratum_order:
        environment_runs = runs_by_stratum[(environment, prompt_mode)]
        if (
            expected_model_count is not None
            and prompt_mode in (None, "full")
            and len(environment_runs) != expected_model_count
        ):
            raise aggregate.InvalidRunError(
                f"{environment} [{prompt_mode or 'default'}]: expected "
                f"{expected_model_count} models, "
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
        diagnostic_values_by_model: dict[str, dict[str, list[float]]] = {}
        diagnostic_rollouts_by_model: dict[str, dict[str, tuple[float, ...]]] = {}
        for model in model_order:
            run = run_by_model[model]
            diagnostic_values_by_model[model] = {}
            diagnostic_rollouts_by_model[model] = diagnostic_rollout_values(run)
            for metric, rollout_values in sorted(
                diagnostic_rollouts_by_model[model].items()
            ):
                per_task_metric_map = diagnostic_task_means(run, metric, rollout_values)
                if set(per_task_metric_map) != set(task_ids):
                    raise aggregate.InvalidRunError(
                        f"{run.run_id}: diagnostic metric {metric} does not cover "
                        "the same task identifiers as reward"
                    )
                diagnostic_values_by_model[model][metric] = [
                    per_task_metric_map[task_id] for task_id in task_ids
                ]
            if {
                "brier",
                "original_snapshot_visible_prior_brier",
            } <= diagnostic_values_by_model[model].keys():
                diagnostic_values_by_model[model][
                    "brier_skill_vs_original_snapshot_visible_prior"
                ] = list(
                    pooled_brier_skill_contributions(
                        diagnostic_values_by_model[model]["brier"],
                        diagnostic_values_by_model[model][
                            "original_snapshot_visible_prior_brier"
                        ],
                    )
                )
                diagnostic_rollouts_by_model[model][
                    "brier_skill_vs_original_snapshot_visible_prior"
                ] = pooled_brier_skill_contributions(
                    diagnostic_rollouts_by_model[model]["brier"],
                    diagnostic_rollouts_by_model[model][
                        "original_snapshot_visible_prior_brier"
                    ],
                )
        bootstrap_by_model: dict[str, list[float]] = {
            model: [] for model in model_order
        }
        diagnostic_bootstrap_by_model: dict[str, dict[str, list[float]]] = {
            model: {metric: [] for metric in diagnostic_values_by_model[model]}
            for model in model_order
        }
        generator = random.Random(
            stable_seed(seed, f"{environment}\0{prompt_mode or 'default'}")
        )
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
                for metric, diagnostic_values in diagnostic_values_by_model[
                    model
                ].items():
                    if metric == "brier_skill_vs_original_snapshot_visible_prior":
                        diagnostic_bootstrap_by_model[model][metric].append(
                            aggregate.pooled_brier_skill(
                                [
                                    diagnostic_values_by_model[model]["brier"][position]
                                    for position in selected
                                ],
                                [
                                    diagnostic_values_by_model[model][
                                        "original_snapshot_visible_prior_brier"
                                    ][position]
                                    for position in selected
                                ],
                            )
                        )
                    else:
                        diagnostic_bootstrap_by_model[model][metric].append(
                            sum(diagnostic_values[position] for position in selected)
                            / len(selected)
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
                    prompt_mode=prompt_mode,
                )
            )
            for metric, per_task_metric in diagnostic_values_by_model[model].items():
                rollout_values = diagnostic_rollouts_by_model[model][metric]
                bootstrapped_metric = diagnostic_bootstrap_by_model[model][metric]
                diagnostics.append(
                    DiagnosticEstimate(
                        environment=environment,
                        model=model,
                        run_id=run.run_id,
                        metric=metric,
                        task_count=len(per_task_metric),
                        rollout_count=len(rollout_values),
                        rollout_mean=statistics.fmean(rollout_values),
                        rollout_std=statistics.pstdev(rollout_values),
                        task_mean=statistics.fmean(per_task_metric),
                        task_mean_std=statistics.pstdev(per_task_metric),
                        ci_low=percentile(bootstrapped_metric, lower_probability),
                        ci_high=percentile(bootstrapped_metric, upper_probability),
                        cluster_count=len(cluster_order),
                        resampling_unit=(
                            "template then variant" if hierarchical else "task"
                        ),
                        prompt_mode=prompt_mode,
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
            task_differences = [
                value_a - value_b
                for value_a, value_b in zip(
                    values_by_model[model_a],
                    values_by_model[model_b],
                    strict=True,
                )
            ]
            observed = statistics.fmean(task_differences)
            bootstrapped_differences = [
                value_a - value_b
                for value_a, value_b in zip(
                    bootstrap_by_model[model_a],
                    bootstrap_by_model[model_b],
                    strict=True,
                )
            ]
            ci_low = percentile(bootstrapped_differences, lower_probability)
            ci_high = percentile(bootstrapped_differences, upper_probability)
            interpretation = (
                f"{model_a} higher"
                if ci_low > 0.0
                else f"{model_b} higher"
                if ci_high < 0.0
                else "interval includes 0"
            )
            raw_p_value = paired_cluster_sign_flip_p_value(
                task_differences,
                positions_by_cluster=positions_by_cluster,
                samples=bootstrap_samples,
                seed=stable_seed(
                    seed,
                    f"{environment}\0{prompt_mode}\0{model_a}\0{model_b}\0sign-flip",
                ),
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
                    raw_p_value=raw_p_value,
                    holm_adjusted_p_value=raw_p_value,
                    multiplicity_family_size=0,
                    exploratory=True,
                    holm_interpretation="pending global Holm adjustment",
                    prompt_mode=prompt_mode,
                )
            )

    adjusted_p_values = holm_adjusted_p_values(
        tuple(item.raw_p_value for item in pairwise)
    )
    family_size = len(pairwise)
    if (
        expected_pairwise_family_size is not None
        and family_size != expected_pairwise_family_size
    ):
        raise aggregate.InvalidRunError(
            "primary pairwise comparison family does not match the predeclared "
            f"size: expected {expected_pairwise_family_size}, found {family_size}"
        )
    pairwise = [
        replace(
            item,
            holm_adjusted_p_value=adjusted_p_value,
            multiplicity_family_size=family_size,
            holm_interpretation=(
                f"{item.model_a} higher after Holm adjustment"
                if adjusted_p_value <= 0.05 and item.mean_difference > 0.0
                else f"{item.model_b} higher after Holm adjustment"
                if adjusted_p_value <= 0.05 and item.mean_difference < 0.0
                else "not significant after Holm adjustment"
            ),
        )
        for item, adjusted_p_value in zip(pairwise, adjusted_p_values, strict=True)
    ]
    prompt_ablations = analyze_prompt_ablations(
        runs,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        lower_probability=lower_probability,
        upper_probability=upper_probability,
    )

    return StudyAnalysis(
        bootstrap_samples=bootstrap_samples,
        confidence_level=0.95,
        seed=seed,
        summaries=tuple(summaries),
        diagnostics=tuple(diagnostics),
        pairwise=tuple(pairwise),
        prompt_ablations=prompt_ablations,
    )


def render_markdown(analysis: StudyAnalysis) -> str:
    lines = [
        "## Clustered bootstrap summary",
        "",
        "| Environment | Prompt mode | Model | Run ID | Tasks x rollouts | Resampling | Reward mean | Rollout std | 95% cluster CI | Task-mean std | Zero-reward tasks | Recovered |",
        "| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in analysis.summaries:
        lines.append(
            "| "
            + " | ".join(
                [
                    summary.environment,
                    summary.prompt_mode or "—",
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

    if analysis.diagnostics:
        lines.extend(
            [
                "",
                "## Clustered diagnostic metrics",
                "",
                "| Environment | Prompt mode | Model | Run ID | Metric | Tasks | Rollouts | Rollout mean | Rollout std | Task mean | 95% cluster CI | Task-mean std | Resampling |",
                "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for diagnostic in analysis.diagnostics:
            lines.append(
                "| "
                + " | ".join(
                    [
                        diagnostic.environment,
                        diagnostic.prompt_mode or "—",
                        diagnostic.model,
                        diagnostic.run_id,
                        diagnostic.metric,
                        str(diagnostic.task_count),
                        str(diagnostic.rollout_count),
                        f"{diagnostic.rollout_mean:.3f}",
                        f"{diagnostic.rollout_std:.3f}",
                        f"{diagnostic.task_mean:.3f}",
                        f"[{diagnostic.ci_low:.3f}, {diagnostic.ci_high:.3f}]",
                        f"{diagnostic.task_mean_std:.3f}",
                        (f"{diagnostic.resampling_unit} ({diagnostic.cluster_count})"),
                    ]
                )
                + " |"
            )

    if analysis.prompt_ablations:
        lines.extend(
            [
                "",
                "## Predict paired prompt ablations (exploratory; unadjusted)",
                "",
                "| Environment | Model | Metric | Reference | Mode | Tasks | Reference mean | Mode mean | Mean full - mode | 95% paired task bootstrap CI | Interpretation |",
                "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for comparison in analysis.prompt_ablations:
            lines.append(
                "| "
                + " | ".join(
                    [
                        comparison.environment,
                        comparison.model,
                        comparison.metric,
                        comparison.reference_mode,
                        comparison.prompt_mode,
                        str(comparison.task_count),
                        f"{comparison.reference_mean:.3f}",
                        f"{comparison.prompt_mean:.3f}",
                        f"{comparison.mean_difference:+.3f}",
                        f"[{comparison.ci_low:+.3f}, {comparison.ci_high:+.3f}]",
                        comparison.interpretation,
                    ]
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "## Paired clustered model differences (exploratory)",
            "",
            "| Environment | Prompt mode | Model A | Model B | Mean A - B | 95% paired clustered bootstrap CI | Raw p | Holm-adjusted p | Unadjusted CI reading | Exploratory Holm reading |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for difference in analysis.pairwise:
        lines.append(
            "| "
            + " | ".join(
                [
                    difference.environment,
                    difference.prompt_mode or "—",
                    difference.model_a,
                    difference.model_b,
                    f"{difference.mean_difference:+.3f}",
                    f"[{difference.ci_low:+.3f}, {difference.ci_high:+.3f}]",
                    f"{difference.raw_p_value:.4f}",
                    f"{difference.holm_adjusted_p_value:.4f}",
                    difference.interpretation,
                    difference.holm_interpretation,
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
                "comparisons reuse the same resample. Diagnostic intervals use "
                "the same task/template resampling as reward. Predict prompt "
                "ablations require matching native comparison signatures and "
                "compare raw named probability_reward scores for each mode with "
                "full on the same task-answer roster; their percentile intervals "
                "are exploratory and are not multiplicity-adjusted. "
                "Uniform Brier skill uses the three-class loss of 1/3; "
                "original-snapshot-prior skill is the ratio of equally weighted "
                "pooled model loss to evaluator-side original visible-matrix "
                "prior loss, recomputed inside each bootstrap resample; that "
                "reference is unavailable to a text-only agent, and zero pooled "
                "reference loss makes the score undefined. Pairwise model "
                "p-values use deterministic Monte Carlo paired sign flips at "
                "the independent-cluster level. This exploratory randomization "
                "tests a sign-symmetric/exchangeable paired cluster-effect null; "
                "equality of means alone is insufficient. Primary task p-values "
                "are adjusted together as one global family of "
                f"{len(analysis.pairwise)} comparisons using Holm's method; "
                "component and prompt-ablation intervals are pointwise, "
                "unadjusted, descriptive intervals rather than simultaneous "
                "confidence bands. These model comparisons are exploratory "
                "rather than confirmatory. "
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
            "diagnostic_intervals": (
                "paired hierarchical cluster percentile bootstrap"
            ),
            "prompt_ablation_intervals": (
                "paired task percentile bootstrap of full minus prompt mode"
            ),
            "prompt_ablation_provenance": (
                "matching native comparison signature over model, client, "
                "sampling, concurrency, task answers, and reward contract"
            ),
            "brier_skill_references": {
                "brier_skill_vs_uniform": (
                    "uniform three-class normalized Brier = 1/3"
                ),
                "brier_skill_vs_original_snapshot_visible_prior": (
                    "equally weighted pooled model loss relative to evaluator-side "
                    "original full-snapshot visible-matrix class-frequency loss; "
                    "fixed across modes and unavailable to text-only agents; "
                    "undefined when pooled reference loss is zero"
                ),
            },
            "pairwise_p_values": (
                "paired cluster sign-flip Monte Carlo randomization under a "
                "sign-symmetric/exchangeable paired cluster-effect null; equality "
                "of means alone is insufficient"
            ),
            "pairwise_multiplicity": (
                "global Holm adjustment of exploratory paired cluster "
                "sign-flip p-values"
            ),
            "pairwise_family_size": len(analysis.pairwise),
            "pairwise_family_definition": (
                "all primary pairwise model contrasts across the three declared "
                "tasks; with four models this is 3 * choose(4, 2) = 18; prompt "
                "ablations and component diagnostics are excluded"
            ),
            "pairwise_status": "exploratory",
            "interval_scope": (
                "pointwise percentile intervals; component and prompt-ablation "
                "intervals are unadjusted and are not simultaneous bands"
            ),
            "bootstrap_samples": analysis.bootstrap_samples,
            "confidence_level": analysis.confidence_level,
            "seed": analysis.seed,
        },
        "summaries": [asdict(item) for item in analysis.summaries],
        "diagnostics": [asdict(item) for item in analysis.diagnostics],
        "prompt_ablations": [asdict(item) for item in analysis.prompt_ablations],
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
            expected_pairwise_family_size=args.expected_pairwise_family_size,
            require_no_recoveries=args.require_no_recoveries,
            task_cluster_labels=cluster_labels,
        )
        if args.csv is not None:
            write_csv(args.csv.resolve(), analysis.summaries)
        if args.diagnostics_csv is not None:
            write_csv(args.diagnostics_csv.resolve(), analysis.diagnostics)
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
