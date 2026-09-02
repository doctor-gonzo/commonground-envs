"""Aggregate complete local Common Ground evaluation runs.

Native ``verifiers==0.3.0`` runs contain ``config.toml`` and one episode per
rollout in ``traces.jsonl``. Saved ``vf-eval`` runs can instead contain
``metadata.json``/``results.jsonl``; both the historical direct layout and the
current nested ``evals/<environment>/<run>`` layout remain readable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIRS_GLOBS = (
    "environments/*/outputs/evals/*/*",
    "environments/*/outputs/*/*",
    "outputs/*/*/evals/*/*",
    "outputs/*/*",
)
LEGACY_ARTIFACTS = ("metadata.json", "results.jsonl")
NATIVE_ARTIFACTS = ("config.toml", "traces.jsonl")
METRIC_NAMES = (
    "vote_accuracy",
    "brier",
    "original_snapshot_visible_prior_brier",
    "brier_skill_vs_original_snapshot_visible_prior",
    "finding_localization_recall",
    "finding_type_accuracy",
    "finding_diagnosis_recall",
    "finding_relation_recall",
    "finding_f1",
    "question_utility",
    "question_format_valid",
    "question_top1_selection_accuracy",
    "question_grounding_recall",
    "question_grounded_stance_recall",
    "question_evidence_match_recall",
    "question_evidence_matched_stance_accuracy",
    # Historical 0.5.x traces used this name for end-to-end grounded recall.
    "question_stance_accuracy",
)
ELICIT_TASK_MODES = frozenset({"find", "elicit-ask"})
PREDICT_PROMPT_MODES = ("full", "matrix-only", "text-only", "shuffled-text")
PREDICT_ABLATION_SIGNATURE_SCHEMA = "commonground-predict-ablation-v1"


class InvalidRunError(ValueError):
    """Raised when a saved run cannot support a complete baseline row."""


@dataclass(frozen=True)
class Summary:
    model: str
    environment: str
    run_id: str
    rollout_count: int
    recovered_rollout_count: int
    reward_mean: float
    reward_std: float
    metrics: Mapping[str, tuple[float, float]]
    prompt_mode: str | None = None


@dataclass(frozen=True)
class CompleteRun:
    model: str
    environment: str
    run_id: str
    descriptor_timestamp_ns: int
    recovered_rollout_count: int
    task_ids: tuple[int, ...]
    rewards: tuple[float, ...]
    metrics: Mapping[str, tuple[float, ...]]
    prompt_mode: str | None = None
    comparison_signature: str | None = None
    task_answer_digests: tuple[tuple[int, str], ...] = ()
    raw_reward_scores: Mapping[str, tuple[float, ...]] = field(default_factory=dict)
    raw_reward_weights: Mapping[str, tuple[float, ...]] = field(default_factory=dict)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate complete Common Ground baseline eval outputs."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Repository root containing outputs or legacy environment evals (default: repo root).",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Also write machine-readable summary rows to this CSV path.",
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help=(
            "Include every complete run instead of only the newest "
            "environment/model/prompt-mode key."
        ),
    )
    return parser.parse_args(argv)


def discover_result_paths(root: Path) -> list[Path]:
    """Find complete native/legacy run pairs and reject partial artifacts."""

    run_dirs = sorted(
        {
            path
            for pattern in DEFAULT_RUN_DIRS_GLOBS
            for path in root.glob(pattern)
            if path.is_dir()
        },
        key=lambda path: path.as_posix(),
    )
    result_paths: list[Path] = []
    partial_run_dirs: list[Path] = []
    log_only_run_dirs: list[Path] = []
    for run_dir in run_dirs:
        legacy_present = tuple((run_dir / name).is_file() for name in LEGACY_ARTIFACTS)
        native_present = tuple((run_dir / name).is_file() for name in NATIVE_ARTIFACTS)
        legacy_complete = all(legacy_present)
        native_complete = all(native_present)
        if legacy_complete and native_complete:
            raise InvalidRunError(
                f"ambiguous eval run contains both native and legacy artifacts: {run_dir}"
            )
        if (legacy_complete and any(native_present)) or (
            native_complete and any(legacy_present)
        ):
            partial_run_dirs.append(run_dir)
        elif legacy_complete:
            result_paths.append(run_dir / LEGACY_ARTIFACTS[1])
        elif native_complete:
            result_paths.append(run_dir / NATIVE_ARTIFACTS[1])
        elif any((*legacy_present, *native_present)):
            partial_run_dirs.append(run_dir)
        elif any(run_dir.glob("*.log")):
            log_only_run_dirs.append(run_dir)

    if partial_run_dirs:
        rendered = ", ".join(path.as_posix() for path in partial_run_dirs)
        raise InvalidRunError(f"partially saved eval run(s): {rendered}")
    if not result_paths:
        message = f"no saved eval results found under {root}"
        if log_only_run_dirs:
            message += (
                f"; found {len(log_only_run_dirs)} log-only run dir(s). "
                "Legacy vf-eval requires --save-results to write metadata.json and "
                "results.jsonl; native evals write config.toml and traces.jsonl"
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
                prompt_mode_sort_key(run.prompt_mode),
                run.descriptor_timestamp_ns,
                run.run_id,
            ),
        )
    ]


def newest_runs(runs: Sequence[CompleteRun]) -> list[CompleteRun]:
    """Select the latest result artifact per environment/model/prompt-mode key."""

    newest: dict[tuple[str, str, str | None], CompleteRun] = {}
    for run in runs:
        key = (run.environment, run.model, run.prompt_mode)
        incumbent = newest.get(key)
        if incumbent is None or (
            run.descriptor_timestamp_ns,
            run.run_id,
        ) > (
            incumbent.descriptor_timestamp_ns,
            incumbent.run_id,
        ):
            newest[key] = run
    return list(newest.values())


def pooled_brier_skill(
    brier_values: Sequence[float], reference_values: Sequence[float]
) -> float:
    """Return skill from aligned pooled losses, using equal rollout weights."""

    if not brier_values or len(brier_values) != len(reference_values):
        raise InvalidRunError(
            "Brier and original-snapshot-prior losses must be non-empty and aligned"
        )
    reference_mean = statistics.fmean(reference_values)
    if reference_mean <= 0.0:
        raise InvalidRunError(
            "original-snapshot-prior Brier reference must have positive pooled loss"
        )
    return 1.0 - statistics.fmean(brier_values) / reference_mean


def summarize_run(run: CompleteRun) -> Summary:
    metrics = {
        metric_name: summarize(run.metrics[metric_name])
        for metric_name in METRIC_NAMES
        if metric_name in run.metrics
    }
    if (
        "brier" in run.metrics
        and "original_snapshot_visible_prior_brier" in run.metrics
    ):
        brier_values = run.metrics["brier"]
        reference_values = run.metrics["original_snapshot_visible_prior_brier"]
        pooled_skill = pooled_brier_skill(brier_values, reference_values)
        reference_mean = statistics.fmean(reference_values)
        skill_contributions = tuple(
            1.0 - value / reference_mean for value in brier_values
        )
        metrics["brier_skill_vs_original_snapshot_visible_prior"] = (
            pooled_skill,
            statistics.pstdev(skill_contributions),
        )
    return Summary(
        model=run.model,
        environment=run.environment,
        run_id=run.run_id,
        rollout_count=len(run.rewards),
        recovered_rollout_count=run.recovered_rollout_count,
        reward_mean=statistics.fmean(run.rewards),
        reward_std=statistics.pstdev(run.rewards),
        metrics=metrics,
        prompt_mode=run.prompt_mode,
    )


def load_complete_run(result_path: Path) -> CompleteRun:
    if result_path.name == LEGACY_ARTIFACTS[1]:
        return load_legacy_run(result_path)
    if result_path.name == NATIVE_ARTIFACTS[1]:
        return load_native_run(result_path)
    raise InvalidRunError(f"unsupported saved eval artifact: {result_path}")


def load_legacy_run(result_path: Path) -> CompleteRun:
    """Load one complete saved-result ``vf-eval`` run."""

    metadata_path = result_path.with_name("metadata.json")
    metadata = load_json_object(metadata_path)
    base_environment = require_nonempty_string(metadata, "env_id", metadata_path)
    env_args = metadata.get("env_args")
    legacy_mode: Any = None
    if is_elicit_environment(base_environment):
        if not isinstance(env_args, Mapping):
            raise InvalidRunError(
                f"{metadata_path}: env_args must identify the Elicit task mode"
            )
        legacy_mode = env_args.get("task")
    environment, task_mode = qualify_elicit_environment(
        base_environment, legacy_mode, metadata_path
    )
    prompt_mode, prompt_mode_explicit = resolve_predict_prompt_mode(
        base_environment,
        env_args,
        metadata_path,
    )
    metric_contract_environment = legacy_metric_contract_environment(
        base_environment, metadata, metadata_path
    )
    expected_legacy_signals = expected_legacy_metrics(
        metric_contract_environment, task_mode, metadata_path
    )
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
    task_ids: list[int] = []
    rewards: list[float] = []
    metric_values: dict[str, list[float]] = {}
    expected_metric_keys: tuple[str, ...] | None = None
    for line_number, output in enumerate(outputs, start=1):
        if output.get("is_completed") is not True or output.get("error") is not None:
            raise InvalidRunError(
                f"{result_path}:{line_number}: rollout must be completed without error"
            )
        example_id = output.get("example_id")
        if isinstance(example_id, bool) or not isinstance(example_id, int):
            raise InvalidRunError(
                f"{result_path}:{line_number}: example_id must be an integer"
            )
        counts_by_example[example_id] = counts_by_example.get(example_id, 0) + 1
        task_ids.append(example_id)
        reward = require_number(output, "reward", result_path, line_number)
        if not 0.0 <= reward <= 1.0:
            raise InvalidRunError(
                f"{result_path}:{line_number}: reward must be within [0, 1]"
            )
        rewards.append(reward)
        require_task_mode(output.get("info"), task_mode, result_path, line_number)
        require_prompt_mode(
            output.get("info"),
            prompt_mode,
            required=prompt_mode_explicit,
            path=result_path,
            line_number=line_number,
        )

        raw_metrics = output.get("metrics")
        if not isinstance(raw_metrics, Mapping):
            raise InvalidRunError(
                f"{result_path}:{line_number}: metrics must be an object"
            )
        raw_metric_keys = validated_mapping_keys(
            raw_metrics, "metrics", result_path, line_number
        )
        known_metric_keys = tuple(name for name in METRIC_NAMES if name in raw_metrics)
        if known_metric_keys != expected_legacy_signals:
            raise InvalidRunError(
                f"{result_path}:{line_number}: expected Common Ground metrics "
                f"{list(expected_legacy_signals)}, found {list(known_metric_keys)}"
            )
        if expected_metric_keys is None:
            expected_metric_keys = raw_metric_keys
        elif raw_metric_keys != expected_metric_keys:
            raise InvalidRunError(
                f"{result_path}:{line_number}: inconsistent logged metrics"
            )
        for metric_name in raw_metric_keys:
            metric_value = require_number(
                raw_metrics, metric_name, result_path, line_number
            )
            if metric_name in METRIC_NAMES:
                validate_metric_domain(
                    metric_name, metric_value, result_path, line_number
                )
                metric_values.setdefault(metric_name, []).append(metric_value)

    validate_rollout_distribution(
        counts_by_example,
        num_examples=num_examples,
        rollouts_per_example=rollouts_per_example,
        result_path=result_path,
    )

    return CompleteRun(
        model=model,
        environment=environment,
        run_id=result_path.parent.name,
        descriptor_timestamp_ns=metadata_path.stat().st_mtime_ns,
        recovered_rollout_count=0,
        task_ids=tuple(task_ids),
        rewards=tuple(rewards),
        metrics={name: tuple(values) for name, values in metric_values.items()},
        prompt_mode=prompt_mode,
    )


def load_native_run(result_path: Path) -> CompleteRun:
    """Load one complete single-agent Verifiers 0.3.x episode run."""

    config_path = result_path.with_name("config.toml")
    config = load_toml_object(config_path)
    model = require_nonempty_string(config, "model", config_path)
    num_tasks = require_positive_int(config, "num_tasks", config_path)
    num_rollouts = require_positive_int(config, "num_rollouts", config_path)
    base_environment, taskset = native_environment(config, config_path)
    native_mode = taskset.get("task_mode") if taskset is not None else None
    environment, task_mode = qualify_elicit_environment(
        base_environment, native_mode, config_path
    )
    prompt_mode, prompt_mode_explicit = resolve_predict_prompt_mode(
        base_environment,
        taskset,
        config_path,
    )
    validate_native_baseline_profile(
        config,
        base_environment=base_environment,
        taskset=taskset,
        task_mode=task_mode,
        config_path=config_path,
    )
    expected_reward_signals, expected_metric_signals = expected_native_signals(
        base_environment, task_mode, config_path
    )
    declared_predict_ablation = prompt_mode_explicit
    ablation_config_provenance: dict[str, Any] | None = None
    if declared_predict_ablation:
        if expected_reward_signals != ("probability_reward",):
            raise InvalidRunError(
                f"{config_path}: declared Predict ablations require the "
                "probability_reward contract"
            )
        ablation_config_provenance = require_predict_ablation_config_provenance(
            config,
            base_environment=base_environment,
            taskset=taskset,
            model=model,
            num_tasks=num_tasks,
            num_rollouts=num_rollouts,
            config_path=config_path,
        )
    expected_count = num_tasks * num_rollouts

    episodes = load_json_lines(result_path)
    if len(episodes) != expected_count:
        raise InvalidRunError(
            f"{result_path}: expected {expected_count} rollouts "
            f"({num_tasks} tasks x {num_rollouts}), found {len(episodes)}"
        )

    counts_by_example: dict[int, int] = {}
    task_ids: list[int] = []
    episode_ids: set[str] = set()
    trace_ids: set[str] = set()
    run_ids: set[str] = set()
    rewards: list[float] = []
    raw_reward_scores: dict[str, list[float]] = {}
    raw_reward_weights: dict[str, list[float]] = {}
    metric_values: dict[str, list[float]] = {}
    expected_reward_keys: tuple[str, ...] | None = None
    expected_metric_keys: tuple[str, ...] | None = None
    task_answer_digests: dict[int, str] = {}
    trace_agent_provenance: dict[str, Any] | None = None
    recovered_rollout_count = 0

    for line_number, episode in enumerate(episodes, start=1):
        trace = native_trace_from_episode(
            episode,
            expected_environment=base_environment,
            expected_model=model,
            result_path=result_path,
            line_number=line_number,
            episode_ids=episode_ids,
        )
        if declared_predict_ablation:
            if ablation_config_provenance is None:  # pragma: no cover - invariant
                raise InvalidRunError(
                    f"{result_path}: declared Predict ablation lacks config provenance"
                )
            current_trace_provenance = require_predict_ablation_trace_provenance(
                trace,
                expected_model=model,
                expected_client=ablation_config_provenance["client"],
                expected_sampling=ablation_config_provenance["sampling"],
                result_path=result_path,
                line_number=line_number,
            )
            if trace_agent_provenance is None:
                trace_agent_provenance = current_trace_provenance
            elif current_trace_provenance != trace_agent_provenance:
                raise InvalidRunError(
                    f"{result_path}:{line_number}: declared Predict ablation "
                    "trace agent provenance is inconsistent"
                )
        episode_errors = episode.get("errors")
        trace_errors = trace.get("errors")
        if not isinstance(episode_errors, list) or not isinstance(trace_errors, list):
            raise InvalidRunError(
                f"{result_path}:{line_number}: errors fields must be lists"
            )
        call_recovery = False
        if "calls" in trace:
            calls = trace.get("calls")
            if not isinstance(calls, list):
                raise InvalidRunError(
                    f"{result_path}:{line_number}: trace.calls must be a list"
                )
            for call in calls:
                if not isinstance(call, Mapping):
                    raise InvalidRunError(
                        f"{result_path}:{line_number}: trace calls must be objects"
                    )
            call_recovery = len(calls) != 1 or any(
                call.get("error") is not None for call in calls
            )
        recovered_rollout_count += bool(episode_errors or trace_errors or call_recovery)
        trace_id = require_nonempty_string(trace, "id", result_path, line_number)
        if trace_id in trace_ids:
            raise InvalidRunError(
                f"{result_path}:{line_number}: duplicate trace id {trace_id!r}"
            )
        trace_ids.add(trace_id)
        if (
            trace.get("is_completed") is not True
            or trace.get("ok") is not True
            or not isinstance(trace.get("errors"), list)
        ):
            raise InvalidRunError(
                f"{result_path}:{line_number}: trace must be completed and ok"
            )

        run = trace.get("run")
        if not isinstance(run, Mapping) or run.get("type") != "eval":
            raise InvalidRunError(
                f"{result_path}:{line_number}: trace.run must identify an eval run"
            )
        run_ids.add(require_nonempty_string(run, "id", result_path, line_number))

        task = trace.get("task")
        if not isinstance(task, Mapping):
            raise InvalidRunError(
                f"{result_path}:{line_number}: trace.task must be an object"
            )
        expected_task_type = (
            "PredictionTask"
            if environment_package_name(base_environment) == "commonground-predict"
            else "ElicitTask"
        )
        if task.get("type") != expected_task_type:
            raise InvalidRunError(
                f"{result_path}:{line_number}: expected task type {expected_task_type}"
            )
        task_data = task.get("data")
        if not isinstance(task_data, Mapping):
            raise InvalidRunError(
                f"{result_path}:{line_number}: trace.task.data must be an object"
            )
        example_id = task_data.get("idx")
        if isinstance(example_id, bool) or not isinstance(example_id, int):
            raise InvalidRunError(
                f"{result_path}:{line_number}: trace.task.data.idx must be an integer"
            )
        counts_by_example[example_id] = counts_by_example.get(example_id, 0) + 1
        task_ids.append(example_id)
        answer_digest = task_answer_digest(
            task_data,
            required=declared_predict_ablation,
            path=result_path,
            line_number=line_number,
        )
        if answer_digest is not None:
            previous_digest = task_answer_digests.setdefault(example_id, answer_digest)
            if previous_digest != answer_digest:
                raise InvalidRunError(
                    f"{result_path}:{line_number}: task {example_id} has inconsistent "
                    "answers across rollouts"
                )
        require_task_mode(task_data.get("info"), task_mode, result_path, line_number)
        require_prompt_mode(
            task_data.get("info"),
            prompt_mode,
            required=prompt_mode_explicit,
            path=result_path,
            line_number=line_number,
        )

        raw_rewards = trace.get("rewards")
        if not isinstance(raw_rewards, Mapping) or not raw_rewards:
            raise InvalidRunError(
                f"{result_path}:{line_number}: rewards must be a non-empty object"
            )
        reward_keys = validated_mapping_keys(
            raw_rewards, "rewards", result_path, line_number
        )
        if reward_keys != expected_reward_signals:
            raise InvalidRunError(
                f"{result_path}:{line_number}: expected rewards "
                f"{list(expected_reward_signals)}, found {list(reward_keys)}"
            )
        if expected_reward_keys is None:
            expected_reward_keys = reward_keys
        elif reward_keys != expected_reward_keys:
            raise InvalidRunError(
                f"{result_path}:{line_number}: inconsistent logged rewards"
            )

        raw_metrics = trace.get("metrics")
        if not isinstance(raw_metrics, Mapping):
            raise InvalidRunError(
                f"{result_path}:{line_number}: metrics must be an object"
            )
        metric_keys = validated_mapping_keys(
            raw_metrics, "metrics", result_path, line_number
        )
        if metric_keys != expected_metric_signals:
            raise InvalidRunError(
                f"{result_path}:{line_number}: expected metrics "
                f"{list(expected_metric_signals)}, found {list(metric_keys)}"
            )
        if expected_metric_keys is None:
            expected_metric_keys = metric_keys
        elif metric_keys != expected_metric_keys:
            raise InvalidRunError(
                f"{result_path}:{line_number}: inconsistent logged metrics"
            )

        overlap = set(reward_keys) & set(metric_keys) & set(METRIC_NAMES)
        if overlap:
            raise InvalidRunError(
                f"{result_path}:{line_number}: signal logged as both reward and metric: "
                f"{sorted(overlap)}"
            )
        reward_total = 0.0
        for reward_name in reward_keys:
            reward = raw_rewards[reward_name]
            if not isinstance(reward, Mapping):
                raise InvalidRunError(
                    f"{result_path}:{line_number}: reward {reward_name!r} must be an object"
                )
            score = require_number(reward, "score", result_path, line_number)
            weight = require_number(reward, "weight", result_path, line_number)
            validate_metric_domain(reward_name, score, result_path, line_number)
            raw_reward_scores.setdefault(reward_name, []).append(score)
            raw_reward_weights.setdefault(reward_name, []).append(weight)
            if (
                declared_predict_ablation
                and reward_name == "probability_reward"
                and weight != 1.0
            ):
                raise InvalidRunError(
                    f"{result_path}:{line_number}: declared Predict ablations require "
                    "probability_reward weight 1.0"
                )
            weighted = score * weight
            if not math.isfinite(weighted):
                raise InvalidRunError(
                    f"{result_path}:{line_number}: reward {reward_name!r} is not finite"
                )
            reward_total += weighted
            if reward_name in METRIC_NAMES:
                metric_values.setdefault(reward_name, []).append(score)
        rewards.append(reward_total)

        for metric_name in metric_keys:
            metric_value = require_number(
                raw_metrics, metric_name, result_path, line_number
            )
            validate_metric_domain(metric_name, metric_value, result_path, line_number)
            if metric_name in METRIC_NAMES:
                metric_values.setdefault(metric_name, []).append(metric_value)

    if len(run_ids) != 1:
        raise InvalidRunError(
            f"{result_path}: traces must share exactly one eval run id, found {sorted(run_ids)}"
        )
    validate_rollout_distribution(
        counts_by_example,
        num_examples=num_tasks,
        rollouts_per_example=num_rollouts,
        result_path=result_path,
    )
    comparison_signature: str | None = None
    if declared_predict_ablation:
        if ablation_config_provenance is None:  # pragma: no cover - invariant
            raise InvalidRunError(
                f"{result_path}: declared Predict ablation lacks config provenance"
            )
        if trace_agent_provenance is None:  # pragma: no cover - positive run size
            raise InvalidRunError(
                f"{result_path}: declared Predict ablation has no trace provenance"
            )
        task_roster = [
            {
                "task_id": task_id,
                "rollouts": counts_by_example[task_id],
                "answer_sha256": task_answer_digests[task_id],
            }
            for task_id in sorted(counts_by_example)
        ]
        # Regression guard: the outcome scores are intentionally excluded, while
        # every model-facing setting and the immutable answer roster stay paired.
        comparison_signature = stable_json_digest(
            {
                **ablation_config_provenance,
                "trace_agent": trace_agent_provenance,
                "tasks": task_roster,
                # Preserve execution order as an additional release-provenance
                # invariant. Pairwise statistics still align observations by
                # task ID, but a reordered run is not evidence that prompt_mode
                # was the only changed factor.
                "task_sequence": task_ids,
                "reward_contract": {
                    "name": "probability_reward",
                    "weight": 1.0,
                },
            },
            f"{result_path}: Predict ablation comparison provenance",
        )
    return CompleteRun(
        model=model,
        environment=environment,
        run_id=next(iter(run_ids)),
        descriptor_timestamp_ns=result_path.stat().st_mtime_ns,
        recovered_rollout_count=recovered_rollout_count,
        task_ids=tuple(task_ids),
        rewards=tuple(rewards),
        metrics={name: tuple(values) for name, values in metric_values.items()},
        prompt_mode=prompt_mode,
        comparison_signature=comparison_signature,
        task_answer_digests=tuple(sorted(task_answer_digests.items())),
        raw_reward_scores={
            name: tuple(values) for name, values in raw_reward_scores.items()
        },
        raw_reward_weights={
            name: tuple(values) for name, values in raw_reward_weights.items()
        },
    )


def native_environment(
    config: Mapping[str, Any], config_path: Path
) -> tuple[str, Mapping[str, Any] | None]:
    """Resolve ``EvalConfig.env_id`` from its saved canonical TOML tables."""

    env = config.get("env")
    if not isinstance(env, Mapping):
        raise InvalidRunError(f"{config_path}: env must be an object")
    raw_env_id = env.get("id", "")
    if not isinstance(raw_env_id, str) or raw_env_id != raw_env_id.strip():
        raise InvalidRunError(f"{config_path}: env.id must be canonical text")

    taskset = env.get("taskset")
    if not isinstance(taskset, Mapping):
        raise InvalidRunError(f"{config_path}: env.taskset must be an object")
    raw_taskset_id = taskset.get("id", "")
    if not isinstance(raw_taskset_id, str) or raw_taskset_id != raw_taskset_id.strip():
        raise InvalidRunError(f"{config_path}: env.taskset.id must be canonical text")

    legacy = config.get("legacy", {})
    if not isinstance(legacy, Mapping):
        raise InvalidRunError(f"{config_path}: legacy must be an object")
    raw_legacy_id = legacy.get("id", "")
    if raw_legacy_id is None:
        raw_legacy_id = ""
    if not isinstance(raw_legacy_id, str) or raw_legacy_id != raw_legacy_id.strip():
        raise InvalidRunError(f"{config_path}: legacy.id must be canonical text")

    if raw_legacy_id and (raw_env_id or raw_taskset_id):
        raise InvalidRunError(
            f"{config_path}: native and legacy environment identities cannot be mixed"
        )
    if raw_taskset_id:
        environment = f"{raw_env_id}+{raw_taskset_id}" if raw_env_id else raw_taskset_id
        return environment, taskset
    if raw_env_id:
        return raw_env_id, taskset
    if raw_legacy_id:
        return raw_legacy_id, None
    raise InvalidRunError(f"{config_path}: no environment identity is configured")


def native_trace_from_episode(
    episode: Mapping[str, Any],
    *,
    expected_environment: str,
    expected_model: str,
    result_path: Path,
    line_number: int,
    episode_ids: set[str],
) -> Mapping[str, Any]:
    """Validate and unwrap one current v1 single-agent episode."""

    if "traces" not in episode or "nodes" in episode:
        raise InvalidRunError(
            f"{result_path}:{line_number}: expected a Verifiers v1 episode object"
        )
    episode_id = require_nonempty_string(episode, "id", result_path, line_number)
    if episode_id in episode_ids:
        raise InvalidRunError(
            f"{result_path}:{line_number}: duplicate episode id {episode_id!r}"
        )
    episode_ids.add(episode_id)
    if episode.get("ok") is not True or not isinstance(episode.get("errors"), list):
        raise InvalidRunError(f"{result_path}:{line_number}: episode must be ok")
    env = episode.get("env")
    if not isinstance(env, Mapping) or env.get("id") != expected_environment:
        raise InvalidRunError(
            f"{result_path}:{line_number}: episode environment does not match config"
        )
    traces = episode.get("traces")
    if not isinstance(traces, list) or len(traces) != 1:
        raise InvalidRunError(
            f"{result_path}:{line_number}: expected exactly one trace per episode"
        )
    trace = traces[0]
    if not isinstance(trace, Mapping):
        raise InvalidRunError(
            f"{result_path}:{line_number}: episode trace must be an object"
        )
    if trace.get("version") != 1:
        raise InvalidRunError(
            f"{result_path}:{line_number}: trace.version must equal 1"
        )
    version_info = trace.get("verifiers")
    if not isinstance(version_info, Mapping):
        raise InvalidRunError(
            f"{result_path}:{line_number}: trace.verifiers must be an object"
        )
    require_nonempty_string(version_info, "version", result_path, line_number)
    agent = trace.get("agent")
    if not isinstance(agent, Mapping):
        raise InvalidRunError(
            f"{result_path}:{line_number}: trace.agent must be an object"
        )
    if agent.get("trainable") is not True:
        raise InvalidRunError(
            f"{result_path}:{line_number}: trace agent must be trainable"
        )
    agent_config = agent.get("config")
    if not isinstance(agent_config, Mapping):
        raise InvalidRunError(
            f"{result_path}:{line_number}: trace.agent.config must be an object"
        )
    agent_model = agent_config.get("model")
    if agent_model is not None and agent_model != expected_model:
        raise InvalidRunError(
            f"{result_path}:{line_number}: trace agent model does not match config"
        )
    return trace


def require_predict_ablation_config_provenance(
    config: Mapping[str, Any],
    *,
    base_environment: str,
    taskset: Mapping[str, Any] | None,
    model: str,
    num_tasks: int,
    num_rollouts: int,
    config_path: Path,
) -> dict[str, Any]:
    """Return the resolved model-facing config for a declared Predict ablation."""

    if taskset is None:  # pragma: no cover - baseline validation already rejects this
        raise InvalidRunError(
            f"{config_path}: declared Predict ablations require a native taskset"
        )
    client = config.get("client")
    if not isinstance(client, Mapping) or not client:
        raise InvalidRunError(
            f"{config_path}: declared Predict ablations require resolved client "
            "provenance"
        )
    identity_values: list[str] = []
    for key in ("provider", "base_url"):
        if key not in client:
            continue
        value = client[key]
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            raise InvalidRunError(
                f"{config_path}: declared Predict ablation client {key} must be "
                "non-empty canonical text"
            )
        identity_values.append(value)
    if not identity_values:
        raise InvalidRunError(
            f"{config_path}: declared Predict ablation client identity is incomplete"
        )
    sampling = config.get("sampling")
    if not isinstance(sampling, Mapping):
        raise InvalidRunError(
            f"{config_path}: declared Predict ablations require resolved sampling "
            "settings"
        )
    max_concurrent = require_positive_int(config, "max_concurrent", config_path)

    concurrency: dict[str, Any] = {"max_concurrent": max_concurrent}
    env = config.get("env")
    if isinstance(env, Mapping):
        if "max_concurrent_agents" in env:
            concurrency["env.max_concurrent_agents"] = env["max_concurrent_agents"]
        interception = env.get("interception")
        if isinstance(interception, Mapping) and "multiplex" in interception:
            concurrency["env.interception.multiplex"] = interception["multiplex"]
    serve = config.get("serve")
    if isinstance(serve, Mapping):
        if "max_concurrent" in serve:
            concurrency["serve.max_concurrent"] = serve["max_concurrent"]
        pool = serve.get("pool")
        if isinstance(pool, Mapping):
            for key in ("max_workers", "multiplex"):
                if key in pool:
                    concurrency[f"serve.pool.{key}"] = pool[key]

    # prompt_mode is the treatment. Keeping it out of this payload is what lets
    # the four otherwise-identical native artifacts share one stable signature.
    taskset_without_prompt_mode = {
        key: value for key, value in taskset.items() if key != "prompt_mode"
    }
    return {
        "schema": PREDICT_ABLATION_SIGNATURE_SCHEMA,
        "environment": base_environment,
        "model": model,
        "client": dict(client),
        "sampling": dict(sampling),
        "concurrency": concurrency,
        "num_tasks": num_tasks,
        "num_rollouts": num_rollouts,
        "shuffle": config.get("shuffle"),
        "taskset": taskset_without_prompt_mode,
    }


def require_predict_ablation_trace_provenance(
    trace: Mapping[str, Any],
    *,
    expected_model: str,
    expected_client: Mapping[str, Any],
    expected_sampling: Mapping[str, Any],
    result_path: Path,
    line_number: int,
) -> dict[str, Any]:
    """Validate the effective agent identity recorded in an ablation trace."""

    agent = trace.get("agent")
    agent_config = agent.get("config") if isinstance(agent, Mapping) else None
    if not isinstance(agent_config, Mapping):  # pragma: no cover - checked upstream
        raise InvalidRunError(
            f"{result_path}:{line_number}: trace agent config must be an object"
        )
    if agent_config.get("model") != expected_model:
        raise InvalidRunError(
            f"{result_path}:{line_number}: declared Predict ablation trace must "
            "record the resolved model"
        )
    client = agent_config.get("client")
    if not isinstance(client, Mapping) or dict(client) != dict(expected_client):
        raise InvalidRunError(
            f"{result_path}:{line_number}: declared Predict ablation trace client "
            "does not match saved config"
        )
    sampling = agent_config.get("sampling")
    if not isinstance(sampling, Mapping) or dict(sampling) != dict(expected_sampling):
        raise InvalidRunError(
            f"{result_path}:{line_number}: declared Predict ablation trace sampling "
            "does not match saved config"
        )
    calls = trace.get("calls")
    if not isinstance(calls, list) or len(calls) != 1:
        raise InvalidRunError(
            f"{result_path}:{line_number}: declared Predict ablation trace must "
            "record exactly one model call"
        )
    call = calls[0]
    if not isinstance(call, Mapping) or call.get("error") is not None:
        raise InvalidRunError(
            f"{result_path}:{line_number}: declared Predict ablation model call "
            "must be a successful object"
        )
    if call.get("model") != expected_model:
        raise InvalidRunError(
            f"{result_path}:{line_number}: declared Predict ablation call model "
            "does not match saved config"
        )
    call_sampling = call.get("sampling")
    if not isinstance(call_sampling, Mapping) or dict(call_sampling) != dict(
        expected_sampling
    ):
        raise InvalidRunError(
            f"{result_path}:{line_number}: declared Predict ablation call sampling "
            "does not match saved config"
        )
    endpoint = call.get("endpoint")
    if (
        not isinstance(endpoint, str)
        or not endpoint.strip()
        or endpoint != endpoint.strip()
    ):
        raise InvalidRunError(
            f"{result_path}:{line_number}: declared Predict ablation call endpoint "
            "must be non-empty canonical text"
        )
    call_identity: dict[str, str] = {}
    for key in ("provider", "base_url"):
        if key not in call:
            continue
        value = call[key]
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            raise InvalidRunError(
                f"{result_path}:{line_number}: declared Predict ablation call {key} "
                "must be non-empty canonical text"
            )
        expected_value = expected_client.get(key)
        if expected_value is not None and value != expected_value:
            raise InvalidRunError(
                f"{result_path}:{line_number}: declared Predict ablation call {key} "
                "does not match saved client config"
            )
        call_identity[key] = value
    return {
        "model": expected_model,
        "client": dict(client),
        "sampling": dict(sampling),
        "call": {
            "model": expected_model,
            "sampling": dict(call_sampling),
            "endpoint": endpoint,
            **call_identity,
        },
    }


def task_answer_digest(
    task_data: Mapping[str, Any],
    *,
    required: bool,
    path: Path,
    line_number: int,
) -> str | None:
    """Hash one saved task answer without retaining its potentially sensitive text."""

    if "answer" not in task_data:
        if required:
            raise InvalidRunError(
                f"{path}:{line_number}: declared Predict ablation trace is missing "
                "the task answer"
            )
        return None
    answer = task_data["answer"]
    if required and not isinstance(answer, Mapping):
        raise InvalidRunError(
            f"{path}:{line_number}: declared Predict ablation answer must be an object"
        )
    return stable_json_digest(answer, f"{path}:{line_number}: task answer")


def stable_json_digest(value: Any, location: str) -> str:
    """Hash a canonical JSON value for stable cross-artifact comparison."""

    try:
        canonical = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise InvalidRunError(f"{location} is not canonical JSON: {error}") from error
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_rollout_distribution(
    counts_by_example: Mapping[int, int],
    *,
    num_examples: int,
    rollouts_per_example: int,
    result_path: Path,
) -> None:
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


def validated_mapping_keys(
    value: Mapping[Any, Any], label: str, path: Path, line_number: int
) -> tuple[str, ...]:
    if any(not isinstance(key, str) or not key for key in value):
        raise InvalidRunError(
            f"{path}:{line_number}: {label} keys must be non-empty strings"
        )
    return tuple(sorted(value))


def environment_package_name(environment: str) -> str:
    taskset_id = environment.rsplit("+", 1)[-1]
    return taskset_id.split("@", 1)[0].rstrip("/").rsplit("/", 1)[-1]


def uses_historical_predict_reward(environment: str) -> bool:
    """Return whether a pinned Predict environment predates the 0.4 reward contract."""

    taskset_id = environment.rsplit("+", 1)[-1]
    _, separator, version = taskset_id.partition("@")
    return bool(separator and version.startswith(("0.1.", "0.2.", "0.3.")))


def uses_snapshot_prior_skill(environment: str) -> bool:
    """Return whether Predict emits the original-snapshot reference loss."""

    taskset_id = environment.rsplit("+", 1)[-1]
    _, separator, version = taskset_id.partition("@")
    return not separator or not version.startswith(
        ("0.1.", "0.2.", "0.3.", "0.4.", "0.5.")
    )


def legacy_metric_contract_environment(
    environment: str, metadata: Mapping[str, Any], path: Path
) -> str:
    """Resolve an unpinned legacy ID against its recorded package version.

    Old ``vf-eval`` artifacts usually saved an unpinned ``env_id`` while
    recording the installed environment version under ``version_info``.  The
    version must drive metric-contract validation so immutable pre-0.6 runs do
    not get reinterpreted as current runs merely because their ID was unpinned.
    """

    taskset_id = environment.rsplit("+", 1)[-1]
    if "@" in taskset_id:
        return environment
    version_info = metadata.get("version_info")
    if version_info is None:
        return environment
    if not isinstance(version_info, Mapping):
        raise InvalidRunError(f"{path}: version_info must be an object")
    env_version = version_info.get("env_version")
    if env_version is None:
        return environment
    if not isinstance(env_version, str) or not env_version.strip():
        raise InvalidRunError(
            f"{path}: version_info.env_version must be a non-empty string"
        )
    return f"{environment}@{env_version.strip()}"


def uses_structured_elicit_diagnostics(environment: str) -> bool:
    """Return whether Elicit uses the expanded 0.5 diagnostic contract."""

    taskset_id = environment.rsplit("+", 1)[-1]
    _, separator, version = taskset_id.partition("@")
    return not separator or not version.startswith(("0.1.", "0.2.", "0.3.", "0.4."))


def uses_grounded_stance_diagnostics(environment: str) -> bool:
    """Return whether Elicit exposes distinct end-to-end and conditional metrics."""

    taskset_id = environment.rsplit("+", 1)[-1]
    _, separator, version = taskset_id.partition("@")
    return not separator or not version.startswith(
        ("0.1.", "0.2.", "0.3.", "0.4.", "0.5.")
    )


def is_elicit_environment(environment: str) -> bool:
    return environment_package_name(environment) == "commonground-elicit"


def validate_native_baseline_profile(
    config: Mapping[str, Any],
    *,
    base_environment: str,
    taskset: Mapping[str, Any] | None,
    task_mode: str | None,
    config_path: Path,
) -> None:
    """Reject native runs whose knobs make baseline rows incomparable."""

    if taskset is None:
        raise InvalidRunError(
            f"{config_path}: Common Ground native baselines require env.taskset"
        )
    package_name = environment_package_name(base_environment)
    if package_name not in {"commonground-elicit", "commonground-predict"}:
        raise InvalidRunError(
            f"{config_path}: unsupported Common Ground taskset {base_environment!r}"
        )
    if config.get("shuffle") is not False:
        raise InvalidRunError(f"{config_path}: baseline shuffle must be false")
    if taskset.get("split") != "eval":
        raise InvalidRunError(f"{config_path}: taskset split must equal 'eval'")
    task_config = taskset.get("task")
    if not isinstance(task_config, Mapping) or task_config.get("judges") != []:
        raise InvalidRunError(
            f"{config_path}: baseline task config must use no auxiliary judges"
        )

    forbidden_overrides = {
        "data_path",
        "docs_count",
        "docs_length",
        "masked_vote_count",
        "min_cluster_count",
        "train_data_path",
    }
    present_overrides = sorted(forbidden_overrides & set(taskset))
    if present_overrides:
        raise InvalidRunError(
            f"{config_path}: baseline taskset has noncanonical overrides "
            f"{present_overrides}"
        )
    if package_name == "commonground-elicit":
        expected = {
            "planted_density": 1.0,
            "distractor_density": 1.0,
            "panel_polarization": 1.0,
            # 0.6 makes the default selection budget one of three issues; older
            # immutable studies used two of three and remain readable.
            "question_count": (
                1 if uses_grounded_stance_diagnostics(base_environment) else 2
            ),
            "task_mode": task_mode,
        }
        drift = {
            key: taskset.get(key)
            for key, expected_value in expected.items()
            if taskset.get(key) != expected_value
        }
        if drift:
            raise InvalidRunError(
                f"{config_path}: noncanonical Elicit baseline settings {drift}"
            )


def expected_native_signals(
    environment: str, task_mode: str | None, path: Path
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    package_name = environment_package_name(environment)
    if package_name == "commonground-predict":
        if uses_historical_predict_reward(environment):
            return ("vote_accuracy",), ("brier",)
        metrics = (
            ("brier", "original_snapshot_visible_prior_brier", "vote_accuracy")
            if uses_snapshot_prior_skill(environment)
            else ("brier", "vote_accuracy")
        )
        return ("probability_reward",), metrics
    if package_name == "commonground-elicit" and task_mode == "find":
        if uses_structured_elicit_diagnostics(environment):
            return ("finding_f1",), (
                "finding_diagnosis_recall",
                "finding_localization_recall",
                "finding_relation_recall",
                "finding_type_accuracy",
                "question_utility",
            )
        return ("finding_f1",), (
            "finding_localization_recall",
            "finding_type_accuracy",
            "question_utility",
        )
    if package_name == "commonground-elicit" and task_mode == "elicit-ask":
        if uses_structured_elicit_diagnostics(environment):
            if uses_grounded_stance_diagnostics(environment):
                return ("question_utility",), (
                    "question_evidence_match_recall",
                    "question_evidence_matched_stance_accuracy",
                    "question_format_valid",
                    "question_grounded_stance_recall",
                    "question_grounding_recall",
                    "question_top1_selection_accuracy",
                )
            return ("question_utility",), (
                "question_format_valid",
                "question_grounding_recall",
                "question_stance_accuracy",
            )
        return ("question_utility",), ()
    raise InvalidRunError(f"{path}: unsupported Common Ground reward profile")


def expected_legacy_metrics(
    environment: str, task_mode: str | None, path: Path
) -> tuple[str, ...]:
    package_name = environment_package_name(environment)
    if package_name == "commonground-predict":
        return (
            ("vote_accuracy", "brier", "original_snapshot_visible_prior_brier")
            if uses_snapshot_prior_skill(environment)
            else ("vote_accuracy", "brier")
        )
    if package_name == "commonground-elicit" and task_mode in ELICIT_TASK_MODES:
        signal_names = {"question_utility"}
        if task_mode == "find":
            signal_names.add("finding_f1")
        if uses_structured_elicit_diagnostics(environment):
            _, diagnostic_names = expected_native_signals(environment, task_mode, path)
            signal_names.update(diagnostic_names)
        return tuple(name for name in METRIC_NAMES if name in signal_names)
    raise InvalidRunError(f"{path}: unsupported Common Ground metric profile")


def validate_metric_domain(
    name: str, value: float, path: Path, line_number: int
) -> None:
    if name == "brier_skill_vs_original_snapshot_visible_prior":
        # Skill is unbounded below when the original-snapshot reference is very
        # accurate. A zero pooled reference is rejected during derivation, so
        # finiteness is the remaining per-value fail-closed domain check.
        return
    upper = 1.0
    if not 0.0 <= value <= upper:
        raise InvalidRunError(
            f"{path}:{line_number}: {name} must be within [0, {upper:g}]"
        )


def qualify_elicit_environment(
    environment: str, raw_mode: Any, path: Path
) -> tuple[str, str | None]:
    if not is_elicit_environment(environment):
        return environment, None
    if raw_mode not in ELICIT_TASK_MODES:
        raise InvalidRunError(
            f"{path}: Elicit task mode must be one of {sorted(ELICIT_TASK_MODES)}"
        )
    return f"{environment}:{raw_mode}", raw_mode


def require_task_mode(
    raw_info: Any,
    expected_mode: str | None,
    path: Path,
    line_number: int,
) -> None:
    if expected_mode is None:
        return
    if not isinstance(raw_info, Mapping) or raw_info.get("task_label") != expected_mode:
        raise InvalidRunError(
            f"{path}:{line_number}: trace task mode does not match run config"
        )


def resolve_predict_prompt_mode(
    environment: str,
    raw_args: Any,
    path: Path,
) -> tuple[str | None, bool]:
    """Resolve Predict's prompt view while keeping historical runs readable."""

    if environment_package_name(environment) != "commonground-predict":
        return None, False
    if raw_args is None:
        return "full", False
    if not isinstance(raw_args, Mapping):
        raise InvalidRunError(
            f"{path}: Predict environment arguments must be an object"
        )
    explicit = "prompt_mode" in raw_args
    prompt_mode = raw_args.get("prompt_mode", "full")
    if prompt_mode not in PREDICT_PROMPT_MODES:
        raise InvalidRunError(
            f"{path}: Predict prompt mode must be one of {list(PREDICT_PROMPT_MODES)}"
        )
    return str(prompt_mode), explicit


def require_prompt_mode(
    raw_info: Any,
    expected_mode: str | None,
    *,
    required: bool,
    path: Path,
    line_number: int,
) -> None:
    if expected_mode is None:
        return
    if not isinstance(raw_info, Mapping):
        if not required:
            return
        raise InvalidRunError(
            f"{path}:{line_number}: trace prompt mode does not match run config"
        )
    raw_mode = raw_info.get("prompt_mode")
    if raw_mode is None and not required:
        return
    if raw_mode != expected_mode:
        raise InvalidRunError(
            f"{path}:{line_number}: trace prompt mode does not match run config"
        )


def prompt_mode_sort_key(prompt_mode: str | None) -> tuple[int, str]:
    if prompt_mode is None:
        return (-1, "")
    try:
        return (PREDICT_PROMPT_MODES.index(prompt_mode), prompt_mode)
    except ValueError:
        return (len(PREDICT_PROMPT_MODES), prompt_mode)


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = strict_json_loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise InvalidRunError(
            f"missing metadata beside saved results: {path}"
        ) from error
    except (json.JSONDecodeError, ValueError) as error:
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
            value = strict_json_loads(line)
        except (json.JSONDecodeError, ValueError) as error:
            raise InvalidRunError(
                f"invalid JSON in {path}:{line_number}: {error}"
            ) from error
        if not isinstance(value, dict):
            raise InvalidRunError(f"{path}:{line_number}: expected a JSON object")
        outputs.append(value)
    return outputs


def strict_json_loads(text: str) -> Any:
    def reject_constant(constant: str) -> None:
        raise ValueError(f"non-finite JSON number {constant}")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key {key!r}")
            value[key] = item
        return value

    return json.loads(
        text,
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicate_keys,
    )


def load_toml_object(path: Path) -> dict[str, Any]:
    try:
        value = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise InvalidRunError(f"missing config beside saved traces: {path}") from error
    except tomllib.TOMLDecodeError as error:
        raise InvalidRunError(f"invalid TOML in {path}: {error}") from error
    if not isinstance(value, dict):
        raise InvalidRunError(f"{path}: expected a TOML object")
    return value


def require_nonempty_string(
    value: Mapping[str, Any],
    key: str,
    path: Path,
    line_number: int | None = None,
) -> str:
    field = value.get(key)
    location = f"{path}:{line_number}" if line_number is not None else str(path)
    if (
        not isinstance(field, str)
        or not field
        or field != field.strip()
        or "|" in field
        or any(ord(character) < 32 for character in field)
    ):
        raise InvalidRunError(
            f"{location}: {key} must be a non-empty safe canonical string"
        )
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
    if (
        isinstance(field, bool)
        or not isinstance(field, int | float)
        or not math.isfinite(field)
    ):
        raise InvalidRunError(f"{path}:{line_number}: {key} must be finite numeric")
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
        "Prompt mode",
        "Run ID",
        "Rollouts",
        "Recovered rollouts",
        "Reward (mean ± std)",
    ]
    headers.extend(f"{name} (mean ± std)" for name in METRIC_NAMES)
    alignments = ["---", "---", "---", "---", "---:", "---:", "---:"]
    alignments.extend("---:" for _ in METRIC_NAMES)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(alignments) + " |",
    ]
    for summary in summaries:
        row = [
            summary.model,
            summary.environment,
            summary.prompt_mode or "—",
            summary.run_id,
            str(summary.rollout_count),
            str(summary.recovered_rollout_count),
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
        "prompt_mode",
        "run_id",
        "rollouts",
        "recovered_rollouts",
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
                "prompt_mode": summary.prompt_mode or "",
                "run_id": summary.run_id,
                "rollouts": summary.rollout_count,
                "recovered_rollouts": summary.recovered_rollout_count,
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
