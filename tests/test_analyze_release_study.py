from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import aggregate_baselines as aggregate  # noqa: E402


def load_script() -> Any:
    script_path = SCRIPTS / "analyze_release_study.py"
    spec = importlib.util.spec_from_file_location("analyze_release_study", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


analysis_module = load_script()


def complete_run(
    *,
    model: str,
    rewards: tuple[float, ...],
    task_ids: tuple[int, ...] = (0, 0, 1, 1),
    recovered: int = 0,
    environment: str = "commonground-predict",
    metrics: Mapping[str, tuple[float, ...]] | None = None,
    prompt_mode: str | None = None,
    comparison_signature: str | None = None,
    raw_probability_rewards: tuple[float, ...] | None = None,
    probability_reward_weights: tuple[float, ...] | None = None,
    include_ablation_provenance: bool = True,
) -> aggregate.CompleteRun:
    task_answer_digests: tuple[tuple[int, str], ...] = ()
    raw_reward_scores: Mapping[str, tuple[float, ...]] = {}
    raw_reward_weights: Mapping[str, tuple[float, ...]] = {}
    if prompt_mode is not None and include_ablation_provenance:
        comparison_signature = (
            comparison_signature
            or hashlib.sha256(
                f"fixture-ablation\0{environment}\0{model}".encode()
            ).hexdigest()
        )
        task_answer_digests = tuple(
            (
                task_id,
                hashlib.sha256(
                    f"fixture-answer\0{environment}\0{task_id}".encode()
                ).hexdigest(),
            )
            for task_id in sorted(set(task_ids))
        )
        raw_reward_scores = {
            "probability_reward": (
                rewards if raw_probability_rewards is None else raw_probability_rewards
            )
        }
        raw_reward_weights = {
            "probability_reward": (
                (1.0,) * len(rewards)
                if probability_reward_weights is None
                else probability_reward_weights
            )
        }
    return aggregate.CompleteRun(
        model=model,
        environment=environment,
        run_id=f"run-{model}",
        descriptor_timestamp_ns=1,
        recovered_rollout_count=recovered,
        task_ids=task_ids,
        rewards=rewards,
        metrics={} if metrics is None else metrics,
        prompt_mode=prompt_mode,
        comparison_signature=comparison_signature,
        task_answer_digests=task_answer_digests,
        raw_reward_scores=raw_reward_scores,
        raw_reward_weights=raw_reward_weights,
    )


def test_task_cluster_analysis_is_deterministic_and_paired() -> None:
    runs = [
        complete_run(model="model/a", rewards=(1.0, 1.0, 0.0, 0.0)),
        complete_run(model="model/b", rewards=(0.5, 0.5, 0.5, 0.5)),
    ]

    first = analysis_module.analyze_runs(
        runs,
        bootstrap_samples=2_000,
        seed=7,
        expected_model_count=2,
        expected_task_count=2,
        expected_rollouts_per_task=2,
        require_no_recoveries=True,
    )
    second = analysis_module.analyze_runs(
        runs,
        bootstrap_samples=2_000,
        seed=7,
        expected_model_count=2,
        expected_task_count=2,
        expected_rollouts_per_task=2,
        require_no_recoveries=True,
    )

    assert first == second
    assert [item.reward_mean for item in first.summaries] == [0.5, 0.5]
    assert first.summaries[0].zero_reward_tasks == 1
    [difference] = first.pairwise
    assert difference.mean_difference == pytest.approx(0.0)
    assert difference.ci_low < 0.0 < difference.ci_high
    assert difference.interpretation == "interval includes 0"


def test_analysis_rejects_unpaired_tasks_and_recovered_rollouts() -> None:
    unpaired = [
        complete_run(model="model/a", rewards=(1.0, 1.0, 0.0, 0.0)),
        complete_run(
            model="model/b",
            rewards=(1.0, 1.0, 0.0, 0.0),
            task_ids=(0, 0, 2, 2),
        ),
    ]
    with pytest.raises(aggregate.InvalidRunError, match="same task identifiers"):
        analysis_module.analyze_runs(
            unpaired,
            bootstrap_samples=1_000,
            seed=1,
        )

    with pytest.raises(aggregate.InvalidRunError, match="recovered rollouts"):
        analysis_module.analyze_runs(
            [complete_run(model="model/a", rewards=(1.0, 1.0, 0.0, 0.0), recovered=1)],
            bootstrap_samples=1_000,
            seed=1,
            require_no_recoveries=True,
        )


def test_markdown_states_task_level_method() -> None:
    study = analysis_module.analyze_runs(
        [complete_run(model="model/a", rewards=(1.0, 1.0, 0.0, 0.0))],
        bootstrap_samples=1_000,
        seed=3,
    )

    rendered = analysis_module.render_markdown(study)

    assert "Clustered bootstrap summary" in rendered
    assert "Paired clustered model differences" in rendered
    assert "1,000 deterministic percentile bootstrap resamples" in rendered
    assert "repeated rollouts" in rendered
    summary_header, summary_divider, summary_row = rendered.splitlines()[2:5]
    assert summary_header.count("|") == summary_divider.count("|")
    assert summary_row.count("|") == summary_header.count("|")


def test_predict_prompt_ablations_are_preserved_and_compared_with_full(
    tmp_path: Path,
) -> None:
    task_ids = (0, 0, 1, 1)
    mode_values = {
        "full": (0.9, 0.9, 0.7, 0.7),
        "matrix-only": (0.8, 0.8, 0.6, 0.6),
        "text-only": (0.6, 0.6, 0.4, 0.4),
        "shuffled-text": (0.7, 0.7, 0.5, 0.5),
    }
    runs = [
        complete_run(
            model="model/a",
            rewards=rewards,
            task_ids=task_ids,
            prompt_mode=prompt_mode,
            metrics={
                "brier": tuple(1.0 - reward for reward in rewards),
                "original_snapshot_visible_prior_brier": (0.1, 0.1, 0.4, 0.4),
                "vote_accuracy": tuple(float(reward >= 0.7) for reward in rewards),
            },
        )
        for prompt_mode, rewards in mode_values.items()
    ]

    study = analysis_module.analyze_runs(
        runs,
        bootstrap_samples=2_000,
        seed=41,
        expected_model_count=1,
        expected_task_count=2,
        expected_rollouts_per_task=2,
    )

    assert {item.prompt_mode for item in study.summaries} == set(mode_values)
    assert {item.prompt_mode for item in study.diagnostics} == set(mode_values)
    assert len(study.prompt_ablations) == 15
    assert all(item.exploratory for item in study.prompt_ablations)
    comparisons = {
        (item.prompt_mode, item.metric): item for item in study.prompt_ablations
    }
    reward = comparisons[("matrix-only", "probability_reward")]
    assert reward.reference_mode == "full"
    assert reward.reference_mean == pytest.approx(0.8)
    assert reward.prompt_mean == pytest.approx(0.7)
    assert reward.mean_difference == pytest.approx(0.1)
    assert reward.ci_low == pytest.approx(0.1)
    assert reward.ci_high == pytest.approx(0.1)
    brier = comparisons[("matrix-only", "brier")]
    assert brier.mean_difference == pytest.approx(-0.1)
    brier_skill = comparisons[("matrix-only", "brier_skill_vs_uniform")]
    assert brier_skill.mean_difference == pytest.approx(0.3)
    snapshot_skill = comparisons[
        ("matrix-only", "brier_skill_vs_original_snapshot_visible_prior")
    ]
    assert snapshot_skill.mean_difference == pytest.approx(0.4)

    rendered = analysis_module.render_markdown(study)
    assert "Predict paired prompt ablations (exploratory; unadjusted)" in rendered
    assert "full - mode" in rendered

    destination = tmp_path / "prompt-ablation-analysis.json"
    analysis_module.write_json(destination, study)
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert len(payload["prompt_ablations"]) == 15
    assert {item["prompt_mode"] for item in payload["summaries"]} == set(mode_values)
    assert payload["method"]["prompt_ablation_intervals"] == (
        "paired task percentile bootstrap of full minus prompt mode"
    )
    assert payload["method"]["brier_skill_references"] == {
        "brier_skill_vs_uniform": "uniform three-class normalized Brier = 1/3",
        "brier_skill_vs_original_snapshot_visible_prior": (
            "equally weighted pooled model loss relative to evaluator-side "
            "original full-snapshot visible-matrix class-frequency loss; fixed "
            "across modes and unavailable to text-only agents; undefined when "
            "pooled reference loss is zero"
        ),
    }
    assert payload["method"]["prompt_ablation_provenance"].startswith(
        "matching native comparison signature"
    )
    assert {item["comparison_signature"] for item in payload["prompt_ablations"]} == {
        runs[0].comparison_signature
    }


def test_predict_prompt_ablation_requires_complete_modes_and_paired_tasks() -> None:
    partial = [
        complete_run(
            model="model/a",
            rewards=(1.0, 1.0, 0.0, 0.0),
            prompt_mode=prompt_mode,
            metrics={
                "brier": (0.0, 0.0, 1.0, 1.0),
                "original_snapshot_visible_prior_brier": (0.25, 0.25, 0.25, 0.25),
                "vote_accuracy": (1.0, 1.0, 0.0, 0.0),
            },
        )
        for prompt_mode in ("full", "matrix-only")
    ]
    with pytest.raises(aggregate.InvalidRunError, match="complete prompt-mode set"):
        analysis_module.analyze_runs(partial, bootstrap_samples=1_000, seed=43)

    complete = [
        complete_run(
            model="model/a",
            rewards=(1.0, 1.0, 0.0, 0.0),
            prompt_mode=prompt_mode,
            task_ids=(0, 0, 2, 2) if prompt_mode == "text-only" else (0, 0, 1, 1),
            metrics={
                "brier": (0.0, 0.0, 1.0, 1.0),
                "original_snapshot_visible_prior_brier": (0.25, 0.25, 0.25, 0.25),
                "vote_accuracy": (1.0, 1.0, 0.0, 0.0),
            },
        )
        for prompt_mode in ("full", "matrix-only", "text-only", "shuffled-text")
    ]
    with pytest.raises(aggregate.InvalidRunError, match="same task identifiers"):
        analysis_module.analyze_runs(complete, bootstrap_samples=1_000, seed=43)


def test_predict_prompt_ablation_fails_closed_without_comparable_provenance() -> None:
    modes = ("full", "matrix-only", "text-only", "shuffled-text")
    missing = [
        complete_run(
            model="model/a",
            rewards=(0.8, 0.8, 0.6, 0.6),
            prompt_mode=prompt_mode,
            metrics={
                "brier": (0.2, 0.2, 0.4, 0.4),
                "original_snapshot_visible_prior_brier": (0.25, 0.25, 0.25, 0.25),
                "vote_accuracy": (1.0, 1.0, 0.0, 0.0),
            },
            include_ablation_provenance=prompt_mode != "text-only",
        )
        for prompt_mode in modes
    ]
    with pytest.raises(
        aggregate.InvalidRunError, match="missing a stable comparison signature"
    ):
        analysis_module.analyze_runs(missing, bootstrap_samples=1_000, seed=47)

    mismatched = [
        complete_run(
            model="model/a",
            rewards=(0.8, 0.8, 0.6, 0.6),
            prompt_mode=prompt_mode,
            comparison_signature=(
                "b" * 64 if prompt_mode == "matrix-only" else "a" * 64
            ),
            metrics={
                "brier": (0.2, 0.2, 0.4, 0.4),
                "original_snapshot_visible_prior_brier": (0.25, 0.25, 0.25, 0.25),
                "vote_accuracy": (1.0, 1.0, 0.0, 0.0),
            },
        )
        for prompt_mode in modes
    ]
    with pytest.raises(
        aggregate.InvalidRunError,
        match="differ in model/client/sampling/concurrency",
    ):
        analysis_module.analyze_runs(mismatched, bootstrap_samples=1_000, seed=47)


def test_predict_prompt_ablation_rejects_weighted_reward_mislabeling() -> None:
    runs = [
        complete_run(
            model="model/a",
            rewards=(0.8, 0.8, 0.6, 0.6),
            prompt_mode=prompt_mode,
            probability_reward_weights=(
                (0.5,) * 4 if prompt_mode == "matrix-only" else (1.0,) * 4
            ),
            metrics={
                "brier": (0.2, 0.2, 0.4, 0.4),
                "original_snapshot_visible_prior_brier": (0.25, 0.25, 0.25, 0.25),
                "vote_accuracy": (1.0, 1.0, 0.0, 0.0),
            },
        )
        for prompt_mode in ("full", "matrix-only", "text-only", "shuffled-text")
    ]

    with pytest.raises(
        aggregate.InvalidRunError, match=r"probability_reward weight 1\.0"
    ):
        analysis_module.analyze_runs(runs, bootstrap_samples=1_000, seed=53)

    inconsistent_totals = [
        complete_run(
            model="model/a",
            rewards=(0.8, 0.8, 0.6, 0.6),
            prompt_mode=prompt_mode,
            raw_probability_rewards=(
                (0.4, 0.4, 0.3, 0.3)
                if prompt_mode == "matrix-only"
                else (0.8, 0.8, 0.6, 0.6)
            ),
            metrics={
                "brier": (0.2, 0.2, 0.4, 0.4),
                "original_snapshot_visible_prior_brier": (0.25, 0.25, 0.25, 0.25),
                "vote_accuracy": (1.0, 1.0, 0.0, 0.0),
            },
        )
        for prompt_mode in ("full", "matrix-only", "text-only", "shuffled-text")
    ]
    with pytest.raises(
        aggregate.InvalidRunError,
        match="weighted reward total does not match the raw probability_reward",
    ):
        analysis_module.analyze_runs(
            inconsistent_totals, bootstrap_samples=1_000, seed=53
        )


def test_diagnostic_metrics_receive_clustered_task_mean_intervals() -> None:
    run = complete_run(
        model="model/a",
        rewards=(1.0, 1.0, 0.0, 0.0),
        metrics={
            "vote_accuracy": (1.0, 0.0, 0.0, 0.0),
            "brier": (0.0, 0.5, 1.0, 1.0),
            "original_snapshot_visible_prior_brier": (0.1, 0.1, 0.5, 0.5),
        },
    )

    first = analysis_module.analyze_runs(
        [run],
        bootstrap_samples=2_000,
        seed=19,
    )
    second = analysis_module.analyze_runs(
        [run],
        bootstrap_samples=2_000,
        seed=19,
    )

    assert first == second
    assert [item.metric for item in first.diagnostics] == [
        "brier",
        "brier_skill_vs_uniform",
        "original_snapshot_visible_prior_brier",
        "vote_accuracy",
        "brier_skill_vs_original_snapshot_visible_prior",
    ]
    by_metric = {item.metric: item for item in first.diagnostics}
    brier_skill = by_metric["brier_skill_vs_uniform"]
    assert brier_skill.rollout_mean == pytest.approx(-0.875)
    assert brier_skill.task_mean == pytest.approx(-0.875)
    assert brier_skill.task_mean_std == pytest.approx(1.125)
    snapshot_skill = by_metric["brier_skill_vs_original_snapshot_visible_prior"]
    assert snapshot_skill.rollout_mean == pytest.approx(1.0 - 0.625 / 0.3)
    assert snapshot_skill.task_mean == pytest.approx(1.0 - 0.625 / 0.3)
    assert snapshot_skill.task_mean_std == pytest.approx(1.25)
    # The two possible single-task resamples have different reference losses.
    # These bounds therefore prove that each bootstrap draw recomputes the
    # pooled ratio instead of averaging fixed-denominator task contributions.
    assert snapshot_skill.ci_low == pytest.approx(-1.5)
    assert snapshot_skill.ci_high == pytest.approx(-1.0)
    vote_accuracy = by_metric["vote_accuracy"]
    assert vote_accuracy.rollout_mean == pytest.approx(0.25)
    assert vote_accuracy.task_mean == pytest.approx(0.25)
    assert vote_accuracy.task_mean_std == pytest.approx(0.25)
    assert vote_accuracy.ci_low == pytest.approx(0.0)
    assert vote_accuracy.ci_high == pytest.approx(0.5)
    assert vote_accuracy.cluster_count == 2
    assert vote_accuracy.resampling_unit == "task"

    rendered = analysis_module.render_markdown(first)
    assert "Clustered diagnostic metrics" in rendered
    assert "vote_accuracy" in rendered
    assert "95% cluster CI" in rendered


def test_diagnostic_metric_alignment_is_required() -> None:
    run = complete_run(
        model="model/a",
        rewards=(1.0, 1.0, 0.0, 0.0),
        metrics={"vote_accuracy": (1.0, 0.0, 0.0)},
    )

    with pytest.raises(
        aggregate.InvalidRunError,
        match="vote_accuracy values and task identifiers are not aligned",
    ):
        analysis_module.analyze_runs(
            [run],
            bootstrap_samples=1_000,
            seed=5,
        )


def test_diagnostics_do_not_change_existing_reward_evidence() -> None:
    bare_runs = [
        complete_run(model="model/a", rewards=(1.0, 1.0, 0.0, 0.0)),
        complete_run(model="model/b", rewards=(0.5, 0.5, 0.5, 0.5)),
    ]
    instrumented_runs = [
        complete_run(
            model=run.model,
            rewards=run.rewards,
            metrics={"vote_accuracy": run.rewards},
        )
        for run in bare_runs
    ]

    bare = analysis_module.analyze_runs(
        bare_runs,
        bootstrap_samples=2_000,
        seed=17,
    )
    instrumented = analysis_module.analyze_runs(
        instrumented_runs,
        bootstrap_samples=2_000,
        seed=17,
    )

    assert instrumented.summaries == bare.summaries
    assert instrumented.pairwise == bare.pairwise


def test_holm_adjustment_is_deterministic_and_monotone() -> None:
    adjusted = analysis_module.holm_adjusted_p_values((0.01, 0.03, 0.04))

    assert adjusted == pytest.approx((0.03, 0.06, 0.06))


def test_existing_estimate_constructors_remain_backward_compatible() -> None:
    pairwise = analysis_module.PairwiseEstimate(
        "environment",
        "model/a",
        "model/b",
        0.1,
        -0.1,
        0.2,
        "interval includes 0",
    )
    study = analysis_module.StudyAnalysis(1_000, 0.95, 7, (), (pairwise,))

    assert study.pairwise == (pairwise,)
    assert study.diagnostics == ()
    assert pairwise.raw_p_value == 1.0
    assert pairwise.holm_adjusted_p_value == 1.0
    assert pairwise.exploratory is True


def test_pairwise_results_are_explicitly_exploratory_and_holm_adjusted() -> None:
    task_ids = tuple(range(8))
    runs = [
        complete_run(
            model="model/a",
            rewards=(1.0,) * 8,
            task_ids=task_ids,
        ),
        complete_run(
            model="model/b",
            rewards=(0.5,) * 8,
            task_ids=task_ids,
        ),
        complete_run(
            model="model/c",
            rewards=(0.0,) * 8,
            task_ids=task_ids,
        ),
    ]

    study = analysis_module.analyze_runs(
        runs,
        bootstrap_samples=4_000,
        seed=23,
        expected_model_count=3,
        expected_task_count=8,
        expected_rollouts_per_task=1,
        expected_pairwise_family_size=3,
    )

    assert len(study.pairwise) == 3
    assert all(item.exploratory for item in study.pairwise)
    assert all(item.multiplicity_family_size == 3 for item in study.pairwise)
    assert all(
        0.0 < item.raw_p_value <= item.holm_adjusted_p_value <= 1.0
        for item in study.pairwise
    )

    rendered = analysis_module.render_markdown(study)
    assert "Paired clustered model differences (exploratory)" in rendered
    assert "Raw p" in rendered
    assert "Holm-adjusted p" in rendered
    assert "exploratory rather than confirmatory" in rendered

    with pytest.raises(
        aggregate.InvalidRunError,
        match="does not match the predeclared size",
    ):
        analysis_module.analyze_runs(
            runs,
            bootstrap_samples=1_000,
            seed=23,
            expected_pairwise_family_size=18,
        )


def test_pairwise_sign_flip_draws_one_sign_per_multi_task_cluster() -> None:
    task_ids = tuple(range(8))
    runs = [
        complete_run(model="model/a", rewards=(1.0,) * 8, task_ids=task_ids),
        complete_run(model="model/b", rewards=(0.0,) * 8, task_ids=task_ids),
    ]
    clusters = {task_id: f"cluster-{task_id // 2}" for task_id in task_ids}

    clustered = analysis_module.analyze_runs(
        runs,
        bootstrap_samples=8_000,
        seed=59,
        task_cluster_labels={"commonground-predict": clusters},
    )
    taskwise = analysis_module.analyze_runs(
        runs,
        bootstrap_samples=8_000,
        seed=59,
    )

    [clustered_pair] = clustered.pairwise
    [taskwise_pair] = taskwise.pairwise
    assert clustered.summaries[0].cluster_count == 4
    assert clustered_pair.raw_p_value > taskwise_pair.raw_p_value + 0.08
    assert clustered_pair.raw_p_value == pytest.approx(0.125, abs=0.02)
    assert taskwise_pair.raw_p_value == pytest.approx(2 / 256, abs=0.006)


def test_holm_family_spans_all_environments_in_one_analysis() -> None:
    runs = [
        complete_run(
            model=model,
            rewards=rewards,
            environment=environment,
        )
        for environment in ("environment/one", "environment/two")
        for model, rewards in (
            ("model/a", (1.0, 1.0, 0.5, 0.5)),
            ("model/b", (0.5, 0.5, 0.0, 0.0)),
        )
    ]

    study = analysis_module.analyze_runs(
        runs,
        bootstrap_samples=1_000,
        seed=31,
        expected_model_count=2,
    )

    assert len(study.pairwise) == 2
    assert all(item.multiplicity_family_size == 2 for item in study.pairwise)


def test_json_reports_diagnostics_and_multiplicity_method(tmp_path: Path) -> None:
    run = complete_run(
        model="model/a",
        rewards=(1.0, 1.0, 0.0, 0.0),
        metrics={"vote_accuracy": (1.0, 0.0, 0.0, 0.0)},
    )
    study = analysis_module.analyze_runs(
        [run],
        bootstrap_samples=1_000,
        seed=29,
    )
    destination = tmp_path / "analysis.json"

    analysis_module.write_json(destination, study)

    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["method"]["diagnostic_intervals"] == (
        "paired hierarchical cluster percentile bootstrap"
    )
    assert payload["method"]["pairwise_multiplicity"] == (
        "global Holm adjustment of exploratory paired cluster sign-flip p-values"
    )
    assert payload["method"]["pairwise_p_values"] == (
        "paired cluster sign-flip Monte Carlo randomization under a sign-"
        "symmetric/exchangeable paired cluster-effect null; equality of means "
        "alone is insufficient"
    )
    assert payload["method"]["pairwise_family_definition"].startswith(
        "all primary pairwise model contrasts"
    )
    assert payload["method"]["interval_scope"].startswith(
        "pointwise percentile intervals"
    )
    assert payload["diagnostics"][0]["metric"] == "vote_accuracy"


def test_elicit_hierarchical_bootstrap_resamples_templates_then_variants() -> None:
    task_ids = tuple(task_id for task_id in range(20) for _ in range(2))
    rewards = tuple(
        reward
        for task_id in range(20)
        for reward in ((1.0, 1.0) if task_id < 10 else (0.0, 0.0))
    )
    run = complete_run(
        model="model/a",
        rewards=rewards,
        task_ids=task_ids,
        environment="commonground-elicit:find",
        metrics={"finding_localization_recall": rewards},
    )
    task_level = analysis_module.analyze_runs([run], bootstrap_samples=2_000, seed=11)
    hierarchical = analysis_module.analyze_runs(
        [run],
        bootstrap_samples=2_000,
        seed=11,
        task_cluster_labels={
            "commonground-elicit:find": {
                task_id: "template-a" if task_id < 10 else "template-b"
                for task_id in range(20)
            }
        },
    )

    [task_summary] = task_level.summaries
    [hierarchical_summary] = hierarchical.summaries
    assert hierarchical_summary.cluster_count == 2
    assert hierarchical_summary.resampling_unit == "template then variant"
    assert hierarchical_summary.ci_low < task_summary.ci_low
    assert hierarchical_summary.ci_high > task_summary.ci_high
    [task_diagnostic] = task_level.diagnostics
    [hierarchical_diagnostic] = hierarchical.diagnostics
    assert hierarchical_diagnostic.cluster_count == 2
    assert hierarchical_diagnostic.resampling_unit == "template then variant"
    assert hierarchical_diagnostic.ci_low < task_diagnostic.ci_low
    assert hierarchical_diagnostic.ci_high > task_diagnostic.ci_high


def test_elicit_subset_accepts_split_wide_cluster_map_but_requires_task_labels() -> (
    None
):
    task_ids = (0, 0, 1, 1)
    run = complete_run(
        model="model/a",
        rewards=(1.0, 1.0, 0.0, 0.0),
        task_ids=task_ids,
        environment="commonground-elicit:find",
    )

    study = analysis_module.analyze_runs(
        [run],
        bootstrap_samples=1_000,
        seed=17,
        task_cluster_labels={
            "commonground-elicit:find": {
                0: "template-a",
                1: "template-a",
                2: "unused-template-b",
            }
        },
    )

    [summary] = study.summaries
    assert summary.cluster_count == 1
    assert summary.resampling_unit == "template then variant"

    with pytest.raises(aggregate.InvalidRunError, match=r"missing=\[1\]"):
        analysis_module.analyze_runs(
            [run],
            bootstrap_samples=1_000,
            seed=17,
            task_cluster_labels={
                "commonground-elicit:find": {
                    0: "template-a",
                    2: "unused-template-b",
                }
            },
        )


def test_template_cluster_loader_uses_row_order(tmp_path: Path) -> None:
    split = tmp_path / "eval.jsonl"
    split.write_text(
        "\n".join(
            [
                '{"provenance":{"template_id":"template-a"}}',
                '{"provenance":{"template_id":"template-a"}}',
                '{"provenance":{"template_id":"template-b"}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert analysis_module.load_template_clusters(split) == {
        0: "template-a",
        1: "template-a",
        2: "template-b",
    }
