from __future__ import annotations

import importlib.util
import sys
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
) -> aggregate.CompleteRun:
    return aggregate.CompleteRun(
        model=model,
        environment="commonground-predict",
        run_id=f"run-{model}",
        descriptor_timestamp_ns=1,
        recovered_rollout_count=recovered,
        task_ids=task_ids,
        rewards=rewards,
        metrics={},
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

    assert "Task-cluster bootstrap summary" in rendered
    assert "1,000 deterministic percentile bootstrap resamples of tasks" in rendered
    assert "repeated rollouts" in rendered
