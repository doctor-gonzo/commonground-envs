from __future__ import annotations

import csv
import importlib.util
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest
from verifiers.utils.eval_utils import load_toml_config

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "eval_outputs"


def load_script() -> Any:
    script_path = ROOT / "scripts" / "aggregate_baselines.py"
    spec = importlib.util.spec_from_file_location("aggregate_baselines", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


aggregate = load_script()


def test_baseline_sweep_uses_installed_multi_eval_toml_schema() -> None:
    config_path = ROOT / "configs" / "eval" / "baseline-sweep.toml"

    configs = load_toml_config(config_path)

    assert [(config["env_id"], config["num_examples"]) for config in configs] == [
        ("commonground-predict", 20),
        ("commonground-elicit", 20),
    ]
    assert {config["rollouts_per_example"] for config in configs} == {3}
    assert all(config["save_results"] is True for config in configs)
    assert all("model" not in config for config in configs)


def test_aggregate_complete_runs_in_deterministic_order() -> None:
    summaries = aggregate.load_summaries(FIXTURE_ROOT)

    assert [(summary.model, summary.environment) for summary in summaries] == [
        ("fixture/offline-oracle", "commonground-elicit"),
        ("fixture/offline-oracle", "commonground-predict"),
    ]
    assert summaries[0].rollout_count == 6
    assert summaries[0].run_id == "d4c3b2a1"
    assert summaries[0].reward_mean == 2 / 3
    assert summaries[0].reward_std == pytest.approx(math.sqrt(2) / 3)
    assert summaries[0].metrics["finding_f1"] == pytest.approx(
        (2 / 3, math.sqrt(2) / 3)
    )
    question_utility = 0.6230234154761808
    assert summaries[0].metrics["question_utility"] == pytest.approx(
        (
            2 * question_utility / 3,
            question_utility * math.sqrt(2) / 3,
        )
    )
    assert summaries[1].reward_mean == 2 / 3
    assert summaries[1].run_id == "a1b2c3d4"
    assert summaries[1].reward_std == pytest.approx(math.sqrt(2) / 3)
    assert summaries[1].metrics["vote_accuracy"] == pytest.approx(
        (2 / 3, math.sqrt(2) / 3)
    )
    assert summaries[1].metrics["brier"] == pytest.approx((1 / 3, math.sqrt(2) / 3))

    assert aggregate.render_markdown(summaries) == "\n".join(
        [
            "| Model | Environment | Run ID | Rollouts | Reward (mean ± std) | vote_accuracy (mean ± std) | brier (mean ± std) | finding_f1 (mean ± std) | question_utility (mean ± std) |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            "| fixture/offline-oracle | commonground-elicit | d4c3b2a1 | 6 | 0.667 ± 0.471 | — | — | 0.667 ± 0.471 | 0.415 ± 0.294 |",
            "| fixture/offline-oracle | commonground-predict | a1b2c3d4 | 6 | 0.667 ± 0.471 | 0.667 ± 0.471 | 0.333 ± 0.471 | — | — |",
        ]
    )


def test_newest_metadata_timestamp_wins_unless_all_runs_requested(
    tmp_path: Path,
) -> None:
    fixture_copy = tmp_path / "fixture"
    shutil.copytree(FIXTURE_ROOT, fixture_copy)
    older_result = next(
        fixture_copy.glob("**/commonground-predict--*/**/results.jsonl")
    )
    newer_run = older_result.parent.parent / "e5f6a7b8"
    shutil.copytree(older_result.parent, newer_run)
    os.utime(older_result.with_name("metadata.json"), ns=(1_000_000_000, 1_000_000_000))
    os.utime(newer_run / "metadata.json", ns=(2_000_000_000, 2_000_000_000))

    newest = aggregate.load_summaries(fixture_copy)
    predict = next(
        summary for summary in newest if summary.environment == "commonground-predict"
    )
    assert predict.run_id == "e5f6a7b8"

    all_runs = aggregate.load_summaries(fixture_copy, all_runs=True)
    assert [
        summary.run_id
        for summary in all_runs
        if summary.environment == "commonground-predict"
    ] == ["a1b2c3d4", "e5f6a7b8"]


def test_log_only_real_run_layout_reports_save_results_fix(tmp_path: Path) -> None:
    run_dir = (
        tmp_path
        / "environments"
        / "commonground_predict"
        / "outputs"
        / "evals"
        / "commonground-predict--openai--gpt-4.1-mini"
        / "a1b2c3d4"
    )
    run_dir.mkdir(parents=True)
    (run_dir / "env_server.log").write_text("saved log only\n", encoding="utf-8")

    with pytest.raises(
        aggregate.InvalidRunError,
        match=r"found 1 log-only run dir.*--save-results",
    ):
        aggregate.load_summaries(tmp_path)


def test_incomplete_run_is_rejected_without_partial_summary(tmp_path: Path) -> None:
    fixture_copy = tmp_path / "fixture"
    shutil.copytree(FIXTURE_ROOT, fixture_copy)
    result_path = next(fixture_copy.glob("**/commonground-predict--*/**/results.jsonl"))
    lines = result_path.read_text(encoding="utf-8").splitlines()
    result_path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")

    with pytest.raises(
        aggregate.InvalidRunError, match=r"expected 6 rollouts.*found 5"
    ):
        aggregate.load_summaries(fixture_copy)


def test_cli_prints_markdown_and_optionally_writes_csv(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    csv_path = tmp_path / "baselines.csv"

    assert aggregate.main(["--root", str(FIXTURE_ROOT), "--csv", str(csv_path)]) == 0

    output = capsys.readouterr().out
    assert output.startswith("| Model | Environment | Run ID | Rollouts |")
    with csv_path.open(encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert [(row["model"], row["environment"]) for row in rows] == [
        ("fixture/offline-oracle", "commonground-elicit"),
        ("fixture/offline-oracle", "commonground-predict"),
    ]
    assert rows[0]["brier_mean"] == ""
    assert rows[0]["run_id"] == "d4c3b2a1"
    assert rows[1]["vote_accuracy_mean"] == "0.6666666666666666"
    assert rows[1]["brier_mean"] == "0.3333333333333333"
