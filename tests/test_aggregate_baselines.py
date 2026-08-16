from __future__ import annotations

import csv
import importlib.util
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
    assert all("model" not in config for config in configs)


def test_aggregate_complete_runs_in_deterministic_order() -> None:
    summaries = aggregate.load_summaries(FIXTURE_ROOT)

    assert [(summary.model, summary.environment) for summary in summaries] == [
        ("openai/gpt-4.1-mini", "commonground-elicit"),
        ("openai/gpt-4.1-mini", "commonground-predict"),
    ]
    assert summaries[0].rollout_count == 6
    assert summaries[0].reward_mean == 0.5
    assert summaries[0].reward_std == 0.408248290463863
    assert summaries[0].metrics["finding_f1"] == (0.5, 0.408248290463863)
    assert summaries[0].metrics["question_utility"] == (0.25, 0.2041241452319315)
    assert summaries[1].reward_mean == 0.5
    assert summaries[1].reward_std == 0.2041241452319315
    assert summaries[1].metrics["brier"] == (0.5, 0.408248290463863)

    assert aggregate.render_markdown(summaries) == "\n".join(
        [
            "| Model | Environment | Rollouts | Reward (mean ± std) | brier (mean ± std) | finding_f1 (mean ± std) | question_utility (mean ± std) |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            "| openai/gpt-4.1-mini | commonground-elicit | 6 | 0.500 ± 0.408 | — | 0.500 ± 0.408 | 0.250 ± 0.204 |",
            "| openai/gpt-4.1-mini | commonground-predict | 6 | 0.500 ± 0.204 | 0.500 ± 0.408 | — | — |",
        ]
    )


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
    assert output.startswith("| Model | Environment | Rollouts |")
    with csv_path.open(encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert [(row["model"], row["environment"]) for row in rows] == [
        ("openai/gpt-4.1-mini", "commonground-elicit"),
        ("openai/gpt-4.1-mini", "commonground-predict"),
    ]
    assert rows[0]["brier_mean"] == ""
    assert rows[1]["brier_mean"] == "0.5"
