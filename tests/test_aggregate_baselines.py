from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import shutil
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest
from verifiers.v1.configs.cli.eval import EvalConfig

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


def write_native_run(
    root: Path,
    *,
    environment: str,
    run_id: str,
    task_mode: str | None = None,
    model: str = "fixture/native-oracle",
    num_tasks: int = 2,
    num_rollouts: int = 3,
    reward_weight: float = 1.0,
) -> Path:
    output_name = environment.replace("/", "--").replace("@", "--")
    run_dir = (
        root
        / "outputs"
        / f"{output_name}--fixture--native-oracle--null"
        / f"directory-{run_id}"
    )
    run_dir.mkdir(parents=True)
    taskset_lines = [
        f'id = "{environment}"',
        'split = "eval"',
    ]
    if task_mode is not None:
        taskset_lines.extend(
            [
                f'task_mode = "{task_mode}"',
                "planted_density = 1.0",
                "distractor_density = 1.0",
                "panel_polarization = 1.0",
                "question_count = 2",
            ]
        )
    config = "\n".join(
        [
            f'model = "{model}"',
            f"num_tasks = {num_tasks}",
            f"num_rollouts = {num_rollouts}",
            "shuffle = false",
            "",
            "[env.taskset]",
            *taskset_lines,
            "",
            "[env.taskset.task]",
            "judges = []",
            "",
        ]
    )
    (run_dir / "config.toml").write_text(config, encoding="utf-8")

    is_predict = (
        aggregate.environment_package_name(environment) == "commonground-predict"
    )
    reward_name = (
        "vote_accuracy"
        if is_predict and environment.endswith("@0.3.0")
        else "probability_reward"
        if is_predict
        else "question_utility"
        if task_mode == "elicit-ask"
        else "finding_f1"
    )
    rows: list[dict[str, Any]] = []
    for task_index in range(num_tasks):
        for rollout_index in range(num_rollouts):
            score = 1.0 if rollout_index % 3 != 1 else 0.0
            metrics: dict[str, float] = {}
            if is_predict:
                metrics["brier"] = 1.0 - score
                if not environment.endswith("@0.3.0"):
                    metrics["vote_accuracy"] = score
            elif task_mode == "find":
                metrics["finding_localization_recall"] = score
                metrics["finding_type_accuracy"] = score
                metrics["question_utility"] = score / 2
            task_info = {"task_label": task_mode} if task_mode is not None else {}
            trace_id = f"trace-{task_index}-{rollout_index}"
            trace = {
                "version": 1,
                "id": trace_id,
                "verifiers": {"version": "0.3.0"},
                "run": {"type": "eval", "id": run_id},
                "task": {
                    "type": "PredictionTask" if is_predict else "ElicitTask",
                    "data": {"idx": task_index, "info": task_info},
                },
                "agent": {"config": {}, "trainable": True},
                "rewards": {reward_name: {"score": score, "weight": reward_weight}},
                "metrics": metrics,
                "is_completed": True,
                "ok": True,
                "errors": [],
            }
            rows.append(
                {
                    "id": f"episode-{task_index}-{rollout_index}",
                    "env": {"id": environment},
                    "ok": True,
                    "errors": [],
                    "traces": [trace],
                }
            )
    traces_path = run_dir / "traces.jsonl"
    write_native_rows(traces_path, rows)
    return traces_path


def read_native_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_native_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(f"{json.dumps(row, sort_keys=True)}\n" for row in rows),
        encoding="utf-8",
    )


def test_baseline_config_uses_installed_native_v1_schema() -> None:
    config_path = ROOT / "configs" / "eval" / "baseline-sweep.toml"

    raw_config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    config = EvalConfig.model_validate(raw_config)

    assert config.env.taskset.id == "commonground-predict"
    assert config.num_tasks == 100
    assert config.num_rollouts == 5
    assert config.shuffle is False
    assert config.push is False
    assert "model" not in raw_config


def test_native_v1_run_aggregates_weighted_rewards_and_named_signals(
    tmp_path: Path,
) -> None:
    write_native_run(
        tmp_path,
        environment="commonground-predict",
        run_id="native-run-id",
        reward_weight=0.5,
    )

    [summary] = aggregate.load_summaries(tmp_path)

    assert summary.environment == "commonground-predict"
    assert summary.run_id == "native-run-id"
    assert summary.rollout_count == 6
    assert summary.reward_mean == pytest.approx(1 / 3)
    assert summary.metrics["vote_accuracy"] == pytest.approx((2 / 3, math.sqrt(2) / 3))
    assert summary.metrics["brier"] == pytest.approx((1 / 3, math.sqrt(2) / 3))

    run = aggregate.load_complete_run(next(tmp_path.glob("outputs/*/*/traces.jsonl")))
    assert run.task_ids == (0, 0, 0, 1, 1, 1)


def test_native_predict_030_reward_contract_remains_readable(tmp_path: Path) -> None:
    write_native_run(
        tmp_path,
        environment="charliethompson/commonground-predict@0.3.0",
        run_id="historical-native-run",
    )

    [summary] = aggregate.load_summaries(tmp_path)

    assert summary.reward_mean == pytest.approx(2 / 3)
    assert summary.metrics["vote_accuracy"] == pytest.approx((2 / 3, math.sqrt(2) / 3))
    assert summary.metrics["brier"] == pytest.approx((1 / 3, math.sqrt(2) / 3))


def test_native_elicit_modes_never_collapse_to_one_environment_key(
    tmp_path: Path,
) -> None:
    write_native_run(
        tmp_path,
        environment="commonground-elicit",
        task_mode="find",
        run_id="native-find",
    )
    write_native_run(
        tmp_path,
        environment="commonground-elicit",
        task_mode="elicit-ask",
        run_id="native-ask",
    )

    summaries = aggregate.load_summaries(tmp_path)

    assert [(summary.environment, summary.run_id) for summary in summaries] == [
        ("commonground-elicit:elicit-ask", "native-ask"),
        ("commonground-elicit:find", "native-find"),
    ]
    assert summaries[0].metrics["question_utility"][0] == pytest.approx(2 / 3)
    assert summaries[1].metrics["finding_f1"][0] == pytest.approx(2 / 3)
    assert summaries[1].metrics["question_utility"][0] == pytest.approx(1 / 3)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("missing-rollout", r"expected 6 rollouts.*found 5"),
        ("failed-episode", r"episode must be ok"),
        ("null-reward", r"reward 'probability_reward' must be an object"),
        ("duplicate-episode", r"duplicate episode id"),
        ("wrong-distribution", r"expected 2 distinct examples, found 1"),
        ("mixed-run-ids", r"exactly one eval run id"),
        ("unexpected-metric", r"expected metrics.*brier"),
    ],
)
def test_native_v1_complete_run_validation_fails_closed(
    tmp_path: Path, case: str, message: str
) -> None:
    traces_path = write_native_run(
        tmp_path,
        environment="commonground-predict",
        run_id="native-invalid",
    )
    rows = read_native_rows(traces_path)
    if case == "missing-rollout":
        rows.pop()
    elif case == "failed-episode":
        rows[0]["ok"] = False
    elif case == "null-reward":
        rows[0]["traces"][0]["rewards"]["probability_reward"] = None
    elif case == "duplicate-episode":
        rows[1]["id"] = rows[0]["id"]
    elif case == "wrong-distribution":
        for row in rows:
            row["traces"][0]["task"]["data"]["idx"] = 0
    elif case == "mixed-run-ids":
        rows[0]["traces"][0]["run"]["id"] = "another-run"
    elif case == "unexpected-metric":
        rows[0]["traces"][0]["metrics"] = {}
    write_native_rows(traces_path, rows)

    with pytest.raises(aggregate.InvalidRunError, match=message):
        aggregate.load_summaries(tmp_path)


def test_native_elicit_trace_mode_must_match_config(tmp_path: Path) -> None:
    traces_path = write_native_run(
        tmp_path,
        environment="commonground-elicit",
        task_mode="find",
        run_id="mode-mismatch",
    )
    rows = read_native_rows(traces_path)
    rows[0]["traces"][0]["task"]["data"]["info"]["task_label"] = "elicit-ask"
    write_native_rows(traces_path, rows)

    with pytest.raises(
        aggregate.InvalidRunError, match=r"task mode does not match run config"
    ):
        aggregate.load_summaries(tmp_path)


def test_native_successful_retry_error_history_remains_aggregatable(
    tmp_path: Path,
) -> None:
    traces_path = write_native_run(
        tmp_path,
        environment="commonground-predict",
        run_id="successful-retry",
        num_tasks=1,
        num_rollouts=1,
    )
    rows = read_native_rows(traces_path)
    error = {"type": "ProviderError", "message": "transient retry"}
    rows[0]["errors"] = [error]
    rows[0]["traces"][0]["errors"] = [error]
    write_native_rows(traces_path, rows)

    [summary] = aggregate.load_summaries(tmp_path)
    assert summary.run_id == "successful-retry"
    assert summary.recovered_rollout_count == 1


def test_native_successful_model_call_retry_is_counted_as_recovered(
    tmp_path: Path,
) -> None:
    traces_path = write_native_run(
        tmp_path,
        environment="commonground-predict",
        run_id="successful-call-retry",
        num_tasks=1,
        num_rollouts=1,
    )
    rows = read_native_rows(traces_path)
    rows[0]["traces"][0]["calls"] = [
        {"error": {"type": "ProviderError", "message": "transient retry"}},
        {"model": "fixture/native-oracle", "finish_reason": "stop"},
    ]
    write_native_rows(traces_path, rows)

    [summary] = aggregate.load_summaries(tmp_path)
    assert summary.run_id == "successful-call-retry"
    assert summary.recovered_rollout_count == 1


def test_native_qualified_hub_id_preserves_provenance(tmp_path: Path) -> None:
    environment = "public-org/commonground-elicit@0.2.0"
    write_native_run(
        tmp_path,
        environment=environment,
        task_mode="find",
        run_id="qualified-hub-id",
        num_tasks=1,
        num_rollouts=1,
    )

    [summary] = aggregate.load_summaries(tmp_path)
    assert summary.environment == f"{environment}:find"


def test_native_config_only_directory_is_rejected(tmp_path: Path) -> None:
    traces_path = write_native_run(
        tmp_path,
        environment="commonground-predict",
        run_id="config-only",
        num_tasks=1,
        num_rollouts=1,
    )
    traces_path.unlink()

    with pytest.raises(aggregate.InvalidRunError, match=r"partially saved eval run"):
        aggregate.load_summaries(tmp_path)


def test_native_partial_and_mixed_format_directories_are_rejected(
    tmp_path: Path,
) -> None:
    traces_path = write_native_run(
        tmp_path,
        environment="commonground-predict",
        run_id="ambiguous",
    )
    run_dir = traces_path.parent
    (run_dir / "metadata.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(aggregate.InvalidRunError, match=r"partially saved eval run"):
        aggregate.load_summaries(tmp_path)

    (run_dir / "results.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(aggregate.InvalidRunError, match=r"both native and legacy"):
        aggregate.load_summaries(tmp_path)


@pytest.mark.parametrize(
    "invalid_fragment",
    [
        '"ok":NaN',
        '"ok":true,"ok":true',
    ],
)
def test_native_json_rejects_nonfinite_values_and_duplicate_keys(
    tmp_path: Path, invalid_fragment: str
) -> None:
    traces_path = write_native_run(
        tmp_path,
        environment="commonground-predict",
        run_id="invalid-json",
        num_tasks=1,
        num_rollouts=1,
    )
    traces_path.write_text(f"{{{invalid_fragment}}}\n", encoding="utf-8")

    with pytest.raises(aggregate.InvalidRunError, match=r"invalid JSON"):
        aggregate.load_summaries(tmp_path)


def test_aggregate_complete_runs_in_deterministic_order() -> None:
    summaries = aggregate.load_summaries(FIXTURE_ROOT)

    assert [(summary.model, summary.environment) for summary in summaries] == [
        ("fixture/offline-oracle", "commonground-elicit:find"),
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
            "| Model | Environment | Run ID | Rollouts | Recovered rollouts | Reward (mean ± std) | vote_accuracy (mean ± std) | brier (mean ± std) | finding_localization_recall (mean ± std) | finding_type_accuracy (mean ± std) | finding_f1 (mean ± std) | question_utility (mean ± std) |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            "| fixture/offline-oracle | commonground-elicit:find | d4c3b2a1 | 6 | 0 | 0.667 ± 0.471 | — | — | — | — | 0.667 ± 0.471 | 0.415 ± 0.294 |",
            "| fixture/offline-oracle | commonground-predict | a1b2c3d4 | 6 | 0 | 0.667 ± 0.471 | 0.667 ± 0.471 | 0.333 ± 0.471 | — | — | — | — |",
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
        ("fixture/offline-oracle", "commonground-elicit:find"),
        ("fixture/offline-oracle", "commonground-predict"),
    ]
    assert rows[0]["brier_mean"] == ""
    assert rows[0]["recovered_rollouts"] == "0"
    assert rows[0]["run_id"] == "d4c3b2a1"
    assert rows[1]["vote_accuracy_mean"] == "0.6666666666666666"
    assert rows[1]["brier_mean"] == "0.3333333333333333"
