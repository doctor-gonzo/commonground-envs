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
    prompt_mode: str | None = None,
    model: str = "fixture/native-oracle",
    num_tasks: int = 2,
    num_rollouts: int = 3,
    reward_weight: float = 1.0,
    client_base_url: str = "https://fixture.invalid/v1",
    call_endpoint: str = "/chat/completions",
    sampling_temperature: float = 0.2,
    max_concurrent: int = 8,
    include_ablation_provenance: bool = True,
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
        question_count = (
            1 if aggregate.uses_grounded_stance_diagnostics(environment) else 2
        )
        taskset_lines.extend(
            [
                f'task_mode = "{task_mode}"',
                "planted_density = 1.0",
                "distractor_density = 1.0",
                "panel_polarization = 1.0",
                f"question_count = {question_count}",
            ]
        )
    if prompt_mode is not None:
        taskset_lines.append(f'prompt_mode = "{prompt_mode}"')
    config_lines = [
        f'model = "{model}"',
        f"num_tasks = {num_tasks}",
        f"num_rollouts = {num_rollouts}",
        "shuffle = false",
    ]
    if include_ablation_provenance:
        config_lines.append(f"max_concurrent = {max_concurrent}")
    config_lines.extend(
        [
            "",
            "[env.taskset]",
            *taskset_lines,
            "",
            "[env.taskset.task]",
            "judges = []",
            "",
        ]
    )
    client_config = {
        "base_url": client_base_url,
        "api_key_var": "FIXTURE_API_KEY",
        "type": "eval",
    }
    sampling_config = {
        "temperature": sampling_temperature,
        "max_tokens": 256,
    }
    if include_ablation_provenance:
        config_lines.extend(
            [
                "[client]",
                f'base_url = "{client_base_url}"',
                'api_key_var = "FIXTURE_API_KEY"',
                'type = "eval"',
                "",
                "[sampling]",
                f"temperature = {sampling_temperature}",
                "max_tokens = 256",
                "",
            ]
        )
    config = "\n".join(config_lines)
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
                if aggregate.uses_snapshot_prior_skill(environment):
                    metrics["original_snapshot_visible_prior_brier"] = (
                        0.25 + 0.25 * task_index
                    )
            elif task_mode == "find":
                metrics["finding_localization_recall"] = score
                metrics["finding_type_accuracy"] = score
                metrics["question_utility"] = score / 2
                if aggregate.uses_structured_elicit_diagnostics(environment):
                    metrics["finding_diagnosis_recall"] = score
                    metrics["finding_relation_recall"] = score
            elif (
                task_mode == "elicit-ask"
                and aggregate.uses_structured_elicit_diagnostics(environment)
            ):
                metrics["question_format_valid"] = score
                metrics["question_grounding_recall"] = score
                if aggregate.uses_grounded_stance_diagnostics(environment):
                    metrics["question_grounded_stance_recall"] = score
                    metrics["question_evidence_match_recall"] = score
                    metrics["question_evidence_matched_stance_accuracy"] = score
                    metrics["question_top1_selection_accuracy"] = score
                else:
                    metrics["question_stance_accuracy"] = score
            task_info = {"task_label": task_mode} if task_mode is not None else {}
            if prompt_mode is not None:
                task_info["prompt_mode"] = prompt_mode
            agent_config = (
                {
                    "model": model,
                    "client": client_config,
                    "sampling": sampling_config,
                }
                if include_ablation_provenance
                else {}
            )
            trace_id = f"trace-{task_index}-{rollout_index}"
            trace = {
                "version": 1,
                "id": trace_id,
                "verifiers": {"version": "0.3.0"},
                "run": {"type": "eval", "id": run_id},
                "task": {
                    "type": "PredictionTask" if is_predict else "ElicitTask",
                    "data": {
                        "idx": task_index,
                        "answer": {f"{task_index},0": 1},
                        "info": task_info,
                    },
                },
                "agent": {"config": agent_config, "trainable": True},
                "calls": [
                    {
                        "model": model,
                        "sampling": sampling_config,
                        "endpoint": call_endpoint,
                        "finish_reason": "stop",
                    }
                ],
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
    assert summary.metrics["original_snapshot_visible_prior_brier"] == pytest.approx(
        (0.375, 0.125)
    )
    assert summary.metrics[
        "brier_skill_vs_original_snapshot_visible_prior"
    ] == pytest.approx((1 / 9, (math.sqrt(2) / 3) / 0.375))

    run = aggregate.load_complete_run(next(tmp_path.glob("outputs/*/*/traces.jsonl")))
    assert run.task_ids == (0, 0, 0, 1, 1, 1)
    assert run.raw_reward_scores["probability_reward"] == (1.0, 0.0, 1.0) * 2
    assert run.raw_reward_weights["probability_reward"] == (0.5,) * 6
    assert run.comparison_signature is None


def test_nested_vf_eval_saved_results_include_current_find_diagnostics(
    tmp_path: Path,
) -> None:
    run_dir = (
        tmp_path
        / "outputs"
        / "find--gpt41"
        / "run"
        / "evals"
        / "commonground-elicit--openai--gpt-4.1"
        / "c51ffbc1"
    )
    run_dir.mkdir(parents=True)
    metadata = {
        "env_id": "commonground-elicit",
        "env_args": {"task": "find"},
        "model": "openai/gpt-4.1",
        "num_examples": 1,
        "rollouts_per_example": 2,
        "version_info": {
            "vf_version": "0.3.0",
            "env_version": "0.6.0",
        },
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata) + "\n", encoding="utf-8"
    )
    rows = []
    for score in (0.25, 0.75):
        rows.append(
            {
                "example_id": 0,
                "reward": score,
                "metrics": {
                    "finding_f1": score,
                    "finding_localization_recall": score,
                    "finding_type_accuracy": score,
                    "finding_diagnosis_recall": score,
                    "finding_relation_recall": score,
                    "question_utility": 0.0,
                    "num_turns": 1.0,
                },
                "info": {"task_label": "find"},
                "is_completed": True,
                "is_truncated": False,
                "error": None,
            }
        )
    (run_dir / "results.jsonl").write_text(
        "".join(f"{json.dumps(row)}\n" for row in rows), encoding="utf-8"
    )

    [summary] = aggregate.load_summaries(tmp_path)

    assert summary.model == "openai/gpt-4.1"
    assert summary.environment == "commonground-elicit:find"
    assert summary.run_id == "c51ffbc1"
    assert summary.rollout_count == 2
    assert summary.reward_mean == 0.5
    assert set(summary.metrics) == {
        "finding_diagnosis_recall",
        "finding_f1",
        "finding_localization_recall",
        "finding_relation_recall",
        "finding_type_accuracy",
        "question_utility",
    }
    assert summary.metrics["finding_f1"] == (0.5, 0.25)


def test_snapshot_prior_skill_uses_pooled_losses_and_can_be_negative(
    tmp_path: Path,
) -> None:
    traces_path = write_native_run(
        tmp_path,
        environment="commonground-predict",
        run_id="negative-skill",
        num_tasks=1,
        num_rollouts=1,
    )
    rows = read_native_rows(traces_path)
    rows[0]["traces"][0]["metrics"]["brier"] = 1.0
    rows[0]["traces"][0]["metrics"]["original_snapshot_visible_prior_brier"] = 0.25
    write_native_rows(traces_path, rows)

    [summary] = aggregate.load_summaries(tmp_path)
    assert summary.metrics["brier_skill_vs_original_snapshot_visible_prior"] == (
        -3.0,
        0.0,
    )


def test_original_snapshot_prior_skill_rejects_zero_pooled_reference() -> None:
    with pytest.raises(
        aggregate.InvalidRunError,
        match="reference must have positive pooled loss",
    ):
        aggregate.pooled_brier_skill([0.0, 0.0], [0.0, 0.0])


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


def test_native_predict_prompt_modes_never_collapse_to_one_run_key(
    tmp_path: Path,
) -> None:
    modes = ("full", "matrix-only", "text-only", "shuffled-text")
    for index, prompt_mode in enumerate(modes):
        write_native_run(
            tmp_path,
            environment="commonground-predict",
            prompt_mode=prompt_mode,
            run_id=f"predict-mode-{index}",
        )

    summaries = aggregate.load_summaries(tmp_path)

    assert {(summary.prompt_mode, summary.run_id) for summary in summaries} == {
        (prompt_mode, f"predict-mode-{index}")
        for index, prompt_mode in enumerate(modes)
    }
    assert all(summary.environment == "commonground-predict" for summary in summaries)
    runs = [
        aggregate.load_complete_run(path)
        for path in sorted(tmp_path.glob("outputs/*/*/traces.jsonl"))
    ]
    assert len({run.comparison_signature for run in runs}) == 1
    assert all(run.comparison_signature is not None for run in runs)
    assert all(len(run.task_answer_digests) == 2 for run in runs)
    assert all(
        run.raw_reward_scores["probability_reward"] == (1.0, 0.0, 1.0) * 2
        for run in runs
    )


@pytest.mark.parametrize(
    "provenance_change",
    ["client", "sampling", "concurrency", "task_ids", "answer", "call_endpoint"],
)
def test_predict_ablation_comparison_signature_covers_saved_provenance(
    tmp_path: Path, provenance_change: str
) -> None:
    full_path = write_native_run(
        tmp_path,
        environment="commonground-predict",
        prompt_mode="full",
        run_id="full-provenance",
    )
    changed_kwargs: dict[str, Any] = {}
    if provenance_change == "client":
        changed_kwargs["client_base_url"] = "https://other-fixture.invalid/v1"
    elif provenance_change == "sampling":
        changed_kwargs["sampling_temperature"] = 0.7
    elif provenance_change == "concurrency":
        changed_kwargs["max_concurrent"] = 3
    elif provenance_change == "call_endpoint":
        changed_kwargs["call_endpoint"] = "/responses"
    changed_path = write_native_run(
        tmp_path,
        environment="commonground-predict",
        prompt_mode="matrix-only",
        run_id="changed-provenance",
        **changed_kwargs,
    )
    if provenance_change in {"task_ids", "answer"}:
        rows = read_native_rows(changed_path)
        for row in rows:
            task_data = row["traces"][0]["task"]["data"]
            if provenance_change == "task_ids":
                task_data["idx"] += 10
            elif task_data["idx"] == 0:
                task_data["answer"] = {"0,0": -1}
        write_native_rows(changed_path, rows)

    full = aggregate.load_complete_run(full_path)
    changed = aggregate.load_complete_run(changed_path)

    assert full.comparison_signature != changed.comparison_signature


def test_predict_ablation_signature_preserves_task_execution_order(
    tmp_path: Path,
) -> None:
    full_path = write_native_run(
        tmp_path,
        environment="commonground-predict",
        prompt_mode="full",
        run_id="ordered-provenance",
    )
    reordered_path = write_native_run(
        tmp_path,
        environment="commonground-predict",
        prompt_mode="matrix-only",
        run_id="reordered-provenance",
    )
    rows = read_native_rows(reordered_path)
    write_native_rows(reordered_path, list(reversed(rows)))

    full = aggregate.load_complete_run(full_path)
    reordered = aggregate.load_complete_run(reordered_path)

    assert full.comparison_signature != reordered.comparison_signature


def test_declared_predict_ablation_requires_complete_native_provenance(
    tmp_path: Path,
) -> None:
    write_native_run(
        tmp_path,
        environment="commonground-predict",
        prompt_mode="matrix-only",
        run_id="missing-provenance",
        include_ablation_provenance=False,
    )

    with pytest.raises(
        aggregate.InvalidRunError, match="require resolved client provenance"
    ):
        aggregate.load_summaries(tmp_path)


@pytest.mark.parametrize("client_base_url", ["", "   "])
def test_declared_predict_ablation_requires_nonempty_client_identity(
    tmp_path: Path, client_base_url: str
) -> None:
    write_native_run(
        tmp_path,
        environment="commonground-predict",
        prompt_mode="matrix-only",
        run_id="empty-client-identity",
        client_base_url=client_base_url,
    )

    with pytest.raises(
        aggregate.InvalidRunError, match="client base_url must be non-empty"
    ):
        aggregate.load_summaries(tmp_path)


def test_declared_predict_ablation_requires_saved_task_answers(tmp_path: Path) -> None:
    traces_path = write_native_run(
        tmp_path,
        environment="commonground-predict",
        prompt_mode="matrix-only",
        run_id="missing-answer-provenance",
    )
    rows = read_native_rows(traces_path)
    del rows[0]["traces"][0]["task"]["data"]["answer"]
    write_native_rows(traces_path, rows)

    with pytest.raises(
        aggregate.InvalidRunError, match="trace is missing the task answer"
    ):
        aggregate.load_summaries(tmp_path)


def test_declared_predict_ablation_requires_canonical_reward_weight(
    tmp_path: Path,
) -> None:
    write_native_run(
        tmp_path,
        environment="commonground-predict",
        prompt_mode="matrix-only",
        run_id="weighted-ablation",
        reward_weight=0.5,
    )

    with pytest.raises(
        aggregate.InvalidRunError, match=r"probability_reward weight 1\.0"
    ):
        aggregate.load_summaries(tmp_path)


def test_declared_predict_ablation_trace_settings_must_match_config(
    tmp_path: Path,
) -> None:
    traces_path = write_native_run(
        tmp_path,
        environment="commonground-predict",
        prompt_mode="matrix-only",
        run_id="trace-provenance-mismatch",
    )
    rows = read_native_rows(traces_path)
    rows[0]["traces"][0]["agent"]["config"]["sampling"]["temperature"] = 0.9
    write_native_rows(traces_path, rows)

    with pytest.raises(
        aggregate.InvalidRunError, match="trace sampling does not match saved config"
    ):
        aggregate.load_summaries(tmp_path)


@pytest.mark.parametrize("field", ["model", "sampling"])
def test_declared_predict_ablation_call_settings_must_match_config(
    tmp_path: Path, field: str
) -> None:
    traces_path = write_native_run(
        tmp_path,
        environment="commonground-predict",
        prompt_mode="matrix-only",
        run_id="call-provenance-mismatch",
    )
    rows = read_native_rows(traces_path)
    call = rows[0]["traces"][0]["calls"][0]
    if field == "model":
        call["model"] = "different/model"
    else:
        call["sampling"]["temperature"] = 0.99
    write_native_rows(traces_path, rows)

    with pytest.raises(
        aggregate.InvalidRunError, match=rf"call {field} does not match saved config"
    ):
        aggregate.load_summaries(tmp_path)


def test_declared_predict_ablation_requires_one_recorded_call(tmp_path: Path) -> None:
    traces_path = write_native_run(
        tmp_path,
        environment="commonground-predict",
        prompt_mode="matrix-only",
        run_id="missing-call-provenance",
    )
    rows = read_native_rows(traces_path)
    del rows[0]["traces"][0]["calls"]
    write_native_rows(traces_path, rows)

    with pytest.raises(
        aggregate.InvalidRunError, match="record exactly one model call"
    ):
        aggregate.load_summaries(tmp_path)


def test_native_predict_trace_prompt_mode_must_match_config(tmp_path: Path) -> None:
    traces_path = write_native_run(
        tmp_path,
        environment="commonground-predict",
        prompt_mode="matrix-only",
        run_id="prompt-mode-mismatch",
    )
    rows = read_native_rows(traces_path)
    rows[0]["traces"][0]["task"]["data"]["info"]["prompt_mode"] = "full"
    write_native_rows(traces_path, rows)

    with pytest.raises(
        aggregate.InvalidRunError, match=r"prompt mode does not match run config"
    ):
        aggregate.load_summaries(tmp_path)


def test_historical_predict_run_without_prompt_mode_defaults_to_full() -> None:
    summaries = aggregate.load_summaries(FIXTURE_ROOT)
    predict = next(
        summary
        for summary in summaries
        if summary.environment == "commonground-predict"
    )

    assert predict.prompt_mode == "full"


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
            task_data = row["traces"][0]["task"]["data"]
            task_data["idx"] = 0
            task_data["answer"] = {"0,0": 1}
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
            "| Model | Environment | Prompt mode | Run ID | Rollouts | Recovered rollouts | Reward (mean ± std) | vote_accuracy (mean ± std) | brier (mean ± std) | original_snapshot_visible_prior_brier (mean ± std) | brier_skill_vs_original_snapshot_visible_prior (mean ± std) | finding_localization_recall (mean ± std) | finding_type_accuracy (mean ± std) | finding_diagnosis_recall (mean ± std) | finding_relation_recall (mean ± std) | finding_f1 (mean ± std) | question_utility (mean ± std) | question_format_valid (mean ± std) | question_top1_selection_accuracy (mean ± std) | question_grounding_recall (mean ± std) | question_grounded_stance_recall (mean ± std) | question_evidence_match_recall (mean ± std) | question_evidence_matched_stance_accuracy (mean ± std) | question_stance_accuracy (mean ± std) |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            "| fixture/offline-oracle | commonground-elicit:find | — | d4c3b2a1 | 6 | 0 | 0.667 ± 0.471 | — | — | — | — | — | — | — | — | 0.667 ± 0.471 | 0.415 ± 0.294 | — | — | — | — | — | — | — |",
            "| fixture/offline-oracle | commonground-predict | full | a1b2c3d4 | 6 | 0 | 0.667 ± 0.471 | 0.667 ± 0.471 | 0.333 ± 0.471 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |",
        ]
    )


def test_060_ask_signals_use_evidence_matched_stance_metric(
    tmp_path: Path,
) -> None:
    rewards, metrics = aggregate.expected_native_signals(
        "charliethompson/commonground-elicit@0.6.0",
        "elicit-ask",
        tmp_path / "config.toml",
    )

    assert rewards == ("question_utility",)
    assert metrics == (
        "question_evidence_match_recall",
        "question_evidence_matched_stance_accuracy",
        "question_format_valid",
        "question_grounded_stance_recall",
        "question_grounding_recall",
        "question_top1_selection_accuracy",
    )


def test_060_saved_result_metrics_track_expanded_elicit_contracts(
    tmp_path: Path,
) -> None:
    find_metrics = aggregate.expected_legacy_metrics(
        "charliethompson/commonground-elicit@0.6.0",
        "find",
        tmp_path / "find-metadata.json",
    )
    ask_metrics = aggregate.expected_legacy_metrics(
        "charliethompson/commonground-elicit@0.6.0",
        "elicit-ask",
        tmp_path / "ask-metadata.json",
    )

    assert find_metrics == (
        "finding_localization_recall",
        "finding_type_accuracy",
        "finding_diagnosis_recall",
        "finding_relation_recall",
        "finding_f1",
        "question_utility",
    )
    assert ask_metrics == (
        "question_utility",
        "question_format_valid",
        "question_top1_selection_accuracy",
        "question_grounding_recall",
        "question_grounded_stance_recall",
        "question_evidence_match_recall",
        "question_evidence_matched_stance_accuracy",
    )
    assert aggregate.expected_legacy_metrics(
        "charliethompson/commonground-elicit@0.4.1",
        "find",
        tmp_path / "historical-metadata.json",
    ) == ("finding_f1", "question_utility")


def test_060_predict_signals_add_prompt_visible_climatology_loss(
    tmp_path: Path,
) -> None:
    current_rewards, current_metrics = aggregate.expected_native_signals(
        "charliethompson/commonground-predict@0.6.0",
        None,
        tmp_path / "current.toml",
    )
    historical_rewards, historical_metrics = aggregate.expected_native_signals(
        "charliethompson/commonground-predict@0.5.0",
        None,
        tmp_path / "historical.toml",
    )

    assert current_rewards == historical_rewards == ("probability_reward",)
    assert current_metrics == (
        "brier",
        "original_snapshot_visible_prior_brier",
        "vote_accuracy",
    )
    assert historical_metrics == ("brier", "vote_accuracy")


@pytest.mark.parametrize(
    ("environment", "question_count"),
    [
        ("charliethompson/commonground-elicit@0.5.0", 2),
        ("charliethompson/commonground-elicit@0.6.0", 1),
        ("charliethompson/commonground-elicit@0.7.0", 1),
        ("commonground-elicit", 1),
    ],
)
def test_native_elicit_question_budget_tracks_immutable_contract(
    tmp_path: Path, environment: str, question_count: int
) -> None:
    taskset: dict[str, Any] = {
        "split": "eval",
        "task": {"judges": []},
        "task_mode": "elicit-ask",
        "planted_density": 1.0,
        "distractor_density": 1.0,
        "panel_polarization": 1.0,
        "question_count": question_count,
    }

    aggregate.validate_native_baseline_profile(
        {"shuffle": False},
        base_environment=environment,
        taskset=taskset,
        task_mode="elicit-ask",
        config_path=tmp_path / "config.toml",
    )

    taskset["question_count"] = 2 if question_count == 1 else 1
    with pytest.raises(aggregate.InvalidRunError, match="noncanonical Elicit"):
        aggregate.validate_native_baseline_profile(
            {"shuffle": False},
            base_environment=environment,
            taskset=taskset,
            task_mode="elicit-ask",
            config_path=tmp_path / "config.toml",
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
    assert output.startswith(
        "| Model | Environment | Prompt mode | Run ID | Rollouts |"
    )
    with csv_path.open(encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert [(row["model"], row["environment"]) for row in rows] == [
        ("fixture/offline-oracle", "commonground-elicit:find"),
        ("fixture/offline-oracle", "commonground-predict"),
    ]
    assert rows[0]["brier_mean"] == ""
    assert rows[0]["prompt_mode"] == ""
    assert rows[0]["recovered_rollouts"] == "0"
    assert rows[0]["run_id"] == "d4c3b2a1"
    assert rows[1]["vote_accuracy_mean"] == "0.6666666666666666"
    assert rows[1]["brier_mean"] == "0.3333333333333333"
    assert rows[1]["prompt_mode"] == "full"
