from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import random
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest
from verifiers.v1.configs.cli.eval import EvalConfig

ROOT = Path(__file__).resolve().parents[3]
REPOSITORY_SCRIPTS = (
    ROOT / "scripts" / "compute_floors.py",
    ROOT / "scripts" / "ingest_snapshots.py",
    ROOT
    / "environments"
    / "commonground_predict"
    / "scripts"
    / "generate_synthetic_eval.py",
)
if not all(path.is_file() for path in REPOSITORY_SCRIPTS):
    pytest.skip(
        "repository-only release tests require the complete monorepo checkout",
        allow_module_level=True,
    )

DATA_DIR = Path(__file__).resolve().parents[1] / "commonground_predict" / "data"
SYNTHETIC_SPLIT = DATA_DIR / "eval_synthetic.jsonl"
TRAIN_SPLIT = DATA_DIR / "train_synthetic.jsonl"
PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"
README = Path(__file__).resolve().parents[1] / "README.md"
PREDICT_ABLATION_CONFIG = ROOT / "configs" / "eval" / "predict-ablation.toml"


def load_module(module_name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compute_floors_module = load_module(
    "commonground_compute_floors", ROOT / "scripts" / "compute_floors.py"
)
ingest_snapshots_module = load_module(
    "commonground_ingest_snapshots", ROOT / "scripts" / "ingest_snapshots.py"
)
eval_generator_module = load_module(
    "commonground_predict_eval_generator",
    ROOT
    / "environments"
    / "commonground_predict"
    / "scripts"
    / "generate_synthetic_eval.py",
)
train_generator_module = load_module(
    "commonground_predict_train_generator",
    ROOT
    / "environments"
    / "commonground_predict"
    / "scripts"
    / "generate_synthetic_train.py",
)
compute_floors = compute_floors_module.compute_floors
render_markdown = compute_floors_module.render_markdown
ingest_main = ingest_snapshots_module.main


def test_hub_pyproject_contract() -> None:
    document = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    tags = document["project"].get("tags")

    assert isinstance(tags, list)
    assert tags
    assert all(isinstance(tag, str) and tag.strip() for tag in tags)

    # PI's Hub action installs the pushed env directory in isolation, where
    # workspace sources make the otherwise portable package uninstallable.
    assert "sources" not in document.get("tool", {}).get("uv", {})


def test_predict_ablation_config_uses_canonical_eval_contract() -> None:
    document = tomllib.loads(PREDICT_ABLATION_CONFIG.read_text(encoding="utf-8"))

    assert document["num_tasks"] == 100
    assert document["num_rollouts"] == 5
    assert document["shuffle"] is False
    assert document["push"] is False
    assert document["env"]["taskset"] == {
        "id": "commonground-predict",
        "split": "eval",
        "prompt_mode": "full",
    }
    config = EvalConfig.model_validate(document)
    assert config.env.taskset.id == "commonground-predict"
    assert config.env.taskset.prompt_mode == "full"  # type: ignore[attr-defined]


def test_statement_bank_is_enterprise_ai_policy() -> None:
    statements = eval_generator_module.STATEMENT_BANK

    assert len(statements) == 20
    assert len(set(statements)) == 20
    assert all("assistant" in statement.casefold() for statement in statements)
    workplace_markers = {
        "support",
        "coding",
        "communications",
        "hr",
        "sales",
        "finance",
        "contract",
        "operations",
        "data classification",
        "customer fields",
        "analytics",
        "customer-facing",
    }
    assert all(
        any(marker in statement.casefold() for marker in workplace_markers)
        for statement in statements[:15]
    )
    governance_markers = (
        "business owner",
        "risk review",
        "contest",
        "evaluations",
        "audit period",
    )
    assert all(
        marker in statement.casefold()
        for marker, statement in zip(governance_markers, statements[15:], strict=True)
    )


def test_bundled_eval_split_matches_seeded_policy_generator() -> None:
    expected = render_generated_split(
        generator=eval_generator_module,
        seed=eval_generator_module.SEED,
        snapshot_count=eval_generator_module.SNAPSHOT_COUNT,
        session_index_offset=0,
    )

    assert SYNTHETIC_SPLIT.read_bytes() == expected


def test_bundled_train_split_matches_seeded_policy_generator() -> None:
    expected = render_generated_split(
        generator=train_generator_module,
        seed=train_generator_module.TRAIN_SEED,
        snapshot_count=train_generator_module.TRAIN_SNAPSHOT_COUNT,
        session_index_offset=train_generator_module.SESSION_INDEX_OFFSET,
    )

    assert TRAIN_SPLIT.read_bytes() == expected


def test_readme_records_exact_candidate_split_hashes() -> None:
    readme = README.read_text(encoding="utf-8")

    for split in (TRAIN_SPLIT, SYNTHETIC_SPLIT):
        assert hashlib.sha256(split.read_bytes()).hexdigest() in readme


def render_generated_split(
    *,
    generator: Any,
    seed: int,
    snapshot_count: int,
    session_index_offset: int,
) -> bytes:
    rng = random.Random(seed)
    snapshots = [
        generator.make_snapshot(rng, session_index_offset + index)
        for index in range(snapshot_count)
    ]
    return "".join(
        json.dumps(snapshot, separators=(",", ":")) + "\n" for snapshot in snapshots
    ).encode()


def test_compute_floors_reproduces_rethemed_synthetic_values() -> None:
    floors = compute_floors(SYNTHETIC_SPLIT, masked_vote_count=8, seed="42")

    snapshot_reference_brier = floors["snapshot-visible-prior"]["brier"]
    assert snapshot_reference_brier > 0
    for metrics in floors.values():
        assert metrics[
            "brier_skill_vs_original_snapshot_visible_prior"
        ] == pytest.approx(1.0 - metrics["brier"] / snapshot_reference_brier)
    floors_without_snapshot_skill = {
        name: {
            metric: value
            for metric, value in metrics.items()
            if metric != "brier_skill_vs_original_snapshot_visible_prior"
        }
        for name, metrics in floors.items()
    }
    assert floors_without_snapshot_skill == {
        "uniform-probability": {
            "vote_accuracy": 0.59,
            "probability_reward": 0.6666666666666659,
            "brier": 0.3333333333333329,
            "brier_skill_vs_uniform": 0.0,
        },
        "snapshot-visible-prior": {
            "vote_accuracy": 0.5875,
            "probability_reward": 0.7245207541099122,
            "brier": 0.2754792458900878,
            "brier_skill_vs_uniform": 0.1735622623297369,
        },
        "global-visible-prior": {
            "vote_accuracy": 0.59,
            "probability_reward": 0.7168234975843193,
            "brier": 0.28317650241568093,
            "brier_skill_vs_uniform": 0.15047049275295743,
        },
        "train-global-prior": {
            "vote_accuracy": 0.16625,
            "probability_reward": 0.6693716353600249,
            "brier": 0.33062836463997497,
            "brier_skill_vs_uniform": 0.00811490608007531,
        },
        "train-text-naive-bayes": {
            "vote_accuracy": 0.24375,
            "probability_reward": 0.4012391358812117,
            "brier": 0.598760864118788,
            "brier_skill_vs_uniform": -0.7962825923563643,
        },
        "always-agree": {
            "vote_accuracy": 0.59,
            "probability_reward": 0.59,
            "brier": 0.41,
            "brier_skill_vs_uniform": -0.22999999999999993,
        },
        "visible-majority": {
            "vote_accuracy": 0.58125,
            "probability_reward": 0.58125,
            "brier": 0.41875,
            "brier_skill_vs_uniform": -0.2562499999999999,
        },
        "statement-visible-frequency": {
            "vote_accuracy": 0.58125,
            "probability_reward": 0.7597568238869272,
            "brier": 0.2402431761130726,
            "brier_skill_vs_uniform": 0.27927047166078267,
        },
        "nearest-participant": {
            "vote_accuracy": 0.81875,
            "probability_reward": 0.81875,
            "brier": 0.18125,
            "brier_skill_vs_uniform": 0.45625,
        },
        "five-neighbor": {
            "vote_accuracy": 0.89125,
            "probability_reward": 0.89125,
            "brier": 0.10875,
            "brier_skill_vs_uniform": 0.67375,
        },
        "five-neighbor-frequency": {
            "vote_accuracy": 0.89125,
            "probability_reward": 0.8888000000000001,
            "brier": 0.11119999999999995,
            "brier_skill_vs_uniform": 0.6663999999999999,
        },
        "distance-weighted-five-neighbor": {
            "vote_accuracy": 0.9,
            "probability_reward": 0.8810775996847495,
            "brier": 0.11892240031525013,
            "brier_skill_vs_uniform": 0.6432327990542497,
        },
        "best-constant-oracle": {
            "vote_accuracy": 0.63125,
            "probability_reward": 0.63125,
            "brier": 0.36875,
            "brier_skill_vs_uniform": -0.10624999999999991,
        },
        "cluster-pattern-oracle": {
            "vote_accuracy": 0.91625,
            "probability_reward": 0.91625,
            "brier": 0.08375,
            "brier_skill_vs_uniform": 0.74875,
        },
    }
    rendered = render_markdown(floors)
    assert rendered == "\n".join(
        [
            "| Comparator class | Comparator | probability reward | vote_accuracy | normalized Brier | Brier skill vs uniform | Brier skill vs snapshot prior |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            "| No-input | Uniform probability | 0.667 | 0.590 | 0.333 | 0.000 | -0.210 |",
            "| Prompt-observable matrix-only | Per-snapshot visible class prior | 0.725 | 0.588 | 0.275 | 0.174 | 0.000 |",
            "| Evaluation-corpus visible (transductive) | Global visible class prior | 0.717 | 0.590 | 0.283 | 0.150 | -0.028 |",
            "| Train-split no-text | Global empirical class prior | 0.669 | 0.166 | 0.331 | 0.008 | -0.200 |",
            "| Train-split text-only | Bag-of-words vote probabilities | 0.401 | 0.244 | 0.599 | -0.796 | -1.174 |",
            "| No-input | Always agree | 0.590 | 0.590 | 0.410 | -0.230 | -0.488 |",
            "| Prompt-observable matrix-only | Per-statement visible majority | 0.581 | 0.581 | 0.419 | -0.256 | -0.520 |",
            "| Prompt-observable matrix-only | Per-statement visible class frequencies | 0.760 | 0.581 | 0.240 | 0.279 | 0.128 |",
            "| Prompt-observable matrix-only | Nearest participant (1-NN) | 0.819 | 0.819 | 0.181 | 0.456 | 0.342 |",
            "| Prompt-observable matrix-only | Five-neighbor vote | 0.891 | 0.891 | 0.109 | 0.674 | 0.605 |",
            "| Prompt-observable matrix-only | Five-neighbor vote frequencies | 0.889 | 0.891 | 0.111 | 0.666 | 0.596 |",
            "| Prompt-observable matrix-only | Distance-weighted 5-NN with smoothing | 0.881 | 0.900 | 0.119 | 0.643 | 0.568 |",
            "| Held-out-label diagnostic | Per-snapshot best constant | 0.631 | 0.631 | 0.369 | -0.106 | -0.339 |",
            "| Generator diagnostic | Latent cluster-pattern replay | 0.916 | 0.916 | 0.084 | 0.749 | 0.696 |",
        ]
    )
    assert rendered in README.read_text(encoding="utf-8")


def test_train_and_eval_use_disjoint_generator_families() -> None:
    assert (
        eval_generator_module.GENERATOR_FAMILY
        != train_generator_module.GENERATOR_FAMILY
    )
    eval_row = json.loads(SYNTHETIC_SPLIT.read_text(encoding="utf-8").splitlines()[0])
    train_row = json.loads(TRAIN_SPLIT.read_text(encoding="utf-8").splitlines()[0])
    assert (
        eval_row["meta"]["generator_family"] == eval_generator_module.GENERATOR_FAMILY
    )
    assert (
        train_row["meta"]["generator_family"] == train_generator_module.GENERATOR_FAMILY
    )


def test_semantic_dimension_not_statement_index_causes_latent_vote() -> None:
    core = sys.modules["synthetic_core"]
    profile = {"evidence": 0.9, "privacy": -0.9}
    evidence = core.StatementSpec("same presentation index", "evidence")
    privacy = core.StatementSpec("same presentation index", "privacy")

    assert core.semantic_vote(random.Random(7), profile, evidence) == 1
    assert core.semantic_vote(random.Random(7), profile, privacy) == -1


def test_cluster_pattern_oracle_replays_hidden_generator_signal() -> None:
    snapshots = [
        json.loads(line)
        for line in SYNTHETIC_SPLIT.read_text(encoding="utf-8").splitlines()
    ]
    recovered = compute_floors_module.replay_cluster_patterns(snapshots)
    rng = random.Random(eval_generator_module.SEED)
    _, expected_first = eval_generator_module.make_snapshot_with_cluster_patterns(
        rng, 0
    )

    assert recovered is not None
    assert recovered[snapshots[0]["session_id"]] == expected_first


def test_compute_floors_is_deterministic_for_same_seed(tmp_path: Path) -> None:
    snapshot = valid_real_snapshot()
    source = write_jsonl(tmp_path / "unmasked.jsonl", [snapshot])

    first = compute_floors(source, masked_vote_count=4, seed="repeatable")
    second = compute_floors(source, masked_vote_count=4, seed="repeatable")

    assert first == second
    assert "cluster-pattern-oracle" not in first


def test_ingest_accepts_real_export_and_writes_manifest_without_touching_bundled_splits(
    tmp_path: Path,
) -> None:
    synthetic_before = (DATA_DIR / "eval_synthetic.jsonl").read_bytes()
    demo_before = (DATA_DIR / "eval_ce_demo.jsonl").read_bytes()
    source = write_jsonl(tmp_path / "export.jsonl", [valid_real_snapshot()])
    output = tmp_path / "eval_real.jsonl"
    manifest = tmp_path / "eval_real.manifest.json"

    result = ingest_main(
        [str(source), "--output", str(output), "--manifest", str(manifest)]
    )

    assert result == 0
    assert (
        json.loads(output.read_text(encoding="utf-8"))["session_id"] == "real-session-a"
    )
    assert json.loads(manifest.read_text(encoding="utf-8")) == {
        "contract": "commonground-human-snapshot-v2",
        "output_file": "eval_real.jsonl",
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "snapshot_count": 1,
        "snapshots": [
            {
                "cluster_count": 2,
                "consent_scope": "public-benchmark",
                "exporter_version": "1.2.3",
                "k_anonymity": 5,
                "participant_count": 10,
                "privacy_review": {
                    "attested": True,
                    "checks": [
                        "direct-identifiers",
                        "free-text",
                        "participant-pseudonyms",
                    ],
                    "reviewed_at": "2026-08-26",
                },
                "redistribution_rights_approved": True,
                "schema_version": "commonground-human-snapshot-v2",
                "session_id": "real-session-a",
                "source_commit": "a" * 40,
                "source_index": 0,
                "source_line": 1,
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "statement_count": 2,
            }
        ],
        "version": 2,
    }
    assert (DATA_DIR / "eval_synthetic.jsonl").read_bytes() == synthetic_before
    assert (DATA_DIR / "eval_ce_demo.jsonl").read_bytes() == demo_before


@pytest.mark.parametrize(
    ("unsafe_text", "label"),
    [
        ("Contact 0x1234567890abcdef1234567890abcdef12345678", "address"),
        ("Contact sample@example.invalid", "email"),
        ("Contact short.eth", "ENS"),
    ],
)
def test_ingest_rejects_each_statement_pii_pattern(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    unsafe_text: str,
    label: str,
) -> None:
    snapshot = valid_real_snapshot()
    snapshot["statements"][0]["text"] = unsafe_text

    result = run_rejected_ingest(tmp_path, snapshot)

    assert result == 1, label
    assert "statement 0.text contains" in capsys.readouterr().err


def test_ingest_rejects_meta_k_below_floor(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    snapshot = valid_real_snapshot()
    snapshot["meta"]["k_anonymity"] = 4

    assert run_rejected_ingest(tmp_path, snapshot) == 1
    assert "meta.k_anonymity must be at least 5" in capsys.readouterr().err


def test_ingest_rejects_cluster_below_declared_k(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    snapshot = valid_real_snapshot()
    move_participant(snapshot, participant_index=4, from_cluster=0, to_cluster=1)

    assert run_rejected_ingest(tmp_path, snapshot) == 1
    assert "cluster 0 must contain at least k=5" in capsys.readouterr().err


def test_ingest_rejects_ragged_vote_matrix(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    snapshot = valid_real_snapshot()
    snapshot["votes"][0].pop()

    assert run_rejected_ingest(tmp_path, snapshot) == 1
    assert "votes row 0 must contain 2 cells" in capsys.readouterr().err


def test_ingest_rejects_transposed_vote_matrix(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    snapshot = valid_real_snapshot()
    snapshot["votes"] = [
        list(column) for column in zip(*snapshot["votes"], strict=True)
    ]

    assert run_rejected_ingest(tmp_path, snapshot) == 1
    assert "votes must be participant-major" in capsys.readouterr().err


def test_ingest_rejects_nonempty_masked_cells(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    snapshot = valid_real_snapshot()
    snapshot["masked_cells"] = [[0, 0]]

    assert run_rejected_ingest(tmp_path, snapshot) == 1
    assert "human snapshot requires empty masked_cells" in capsys.readouterr().err


def test_ingest_rejects_duplicate_session_ids(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    snapshot = valid_real_snapshot()
    source = write_jsonl(tmp_path / "duplicates.jsonl", [snapshot, snapshot])

    result = ingest_main(
        [
            str(source),
            "--output",
            str(tmp_path / "output.jsonl"),
            "--manifest",
            str(tmp_path / "manifest.json"),
        ]
    )

    assert result == 1
    assert "duplicate session_id: real-session-a" in capsys.readouterr().err
    assert not (tmp_path / "output.jsonl").exists()


def test_ingest_skip_invalid_writes_only_valid_snapshots(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    valid = valid_real_snapshot()
    invalid = copy.deepcopy(valid)
    invalid["session_id"] = "invalid-session"
    invalid["masked_cells"] = [[0, 0]]
    source = write_jsonl(tmp_path / "mixed.jsonl", [valid, invalid])
    output = tmp_path / "output.jsonl"
    manifest = tmp_path / "manifest.json"

    result = ingest_main(
        [
            str(source),
            "--skip-invalid",
            "--output",
            str(output),
            "--manifest",
            str(manifest),
        ]
    )

    assert result == 0
    assert "human snapshot requires empty masked_cells" in capsys.readouterr().err
    assert [
        json.loads(line)["session_id"] for line in output.read_text().splitlines()
    ] == ["real-session-a"]


def test_ingest_rejects_synthetic_provenance(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    snapshot = valid_real_snapshot()
    snapshot["meta"]["synthetic"] = True

    assert run_rejected_ingest(tmp_path, snapshot) == 1
    assert "meta.synthetic must be false" in capsys.readouterr().err


def test_ingest_rejects_identifier_shaped_participant(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    snapshot = valid_real_snapshot()
    snapshot["participants"][0] = "sample@example.invalid"
    snapshot["clusters"][0]["members"][0] = "sample@example.invalid"

    assert run_rejected_ingest(tmp_path, snapshot) == 1
    assert "participants must use positional pseudonyms" in capsys.readouterr().err


def test_ingest_rejects_unknown_nested_fields_instead_of_republishing_them(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    snapshot = valid_real_snapshot()
    snapshot["meta"]["source_path"] = "/private/operator/session.json"
    snapshot["stats"]["operator_email"] = "sample@example.invalid"

    assert run_rejected_ingest(tmp_path, snapshot) == 1
    assert "fields mismatch" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        (
            lambda snapshot: snapshot["statements"][0].__setitem__("index", 0.0),
            "positional index",
        ),
        (
            lambda snapshot: snapshot.__setitem__("clusters", [0] * 5 + [1] * 5),
            "cluster 0 must be an object",
        ),
        (
            lambda snapshot: snapshot["votes"][0].__setitem__(0, 2),
            "invalid vote at 0,0",
        ),
        (
            lambda snapshot: snapshot["votes"][0].__setitem__(0, 1.0),
            "invalid vote at 0,0",
        ),
        (
            lambda snapshot: snapshot["votes"][0].__setitem__(0, True),
            "invalid vote at 0,0",
        ),
        (
            lambda snapshot: snapshot["votes"][0].__setitem__(0, "1"),
            "invalid vote at 0,0",
        ),
    ],
)
def test_ingest_rejects_permissive_schema_types(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    mutation: Any,
    expected_error: str,
) -> None:
    snapshot = valid_real_snapshot()
    mutation(snapshot)

    assert run_rejected_ingest(tmp_path, snapshot) == 1
    assert expected_error in capsys.readouterr().err


def test_ingest_rejects_nonstandard_json_constants(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    snapshot = valid_real_snapshot()
    encoded = json.dumps(snapshot, sort_keys=True).replace('"seed": 42', '"seed": NaN')
    source = tmp_path / "nan.jsonl"
    source.write_text(encoded + "\n", encoding="utf-8")

    result = ingest_main(
        [
            str(source),
            "--output",
            str(tmp_path / "output.jsonl"),
            "--manifest",
            str(tmp_path / "manifest.json"),
        ]
    )

    assert result == 1
    assert "non-standard JSON constant is not allowed: NaN" in capsys.readouterr().err


def test_ingest_rejects_malformed_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "malformed.jsonl"
    source.write_text('{"session_id":\n', encoding="utf-8")

    result = ingest_main(
        [
            str(source),
            "--output",
            str(tmp_path / "output.jsonl"),
            "--manifest",
            str(tmp_path / "manifest.json"),
        ]
    )

    assert result == 1
    assert "Expecting value" in capsys.readouterr().err


@pytest.mark.parametrize(
    "protected_name",
    [
        "eval_synthetic.jsonl",
        "EVAL_SYNTHETIC.JSONL",
        "eval_ce_demo.jsonl",
        "EVAL_CE_DEMO.JSONL",
    ],
)
@pytest.mark.parametrize("protected_option", ["--output", "--manifest"])
def test_ingest_refuses_protected_split_as_either_destination(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    protected_name: str,
    protected_option: str,
) -> None:
    source = write_jsonl(tmp_path / "source.jsonl", [valid_real_snapshot()])
    protected = tmp_path / protected_name
    protected.write_text("sentinel\n", encoding="utf-8")
    output = tmp_path / "output.jsonl"
    manifest = tmp_path / "manifest.json"
    arguments = [
        str(source),
        "--output",
        str(output),
        "--manifest",
        str(manifest),
    ]
    destination_index = arguments.index(protected_option) + 1
    arguments[destination_index] = str(protected)

    assert ingest_main(arguments) == 1
    assert protected.read_text(encoding="utf-8") == "sentinel\n"
    assert "refusing to overwrite protected split" in capsys.readouterr().err


def test_ingest_refuses_case_only_output_manifest_alias(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = write_jsonl(tmp_path / "source.jsonl", [valid_real_snapshot()])

    result = ingest_main(
        [
            str(source),
            "--output",
            str(tmp_path / "accepted.jsonl"),
            "--manifest",
            str(tmp_path / "ACCEPTED.JSONL"),
        ]
    )

    assert result == 1
    assert "output and manifest paths must be different" in capsys.readouterr().err


def test_ingest_rolls_back_output_when_manifest_publish_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = write_jsonl(tmp_path / "source.jsonl", [valid_real_snapshot()])
    output = tmp_path / "output.jsonl"
    manifest = tmp_path / "manifest.json"
    output.write_text("old-output\n", encoding="utf-8")
    manifest.write_text("old-manifest\n", encoding="utf-8")
    real_replace = ingest_snapshots_module.os.replace
    publish_count = 0

    def fail_second_publish(source_path: Path, destination_path: Path) -> None:
        nonlocal publish_count
        if str(source_path).endswith(".tmp"):
            publish_count += 1
            if publish_count == 2:
                raise OSError("forced manifest publish failure")
        real_replace(source_path, destination_path)

    monkeypatch.setattr(ingest_snapshots_module.os, "replace", fail_second_publish)

    result = ingest_main(
        [
            str(source),
            "--output",
            str(output),
            "--manifest",
            str(manifest),
        ]
    )

    assert result == 1
    assert output.read_text(encoding="utf-8") == "old-output\n"
    assert manifest.read_text(encoding="utf-8") == "old-manifest\n"


def run_rejected_ingest(tmp_path: Path, snapshot: dict[str, Any]) -> int:
    source = write_jsonl(tmp_path / "rejected.jsonl", [snapshot])
    return ingest_main(
        [
            str(source),
            "--output",
            str(tmp_path / "output.jsonl"),
            "--manifest",
            str(tmp_path / "manifest.json"),
        ]
    )


def write_jsonl(path: Path, snapshots: list[dict[str, Any]]) -> Path:
    path.write_text(
        "".join(json.dumps(snapshot, sort_keys=True) + "\n" for snapshot in snapshots),
        encoding="utf-8",
    )
    return path


def valid_real_snapshot() -> dict[str, Any]:
    participants = [f"p{index:03d}" for index in range(10)]
    return {
        "session_id": "real-session-a",
        "statements": [
            {"index": 0, "text": "Fund the pilot."},
            {"index": 1, "text": "Publish the results."},
        ],
        "participants": participants,
        "votes": [[1, -1] if index % 2 == 0 else [-1, 1] for index in range(10)],
        "masked_cells": [],
        "held_out": {},
        "clusters": [
            {
                "id": 0,
                "members": participants[:5],
                "member_indices": list(range(5)),
                "center": [],
            },
            {
                "id": 1,
                "members": participants[5:],
                "member_indices": list(range(5, 10)),
                "center": [],
            },
        ],
        "stats": {
            "comment": [
                {
                    "commentIndex": statement_index,
                    "agrees": 5,
                    "disagrees": 5,
                    "unsure": 0,
                    "total": 10,
                    "responded": 10,
                    "extremity": None,
                    "divisiveness": None,
                }
                for statement_index in range(2)
            ]
        },
        "meta": {
            "consent_scope": "public-benchmark",
            "exporter_version": "1.2.3",
            "k_anonymity": 5,
            "privacy_review": {
                "attested": True,
                "checks": [
                    "direct-identifiers",
                    "free-text",
                    "participant-pseudonyms",
                ],
                "reviewed_at": "2026-08-26",
            },
            "redistribution_rights_approved": True,
            "schema_version": "commonground-human-snapshot-v2",
            "source": "context-engine-session",
            "source_commit": "a" * 40,
            "synthetic": False,
            "seed": 42,
        },
    }


def move_participant(
    snapshot: dict[str, Any], participant_index: int, from_cluster: int, to_cluster: int
) -> None:
    participant = snapshot["participants"][participant_index]
    source = snapshot["clusters"][from_cluster]
    destination = snapshot["clusters"][to_cluster]
    source["member_indices"].remove(participant_index)
    source["members"].remove(participant)
    destination["member_indices"].append(participant_index)
    destination["members"].append(participant)
