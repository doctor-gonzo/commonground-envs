from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import random
import shutil
import stat
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest
from commonground_scenarios.snapshot_validation import MAX_VOTE_CELLS
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

    expected = {
        "uniform-probability": {
            "vote_accuracy": 0.59,
            "probability_reward": 0.6666666666666659,
            "brier": 0.3333333333333329,
            "brier_skill_vs_uniform": 0.0,
            "brier_skill_vs_empirical_prior": -0.007777531239914426,
        },
        "train-global-prior": {
            "vote_accuracy": 0.16625,
            "probability_reward": 0.6692391693599106,
            "brier": 0.3307608306400896,
            "brier_skill_vs_uniform": 0.0077175080797316645,
            "brier_skill_vs_empirical_prior": 0.0,
        },
        "statement-visible-frequency": {
            "vote_accuracy": 0.58125,
            "probability_reward": 0.7597568238869272,
            "brier": 0.2402431761130726,
            "brier_skill_vs_uniform": 0.27927047166078267,
            "brier_skill_vs_empirical_prior": 0.27366497523859423,
        },
        "distance-weighted-five-neighbor": {
            "vote_accuracy": 0.9,
            "probability_reward": 0.8810775996847495,
            "brier": 0.11892240031525013,
            "brier_skill_vs_uniform": 0.6432327990542497,
            "brier_skill_vs_empirical_prior": 0.6404580310035168,
        },
    }
    assert set(floors) == set(expected)
    for name, metrics in expected.items():
        assert floors[name] == pytest.approx(metrics)
    rendered = render_markdown(floors)
    assert rendered == "\n".join(
        [
            "| Comparator class | Comparator | probability reward | vote_accuracy | normalized Brier | Brier skill vs uniform | Brier skill vs empirical prior |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            "| No-input | Uniform probability | 0.667 | 0.590 | 0.333 | 0.000 | -0.008 |",
            "| Train-split no-text | Global empirical class prior | 0.669 | 0.166 | 0.331 | 0.008 | 0.000 |",
            "| Prompt-observable matrix-only | Per-statement visible class frequencies | 0.760 | 0.581 | 0.240 | 0.279 | 0.274 |",
            "| Prompt-observable matrix-only | Smoothed distance-weighted 5-neighbor frequencies | 0.881 | 0.900 | 0.119 | 0.643 | 0.640 |",
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


def test_compute_floors_is_deterministic_for_same_seed(tmp_path: Path) -> None:
    snapshot = valid_real_snapshot()
    source = write_jsonl(tmp_path / "unmasked.jsonl", [snapshot])

    first = compute_floors(source, masked_vote_count=4, seed="repeatable")
    second = compute_floors(source, masked_vote_count=4, seed="repeatable")

    assert first == second
    assert set(first) == {
        "uniform-probability",
        "statement-visible-frequency",
        "distance-weighted-five-neighbor",
    }


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


def test_ingest_keeps_unicode_line_separators_inside_json_strings(
    tmp_path: Path,
) -> None:
    snapshot = valid_real_snapshot()
    text = "Fund the pilot.\u2028Review it.\u2029Publish it.\u0085Done."
    snapshot["statements"][0]["text"] = text
    source = tmp_path / "export.jsonl"
    # JavaScript's JSON.stringify leaves these characters unescaped.
    source.write_text(json.dumps(snapshot, ensure_ascii=False) + "\n", encoding="utf-8")
    output = tmp_path / "eval_real.jsonl"

    result = ingest_main(
        [str(source), "--output", str(output), "--manifest", str(tmp_path / "m.json")]
    )

    assert result == 0
    assert json.loads(output.read_text(encoding="utf-8"))["statements"][0]["text"] == (
        text
    )


def test_ingest_treats_crlf_and_lone_cr_as_row_ends(tmp_path: Path) -> None:
    session_ids = ["real-session-a", "real-session-b", "real-session-c"]
    rows = []
    for session_id in session_ids:
        snapshot = valid_real_snapshot()
        snapshot["session_id"] = session_id
        rows.append(json.dumps(snapshot))
    source = tmp_path / "export.jsonl"
    # Rows are separated by CRLF and then by a lone CR, not only terminated.
    source.write_bytes(
        (rows[0] + "\r\n" + rows[1] + "\r" + rows[2] + "\n").encode("utf-8")
    )
    output = tmp_path / "eval_real.jsonl"

    result = ingest_main(
        [str(source), "--output", str(output), "--manifest", str(tmp_path / "m.json")]
    )

    assert result == 0
    accepted = output.read_text(encoding="utf-8").split("\n")
    assert [json.loads(row)["session_id"] for row in accepted if row] == session_ids


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


def test_ingest_rejects_duplicate_json_keys(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    snapshot = valid_real_snapshot()
    shadowed = json.dumps({"synthetic": True})
    source = tmp_path / "export.jsonl"
    source.write_text(
        '{"meta":' + shadowed + "," + json.dumps(snapshot)[1:] + "\n", encoding="utf-8"
    )

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
    assert "duplicate JSON key: 'meta'" in capsys.readouterr().err
    assert not (tmp_path / "output.jsonl").exists()


def test_ingest_skips_a_row_with_an_integer_too_long_to_parse(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    oversized = json.dumps(valid_real_snapshot()).replace(
        '"seed": 42', '"seed": ' + "1" * 5_000
    )
    second = valid_real_snapshot()
    second["session_id"] = "real-session-b"
    source = tmp_path / "export.jsonl"
    source.write_text(oversized + "\n" + json.dumps(second) + "\n", encoding="utf-8")
    output = tmp_path / "output.jsonl"

    result = ingest_main(
        [
            str(source),
            "--skip-invalid",
            "--output",
            str(output),
            "--manifest",
            str(tmp_path / "manifest.json"),
        ]
    )

    assert result == 0
    assert f"rejected: {source}:1:" in capsys.readouterr().err
    accepted = [json.loads(row) for row in output.read_text().split("\n") if row]
    assert [row["session_id"] for row in accepted] == ["real-session-b"]


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
    "repository_destination",
    [
        DATA_DIR / "eval_ce_demo.jsonl",
        DATA_DIR / "EVAL_SYNTHETIC.JSONL",
        DATA_DIR / "eval_real.jsonl",
        ROOT / "untracked-intake" / "eval_real.jsonl",
    ],
)
@pytest.mark.parametrize("destination_option", ["--output", "--manifest"])
def test_ingest_refuses_any_destination_inside_the_repository(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    repository_destination: Path,
    destination_option: str,
) -> None:
    source = write_jsonl(tmp_path / "source.jsonl", [valid_real_snapshot()])
    bundled_before = {path: path.read_bytes() for path in DATA_DIR.iterdir()}
    arguments = [
        str(source),
        "--output",
        str(tmp_path / "output.jsonl"),
        "--manifest",
        str(tmp_path / "manifest.json"),
    ]
    destination_index = arguments.index(destination_option) + 1
    arguments[destination_index] = str(repository_destination)
    created_by_test = not repository_destination.exists()

    try:
        assert ingest_main(arguments) == 1
        err = capsys.readouterr().err
        assert "refusing to write inside a repository checkout" in err
        assert {path: path.read_bytes() for path in DATA_DIR.iterdir()} == (
            bundled_before
        )
        assert not (ROOT / "untracked-intake").exists()
    finally:
        # A guard regression must not leave git-ignored files in the checkout.
        if created_by_test:
            repository_destination.unlink(missing_ok=True)
        shutil.rmtree(ROOT / "untracked-intake", ignore_errors=True)


def test_ingest_refuses_other_checkouts_and_symlinked_destinations(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = write_jsonl(tmp_path / "source.jsonl", [valid_real_snapshot()])
    other_checkout = tmp_path / "other-worktree"
    marker = other_checkout / "environments" / "commonground_predict" / "pyproject.toml"
    marker.parent.mkdir(parents=True)
    marker.write_text("", encoding="utf-8")
    outside_target = tmp_path / "outside" / "target.jsonl"
    outside_target.parent.mkdir()
    outside_target.write_text("outside\n", encoding="utf-8")
    link = other_checkout / "eval_real.jsonl"
    link.symlink_to(outside_target)

    for destination in (other_checkout / "data" / "eval_real.jsonl", link):
        arguments = [str(source), "--output", str(destination)]
        arguments += ["--manifest", str(tmp_path / "manifest.json")]
        assert ingest_main(arguments) == 1

    assert "refusing to write inside a repository checkout" in capsys.readouterr().err
    assert link.is_symlink()
    assert outside_target.read_text(encoding="utf-8") == "outside\n"
    assert not (other_checkout / "data").exists()


def test_ingest_refuses_its_own_checkout_without_the_marker(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A sparse checkout can lack the marker; the script's own root still counts.
    sparse_root = tmp_path / "sparse-checkout"
    sparse_root.mkdir()
    monkeypatch.setattr(ingest_snapshots_module, "ROOT", sparse_root)
    monkeypatch.setattr(
        ingest_snapshots_module, "CHECKOUT_MARKER", Path("missing-marker")
    )
    source = write_jsonl(tmp_path / "source.jsonl", [valid_real_snapshot()])
    destination = sparse_root / "scripts" / "eval_real.jsonl"

    result = ingest_main(
        [
            str(source),
            "--output",
            str(destination),
            "--manifest",
            str(tmp_path / "manifest.json"),
        ]
    )

    assert result == 1
    assert "refusing to write inside a repository checkout" in capsys.readouterr().err
    assert not destination.parent.exists()


def test_ingest_accepts_the_largest_valid_vote_matrix(tmp_path: Path) -> None:
    participant_count = 1_000
    statement_count = MAX_VOTE_CELLS // participant_count
    participants = [f"p{index:03d}" for index in range(participant_count)]
    votes: list[list[int | None]] = [
        [None] * statement_count for _ in range(participant_count)
    ]
    votes[0][0] = 1
    snapshot = valid_real_snapshot()
    snapshot.update(
        statements=[
            {"index": index, "text": f"Statement {index}"}
            for index in range(statement_count)
        ],
        participants=participants,
        votes=votes,
        clusters=[
            {
                "id": 0,
                "members": participants,
                "member_indices": list(range(participant_count)),
                "center": [],
            }
        ],
        stats={
            "comment": [
                {
                    "commentIndex": index,
                    "agrees": int(index == 0),
                    "disagrees": 0,
                    "unsure": 0,
                    "total": int(index == 0),
                    "responded": int(index == 0),
                    "extremity": None,
                    "divisiveness": None,
                }
                for index in range(statement_count)
            ]
        },
    )
    source = write_jsonl(tmp_path / "export.jsonl", [snapshot])
    assert source.stat().st_size > 4 * 1024 * 1024

    result = ingest_main(
        [
            str(source),
            "--output",
            str(tmp_path / "output.jsonl"),
            "--manifest",
            str(tmp_path / "manifest.json"),
        ]
    )

    assert result == 0


def test_ingest_requires_explicit_destinations(tmp_path: Path) -> None:
    source = write_jsonl(tmp_path / "source.jsonl", [valid_real_snapshot()])

    with pytest.raises(SystemExit) as exc_info:
        ingest_main([str(source)])

    assert exc_info.value.code == 2


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


@pytest.mark.parametrize(
    "make_destination",
    [
        pytest.param(Path.mkdir, id="directory"),
        pytest.param(
            getattr(os, "mkfifo", None),
            id="pipe",
            marks=pytest.mark.skipif(
                not hasattr(os, "mkfifo"), reason="needs named pipes"
            ),
        ),
    ],
)
@pytest.mark.parametrize("destination_option", ["--output", "--manifest"])
def test_ingest_refuses_a_non_regular_destination_without_leftovers(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    make_destination: Any,
    destination_option: str,
) -> None:
    source = write_jsonl(tmp_path / "source.jsonl", [valid_real_snapshot()])
    destination = tmp_path / "approved"
    make_destination(destination)
    arguments = [
        str(source),
        "--output",
        str(tmp_path / "output.jsonl"),
        "--manifest",
        str(tmp_path / "manifest.json"),
    ]
    arguments[arguments.index(destination_option) + 1] = str(destination)

    def refuse_backup(path: Path) -> None:
        raise AssertionError(f"tried to back up {path}")

    # Fail at once if the refusal regresses; backing up a pipe blocks forever.
    monkeypatch.setattr(ingest_snapshots_module, "_backup_existing", refuse_backup)

    assert ingest_main(arguments) == 1
    assert "destination is not a regular file" in capsys.readouterr().err
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "approved",
        "source.jsonl",
    ]


@pytest.mark.skipif(
    not hasattr(os, "geteuid") or os.geteuid() == 0,
    reason="needs a user whose file permissions are enforced",
)
def test_ingest_fails_cleanly_on_an_unreadable_destination(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = write_jsonl(tmp_path / "source.jsonl", [valid_real_snapshot()])
    output = tmp_path / "output.jsonl"
    output.write_text("old-output\n", encoding="utf-8")
    output.chmod(0)

    result = ingest_main(
        [
            str(source),
            "--output",
            str(output),
            "--manifest",
            str(tmp_path / "manifest.json"),
        ]
    )

    output.chmod(0o600)
    assert result == 1
    assert "Permission denied" in capsys.readouterr().err
    assert output.read_text(encoding="utf-8") == "old-output\n"
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "output.jsonl",
        "source.jsonl",
    ]


def test_ingest_removes_its_backup_when_copying_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = write_jsonl(tmp_path / "source.jsonl", [valid_real_snapshot()])
    output = tmp_path / "output.jsonl"
    output.write_text("old-output\n", encoding="utf-8")

    def fail_copy(existing_file: Any, backup_file: Any) -> None:
        raise OSError("forced backup copy failure")

    monkeypatch.setattr(ingest_snapshots_module.shutil, "copyfileobj", fail_copy)

    result = ingest_main(
        [
            str(source),
            "--output",
            str(output),
            "--manifest",
            str(tmp_path / "manifest.json"),
        ]
    )

    assert result == 1
    assert output.read_text(encoding="utf-8") == "old-output\n"
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "output.jsonl",
        "source.jsonl",
    ]


def test_ingest_removes_its_backup_when_copying_is_interrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output.jsonl"
    output.write_text("old-output\n", encoding="utf-8")

    def interrupt_copy(existing_file: Any, backup_file: Any) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(ingest_snapshots_module.shutil, "copyfileobj", interrupt_copy)

    with pytest.raises(KeyboardInterrupt):
        ingest_snapshots_module._backup_existing(output)

    assert list(tmp_path.iterdir()) == [output]


def test_ingest_removes_its_temporary_file_when_writing_fails(tmp_path: Path) -> None:
    # A lone surrogate cannot be encoded, so the write fails part way.
    with pytest.raises(UnicodeEncodeError):
        ingest_snapshots_module._write_temporary(tmp_path / "output.jsonl", "\ud800")

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "write_file",
    [
        lambda output: ingest_snapshots_module._write_temporary(output, "human\n"),
        lambda output: ingest_snapshots_module._backup_existing(output),
    ],
    ids=["temporary", "backup"],
)
def test_ingest_writes_temporary_files_through_their_handle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, write_file: Any
) -> None:
    output = tmp_path / "output.jsonl"
    output.write_text("old human output\n", encoding="utf-8")
    victim = tmp_path / "victim.txt"
    victim.write_text("victim\n", encoding="utf-8")
    real_named_temporary_file = ingest_snapshots_module.tempfile.NamedTemporaryFile

    def swap_name_for_symlink(*args: Any, **kwargs: Any) -> Any:
        handle = real_named_temporary_file(*args, **kwargs)
        link = Path(handle.name).with_suffix(".link")
        link.symlink_to(victim)
        os.replace(link, handle.name)
        return handle

    monkeypatch.setattr(
        ingest_snapshots_module.tempfile, "NamedTemporaryFile", swap_name_for_symlink
    )

    write_file(output)

    assert victim.read_text(encoding="utf-8") == "victim\n"


def test_ingest_rolls_back_output_when_manifest_publish_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = write_jsonl(tmp_path / "source.jsonl", [valid_real_snapshot()])
    output = tmp_path / "output.jsonl"
    manifest = tmp_path / "manifest.json"
    output.write_text("old-output\n", encoding="utf-8")
    manifest.write_text("old-manifest\n", encoding="utf-8")
    output.chmod(0o644)
    manifest.chmod(0o644)
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
    # Backups stay owner-only: copying the old file's mode could leave a
    # readable copy of human data in a shared output directory.
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert stat.S_IMODE(manifest.stat().st_mode) == 0o600


def test_ingest_rolls_back_when_interrupted_between_publish_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = write_jsonl(tmp_path / "source.jsonl", [valid_real_snapshot()])
    output = tmp_path / "output.jsonl"
    manifest = tmp_path / "manifest.json"
    output.write_text("old-output\n", encoding="utf-8")
    manifest.write_text("old-manifest\n", encoding="utf-8")
    real_replace = ingest_snapshots_module.os.replace
    publish_count = 0

    def interrupt_second_publish(source_path: Path, destination_path: Path) -> None:
        nonlocal publish_count
        if str(source_path).endswith(".tmp"):
            publish_count += 1
            if publish_count == 2:
                raise KeyboardInterrupt
        real_replace(source_path, destination_path)

    monkeypatch.setattr(ingest_snapshots_module.os, "replace", interrupt_second_publish)

    with pytest.raises(KeyboardInterrupt):
        ingest_main(
            [
                str(source),
                "--output",
                str(output),
                "--manifest",
                str(manifest),
            ]
        )

    assert output.read_text(encoding="utf-8") == "old-output\n"
    assert manifest.read_text(encoding="utf-8") == "old-manifest\n"
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "manifest.json",
        "output.jsonl",
        "source.jsonl",
    ]


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
