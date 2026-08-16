from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = Path(__file__).resolve().parents[1] / "commonground_predict" / "data"
SYNTHETIC_SPLIT = DATA_DIR / "eval_synthetic.jsonl"


def load_script(module_name: str, filename: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, ROOT / "scripts" / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compute_floors_module = load_script("commonground_compute_floors", "compute_floors.py")
ingest_snapshots_module = load_script("commonground_ingest_snapshots", "ingest_snapshots.py")
compute_floors = compute_floors_module.compute_floors
render_markdown = compute_floors_module.render_markdown
ingest_main = ingest_snapshots_module.main


def test_compute_floors_reproduces_published_synthetic_values() -> None:
    floors = compute_floors(SYNTHETIC_SPLIT, masked_vote_count=8, seed="42")

    assert floors == {
        "always-agree": 0.425,
        "visible-majority": 0.49375,
        "best-constant-oracle": 0.525,
    }
    assert render_markdown(floors) == "\n".join(
        [
            "| Baseline | vote_accuracy |",
            "| --- | ---: |",
            "| Always agree | 0.425 |",
            "| Per-statement visible majority | 0.494 |",
            "| Per-snapshot best constant oracle | 0.525 |",
        ]
    )


def test_compute_floors_is_deterministic_for_same_seed(tmp_path: Path) -> None:
    snapshot = valid_real_snapshot()
    source = write_jsonl(tmp_path / "unmasked.jsonl", [snapshot])

    first = compute_floors(source, masked_vote_count=4, seed="repeatable")
    second = compute_floors(source, masked_vote_count=4, seed="repeatable")

    assert first == second


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
    assert json.loads(output.read_text(encoding="utf-8"))[
        "session_id"
    ] == "real-session-a"
    assert json.loads(manifest.read_text(encoding="utf-8")) == {
        "output_file": "eval_real.jsonl",
        "snapshot_count": 1,
        "snapshots": [
            {
                "cluster_count": 2,
                "k_anonymity": 5,
                "participant_count": 10,
                "session_id": "real-session-a",
                "source_index": 0,
                "source_line": 1,
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "statement_count": 2,
            }
        ],
        "version": 1,
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
    assert "statement 0 contains a redacted identifier pattern" in capsys.readouterr().err


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
    assert "cluster size below k=5: [4]" in capsys.readouterr().err


def test_ingest_rejects_ragged_vote_matrix(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    snapshot = valid_real_snapshot()
    snapshot["votes"][0].pop()

    assert run_rejected_ingest(tmp_path, snapshot) == 1
    assert "votes row 0 is ragged" in capsys.readouterr().err


def test_ingest_rejects_transposed_vote_matrix(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    snapshot = valid_real_snapshot()
    snapshot["votes"] = [list(column) for column in zip(*snapshot["votes"], strict=True)]

    assert run_rejected_ingest(tmp_path, snapshot) == 1
    assert "votes must be participant-major" in capsys.readouterr().err


def test_ingest_rejects_nonempty_masked_cells(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    snapshot = valid_real_snapshot()
    snapshot["masked_cells"] = [[0, 0]]

    assert run_rejected_ingest(tmp_path, snapshot) == 1
    assert "real intake requires empty masked_cells" in capsys.readouterr().err


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
    assert "real intake requires empty masked_cells" in capsys.readouterr().err
    assert [json.loads(line)["session_id"] for line in output.read_text().splitlines()] == [
        "real-session-a"
    ]


def test_ingest_rejects_synthetic_provenance(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    snapshot = valid_real_snapshot()
    snapshot["meta"]["synthetic"] = True

    assert run_rejected_ingest(tmp_path, snapshot) == 1
    assert "real intake requires meta.synthetic=false" in capsys.readouterr().err


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
        (lambda snapshot: snapshot["statements"][0].__setitem__("index", 0.0), "positional index"),
        (lambda snapshot: snapshot.__setitem__("clusters", [0] * 5 + [1] * 5), "exporter cluster objects"),
        (lambda snapshot: snapshot["votes"][0].__setitem__(0, 2), "invalid vote at 0,0"),
        (lambda snapshot: snapshot["votes"][0].__setitem__(0, 1.0), "invalid vote at 0,0"),
        (lambda snapshot: snapshot["votes"][0].__setitem__(0, True), "invalid vote at 0,0"),
        (lambda snapshot: snapshot["votes"][0].__setitem__(0, "1"), "invalid vote at 0,0"),
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
                    "extremity": 1.0,
                    "divisiveness": 1.0,
                }
                for statement_index in range(2)
            ]
        },
        "meta": {
            "k_anonymity": 5,
            "source": "context-engine-session",
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
