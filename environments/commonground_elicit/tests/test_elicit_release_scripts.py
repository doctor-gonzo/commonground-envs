from __future__ import annotations

import copy
import importlib.util
import json
import random
import tomllib
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
REPOSITORY_SCRIPTS = (
    ROOT / "scripts" / "generate_elicit_splits.py",
    ROOT / "scripts" / "compute_elicit_floors.py",
)
if not all(path.is_file() for path in REPOSITORY_SCRIPTS):
    pytest.skip(
        "repository-only release tests require the complete monorepo checkout",
        allow_module_level=True,
    )

DATA_DIR = Path(__file__).resolve().parents[1] / "commonground_elicit" / "data"
TRAIN_SPLIT = DATA_DIR / "train_synthetic.jsonl"
EVAL_SPLIT = DATA_DIR / "eval_synthetic_heldout.jsonl"
PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def load_script(module_name: str, filename: str) -> Any:
    spec = importlib.util.spec_from_file_location(
        module_name, ROOT / "scripts" / filename
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


generator = load_script(
    "commonground_generate_elicit_splits", "generate_elicit_splits.py"
)
floors = load_script("commonground_compute_elicit_floors", "compute_elicit_floors.py")


def test_hub_pyproject_contract() -> None:
    document = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    tags = document["project"].get("tags")

    assert isinstance(tags, list)
    assert tags
    assert all(isinstance(tag, str) and tag.strip() for tag in tags)

    # PI's Hub action installs the pushed env directory in isolation, where
    # workspace sources make the otherwise portable package uninstallable.
    assert "sources" not in document.get("tool", {}).get("uv", {})


def test_bundled_splits_reproduce_generator_bytes() -> None:
    expected_train = generator.build_split_bytes(
        generator.TRAIN_TEMPLATES,
        seed_base=generator.TRAIN_SEED_BASE,
    )
    expected_eval = generator.build_split_bytes(
        generator.HELDOUT_TEMPLATES,
        seed_base=generator.EVAL_SEED_BASE,
    )

    assert TRAIN_SPLIT.read_bytes() == expected_train
    assert EVAL_SPLIT.read_bytes() == expected_eval


def test_same_seed_regeneration_is_byte_identical(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    generator.write_splits(first_dir)
    generator.write_splits(second_dir)

    assert (first_dir / TRAIN_SPLIT.name).read_bytes() == (
        second_dir / TRAIN_SPLIT.name
    ).read_bytes()
    assert (first_dir / EVAL_SPLIT.name).read_bytes() == (
        second_dir / EVAL_SPLIT.name
    ).read_bytes()


def test_bundled_splits_are_synthetic_and_template_separated() -> None:
    train = read_jsonl(TRAIN_SPLIT)
    heldout = read_jsonl(EVAL_SPLIT)
    train_template_ids = {scenario["provenance"]["template_id"] for scenario in train}
    heldout_template_ids = {
        scenario["provenance"]["template_id"] for scenario in heldout
    }

    assert len(train) == 40
    assert len(heldout) == 20
    assert train_template_ids == {
        template.template_id for template in generator.TRAIN_TEMPLATES
    }
    assert heldout_template_ids == {
        template.template_id for template in generator.HELDOUT_TEMPLATES
    }
    assert train_template_ids.isdisjoint(heldout_template_ids)
    assert all(
        scenario["provenance"]["synthetic"] is True
        and scenario["provenance"]["generated_at"] == generator.GENERATED_AT
        for scenario in train + heldout
    )
    train_semantic_keys = [
        generator.scenario_semantic_key(scenario) for scenario in train
    ]
    semantic_keys = [generator.scenario_semantic_key(scenario) for scenario in heldout]
    assert len(train_semantic_keys) == len(set(train_semantic_keys)) == 40
    assert len(semantic_keys) == len(set(semantic_keys)) == 20


def test_generator_rejects_duplicate_semantic_tasks() -> None:
    scenario = read_jsonl(EVAL_SPLIT)[0]
    duplicate = copy.deepcopy(scenario)
    duplicate["scenario_id"] = "different-seed-only"

    with pytest.raises(ValueError, match="duplicate semantic task"):
        generator.assert_unique_semantic_tasks([scenario, duplicate])


def test_baseline_generation_depends_only_on_public_projections() -> None:
    scenario = read_jsonl(EVAL_SPLIT)[0]
    altered = copy.deepcopy(scenario)
    altered["planted_items"] = [{"secret": "changed"}]
    documents = floors.public_documents(scenario)
    factions = floors.public_factions(scenario)

    assert documents == floors.public_documents(altered)
    assert factions == floors.public_factions(altered)
    assert all(set(document) == {"doc_id", "text"} for document in documents)
    assert all(
        set(faction) == {"faction_id", "name", "summary"} for faction in factions
    )
    assert floors.random_span_findings(
        documents, random.Random("same")
    ) == floors.random_span_findings(
        floors.public_documents(altered), random.Random("same")
    )
    assert floors.randomly_targeted_questions(
        documents,
        factions,
        random.Random("same"),
        question_count=floors.QUESTION_COUNT,
    ) == floors.randomly_targeted_questions(
        floors.public_documents(altered),
        floors.public_factions(altered),
        random.Random("same"),
        question_count=floors.QUESTION_COUNT,
    )


def test_compute_elicit_floors_reproduces_published_values() -> None:
    computed = floors.compute_elicit_floors(EVAL_SPLIT)

    assert computed == {
        "find/random-span": 2 / 15,
        "find/vague-sounding": 39 / 200,
        "elicit-ask/template-question": 0.0,
        "elicit-ask/randomly-targeted": 0.0,
    }
    assert floors.render_markdown(computed) == "\n".join(
        [
            "| Task | Baseline | mean reward |",
            "| --- | --- | ---: |",
            "| find | Random visible spans | 0.133 |",
            "| find | Flag vague-sounding spans | 0.195 |",
            "| elicit-ask | Template clarity questions | 0.000 |",
            "| elicit-ask | Randomly targeted questions | 0.000 |",
        ]
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
