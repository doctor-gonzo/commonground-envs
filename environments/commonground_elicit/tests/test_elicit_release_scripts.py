from __future__ import annotations

import copy
import importlib.util
import json
import tomllib
from pathlib import Path
from typing import Any

import pytest
from commonground_scenarios import get_template

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


def test_every_bundled_contradiction_matches_its_authored_relationship() -> None:
    for scenario in (*read_jsonl(TRAIN_SPLIT), *read_jsonl(EVAL_SPLIT)):
        template = get_template(scenario["provenance"]["template_id"])
        authored = next(
            plant
            for plant in template.planted_items
            if plant["type"] == "contradiction"
        )
        contradiction = next(
            plant
            for plant in scenario["planted_items"]
            if plant["type"] == "contradiction"
        )
        related = contradiction["related_evidence"]
        related_document = next(
            document
            for document in scenario["documents"]
            if document["doc_id"] == related["doc_id"]
        )

        assert related["quote"].startswith(
            authored["related_anchor_quote"].removesuffix(".")
        )
        assert related["quote"] in related_document["text"]
        assert related["doc_id"] != contradiction["doc_id"]
        assert related["quote"] not in {
            plant["anchor_quote"] for plant in scenario["planted_items"]
        }
        assert related["quote"] not in {
            distractor["anchor_quote"] for distractor in scenario["distractors"]
        }


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

    assert len(train) == 100
    assert len(heldout) == 100
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
        generator.policy_issue_semantics(scenario) for scenario in train
    ]
    semantic_keys = [generator.policy_issue_semantics(scenario) for scenario in heldout]
    assert len(train_semantic_keys) == len(set(train_semantic_keys)) == 100
    assert len(semantic_keys) == len(set(semantic_keys)) == 100
    generator.assert_split_integrity(train, heldout)
    assert {
        scenario["provenance"]["generator_family"] for scenario in train
    }.isdisjoint({scenario["provenance"]["generator_family"] for scenario in heldout})
    assert (
        len({generator.instance_fingerprint(scenario) for scenario in heldout}) == 100
    )
    assert {len(scenario["factions"]) for scenario in heldout} == {3, 4, 5}
    assert all(
        document["doc_id"].startswith("doc-")
        for scenario in heldout
        for document in scenario["documents"]
    )
    assert all(
        document["title"].split()[0]
        in {"Operations", "Service", "Implementation", "Review", "Field", "Decision"}
        for scenario in heldout
        for document in scenario["documents"]
    )
    assert all(
        "For decisions involving" not in faction["summary"]
        and "leans toward yes" not in faction["summary"]
        and "leans toward no" not in faction["summary"]
        for scenario in heldout
        for faction in scenario["factions"]
    )
    similarity, train_id, eval_id = generator.maximum_cross_split_tfidf_cosine(
        train, heldout
    )
    assert similarity <= generator.MAX_CROSS_SPLIT_TFIDF_COSINE, (
        similarity,
        train_id,
        eval_id,
    )


def test_generator_rejects_duplicate_semantic_tasks() -> None:
    scenario = read_jsonl(EVAL_SPLIT)[0]
    duplicate = copy.deepcopy(scenario)
    duplicate["scenario_id"] = "different-seed-only"

    with pytest.raises(ValueError, match="duplicate semantic task"):
        generator.assert_unique_policy_issue_semantics([scenario, duplicate])


def test_policy_semantic_hash_ignores_opaque_identity_and_neutral_layout() -> None:
    scenario = read_jsonl(EVAL_SPLIT)[0]
    relabeled = copy.deepcopy(scenario)
    relabeled["scenario_id"] = "identity-only-relabel"
    doc_ids = {
        document["doc_id"]: f"doc-relabeled-{index}"
        for index, document in enumerate(relabeled["documents"])
    }
    for index, document in enumerate(relabeled["documents"]):
        document["doc_id"] = doc_ids[document["doc_id"]]
        document["title"] = f"Neutral title {index}"
        document["style"] = "neutral"
    faction_ids = {
        faction["faction_id"]: f"group-relabeled-{index}"
        for index, faction in enumerate(relabeled["factions"])
    }
    for faction in relabeled["factions"]:
        faction["faction_id"] = faction_ids[faction["faction_id"]]
    relabeled["persona_panel"]["faction_ids"] = [
        faction_ids[faction_id]
        for faction_id in relabeled["persona_panel"]["faction_ids"]
    ]
    for index, plant in enumerate(relabeled["planted_items"]):
        plant["plant_id"] = f"issue-relabeled-{index}"
        plant["doc_id"] = doc_ids[plant["doc_id"]]
        plant["target_stances"] = {
            faction_ids[faction_id]: stance
            for faction_id, stance in plant["target_stances"].items()
        }
        if plant["related_evidence"] is not None:
            plant["related_evidence"]["doc_id"] = doc_ids[
                plant["related_evidence"]["doc_id"]
            ]
    for distractor in relabeled["distractors"]:
        distractor["doc_id"] = doc_ids[distractor["doc_id"]]

    assert generator.instance_fingerprint(relabeled) != generator.instance_fingerprint(
        scenario
    )
    assert generator.policy_issue_semantics(
        relabeled
    ) == generator.policy_issue_semantics(scenario)


def test_policy_semantic_hash_changes_with_unresolved_decision() -> None:
    scenario = read_jsonl(EVAL_SPLIT)[0]
    changed = copy.deepcopy(scenario)
    changed["planted_items"][0]["canonical_question"] = (
        "Should a materially different policy decision control?"
    )

    assert generator.policy_issue_semantics(
        changed
    ) != generator.policy_issue_semantics(scenario)


def test_minimal_model_free_baselines_and_corpus_diagnostics(tmp_path: Path) -> None:
    _, eval_path = generator.write_splits(tmp_path)

    computed = floors.compute_elicit_floors(eval_path)

    assert set(computed) == {
        "find/random-visible-sentences",
        "find/longest-visible-sentences",
        "find/exact-ceiling",
        "elicit-ask/uniform-random-candidate",
        "elicit-ask/runner-up-candidate",
        "elicit-ask/exact-answer-ceiling",
        "audit/top1-tie-rate",
        "audit/top1-normalized-margin-min",
        "audit/issue-class-min-proportion",
        "audit/issue-class-max-proportion",
    }
    assert computed["find/random-visible-sentences"] < 0.10
    assert computed["find/longest-visible-sentences"] < 0.10
    assert computed["find/exact-ceiling"] == 1.0
    assert 0.0 < computed["elicit-ask/uniform-random-candidate"] < 1.0
    assert 0.0 < computed["elicit-ask/runner-up-candidate"] < 1.0
    assert computed["elicit-ask/exact-answer-ceiling"] == 1.0
    assert computed["audit/top1-tie-rate"] == 0.0
    assert computed["audit/top1-normalized-margin-min"] > 0.0
    assert computed["audit/issue-class-min-proportion"] == pytest.approx(1 / 3)
    assert computed["audit/issue-class-max-proportion"] == pytest.approx(1 / 3)

    rendered = floors.render_markdown(computed)
    assert "Random visible sentences" in rendered
    assert "Uniform candidate + exact components" in rendered
    assert "Exact Ask answer ceiling" in rendered


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
