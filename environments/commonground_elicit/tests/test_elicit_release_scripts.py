from __future__ import annotations

import asyncio
import copy
import importlib.util
import json
import random
import re
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
README = Path(__file__).resolve().parents[1] / "README.md"
DATA_README = DATA_DIR / "README.md"


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

        assert related["quote"] == authored["related_anchor_quote"]
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
    assert len({generator.prompt_fingerprint(scenario) for scenario in heldout}) == 100
    assert len({generator.answer_fingerprint(scenario) for scenario in heldout}) == 100
    assert (
        len({generator.instance_fingerprint(scenario) for scenario in heldout}) == 100
    )
    assert (
        len({generator.canonical_prompt_semantics(scenario) for scenario in heldout})
        == 100
    )
    assert len({generator.structural_signature(scenario) for scenario in heldout}) >= 95
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
    similarity, train_id, eval_id = generator.maximum_cross_split_lexical_jaccard(
        train, heldout
    )
    assert similarity <= generator.MAX_CROSS_SPLIT_LEXICAL_JACCARD, (
        similarity,
        train_id,
        eval_id,
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
        generator.assert_unique_semantic_tasks([scenario, duplicate])


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
    assert generator.canonical_prompt_semantics(
        relabeled
    ) == generator.canonical_prompt_semantics(scenario)
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


def test_removed_summary_codebook_baseline_detects_the_old_leak() -> None:
    scenario = read_jsonl(EVAL_SPLIT)[0]
    factions = floors.public_factions(scenario)
    tendency = {
        "agree": "leans toward yes",
        "disagree": "leans toward no",
        "pass": "has no settled position",
    }
    for faction in factions:
        faction_id = faction["faction_id"]
        clauses = [
            "For decisions involving "
            f"{', '.join(_old_decision_terms(plant))}, "
            f"{tendency[plant['target_stances'][faction_id]]}."
            for plant in scenario["planted_items"]
        ]
        faction["summary"] = f"{faction['summary']} {' '.join(clauses)}"

    questions = floors.leaked_summary_codebook_questions(
        floors.public_documents(scenario),
        factions,
        question_count=floors.QUESTION_COUNT,
    )
    expected = sorted(
        scenario["planted_items"],
        key=lambda plant: (
            -floors.leaked_keyword_decision_value(
                str(plant["canonical_question"]), str(plant["anchor_quote"])
            )
            * floors.panel_disagreement(plant["target_stances"]),
            str(plant["doc_id"]),
            str(plant["anchor_quote"]),
        ),
    )[: floors.QUESTION_COUNT]

    assert len(questions) == floors.QUESTION_COUNT
    assert [question["doc_id"] for question in questions] == [
        plant["doc_id"] for plant in expected
    ]
    assert [question["target_stances"] for question in questions] == [
        plant["target_stances"] for plant in expected
    ]


def test_removed_0_4_principle_parser_recovers_old_stances_but_not_new_prose() -> None:
    scenario = read_jsonl(EVAL_SPLIT)[0]
    current_factions = floors.public_factions(scenario)
    documents_by_id = {
        document["doc_id"]: document["text"] for document in scenario["documents"]
    }
    planted = [
        {
            "doc_id": plant["doc_id"],
            "quote": plant["anchor_quote"],
            "type": plant["type"],
            "question": plant["canonical_question"],
            "yes_choice": plant["canonical_yes_choice"],
            "related_evidence": plant["related_evidence"],
            "target_stances": plant["target_stances"],
            "alternative_stances": plant["alternative_stances"],
            "decision_value": plant["decision_value"],
            "document_text": documents_by_id[plant["doc_id"]],
            "related_document_text": (
                documents_by_id[plant["related_evidence"]["doc_id"]]
                if plant["related_evidence"] is not None
                else None
            ),
        }
        for plant in scenario["planted_items"]
    ]

    assert (
        floors.removed_0_4_principle_codebook_questions(
            planted,
            current_factions,
            question_count=floors.QUESTION_COUNT,
        )
        == []
    )

    old_factions = copy.deepcopy(current_factions)
    for faction in old_factions:
        faction_id = faction["faction_id"]
        clauses = [
            floors._REMOVED_0_4_PRINCIPLES[
                (plant["type"], plant["target_stances"][faction_id])
            ][0]
            for plant in scenario["planted_items"]
        ]
        faction["summary"] = f"Historical fixture. {' '.join(clauses)}"

    questions = floors.removed_0_4_principle_codebook_questions(
        planted,
        old_factions,
        question_count=floors.QUESTION_COUNT,
    )
    score = asyncio.run(
        floors.question_utility(
            floors.completion_for({"questions": questions}),
            {"questions": planted},
            {
                "panel_polarization": 1.0,
                "question_count": floors.QUESTION_COUNT,
                "allow_combined_questions": False,
            },
            floors.ElicitJsonParser("questions"),
        )
    )

    assert score == 1.0

    source_questions = floors.source_template_0_4_codebook_questions(
        floors.public_documents(scenario),
        old_factions,
        question_count=floors.QUESTION_COUNT,
    )
    source_score = asyncio.run(
        floors.question_utility(
            floors.completion_for({"questions": source_questions}),
            {"questions": planted},
            {
                "panel_polarization": 1.0,
                "question_count": floors.QUESTION_COUNT,
                "allow_combined_questions": False,
            },
            floors.ElicitJsonParser("questions"),
        )
    )

    assert source_score == 1.0


def test_compute_elicit_floors_reproduces_published_values() -> None:
    computed = floors.compute_elicit_floors(EVAL_SPLIT)

    assert computed == {
        "find/random-span": 0.07999999999999997,
        "find/vague-sounding": 0.19499999999999995,
        "find/legacy-0.2-codebook": 0.0,
        "elicit-ask/template-question": 0.07825584279459283,
        "elicit-ask/randomly-targeted": 0.06941666666666667,
        "elicit-ask/legacy-0.3-summary-codebook": 0.0,
        "elicit-ask/legacy-0.4-principle-codebook": 0.0,
        "elicit-ask/source-template-0.4-principle-codebook": 0.0,
        "elicit-ask/exact-issue-random-stance": 0.6700602815554378,
        "elicit-ask/exact-issue-exact-stance": 1.0,
    }
    rendered = floors.render_markdown(computed)
    assert rendered == "\n".join(
        [
            "| Comparator class | Task | Comparator | mean reward |",
            "| --- | --- | --- | ---: |",
            "| Prompt-observable | find | Random visible spans | 0.080 |",
            "| Prompt-observable | find | Flag vague-sounding spans | 0.195 |",
            "| Prompt-observable | find | Legacy 0.2 document-ID/position codebook | 0.000 |",
            "| Prompt-observable | elicit-ask | Template clarity questions | 0.078 |",
            "| Prompt-observable | elicit-ask | Randomly targeted questions | 0.069 |",
            "| Prompt-observable | elicit-ask | Removed 0.3 summary/stance codebook | 0.000 |",
            "| Component oracle | elicit-ask | Exact issues + removed 0.4 principle-table parser | 0.000 |",
            "| Source-aware prompt-only | elicit-ask | Public template detector + removed 0.4 principle-table parser | 0.000 |",
            "| Component oracle | elicit-ask | Exact top-K issues + random stances | 0.670 |",
            "| Component oracle | elicit-ask | Exact top-K issues + exact stances (ceiling) | 1.000 |",
        ]
    )
    assert rendered in README.read_text(encoding="utf-8")
    assert rendered in DATA_README.read_text(encoding="utf-8")


def _old_decision_terms(plant: dict[str, Any]) -> list[str]:
    """Reproduce the removed 0.3 signature extraction for a fixture."""

    stopwords = {
        "after",
        "allowed",
        "before",
        "could",
        "decide",
        "does",
        "during",
        "each",
        "from",
        "have",
        "instead",
        "into",
        "must",
        "should",
        "than",
        "that",
        "their",
        "them",
        "this",
        "under",
        "what",
        "when",
        "which",
        "with",
        "without",
        "would",
    }
    terms = [
        token.casefold()
        for token in re.findall(r"[^\W_]+", plant["canonical_question"])
        if len(token) >= 4 and token.casefold() not in stopwords
    ]
    return list(dict.fromkeys(terms))[:8]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
