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
    assert len({generator.prompt_fingerprint(scenario) for scenario in heldout}) == 100
    assert len({generator.answer_fingerprint(scenario) for scenario in heldout}) == 100
    assert (
        len({generator.instance_fingerprint(scenario) for scenario in heldout}) == 100
    )
    assert (
        len({generator.canonical_prompt_semantics(scenario) for scenario in heldout})
        == 100
    )
    # Varied two-to-seven-sentence layouts reduce fixed sentence-count signal.
    # Keep broad independent structure coverage while auditing residual label
    # and length signal directly in the model-free corpus gates below.
    assert len({generator.structural_signature(scenario) for scenario in heldout}) >= 90
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
    assert all(
        set(document) == {"doc_id", "title", "style", "text"} for document in documents
    )
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
    planted = []
    for plant in scenario["planted_items"]:
        decision = floors._canonical_decision_frame(plant, scenario["documents"])
        yes_choice = plant["canonical_yes_choice"]
        planted.append(
            {
                "doc_id": plant["doc_id"],
                "quote": plant["anchor_quote"],
                "type": plant["type"],
                "question": floors._oriented_decision_question(
                    plant["canonical_question"], decision, yes_choice
                ),
                "decision": decision,
                "yes_choice": yes_choice,
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
        )

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
        faction["summary"] = (
            f"{faction['summary']} Historical fixture. {' '.join(clauses)}"
        )

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


def test_compute_elicit_floors_enforces_corpus_quality_gates(
    tmp_path: Path,
) -> None:
    _, eval_path = generator.write_splits(tmp_path)
    computed = floors.compute_elicit_floors(eval_path)

    assert computed["find/removed-fixed-filler-marker"] == 0.0
    assert computed["find/removed-fixed-filler-marker-localization-recall"] == 0.0
    assert computed["find/removed-fixed-filler-marker-type-accuracy"] == 0.0
    assert (
        computed["find/removed-fixed-filler-marker-localization-component-oracle-f1"]
        == 0.0
    )
    assert computed["find/layout-position-length"] < 0.05
    assert computed["find/layout-position-length-localization-recall"] < 0.25
    assert computed["find/layout-position-length-type-accuracy"] < 0.20
    assert (
        computed["find/layout-position-length-localization-component-oracle-f1"] < 0.25
    )
    assert computed["find/longest-visible-sentences"] == 0.0
    # This is a genuine prompt-only attack: it selects the three longest
    # visible sentences without source, template, or retired-pool knowledge.
    # Length-balanced compositional distractors must keep its localization and
    # conditional type signal near zero, not merely its strict semantic F1.
    assert computed["find/longest-visible-sentences-localization-recall"] < 0.10
    assert computed["find/longest-visible-sentences-type-accuracy"] < 0.10
    assert (
        computed["find/longest-visible-sentences-localization-component-oracle-f1"]
        < 0.10
    )
    assert computed["find/shared-predicate-exclusion"] == 0.0
    assert computed["find/shared-predicate-exclusion-localization-recall"] == 0.0
    assert computed["find/shared-predicate-exclusion-type-accuracy"] == 0.0
    assert computed["find/sector-team-marker-localization-recall"] < 0.05
    assert computed["find/sector-team-marker-localization-f1"] < 0.05
    assert computed["find/sector-team-marker-localization-component-oracle-f1"] < 0.05
    # Selection is one of three candidates, so a random selector with perfect
    # per-issue content remains a deliberately strong diagnostic but must leave
    # material headroom below the exact top-one ceiling.
    assert computed["elicit-ask/random-issue-exact-components"] < 0.90
    assert computed["elicit-ask/random-issue-top1-selection-accuracy"] == pytest.approx(
        1 / 3
    )
    assert 0.0 < computed["elicit-ask/runner-up-exact-components"] < 1.0
    assert computed["elicit-ask/runner-up-top1-selection-accuracy"] == 0.0
    assert computed["elicit-ask/public-profile-composition"] == 1.0
    assert computed["elicit-ask/public-profile-top1-selection-accuracy"] == 1.0
    assert computed["elicit-ask/exact-issue-exact-stance"] == 1.0
    assert computed["elicit-ask/exact-issue-top1-selection-accuracy"] == 1.0
    assert computed["find/public-source-replay"] == 1.0
    assert computed["elicit-ask/public-source-replay"] == 1.0
    assert computed["audit/find-old-fixed-marker-scenario-rate"] == 0.0
    assert computed["audit/find-shared-predicate-plant-rate"] == 1.0
    assert computed["audit/find-shared-predicate-distractor-rate"] == 1.0
    assert computed["audit/find-shared-predicate-exclusive-advantage"] == 0.0
    assert computed["audit/find-sector-team-plant-rate"] < 0.05
    assert computed["audit/find-sector-team-distractor-rate"] > 0.0
    assert computed["audit/find-sector-team-exclusive-advantage"] < 0.10
    assert computed["audit/elicit-ask-top-k-tie-rate"] == 0.0
    assert computed["audit/elicit-ask-top-k-gap-min"] > 0.0
    assert (
        computed["audit/elicit-ask-top1-normalized-margin-min"]
        >= floors.MIN_TOP1_NORMALIZED_MARGIN
    )
    for feature in (
        "title",
        "style",
        "sentence-position",
        "sentence-count",
        "document-length",
    ):
        assert computed[f"audit/find-{feature}-majority-label-accuracy"] < 0.45
    assert computed["audit/find-issue-class-min-proportion"] == pytest.approx(1 / 3)
    assert computed["audit/find-issue-class-max-proportion"] == pytest.approx(1 / 3)
    assert computed["audit/find-issue-balanced-chance"] == pytest.approx(1 / 3)
    assert (
        computed["audit/find-combined-title-length-position-loto-balanced-accuracy"]
        < 0.45
    )
    assert computed["audit/find-document-role-class-min-proportion"] == pytest.approx(
        0.20
    )
    assert computed["audit/find-document-role-class-max-proportion"] == pytest.approx(
        0.20
    )
    assert computed["audit/find-document-role-balanced-chance"] == pytest.approx(0.20)
    assert computed["audit/find-document-role-structure-loto-balanced-accuracy"] < 0.30
    assert computed["audit/find-related-document-positive-rate"] == pytest.approx(0.20)
    assert computed[
        "audit/find-related-document-class-min-proportion"
    ] == pytest.approx(0.20)
    assert computed[
        "audit/find-related-document-class-max-proportion"
    ] == pytest.approx(0.80)
    assert computed["audit/find-related-document-balanced-chance"] == pytest.approx(
        0.50
    )
    assert (
        computed["audit/find-related-document-structure-loto-balanced-accuracy"] < 0.60
    )

    rendered = floors.render_markdown(computed)
    assert "Removed fixed filler/type marker" in rendered
    assert "Longest-sentence localization recall" in rendered
    assert "Exclude current shared procedural predicates" in rendered
    assert "shared-predicate rate gap" in rendered
    assert "Sector-team marker localization F1" in rendered
    assert "locator + exact non-localization components" in rendered
    assert "sector-team rate gap" in rendered
    assert "Combined title/length/anchor-position LOTO balanced accuracy" in rendered
    assert "Helper document-role LOTO balanced accuracy" in rendered
    assert "Related-document LOTO balanced accuracy" in rendered
    assert "Source-aware memorization ceiling" in rendered
    assert "Exact uniform-random issue expectation + exact components" in rendered
    assert "Exact runner-up issue + exact per-issue components" in rendered
    assert "Exact grounding + public-profile stance/rank composition" in rendered
    assert "Minimum normalized top-1 utility margin" in rendered
    assert "Top-1 boundary tie rate" in rendered
    readme = (ROOT / "environments" / "commonground_elicit" / "README.md").read_text()
    assert rendered.strip() in readme


def test_localization_component_oracle_preserves_false_positives() -> None:
    scenario = generator.generate_scenario(8200, generator.HELDOUT_TEMPLATES[0])
    answer = floors.exact_answer_for_scenario(scenario)
    exact = answer["findings"][0]
    exact_tokens = re.findall(r"[^\W_]+", exact["quote"])
    assert len(exact_tokens) >= 10
    localized_subspan = " ".join(exact_tokens[:-1])
    wrong_semantics = {
        "doc_id": exact["doc_id"],
        "quote": localized_subspan,
        "type": next(
            issue_type
            for issue_type in floors.FINDING_TYPES
            if issue_type != exact["type"]
        ),
        "diagnosis": "Should an unrelated process be changed?",
        "decision": floors.visible_decision_frame(
            exact["quote"], "Should an unrelated process be changed?"
        ),
        "related_evidence": None,
    }
    planted_spans = {
        (finding["doc_id"], finding["quote"]) for finding in answer["findings"]
    }
    false_doc_id, false_quote = next(
        (document["doc_id"], sentence)
        for document in floors.public_documents(scenario)
        for sentence in floors.split_sentences(document["text"])
        if (document["doc_id"], sentence) not in planted_spans
    )
    false_diagnosis = floors.sentence_question(false_quote)
    false_positive = {
        "doc_id": false_doc_id,
        "quote": false_quote,
        "type": "ambiguity",
        "diagnosis": false_diagnosis,
        "decision": floors.visible_decision_frame(false_quote, false_diagnosis),
        "related_evidence": None,
    }

    completed = floors.localization_component_oracle_findings(
        [wrong_semantics, false_positive], answer["findings"]
    )
    scores = floors.match_findings(completed, answer["findings"])

    assert completed[0]["type"] == exact["type"]
    assert completed[0]["decision"] == exact["decision"]
    assert completed[0]["quote"] == localized_subspan
    assert completed[1] == false_positive
    assert scores["f1"] == scores["localization_f1"] == pytest.approx(0.4)


def test_structural_knn_reports_balanced_accuracy_not_majority_accuracy() -> None:
    rows = [
        (template, ("same-title",), (0.0, 1.0, 10.0), label)
        for template in ("template-a", "template-b")
        for label in ("other", "other", "other", "other", "related")
    ]

    assert floors._leave_one_template_out_knn_balanced_accuracy(rows) == 0.5


def test_structural_knn_holds_out_the_complete_template_family() -> None:
    rows = [
        ("template-a", ("title-a",), (0.0,), "ambiguity"),
        ("template-a", ("title-b",), (0.0,), "gap"),
        ("template-b", ("title-a",), (0.0,), "gap"),
        ("template-b", ("title-b",), (0.0,), "ambiguity"),
    ]

    # Each held-out family reverses the only mapping available in training. An
    # in-sample nearest-neighbor implementation would score one instead.
    assert floors._leave_one_template_out_knn_balanced_accuracy(rows) == 0.0


def test_helper_document_role_audit_fails_when_neutral_role_is_missing() -> None:
    helper_template = next(
        template
        for template in generator.HELDOUT_TEMPLATES
        if template.balance_type_neutral_distractors
    )
    scenario = generator.generate_scenario(8210, helper_template)
    occupied_doc_ids = {str(plant["doc_id"]) for plant in scenario["planted_items"]} | {
        str(plant["related_evidence"]["doc_id"])
        for plant in scenario["planted_items"]
        if plant["related_evidence"] is not None
    }
    neutral_doc_id = next(
        str(document["doc_id"])
        for document in scenario["documents"]
        if str(document["doc_id"]) not in occupied_doc_ids
    )
    scenario["documents"] = [
        document
        for document in scenario["documents"]
        if str(document["doc_id"]) != neutral_doc_id
    ]

    with pytest.raises(ValueError, match="must render five documents"):
        floors.compute_corpus_audits([scenario])


def test_uniform_selection_expectation_enumerates_every_candidate_exactly() -> None:
    scenario = generator.generate_scenario(8200, generator.HELDOUT_TEMPLATES[0])
    answer = floors.exact_answer_for_scenario(scenario)
    planted = answer["questions"]
    info = {
        "panel_polarization": 1.0,
        "question_count": floors.QUESTION_COUNT,
        "allow_combined_questions": False,
    }
    enumerated = [
        asyncio.run(
            floors.question_utility(
                floors.completion_for(
                    {"questions": [floors._candidate_question(plant)]}
                ),
                answer,
                info,
                floors.ElicitJsonParser("questions"),
            )
        )
        for plant in planted
    ]

    assert floors.exact_uniform_selection_expected_reward(planted) == pytest.approx(
        sum(enumerated) / len(enumerated)
    )


def test_public_profile_composition_recovers_top_issue_and_stances() -> None:
    scenario = generator.generate_scenario(8200, generator.HELDOUT_TEMPLATES[0])
    answer = floors.exact_answer_for_scenario(scenario)
    composed = floors.public_profile_composition_questions(
        answer["questions"],
        floors.public_factions(scenario),
        question_count=floors.QUESTION_COUNT,
    )
    exact = floors.exact_issue_exact_stance_questions(
        answer["questions"],
        question_count=floors.QUESTION_COUNT,
    )

    assert composed == exact
    assert floors.top1_selection_accuracy(composed, answer["questions"]) == 1.0


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
