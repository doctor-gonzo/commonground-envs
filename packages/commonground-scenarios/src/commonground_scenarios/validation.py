"""Strict validation for Common Ground scenario documents."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from importlib.resources import files
import hashlib
import json
from math import isfinite
import re
from typing import Any


SCENARIO_FIELDS = {
    "scenario_id",
    "organization",
    "factions",
    "documents",
    "planted_items",
    "distractors",
    "persona_panel",
    "human_feedback",
    "provenance",
}
CE_SNAPSHOT_FIELDS = {
    "session_id",
    "statements",
    "participants",
    "votes",
    "masked_cells",
    "held_out",
    "clusters",
    "stats",
    "meta",
}
PLANT_TYPES = {"ambiguity", "contradiction", "gap"}
STANCES = {"agree", "disagree", "pass"}
ID_PATTERN = re.compile(r"^[a-z][a-z0-9-]*$")
PASS_THRESHOLD = 0.25


class ScenarioValidationError(ValueError):
    """Raised when a scenario violates the committed schema contract."""


def load_scenario_schema() -> dict[str, Any]:
    """Load the packaged JSON Schema document."""

    schema_text = files("commonground_scenarios").joinpath("schema/scenario.schema.json").read_text(
        encoding="utf-8"
    )
    schema = json.loads(schema_text)
    if not isinstance(schema, dict):
        raise ScenarioValidationError("packaged scenario schema must be an object")
    return schema


def validate_scenario(scenario: Mapping[str, Any]) -> None:
    """Validate one synthetic or human-socket scenario without dependencies."""

    root = _exact_object(scenario, SCENARIO_FIELDS, "scenario")
    scenario_id = _identifier(root["scenario_id"], "scenario_id")
    organization = _exact_object(
        root["organization"], {"name", "sector", "fictional"}, "organization"
    )
    _nonempty_text(organization["name"], "organization.name")
    _nonempty_text(organization["sector"], "organization.sector")
    if type(organization["fictional"]) is not bool:
        raise ScenarioValidationError("organization.fictional must be boolean")

    factions = _validate_factions(root["factions"])
    documents = _validate_documents(root["documents"])
    panel = _validate_panel(root["persona_panel"], factions)
    _validate_plants(root["planted_items"], factions, documents)
    _validate_distractors(root["distractors"], root["planted_items"], documents)
    provenance = _validate_provenance(root["provenance"], scenario_id)

    if provenance["synthetic"]:
        if organization["fictional"] is not True:
            raise ScenarioValidationError("synthetic scenarios require a fictional organization")
        if provenance["generation_mode"] not in {"template", "operator-polished"}:
            raise ScenarioValidationError("synthetic scenarios require template generation provenance")
        if panel is None:
            raise ScenarioValidationError("synthetic scenarios require persona_panel")
        if root["human_feedback"] is not None:
            raise ScenarioValidationError("synthetic scenarios cannot contain human_feedback")
    else:
        if organization["fictional"] is not False:
            raise ScenarioValidationError("human scenarios require a non-fictional organization")
        if provenance["generation_mode"] != "human":
            raise ScenarioValidationError("human scenarios require human generation provenance")
        if panel is not None:
            raise ScenarioValidationError("human scenarios replace persona_panel with human_feedback")
        _validate_human_feedback(root["human_feedback"])


def _validate_factions(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list) or not 3 <= len(value) <= 5:
        raise ScenarioValidationError("factions must contain three to five entries")
    factions: dict[str, dict[str, Any]] = {}
    for index, raw_faction in enumerate(value):
        faction = _exact_object(
            raw_faction,
            {"faction_id", "name", "summary", "priors"},
            f"factions[{index}]",
        )
        faction_id = _identifier(faction["faction_id"], f"factions[{index}].faction_id")
        if faction_id in factions:
            raise ScenarioValidationError(f"duplicate faction_id: {faction_id}")
        _nonempty_text(faction["name"], f"factions[{index}].name")
        _nonempty_text(faction["summary"], f"factions[{index}].summary")
        priors = faction["priors"]
        if not isinstance(priors, Mapping) or len(priors) < 3:
            raise ScenarioValidationError(f"factions[{index}].priors must define planted dimensions")
        normalized_priors: dict[str, float] = {}
        for dimension, prior in priors.items():
            dimension_id = _identifier(dimension, f"factions[{index}].priors key")
            if isinstance(prior, bool) or not isinstance(prior, (int, float)):
                raise ScenarioValidationError(f"prior {dimension_id} must be numeric")
            numeric_prior = float(prior)
            if not isfinite(numeric_prior) or not -1 <= numeric_prior <= 1:
                raise ScenarioValidationError(f"prior {dimension_id} must be finite and within [-1, 1]")
            normalized_priors[dimension_id] = numeric_prior
        factions[faction_id] = {**faction, "priors": normalized_priors}
    return factions


def _validate_documents(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list) or not 3 <= len(value) <= 8:
        raise ScenarioValidationError("documents must contain three to eight entries")
    documents: dict[str, dict[str, Any]] = {}
    for index, raw_document in enumerate(value):
        document = _exact_object(
            raw_document, {"doc_id", "title", "style", "text"}, f"documents[{index}]"
        )
        doc_id = _identifier(document["doc_id"], f"documents[{index}].doc_id")
        if doc_id in documents:
            raise ScenarioValidationError(f"duplicate doc_id: {doc_id}")
        for field in ("title", "style", "text"):
            _nonempty_text(document[field], f"documents[{index}].{field}")
        documents[doc_id] = document
    return documents


def _validate_panel(
    value: Any, factions: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any] | None:
    if value is None:
        return None
    panel = _exact_object(
        value, {"vote_rule", "pass_threshold", "faction_ids"}, "persona_panel"
    )
    if panel["vote_rule"] != "dimension-threshold-v1":
        raise ScenarioValidationError("persona_panel.vote_rule must be dimension-threshold-v1")
    threshold = panel["pass_threshold"]
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ScenarioValidationError("persona_panel.pass_threshold must be numeric")
    threshold = float(threshold)
    if threshold != PASS_THRESHOLD:
        raise ScenarioValidationError(f"persona_panel.pass_threshold must equal {PASS_THRESHOLD}")
    faction_ids = panel["faction_ids"]
    if not isinstance(faction_ids, list) or faction_ids != list(factions):
        raise ScenarioValidationError("persona_panel.faction_ids must match faction order")
    return {**panel, "pass_threshold": threshold}


def _validate_plants(
    value: Any,
    factions: Mapping[str, Mapping[str, Any]],
    documents: Mapping[str, Mapping[str, Any]],
) -> None:
    if not isinstance(value, list) or len(value) < 3:
        raise ScenarioValidationError("planted_items must contain the planted answer key")
    seen_ids: set[str] = set()
    seen_types: set[str] = set()
    for index, raw_plant in enumerate(value):
        plant = _exact_object(
            raw_plant,
            {
                "plant_id",
                "doc_id",
                "anchor_quote",
                "type",
                "canonical_question",
                "target_dimension",
                "target_stances",
            },
            f"planted_items[{index}]",
        )
        plant_id = _identifier(plant["plant_id"], f"planted_items[{index}].plant_id")
        if plant_id in seen_ids:
            raise ScenarioValidationError(f"duplicate plant_id: {plant_id}")
        seen_ids.add(plant_id)
        if plant["type"] not in PLANT_TYPES:
            raise ScenarioValidationError(f"invalid planted type: {plant['type']!r}")
        seen_types.add(plant["type"])
        doc_id = _identifier(plant["doc_id"], f"planted_items[{index}].doc_id")
        if doc_id not in documents:
            raise ScenarioValidationError(f"plant references unknown doc_id: {doc_id}")
        anchor = _nonempty_text(plant["anchor_quote"], f"planted_items[{index}].anchor_quote")
        if anchor not in documents[doc_id]["text"]:
            raise ScenarioValidationError(f"plant anchor is absent from document: {plant_id}")
        _nonempty_text(plant["canonical_question"], f"planted_items[{index}].canonical_question")
        dimension = _identifier(
            plant["target_dimension"], f"planted_items[{index}].target_dimension"
        )
        stances = plant["target_stances"]
        if not isinstance(stances, Mapping) or set(stances) != set(factions):
            raise ScenarioValidationError(f"target_stances must cover every faction: {plant_id}")
        if any(stance not in STANCES for stance in stances.values()):
            raise ScenarioValidationError(f"invalid target stance: {plant_id}")
        expected = {
            faction_id: _stance_for(
                faction["priors"].get(dimension), PASS_THRESHOLD, dimension
            )
            for faction_id, faction in factions.items()
        }
        if dict(stances) != expected:
            raise ScenarioValidationError(f"target_stances do not match faction priors: {plant_id}")
        if not {"agree", "disagree"}.issubset(set(stances.values())):
            raise ScenarioValidationError(f"plant must mark latently split factions: {plant_id}")
    if seen_types != PLANT_TYPES:
        raise ScenarioValidationError("planted_items must include ambiguity, contradiction, and gap")


def _validate_distractors(
    value: Any,
    planted_items: Any,
    documents: Mapping[str, Mapping[str, Any]],
) -> None:
    if not isinstance(value, list) or not value:
        raise ScenarioValidationError("distractors must be a non-empty list")
    planted_anchors = {plant["anchor_quote"] for plant in planted_items}
    for index, raw_distractor in enumerate(value):
        distractor = _exact_object(
            raw_distractor, {"doc_id", "anchor_quote", "reason"}, f"distractors[{index}]"
        )
        doc_id = _identifier(distractor["doc_id"], f"distractors[{index}].doc_id")
        if doc_id not in documents:
            raise ScenarioValidationError(f"distractor references unknown doc_id: {doc_id}")
        anchor = _nonempty_text(distractor["anchor_quote"], f"distractors[{index}].anchor_quote")
        if anchor not in documents[doc_id]["text"]:
            raise ScenarioValidationError("distractor anchor is absent from its document")
        if anchor in planted_anchors:
            raise ScenarioValidationError("distractor cannot duplicate a planted anchor")
        _nonempty_text(distractor["reason"], f"distractors[{index}].reason")


def _validate_provenance(value: Any, scenario_id: str) -> dict[str, Any]:
    provenance = _exact_object(
        value,
        {
            "seed",
            "template_id",
            "template_set",
            "generated_at",
            "synthetic",
            "generation_mode",
        },
        "provenance",
    )
    seed = _json_integer(provenance["seed"], "provenance.seed")
    template_id = _identifier(provenance["template_id"], "provenance.template_id")
    expected_scenario_id = scenario_id_for(template_id, seed)
    if scenario_id != expected_scenario_id:
        raise ScenarioValidationError("scenario_id must match provenance template and seed")
    if provenance["template_set"] not in {"train", "heldout"}:
        raise ScenarioValidationError("provenance.template_set must be train or heldout")
    canonical_date(provenance["generated_at"])
    if type(provenance["synthetic"]) is not bool:
        raise ScenarioValidationError("provenance.synthetic must be boolean")
    if provenance["generation_mode"] not in {"template", "operator-polished", "human"}:
        raise ScenarioValidationError("invalid provenance.generation_mode")
    return provenance


def _validate_human_feedback(value: Any) -> None:
    snapshot = _exact_object(value, CE_SNAPSHOT_FIELDS, "human_feedback")
    _nonempty_text(snapshot["session_id"], "human_feedback.session_id")
    statements = snapshot["statements"]
    participants = snapshot["participants"]
    votes = snapshot["votes"]
    if not isinstance(statements, list) or not statements:
        raise ScenarioValidationError("human_feedback.statements must be non-empty")
    for index, statement in enumerate(statements):
        statement_object = _exact_object(
            statement, {"index", "text"}, f"human_feedback.statements[{index}]"
        )
        if _json_integer(
            statement_object["index"], f"human_feedback.statements[{index}].index"
        ) != index:
            raise ScenarioValidationError("human_feedback statement indices must be positional")
        _nonempty_text(statement_object["text"], f"human_feedback.statements[{index}].text")
    if not isinstance(participants, list) or not participants or not all(
        isinstance(participant, str) and participant for participant in participants
    ):
        raise ScenarioValidationError("human_feedback.participants must be non-empty strings")
    if len(set(participants)) != len(participants):
        raise ScenarioValidationError("human_feedback.participants must be unique")
    if not isinstance(votes, list) or len(votes) != len(participants):
        raise ScenarioValidationError("human_feedback.votes must be participant-major")
    statement_count = len(statements)
    for row in votes:
        if not isinstance(row, list) or len(row) != statement_count:
            raise ScenarioValidationError("human_feedback.votes must be rectangular")
        for vote in row:
            if vote is None:
                continue
            if _json_integer(vote, "human_feedback vote") not in {-1, 0, 1}:
                raise ScenarioValidationError("human_feedback contains an invalid vote")
    masked_cells = snapshot["masked_cells"]
    if not isinstance(masked_cells, list):
        raise ScenarioValidationError("human_feedback.masked_cells must be an array")
    normalized_masked_cells: set[str] = set()
    for cell in masked_cells:
        if not isinstance(cell, list) or len(cell) != 2:
            raise ScenarioValidationError("human_feedback contains an invalid masked cell")
        participant_index = _json_integer(cell[0], "human_feedback masked participant index")
        statement_index = _json_integer(cell[1], "human_feedback masked statement index")
        if not 0 <= participant_index < len(participants) or not 0 <= statement_index < statement_count:
            raise ScenarioValidationError("human_feedback contains an invalid masked cell")
        cell_id = f"{participant_index},{statement_index}"
        if cell_id in normalized_masked_cells:
            raise ScenarioValidationError("human_feedback contains a duplicate masked cell")
        normalized_masked_cells.add(cell_id)
        if votes[participant_index][statement_index] is not None:
            raise ScenarioValidationError("human_feedback masked votes must be null")
    held_out = snapshot["held_out"]
    if not isinstance(held_out, Mapping):
        raise ScenarioValidationError("human_feedback.held_out must be an object")
    for cell_id, vote in held_out.items():
        if not isinstance(cell_id, str) or not re.fullmatch(r"\d+,\d+", cell_id):
            raise ScenarioValidationError("human_feedback contains an invalid held-out cell")
        participant_index, statement_index = (int(index) for index in cell_id.split(","))
        if not 0 <= participant_index < len(participants) or not 0 <= statement_index < statement_count:
            raise ScenarioValidationError("human_feedback contains an out-of-bounds held-out cell")
        if _json_integer(vote, "human_feedback held-out vote") not in {-1, 0, 1}:
            raise ScenarioValidationError("human_feedback contains an invalid held-out vote")
    if set(held_out) != normalized_masked_cells:
        raise ScenarioValidationError("human_feedback held_out must match masked_cells")
    clusters = snapshot["clusters"]
    if not isinstance(clusters, list) or not all(isinstance(cluster, Mapping) for cluster in clusters):
        raise ScenarioValidationError("human_feedback.clusters must be an array")
    if not isinstance(snapshot["stats"], Mapping):
        raise ScenarioValidationError("human_feedback.stats must be an object")
    meta = snapshot["meta"]
    if not isinstance(meta, Mapping):
        raise ScenarioValidationError("human_feedback.meta must be an object")
    if meta.get("synthetic") is not False:
        raise ScenarioValidationError("human_feedback.meta.synthetic must be false")


def scenario_id_for(template_id: str, seed: int | float) -> str:
    """Return the deterministic identity bound to template and seed provenance."""

    normalized_template_id = _identifier(template_id, "template_id")
    normalized_seed = _json_integer(seed, "seed")
    digest = hashlib.sha256(f"{normalized_template_id}:{normalized_seed}".encode()).hexdigest()[:12]
    return f"{normalized_template_id}-{digest}"


def canonical_date(value: Any) -> str:
    """Require the canonical JSON Schema full-date representation."""

    text = _nonempty_text(value, "provenance.generated_at")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        raise ScenarioValidationError("provenance.generated_at must use YYYY-MM-DD")
    try:
        parsed = date.fromisoformat(text)
    except ValueError as error:
        raise ScenarioValidationError("provenance.generated_at must be a valid date") from error
    if parsed.isoformat() != text:
        raise ScenarioValidationError("provenance.generated_at must use YYYY-MM-DD")
    return text


def _stance_for(prior: float | None, threshold: float, dimension: str) -> str:
    if prior is None:
        raise ScenarioValidationError(f"faction prior missing target dimension: {dimension}")
    if prior >= threshold:
        return "agree"
    if prior <= -threshold:
        return "disagree"
    return "pass"


def _exact_object(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ScenarioValidationError(f"{label} must be an object")
    if set(value) != fields:
        raise ScenarioValidationError(f"{label} fields mismatch")
    return dict(value)


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not ID_PATTERN.fullmatch(value):
        raise ScenarioValidationError(f"{label} must be a lowercase identifier")
    return value


def _json_integer(value: Any, label: str) -> int:
    """Match JSON Schema integer semantics, including integral-valued numbers."""

    if isinstance(value, bool):
        raise ScenarioValidationError(f"{label} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and isfinite(value) and value.is_integer():
        return int(value)
    raise ScenarioValidationError(f"{label} must be an integer")


def _nonempty_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ScenarioValidationError(f"{label} must be non-empty text")
    return value
