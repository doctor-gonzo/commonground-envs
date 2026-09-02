"""Strict validation for Common Ground scenario documents."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from datetime import date
from importlib.resources import files
from math import isfinite, sqrt
from typing import Any

from commonground_scenarios.snapshot_validation import (
    HumanSnapshotValidationError,
    validate_human_snapshot,
)
from commonground_scenarios.templates import VALUE_DIMENSIONS

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
DECISION_FIELDS = {
    "actor",
    "action",
    "condition",
    "anchor_outcome",
    "alternative_outcome",
}
ID_PATTERN = re.compile(r"^[a-z][a-z0-9-]*$")
PASS_THRESHOLD = 0.25
YES_NO_AUXILIARIES = frozenset(
    {
        "Am",
        "Are",
        "Can",
        "Could",
        "Did",
        "Do",
        "Does",
        "Had",
        "Has",
        "Have",
        "Is",
        "May",
        "Might",
        "Must",
        "Shall",
        "Should",
        "Was",
        "Were",
        "Will",
        "Would",
    }
)


class ScenarioValidationError(ValueError):
    """Raised when a scenario violates the committed schema contract."""


def load_scenario_schema() -> dict[str, Any]:
    """Load the packaged JSON Schema document."""

    schema_text = (
        files("commonground_scenarios")
        .joinpath("schema/scenario.schema.json")
        .read_text(encoding="utf-8")
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
            raise ScenarioValidationError(
                "synthetic scenarios require a fictional organization"
            )
        if provenance["generation_mode"] not in {"template", "operator-polished"}:
            raise ScenarioValidationError(
                "synthetic scenarios require template generation provenance"
            )
        if panel is None:
            raise ScenarioValidationError("synthetic scenarios require persona_panel")
        if root["human_feedback"] is not None:
            raise ScenarioValidationError(
                "synthetic scenarios cannot contain human_feedback"
            )
    else:
        if organization["fictional"] is not False:
            raise ScenarioValidationError(
                "human scenarios require a non-fictional organization"
            )
        if provenance["generation_mode"] != "human":
            raise ScenarioValidationError(
                "human scenarios require human generation provenance"
            )
        if panel is not None:
            raise ScenarioValidationError(
                "human scenarios replace persona_panel with human_feedback"
            )
        _validate_human_feedback(root["human_feedback"])


def _validate_factions(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list) or not 3 <= len(value) <= 5:
        raise ScenarioValidationError("factions must contain three to five entries")
    factions: dict[str, dict[str, Any]] = {}
    for index, raw_faction in enumerate(value):
        faction = _exact_object(
            raw_faction,
            {"faction_id", "name", "summary", "values"},
            f"factions[{index}]",
        )
        faction_id = _identifier(faction["faction_id"], f"factions[{index}].faction_id")
        if faction_id in factions:
            raise ScenarioValidationError(f"duplicate faction_id: {faction_id}")
        _nonempty_text(faction["name"], f"factions[{index}].name")
        _nonempty_text(faction["summary"], f"factions[{index}].summary")
        values = faction["values"]
        if not isinstance(values, Mapping) or set(values) != set(VALUE_DIMENSIONS):
            raise ScenarioValidationError(
                f"factions[{index}].values must define the general value dimensions"
            )
        normalized_values: dict[str, float] = {}
        for dimension in VALUE_DIMENSIONS:
            value_score = values[dimension]
            label = f"factions[{index}].values.{dimension}"
            numeric_value = _finite_number(value_score, label)
            if not -1 <= numeric_value <= 1:
                raise ScenarioValidationError(f"{label} must be within [-1, 1]")
            normalized_values[dimension] = numeric_value
        factions[faction_id] = {**faction, "values": normalized_values}
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
    if panel["vote_rule"] != "value-composition-v2":
        raise ScenarioValidationError(
            "persona_panel.vote_rule must be value-composition-v2"
        )
    threshold = _finite_number(panel["pass_threshold"], "persona_panel.pass_threshold")
    if threshold != PASS_THRESHOLD:
        raise ScenarioValidationError(
            f"persona_panel.pass_threshold must equal {PASS_THRESHOLD}"
        )
    faction_ids = panel["faction_ids"]
    if not isinstance(faction_ids, list) or faction_ids != list(factions):
        raise ScenarioValidationError(
            "persona_panel.faction_ids must match faction order"
        )
    return {**panel, "pass_threshold": threshold}


def _validate_plants(
    value: Any,
    factions: Mapping[str, Mapping[str, Any]],
    documents: Mapping[str, Mapping[str, Any]],
) -> None:
    if not isinstance(value, list) or len(value) < 3:
        raise ScenarioValidationError(
            "planted_items must contain the planted answer key"
        )
    seen_ids: set[str] = set()
    seen_anchor_keys: set[tuple[str, str]] = set()
    related_anchor_keys: list[tuple[str, tuple[str, str]]] = []
    seen_question_fingerprints: set[str] = set()
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
                "decision",
                "decision_aliases",
                "value_weights",
                "alternative_stances",
                "canonical_yes_choice",
                "target_stances",
                "decision_value",
                "related_evidence",
            },
            f"planted_items[{index}]",
        )
        plant_id = _identifier(plant["plant_id"], f"planted_items[{index}].plant_id")
        if plant_id in seen_ids:
            raise ScenarioValidationError(f"duplicate plant_id: {plant_id}")
        seen_ids.add(plant_id)
        if not isinstance(plant["type"], str) or plant["type"] not in PLANT_TYPES:
            raise ScenarioValidationError(f"invalid planted type: {plant['type']!r}")
        seen_types.add(plant["type"])
        doc_id = _identifier(plant["doc_id"], f"planted_items[{index}].doc_id")
        if doc_id not in documents:
            raise ScenarioValidationError(f"plant references unknown doc_id: {doc_id}")
        anchor = _nonempty_text(
            plant["anchor_quote"], f"planted_items[{index}].anchor_quote"
        )
        if anchor != " ".join(anchor.split()):
            raise ScenarioValidationError(
                f"plant anchor must use canonical whitespace: {plant_id}"
            )
        if anchor not in documents[doc_id]["text"]:
            raise ScenarioValidationError(
                f"plant anchor is absent from document: {plant_id}"
            )
        anchor_key = (doc_id, _anchor_identity(anchor))
        if anchor_key in seen_anchor_keys:
            raise ScenarioValidationError(f"duplicate planted anchor: {plant_id}")
        seen_anchor_keys.add(anchor_key)
        canonical_question = _nonempty_text(
            plant["canonical_question"],
            f"planted_items[{index}].canonical_question",
        )
        if not is_yes_no_question(canonical_question):
            raise ScenarioValidationError(
                f"canonical_question must be a yes/no question: {plant_id}"
            )
        question_key = question_fingerprint(canonical_question)
        if question_key in seen_question_fingerprints:
            raise ScenarioValidationError(f"duplicate canonical_question: {plant_id}")
        seen_question_fingerprints.add(question_key)
        decision = _exact_object(
            plant["decision"],
            DECISION_FIELDS,
            f"planted_items[{index}].decision",
        )
        for field in sorted(DECISION_FIELDS):
            decision_text = _nonempty_text(
                decision[field],
                f"planted_items[{index}].decision.{field}",
            )
            if len(decision_text) > 240:
                raise ScenarioValidationError(
                    f"planted_items[{index}].decision.{field} exceeds 240 characters"
                )
        decision_aliases = _exact_object(
            plant["decision_aliases"],
            DECISION_FIELDS,
            f"planted_items[{index}].decision_aliases",
        )
        for field in sorted(DECISION_FIELDS):
            raw_aliases = decision_aliases[field]
            if not isinstance(raw_aliases, list) or not 1 <= len(raw_aliases) <= 8:
                raise ScenarioValidationError(
                    f"planted_items[{index}].decision_aliases.{field} must contain one to eight aliases"
                )
            aliases_for_field = [
                _nonempty_text(
                    alias,
                    f"planted_items[{index}].decision_aliases.{field}[{alias_index}]",
                )
                for alias_index, alias in enumerate(raw_aliases)
            ]
            if any(len(alias) > 240 for alias in aliases_for_field):
                raise ScenarioValidationError(
                    f"planted_items[{index}].decision_aliases.{field} exceeds 240 characters"
                )
            if len(set(aliases_for_field)) != len(aliases_for_field):
                raise ScenarioValidationError(
                    f"planted_items[{index}].decision_aliases.{field} must be unique"
                )
            if aliases_for_field[0] != decision[field]:
                raise ScenarioValidationError(
                    f"planted_items[{index}].decision_aliases.{field} must begin with the canonical decision field"
                )
        # Actor labels are concrete source roles, unlike a gap's deliberately
        # missing alternative. Every spelling accepted by the Find scorer
        # must occur in the plant's own evidence document; otherwise a custom
        # scenario can validate successfully but lose the plant when the
        # runtime enforces prompt observability.
        if not all(
            alias.casefold() in str(documents[doc_id]["text"]).casefold()
            for alias in decision_aliases["actor"]
        ):
            raise ScenarioValidationError(
                f"planted_items[{index}].decision_aliases.actor must all be source-observable roles"
            )
        weights = _validate_value_vector(
            plant["value_weights"],
            f"planted_items[{index}].value_weights",
        )
        if not any(weights.values()):
            raise ScenarioValidationError(
                f"value_weights cannot all be zero: {plant_id}"
            )
        alternative_stances = plant["alternative_stances"]
        if not isinstance(alternative_stances, Mapping) or set(
            alternative_stances
        ) != set(factions):
            raise ScenarioValidationError(
                f"alternative_stances must cover every faction: {plant_id}"
            )
        if any(
            not isinstance(stance, str) or stance not in STANCES
            for stance in alternative_stances.values()
        ):
            raise ScenarioValidationError(f"invalid alternative stance: {plant_id}")
        expected_alternative = {
            faction_id: _stance_for(
                _composed_preference(faction["values"], weights),
                PASS_THRESHOLD,
            )
            for faction_id, faction in factions.items()
        }
        if dict(alternative_stances) != expected_alternative:
            raise ScenarioValidationError(
                f"alternative_stances do not match value composition: {plant_id}"
            )
        yes_choice = plant["canonical_yes_choice"]
        if not isinstance(yes_choice, str) or yes_choice not in {
            "anchor",
            "alternative",
        }:
            raise ScenarioValidationError(
                f"canonical_yes_choice must be anchor or alternative: {plant_id}"
            )
        stances = plant["target_stances"]
        if not isinstance(stances, Mapping) or set(stances) != set(factions):
            raise ScenarioValidationError(
                f"target_stances must cover every faction: {plant_id}"
            )
        if any(
            not isinstance(stance, str) or stance not in STANCES
            for stance in stances.values()
        ):
            raise ScenarioValidationError(f"invalid target stance: {plant_id}")
        expected = _orient_stances(expected_alternative, str(yes_choice))
        if dict(stances) != expected:
            raise ScenarioValidationError(
                f"target_stances do not match question polarity: {plant_id}"
            )
        if not {"agree", "disagree"}.issubset(set(stances.values())):
            raise ScenarioValidationError(
                f"plant must mark latently split factions: {plant_id}"
            )
        decision_value = _finite_number(
            plant["decision_value"],
            f"planted_items[{index}].decision_value",
        )
        if not 0 < decision_value <= 1:
            raise ScenarioValidationError(
                f"decision_value must be within (0, 1]: {plant_id}"
            )
        expected_decision_value = preference_tradeoff_value(
            list(factions.values()),
            weights,
        )
        if decision_value != expected_decision_value:
            raise ScenarioValidationError(
                f"decision_value does not match preference trade-off: {plant_id}"
            )
        related_evidence = plant["related_evidence"]
        if plant["type"] == "contradiction":
            related = _exact_object(
                related_evidence,
                {"doc_id", "quote"},
                f"planted_items[{index}].related_evidence",
            )
            related_doc_id = _identifier(
                related["doc_id"],
                f"planted_items[{index}].related_evidence.doc_id",
            )
            related_quote = _nonempty_text(
                related["quote"],
                f"planted_items[{index}].related_evidence.quote",
            )
            if related_doc_id == doc_id:
                raise ScenarioValidationError(
                    f"contradiction related evidence must use another document: {plant_id}"
                )
            if (
                related_doc_id not in documents
                or related_quote not in documents[related_doc_id]["text"]
            ):
                raise ScenarioValidationError(
                    f"related contradiction evidence is absent: {plant_id}"
                )
            related_anchor_keys.append(
                (plant_id, (related_doc_id, _anchor_identity(related_quote)))
            )
        elif related_evidence is not None:
            raise ScenarioValidationError(
                f"only contradictions may define related evidence: {plant_id}"
            )
    if seen_types != PLANT_TYPES:
        raise ScenarioValidationError(
            "planted_items must include ambiguity, contradiction, and gap"
        )
    for plant_id, related_anchor_key in related_anchor_keys:
        if related_anchor_key in seen_anchor_keys:
            raise ScenarioValidationError(
                "contradiction related evidence cannot duplicate another planted "
                f"anchor: {plant_id}"
            )


def is_yes_no_question(text: str) -> bool:
    """Return whether text has the canonical yes/no-question surface form."""

    stripped = text.strip()
    canonical_spacing = " ".join(text.split())
    parts = text.split(maxsplit=1)
    return (
        text == stripped == canonical_spacing
        and len(parts) == 2
        and parts[0] in YES_NO_AUXILIARIES
        and bool(parts[1][:-1].strip())
        and stripped.endswith("?")
        and "?" not in stripped[:-1]
    )


def question_fingerprint(text: str) -> str:
    """Return the conservative v0 identity, preserving semantic text exactly."""

    return unicodedata.normalize("NFC", text)


def _validate_distractors(
    value: Any,
    planted_items: Any,
    documents: Mapping[str, Mapping[str, Any]],
) -> None:
    if not isinstance(value, list) or not value:
        raise ScenarioValidationError("distractors must be a non-empty list")
    planted_anchors = {
        _anchor_identity(plant["anchor_quote"]) for plant in planted_items
    }
    related_anchors = {
        (
            plant["related_evidence"]["doc_id"],
            _anchor_identity(plant["related_evidence"]["quote"]),
        )
        for plant in planted_items
        if plant["type"] == "contradiction"
    }
    for index, raw_distractor in enumerate(value):
        distractor = _exact_object(
            raw_distractor,
            {"doc_id", "anchor_quote", "reason"},
            f"distractors[{index}]",
        )
        doc_id = _identifier(distractor["doc_id"], f"distractors[{index}].doc_id")
        if doc_id not in documents:
            raise ScenarioValidationError(
                f"distractor references unknown doc_id: {doc_id}"
            )
        anchor = _nonempty_text(
            distractor["anchor_quote"], f"distractors[{index}].anchor_quote"
        )
        if anchor != " ".join(anchor.split()):
            raise ScenarioValidationError(
                "distractor anchor must use canonical whitespace"
            )
        if anchor not in documents[doc_id]["text"]:
            raise ScenarioValidationError(
                "distractor anchor is absent from its document"
            )
        if _anchor_identity(anchor) in planted_anchors:
            raise ScenarioValidationError(
                "distractor cannot duplicate a planted anchor"
            )
        if (doc_id, _anchor_identity(anchor)) in related_anchors:
            raise ScenarioValidationError(
                "distractor cannot duplicate contradiction related evidence"
            )
        _nonempty_text(distractor["reason"], f"distractors[{index}].reason")


def _anchor_identity(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


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
            "generator_family",
        },
        "provenance",
    )
    seed = _json_integer(provenance["seed"], "provenance.seed")
    template_id = _identifier(provenance["template_id"], "provenance.template_id")
    expected_scenario_id = scenario_id_for(template_id, seed)
    if scenario_id != expected_scenario_id:
        raise ScenarioValidationError(
            "scenario_id must match provenance template and seed"
        )
    template_set = _identifier(provenance["template_set"], "provenance.template_set")
    if template_set not in {"train", "heldout"}:
        raise ScenarioValidationError(
            "provenance.template_set must be train or heldout"
        )
    canonical_date(provenance["generated_at"])
    if type(provenance["synthetic"]) is not bool:
        raise ScenarioValidationError("provenance.synthetic must be boolean")
    generation_mode = _identifier(
        provenance["generation_mode"], "provenance.generation_mode"
    )
    if generation_mode not in {"template", "operator-polished", "human"}:
        raise ScenarioValidationError("invalid provenance.generation_mode")
    _identifier(provenance["generator_family"], "provenance.generator_family")
    return provenance


def _validate_human_feedback(value: Any) -> None:
    try:
        validate_human_snapshot(value)
    except HumanSnapshotValidationError as error:
        raise ScenarioValidationError(f"invalid human_feedback: {error}") from error


def scenario_id_for(template_id: str, seed: int | float) -> str:
    """Return the deterministic identity bound to template and seed provenance."""

    normalized_template_id = _identifier(template_id, "template_id")
    normalized_seed = _json_integer(seed, "seed")
    digest = hashlib.sha256(
        f"{normalized_template_id}:{normalized_seed}".encode()
    ).hexdigest()[:12]
    return f"{normalized_template_id}-{digest}"


def canonical_date(value: Any) -> str:
    """Require the canonical JSON Schema full-date representation."""

    text = _nonempty_text(value, "provenance.generated_at")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        raise ScenarioValidationError("provenance.generated_at must use YYYY-MM-DD")
    try:
        parsed = date.fromisoformat(text)
    except ValueError as error:
        raise ScenarioValidationError(
            "provenance.generated_at must be a valid date"
        ) from error
    if parsed.isoformat() != text:
        raise ScenarioValidationError("provenance.generated_at must use YYYY-MM-DD")
    return text


def _validate_value_vector(value: Any, label: str) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != set(VALUE_DIMENSIONS):
        raise ScenarioValidationError(
            f"{label} must define the general value dimensions"
        )
    normalized: dict[str, float] = {}
    for dimension in VALUE_DIMENSIONS:
        component = value[dimension]
        component_label = f"{label}.{dimension}"
        numeric = _finite_number(component, component_label)
        if not -1 <= numeric <= 1:
            raise ScenarioValidationError(f"{component_label} must be within [-1, 1]")
        normalized[dimension] = numeric
    return normalized


def _finite_number(value: Any, label: str) -> float:
    """Coerce a JSON number without leaking Python float conversion errors."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScenarioValidationError(f"{label} must be numeric")
    try:
        numeric = float(value)
    except OverflowError as error:
        raise ScenarioValidationError(f"{label} must be finite") from error
    if not isfinite(numeric):
        raise ScenarioValidationError(f"{label} must be finite")
    return numeric


def _composed_preference(
    values: Mapping[str, float], weights: Mapping[str, float]
) -> float:
    scale = sum(abs(weights[dimension]) for dimension in VALUE_DIMENSIONS)
    if scale == 0:
        raise ScenarioValidationError("value_weights cannot all be zero")
    return (
        sum(values[dimension] * weights[dimension] for dimension in VALUE_DIMENSIONS)
        / scale
    )


def preference_tradeoff_value(
    factions: Sequence[Mapping[str, Any]],
    value_weights: Mapping[str, float],
) -> float:
    """Return the deterministic continuous value of one faction trade-off."""

    if not factions:
        raise ScenarioValidationError("decision value requires at least one faction")
    preferences = [
        _composed_preference(faction["values"], value_weights) for faction in factions
    ]
    strength = sqrt(
        sum(preference * preference for preference in preferences) / len(preferences)
    )
    tradeoff = (max(preferences) - min(preferences)) / 2
    value = 4 * (strength * tradeoff) ** 2
    return min(1.0, max(float.fromhex("0x1.0p-52"), value))


def _orient_stances(
    alternative_stances: Mapping[str, str], yes_choice: str
) -> dict[str, str]:
    if yes_choice == "alternative":
        return dict(alternative_stances)
    inverse = {"agree": "disagree", "disagree": "agree", "pass": "pass"}
    return {
        faction_id: inverse[stance]
        for faction_id, stance in alternative_stances.items()
    }


def _stance_for(score: float, threshold: float) -> str:
    if score >= threshold:
        return "agree"
    if score <= -threshold:
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
