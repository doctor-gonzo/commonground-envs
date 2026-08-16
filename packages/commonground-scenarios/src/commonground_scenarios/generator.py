"""Seeded, offline scenario generation."""

from __future__ import annotations

import copy
import json
import random
from collections.abc import Callable
from typing import Any

from commonground_scenarios.templates import DomainTemplate, get_template
from commonground_scenarios.validation import (
    PASS_THRESHOLD,
    canonical_date,
    scenario_id_for,
    validate_scenario,
)

DEFAULT_GENERATED_AT = "2026-08-15"


def generate_scenario(
    seed: int,
    domain_template: DomainTemplate | str,
    *,
    generated_at: str = DEFAULT_GENERATED_AT,
    prose_polisher: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """Generate one canonical planted scenario without network or model calls.

    ``generated_at`` is explicit and never read from the wall clock. An
    operator may inject ``prose_polisher`` at generation time; it is absent and
    therefore off by default.
    """

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    template = (
        get_template(domain_template)
        if isinstance(domain_template, str)
        else domain_template
    )
    canonical_date(generated_at)
    rng = random.Random(seed)

    documents = copy.deepcopy(list(template.documents))
    rng.shuffle(documents)
    if prose_polisher is not None:
        for document in documents:
            polished = prose_polisher(document["text"])
            if not isinstance(polished, str):
                raise TypeError("prose_polisher must return text")
            document["text"] = polished

    factions = copy.deepcopy(list(template.factions))
    planted_items = copy.deepcopy(list(template.planted_items))
    for planted in planted_items:
        dimension = planted["target_dimension"]
        planted["target_stances"] = {
            faction["faction_id"]: _stance_for(float(faction["priors"][dimension]))
            for faction in factions
        }

    scenario = {
        "scenario_id": scenario_id_for(template.template_id, seed),
        "organization": {
            "name": rng.choice(template.organization_names),
            "sector": template.sector,
            "fictional": True,
        },
        "factions": factions,
        "documents": documents,
        "planted_items": planted_items,
        "distractors": copy.deepcopy(list(template.distractors)),
        "persona_panel": {
            "vote_rule": "dimension-threshold-v1",
            "pass_threshold": PASS_THRESHOLD,
            "faction_ids": [faction["faction_id"] for faction in factions],
        },
        "human_feedback": None,
        "provenance": {
            "seed": seed,
            "template_id": template.template_id,
            "template_set": template.template_set,
            "generated_at": generated_at,
            "synthetic": True,
            "generation_mode": "operator-polished"
            if prose_polisher is not None
            else "template",
        },
    }
    validate_scenario(scenario)
    return scenario


def scenario_to_bytes(scenario: dict[str, Any]) -> bytes:
    """Validate and serialize a scenario as canonical newline-terminated JSON."""

    validate_scenario(scenario)
    return (
        json.dumps(
            scenario,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _stance_for(prior: float) -> str:
    if prior >= PASS_THRESHOLD:
        return "agree"
    if prior <= -PASS_THRESHOLD:
        return "disagree"
    return "pass"
