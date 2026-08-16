"""Deterministic planted scenarios for Common Ground environments."""

from commonground_scenarios.generator import generate_scenario, scenario_to_bytes
from commonground_scenarios.templates import (
    HELDOUT_TEMPLATES,
    TRAIN_TEMPLATES,
    DomainTemplate,
    get_template,
)
from commonground_scenarios.validation import (
    ScenarioValidationError,
    is_yes_no_question,
    load_scenario_schema,
    question_fingerprint,
    validate_scenario,
)

__version__ = "0.1.0"

__all__ = [
    "DomainTemplate",
    "HELDOUT_TEMPLATES",
    "ScenarioValidationError",
    "TRAIN_TEMPLATES",
    "generate_scenario",
    "get_template",
    "is_yes_no_question",
    "load_scenario_schema",
    "question_fingerprint",
    "scenario_to_bytes",
    "validate_scenario",
]
