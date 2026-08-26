"""Deterministic planted scenarios for Common Ground environments."""

from importlib.metadata import PackageNotFoundError, version

from commonground_scenarios.generator import generate_scenario, scenario_to_bytes
from commonground_scenarios.snapshot_validation import (
    HUMAN_SNAPSHOT_SCHEMA_VERSION,
    HumanSnapshotValidationError,
    contains_direct_identifier,
    find_direct_identifier,
    validate_human_snapshot,
)
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

try:
    __version__ = version("commonground-scenarios")
except PackageNotFoundError:  # pragma: no cover - source tree without installation
    __version__ = "0+unknown"

__all__ = [
    "HELDOUT_TEMPLATES",
    "HUMAN_SNAPSHOT_SCHEMA_VERSION",
    "TRAIN_TEMPLATES",
    "DomainTemplate",
    "HumanSnapshotValidationError",
    "ScenarioValidationError",
    "contains_direct_identifier",
    "find_direct_identifier",
    "generate_scenario",
    "get_template",
    "is_yes_no_question",
    "load_scenario_schema",
    "question_fingerprint",
    "scenario_to_bytes",
    "validate_human_snapshot",
    "validate_scenario",
]
