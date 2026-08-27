"""Common Ground document-grounded elicitation environment."""

from importlib.metadata import PackageNotFoundError, version

from commonground_elicit.environment import (
    ElicitHarness,
    ElicitJsonParser,
    ElicitTask,
    ElicitTaskData,
    ElicitTaskset,
    ElicitTasksetConfig,
    finding_f1,
    load_environment,
    load_taskset,
    match_findings,
    normalized_quote_overlap,
    panel_disagreement,
    question_utility,
    question_utility_score,
    render_prompt,
)

try:
    __version__ = version("commonground-elicit")
except PackageNotFoundError:  # pragma: no cover - source tree without installation
    __version__ = "0+unknown"

__all__ = [
    "ElicitHarness",
    "ElicitJsonParser",
    "ElicitTask",
    "ElicitTaskData",
    "ElicitTaskset",
    "ElicitTasksetConfig",
    "finding_f1",
    "load_environment",
    "load_taskset",
    "match_findings",
    "normalized_quote_overlap",
    "panel_disagreement",
    "question_utility",
    "question_utility_score",
    "render_prompt",
]
