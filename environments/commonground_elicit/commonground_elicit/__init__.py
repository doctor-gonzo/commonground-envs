"""Common Ground document-grounded elicitation environment."""

from importlib.metadata import PackageNotFoundError, version

from commonground_elicit.environment import (
    ElicitHarness,
    ElicitJsonParser,
    ElicitTask,
    ElicitTaskData,
    ElicitTaskset,
    ElicitTasksetConfig,
    finding_diagnosis_recall,
    finding_f1,
    finding_localization_recall,
    finding_relation_recall,
    finding_training_reward,
    finding_type_accuracy,
    load_environment,
    load_taskset,
    match_findings,
    normalized_quote_overlap,
    normalized_quote_precision,
    panel_disagreement,
    question_decision_similarity,
    question_format_validity,
    question_grounding_recall,
    question_stance_accuracy,
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
    "finding_diagnosis_recall",
    "finding_f1",
    "finding_localization_recall",
    "finding_relation_recall",
    "finding_training_reward",
    "finding_type_accuracy",
    "load_environment",
    "load_taskset",
    "match_findings",
    "normalized_quote_overlap",
    "normalized_quote_precision",
    "panel_disagreement",
    "question_decision_similarity",
    "question_format_validity",
    "question_grounding_recall",
    "question_stance_accuracy",
    "question_utility",
    "question_utility_score",
    "render_prompt",
]
