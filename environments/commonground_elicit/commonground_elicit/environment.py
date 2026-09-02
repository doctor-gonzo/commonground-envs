"""Verifiers environment for planted document-grounded findings."""

from __future__ import annotations

import copy
import json
import math
import os
import re
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import verifiers as legacy_vf
import verifiers.v1 as vf
from commonground_scenarios import (
    is_yes_no_question,
    validate_scenario,
)
from commonground_scenarios.generator import ACTOR_SUPPORT_REASON, SEMANTIC_SCOPES
from commonground_scenarios.templates import VALUE_DIMENSIONS
from commonground_scenarios.validation import PASS_THRESHOLD, YES_NO_AUXILIARIES
from commonground_score import (
    cluster_separation,
    vote_entropy,
)
from datasets import Dataset
from verifiers.v1.harnesses.null import NullHarness

ENV_ID = "commonground-elicit"
DATA_ENV_VAR = "COMMONGROUND_ELICIT_DATA_PATH"
TRAIN_DATA_ENV_VAR = "COMMONGROUND_ELICIT_TRAIN_DATA_PATH"
DATA_DIR = Path(__file__).resolve().parent / "data"
BUNDLED_TRAIN_PATH = DATA_DIR / "train_synthetic.jsonl"
BUNDLED_EVAL_PATH = DATA_DIR / "eval_synthetic_heldout.jsonl"
BUNDLED_SPLIT_PATHS = {
    "eval": BUNDLED_EVAL_PATH,
    "train": BUNDLED_TRAIN_PATH,
}
FINDING_TYPES = frozenset({"ambiguity", "contradiction", "gap"})
QUOTE_OVERLAP_THRESHOLD = 0.8
PLANT_COVERAGE_THRESHOLD = 0.9
QUOTE_PRECISION_THRESHOLD = 0.8
MAX_COMPLETION_CHARS = 32_768
MAX_JSON_DEPTH = 32
MAX_JSON_NODES = 10_000
VALID_TASKS = frozenset({"find", "elicit-ask"})
VALID_REWARD_MODES = frozenset({"strict", "shaped"})
STANCE_TO_VOTE = {"agree": 1, "disagree": -1, "pass": 0}
QUESTION_GROUNDING_WEIGHT = 0.5
STANCE_ACCURACY_WEIGHT = 0.5
DECISION_FRAME_FIELDS = frozenset(
    {
        "actor",
        "action",
        "condition",
        "anchor_outcome",
        "alternative_outcome",
    }
)
_SEMANTIC_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "at",
        "be",
        "between",
        "by",
        "can",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "should",
        "that",
        "the",
        "this",
        "to",
        "than",
        "unless",
        "when",
        "whenever",
        "whether",
        "who",
        "with",
        "rather",
    }
)
_YES_NO_AUXILIARY_TOKENS = frozenset(
    auxiliary.casefold() for auxiliary in YES_NO_AUXILIARIES
)
_SEMANTIC_EQUIVALENTS = {
    "allow": "permit",
    "allowed": "permit",
    "allows": "permit",
    "authorise": "permit",
    "authorize": "permit",
    # Deterministic active/passive normalization for common policy actions.
    # Keep this explicit: an open-ended stemmer would conflate unrelated
    # authored concepts and make the semantic gate harder to audit.
    "approved": "approve",
    "choosing": "choose",
    "defined": "define",
    "disclosed": "disclose",
    "issued": "issue",
    "permitted": "permit",
    "pausing": "pause",
    "immediately": "immediate",
    "removal": "remove",
    "removed": "remove",
    "released": "release",
    "prohibit": "forbid",
    "prohibited": "forbid",
    "prevent": "forbid",
    "prevents": "forbid",
    "require": "must",
    "required": "must",
    "requires": "must",
    "used": "use",
}
_QUESTION_INFERENCE_TOKENS = frozenset({"choose", "make", "permit", "wait"})
_POLARITY_TOKENS = frozenset(
    {
        "!",
        "!=",
        "cannot",
        "no",
        "neither",
        "never",
        "nor",
        "not",
        "lack",
        "lacking",
        "lacks",
        "without",
        "~",
        "¬",
        "≠",
    }
)
_TOKEN_PATTERN = re.compile(
    r"(?:!=|<=|>=|==)|[!~](?=\s*[^\W_])|-(?=\s*\d)|[^\W_]+|"
    r"[¬≠≤≥=<>±+\N{MINUS SIGN}%$€£¥∉∈∧\N{LOGICAL OR}]",
    flags=re.UNICODE,
)
_ORIENTATION_MARKER_PATTERN = re.compile(
    r"\byes\s+selects\s+the\s+(anchor|alternative)\s+outcome\b",
    flags=re.IGNORECASE,
)
_NEGATION_CONTRACTION_PATTERN = re.compile(r"\b[^\W_]+n['\u2019]t\b", re.IGNORECASE)
_NEGATING_VERB_PATTERN = re.compile(
    r"\b(?:avoid(?:s|ed|ing)?|refus(?:e|es|ed|ing)|declin(?:e|es|ed|ing))\b",
    re.IGNORECASE,
)
_FAIL_TO_PATTERN = re.compile(r"\bfail(?:s|ed|ing)?\s+to\b", re.IGNORECASE)
_SUBMITTED_SLOT_PRECISION_THRESHOLD = 0.75


class _DuplicateJsonKeyError(ValueError):
    """Raised before JSON object construction can erase a repeated key."""


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateJsonKeyError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _reject_non_finite_json_constant(value: str) -> Any:
    """Reject the non-standard NaN and infinity tokens accepted by Python JSON."""

    raise ValueError(f"non-finite JSON constant {value!r}")


class ElicitJsonParser(legacy_vf.Parser):
    """Parse one complete task-specific JSON object from a completion."""

    def __init__(self, preferred_key: str = "findings") -> None:
        super().__init__()
        self.preferred_key = preferred_key

    def parse(self, text: str) -> dict[str, Any]:
        try:
            parsed = extract_json_object(text, preferred_key=self.preferred_key)
        except (RecursionError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def parse_answer(self, completion: Sequence[Any]) -> dict[str, Any]:
        """Parse the last assistant message from legacy or v1 message carriers."""

        for message in reversed(completion):
            role = (
                message.get("role")
                if isinstance(message, Mapping)
                else getattr(message, "role", None)
            )
            if role != "assistant":
                continue
            content = (
                message.get("content")
                if isinstance(message, Mapping)
                else getattr(message, "content", None)
            )
            return self.parse(content if isinstance(content, str) else "")
        return {}


class ElicitTaskData(vf.TaskData):
    """One native Verifiers v1 task row and its hidden deterministic oracle."""

    answer: dict[str, Any]
    info: dict[str, Any]


class ElicitTask(vf.Task[ElicitTaskData]):
    """Score one single-turn completion through native v1 reward decorators."""

    @vf.reward(weight=1.0)
    async def task_reward(self, trace: vf.Trace) -> Mapping[str, float]:
        completion = [{"role": "assistant", "content": trace.last_reply}]
        if self.data.info["task_label"] == "elicit-ask":
            score = await question_utility(
                completion,
                self.data.answer,
                self.data.info,
                ElicitJsonParser("questions"),
            )
            return {"question_utility": score}
        score = await finding_f1(
            completion, self.data.answer, ElicitJsonParser("findings")
        )
        if self.data.info.get("reward_mode") == "shaped":
            score = await finding_training_reward(
                completion, self.data.answer, ElicitJsonParser("findings")
            )
            return {"finding_training_reward": score}
        return {"finding_f1": score}

    @vf.metric
    async def companion_metrics(self, trace: vf.Trace) -> Mapping[str, float]:
        completion = [{"role": "assistant", "content": trace.last_reply}]
        if self.data.info["task_label"] == "elicit-ask":
            parser = ElicitJsonParser("questions")
            return {
                "question_format_valid": await question_format_validity(
                    completion, self.data.answer, self.data.info, parser
                ),
                "question_top1_selection_accuracy": await question_top1_selection_accuracy(
                    completion, self.data.answer, self.data.info, parser
                ),
                "question_grounding_recall": await question_grounding_recall(
                    completion, self.data.answer, self.data.info, parser
                ),
                "question_grounded_stance_recall": await question_grounded_stance_recall(
                    completion, self.data.answer, self.data.info, parser
                ),
                "question_evidence_match_recall": await question_evidence_match_recall(
                    completion, self.data.answer, self.data.info, parser
                ),
                "question_evidence_matched_stance_accuracy": await question_evidence_matched_stance_accuracy(
                    completion, self.data.answer, self.data.info, parser
                ),
            }
        localization = await finding_localization_recall(
            completion,
            self.data.answer,
            ElicitJsonParser("findings"),
        )
        type_score = await finding_type_accuracy(
            completion,
            self.data.answer,
            ElicitJsonParser("findings"),
        )
        question_score = await question_utility(
            completion,
            self.data.answer,
            self.data.info,
            ElicitJsonParser("findings"),
        )
        metrics = {
            "question_utility": question_score,
            "finding_localization_recall": localization,
            "finding_type_accuracy": type_score,
            "finding_diagnosis_recall": await finding_diagnosis_recall(
                completion,
                self.data.answer,
                ElicitJsonParser("findings"),
            ),
            "finding_relation_recall": await finding_relation_recall(
                completion,
                self.data.answer,
                ElicitJsonParser("findings"),
            ),
        }
        if self.data.info.get("reward_mode") == "shaped":
            metrics["finding_f1"] = await finding_f1(
                completion, self.data.answer, ElicitJsonParser("findings")
            )
        return metrics


class ElicitTasksetConfig(vf.TasksetConfig):
    """Load-time difficulty and split controls for the v1 taskset."""

    docs_count: int | None = None
    docs_length: int | None = None
    planted_density: float = 1.0
    distractor_density: float = 1.0
    data_path: Path | None = None
    train_data_path: Path | None = None
    task_mode: Literal["find", "elicit-ask"] = "find"
    panel_polarization: float = 1.0
    question_count: int = 1
    split: Literal["eval", "train"] = "eval"
    reward_mode: Literal["strict", "shaped"] = "strict"


class ElicitHarness(NullHarness):
    """The built-in pure-chat harness, namespaced with this taskset plugin."""


class _CompatibilityRubric:
    """Small adapter for existing local score probes; native runs use ElicitTask."""

    def __init__(self, reward_mode: str = "strict") -> None:
        self.reward_mode = reward_mode

    async def score_rollout(self, state: Mapping[str, Any]) -> None:
        completion = state["completion"]
        answer = state["answer"]
        info = state["info"]
        parsed_info = _parse_mapping_payload(info)
        metrics: dict[str, float] = {}
        if parsed_info.get("task_label") == "elicit-ask":
            reward = await question_utility(
                completion, answer, info, ElicitJsonParser("questions")
            )
            metrics["question_utility"] = reward
            metrics["question_format_valid"] = await question_format_validity(
                completion, answer, info, ElicitJsonParser("questions")
            )
            metrics[
                "question_top1_selection_accuracy"
            ] = await question_top1_selection_accuracy(
                completion, answer, info, ElicitJsonParser("questions")
            )
            metrics["question_grounding_recall"] = await question_grounding_recall(
                completion, answer, info, ElicitJsonParser("questions")
            )
            metrics[
                "question_grounded_stance_recall"
            ] = await question_grounded_stance_recall(
                completion, answer, info, ElicitJsonParser("questions")
            )
            metrics[
                "question_evidence_match_recall"
            ] = await question_evidence_match_recall(
                completion, answer, info, ElicitJsonParser("questions")
            )
            metrics[
                "question_evidence_matched_stance_accuracy"
            ] = await question_evidence_matched_stance_accuracy(
                completion, answer, info, ElicitJsonParser("questions")
            )
        else:
            strict_reward = await finding_f1(
                completion, answer, ElicitJsonParser("findings")
            )
            reward = (
                await finding_training_reward(
                    completion, answer, ElicitJsonParser("findings")
                )
                if self.reward_mode == "shaped"
                else strict_reward
            )
            metrics["finding_f1"] = strict_reward
            metrics["finding_localization_recall"] = await finding_localization_recall(
                completion, answer, ElicitJsonParser("findings")
            )
            metrics["finding_type_accuracy"] = await finding_type_accuracy(
                completion, answer, ElicitJsonParser("findings")
            )
            metrics["finding_diagnosis_recall"] = await finding_diagnosis_recall(
                completion, answer, ElicitJsonParser("findings")
            )
            metrics["finding_relation_recall"] = await finding_relation_recall(
                completion, answer, ElicitJsonParser("findings")
            )
            metrics["question_utility"] = await question_utility(
                completion, answer, info, ElicitJsonParser("findings")
            )
        state["metrics"] = metrics  # type: ignore[index]
        state["reward"] = reward  # type: ignore[index]


class ElicitTaskset(vf.Taskset[ElicitTask, ElicitTasksetConfig]):
    """Native Verifiers v1 taskset for both Common Ground Elicit modes."""

    def __init__(self, config: ElicitTasksetConfig) -> None:
        super().__init__(config)
        validate_difficulty_args(
            docs_count=config.docs_count,
            docs_length=config.docs_length,
            planted_density=config.planted_density,
            distractor_density=config.distractor_density,
            panel_polarization=config.panel_polarization,
            question_count=config.question_count,
            task=config.task_mode,
        )
        bundled_eval_path = _bundled_data_path(config.split)
        configured_eval_path = config.data_path or os.environ.get(DATA_ENV_VAR)
        configured_train_path = config.train_data_path or os.environ.get(
            TRAIN_DATA_ENV_VAR
        )
        self.eval_path = (
            Path(configured_eval_path) if configured_eval_path else bundled_eval_path
        )
        self.train_path = (
            Path(configured_train_path) if configured_train_path else BUNDLED_TRAIN_PATH
        )
        self._train_rows = _scenario_rows(
            load_scenarios(self.train_path),
            docs_count=config.docs_count,
            docs_length=config.docs_length,
            planted_density=config.planted_density,
            distractor_density=config.distractor_density,
            panel_polarization=config.panel_polarization,
            question_count=config.question_count,
            task=config.task_mode,
        )
        self._eval_rows = _scenario_rows(
            load_scenarios(self.eval_path),
            docs_count=config.docs_count,
            docs_length=config.docs_length,
            planted_density=config.planted_density,
            distractor_density=config.distractor_density,
            panel_polarization=config.panel_polarization,
            question_count=config.question_count,
            task=config.task_mode,
        )
        if not self._train_rows or not self._eval_rows:
            raise ValueError(
                "difficulty arguments remove all planted items from the dataset"
            )
        self.env_id = ENV_ID
        self.rubric = _CompatibilityRubric(config.reward_mode)
        self.env_args = {
            "task": config.task_mode,
            "docs_count": config.docs_count,
            "docs_length": config.docs_length,
            "planted_density": config.planted_density,
            "distractor_density": config.distractor_density,
            "panel_polarization": config.panel_polarization,
            "question_count": config.question_count,
            "data_path": str(self.eval_path),
            "train_data_path": str(self.train_path),
            "split": config.split,
            "reward_mode": config.reward_mode,
        }

    def load(self) -> Sequence[ElicitTask]:
        return [
            ElicitTask(
                ElicitTaskData(
                    idx=index,
                    name=str(json.loads(row["info"])["scenario_id"]),
                    prompt=row["prompt"],
                    answer=json.loads(row["answer"]),
                    info={
                        **json.loads(row["info"]),
                        "reward_mode": self.config.reward_mode,
                    },
                )
            )
            for index, row in enumerate(self._eval_rows)
        ]

    def get_dataset(self) -> list[dict[str, Any]]:
        """Return canonical training rows for local compatibility probes."""

        return list(self._train_rows)

    def get_eval_dataset(self) -> list[dict[str, Any]]:
        """Return the selected native-v1 task rows for local compatibility probes."""

        return list(self._eval_rows)


def load_taskset(
    docs_count: int | None = None,
    docs_length: int | None = None,
    planted_density: float = 1.0,
    distractor_density: float = 1.0,
    data_path: str | os.PathLike[str] | None = None,
    *,
    task: str = "find",
    panel_polarization: float = 1.0,
    question_count: int = 1,
    train_data_path: str | os.PathLike[str] | None = None,
    split: str = "eval",
    reward_mode: str = "strict",
    **kwargs: Any,
) -> ElicitTaskset:
    """Build the native Verifiers v1 taskset with the public load controls."""

    validate_difficulty_args(
        docs_count=docs_count,
        docs_length=docs_length,
        planted_density=planted_density,
        distractor_density=distractor_density,
        panel_polarization=panel_polarization,
        question_count=question_count,
        task=task,
    )
    _bundled_data_path(split)
    if reward_mode not in VALID_REWARD_MODES:
        raise ValueError(
            f"unknown reward_mode {reward_mode!r}; valid modes: strict, shaped"
        )
    if kwargs:
        raise TypeError(f"unknown taskset arguments: {sorted(kwargs)}")
    return ElicitTaskset(
        ElicitTasksetConfig(
            id=ENV_ID,
            docs_count=docs_count,
            docs_length=docs_length,
            planted_density=planted_density,
            distractor_density=distractor_density,
            data_path=Path(data_path) if data_path is not None else None,
            train_data_path=(
                Path(train_data_path) if train_data_path is not None else None
            ),
            task_mode=task,
            panel_polarization=panel_polarization,
            question_count=question_count,
            split=split,
            reward_mode=reward_mode,
        )
    )


def load_environment(
    docs_count: int | None = None,
    docs_length: int | None = None,
    planted_density: float = 1.0,
    distractor_density: float = 1.0,
    data_path: str | os.PathLike[str] | None = None,
    *,
    task: str = "find",
    panel_polarization: float = 1.0,
    question_count: int = 1,
    train_data_path: str | os.PathLike[str] | None = None,
    split: str = "eval",
    reward_mode: str = "strict",
    **kwargs: Any,
) -> legacy_vf.SingleTurnEnv:
    """Build the legacy adapter required by Prime Hosted Evaluations."""

    # Hosted Evaluations still call the v0 factory, while native v1 discovers
    # ElicitTaskset through __all__. Keep a genuine Environment here so the
    # hosted runner receives its full server/evaluate lifecycle contract.
    taskset = load_taskset(
        docs_count=docs_count,
        docs_length=docs_length,
        planted_density=planted_density,
        distractor_density=distractor_density,
        data_path=data_path,
        task=task,
        panel_polarization=panel_polarization,
        question_count=question_count,
        train_data_path=train_data_path,
        split=split,
        reward_mode=reward_mode,
    )
    parser = ElicitJsonParser("questions" if task == "elicit-ask" else "findings")
    rubric = (
        legacy_vf.Rubric(
            funcs=[
                question_utility,
                question_format_validity,
                question_top1_selection_accuracy,
                question_grounding_recall,
                question_grounded_stance_recall,
                question_evidence_match_recall,
                question_evidence_matched_stance_accuracy,
            ],
            weights=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            parser=parser,
        )
        if task == "elicit-ask"
        else legacy_vf.Rubric(
            funcs=(
                [
                    finding_training_reward,
                    finding_f1,
                    finding_localization_recall,
                    finding_type_accuracy,
                    finding_diagnosis_recall,
                    finding_relation_recall,
                    question_utility,
                ]
                if reward_mode == "shaped"
                else [
                    finding_f1,
                    finding_localization_recall,
                    finding_type_accuracy,
                    finding_diagnosis_recall,
                    finding_relation_recall,
                    question_utility,
                ]
            ),
            weights=(
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                if reward_mode == "shaped"
                else [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            ),
            parser=parser,
        )
    )
    return legacy_vf.SingleTurnEnv(
        dataset=Dataset.from_list(taskset.get_dataset()),
        eval_dataset=Dataset.from_list(taskset.get_eval_dataset()),
        parser=parser,
        rubric=rubric,
        env_id=ENV_ID,
        env_args=dict(taskset.env_args),
        **kwargs,
    )


def _bundled_data_path(split: str) -> Path:
    """Resolve a named bundled split to its packaged JSONL path."""

    try:
        return BUNDLED_SPLIT_PATHS[split]
    except KeyError:
        valid_names = ", ".join(BUNDLED_SPLIT_PATHS)
        raise ValueError(
            f"unknown split {split!r}; valid splits: {valid_names}"
        ) from None


def load_scenarios(path: Path) -> list[dict[str, Any]]:
    """Load and validate canonical scenario JSONL."""

    if not path.is_file():
        raise FileNotFoundError(path)

    scenarios: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            scenario = json.loads(line)
            validate_scenario(scenario)
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            raise ValueError(
                f"invalid scenario at {path}:{line_number}: {error}"
            ) from error
        scenarios.append(scenario)
    if not scenarios:
        raise ValueError(f"no scenarios loaded from {path}")
    return scenarios


def _scenario_rows(
    scenarios: Sequence[Mapping[str, Any]],
    *,
    docs_count: int | None,
    docs_length: int | None,
    planted_density: float,
    distractor_density: float,
    panel_polarization: float,
    question_count: int,
    task: str,
) -> list[dict[str, Any]]:
    """Apply the same difficulty contract to one scenario split."""

    candidate_rows = [
        scenario_to_row(
            scenario,
            docs_count=docs_count,
            docs_length=docs_length,
            planted_density=planted_density,
            distractor_density=distractor_density,
            panel_polarization=panel_polarization,
            question_count=question_count,
            task=task,
        )
        for scenario in scenarios
    ]
    return [
        row
        for row in candidate_rows
        if _row_has_required_plants(
            row,
            task=task,
            question_count=question_count,
        )
    ]


def _row_has_required_plants(
    row: Mapping[str, Any], *, task: str, question_count: int
) -> bool:
    """Check plant availability through the canonical hidden answer payload."""

    answer = json.loads(str(row["answer"]))
    findings = answer.get("findings", [])
    questions = answer.get("questions", [])
    if task == "elicit-ask":
        return bool(questions) and len(questions) >= question_count
    return bool(findings)


def scenario_to_row(
    scenario: Mapping[str, Any],
    *,
    docs_count: int | None,
    docs_length: int | None,
    planted_density: float,
    distractor_density: float,
    panel_polarization: float,
    question_count: int,
    task: str,
) -> dict[str, Any]:
    """Turn one validated scenario into a prompt and hidden answer row."""

    validate_scenario(scenario)
    documents, visible_plants = build_document_view(
        scenario,
        docs_count=docs_count,
        docs_length=docs_length,
        planted_density=planted_density,
        distractor_density=distractor_density,
    )
    effective_question_count = min(question_count, len(visible_plants))
    findings_answer = [
        {
            "doc_id": plant["doc_id"],
            "quote": plant["anchor_quote"],
            "type": plant["type"],
            "diagnosis": _oriented_decision_question(
                str(plant["canonical_question"]),
                _canonical_decision_frame(plant, documents),
                str(plant["canonical_yes_choice"]),
            ),
            "decision": _canonical_decision_frame(plant, documents),
            "decision_aliases": copy.deepcopy(plant["decision_aliases"]),
            "related_evidence": plant["related_evidence"],
            "related_document_text": (
                next(
                    document["text"]
                    for document in documents
                    if document["doc_id"] == plant["related_evidence"]["doc_id"]
                )
                if plant["related_evidence"] is not None
                else None
            ),
            "document_text": next(
                document["text"]
                for document in documents
                if document["doc_id"] == plant["doc_id"]
            ),
        }
        for plant in visible_plants
    ]
    question_oracle = [
        {
            "doc_id": plant["doc_id"],
            "quote": plant["anchor_quote"],
            "type": plant["type"],
            "question": _oriented_decision_question(
                str(plant["canonical_question"]),
                _canonical_decision_frame(plant, documents),
                str(plant["canonical_yes_choice"]),
            ),
            "question_aliases": [
                _oriented_decision_question(
                    str(alias),
                    _canonical_decision_frame(plant, documents),
                    str(plant["canonical_yes_choice"]),
                )
                for alias in plant["canonical_question_aliases"]
            ],
            "decision": _canonical_decision_frame(plant, documents),
            "decision_aliases": copy.deepcopy(plant["decision_aliases"]),
            "value_weights": {
                dimension: float(plant["value_weights"][dimension])
                for dimension in VALUE_DIMENSIONS
            },
            "yes_choice": plant["canonical_yes_choice"],
            "target_stances": dict(plant["target_stances"]),
            "alternative_stances": dict(plant["alternative_stances"]),
            "related_evidence": plant["related_evidence"],
            "related_document_text": (
                next(
                    document["text"]
                    for document in documents
                    if document["doc_id"] == plant["related_evidence"]["doc_id"]
                )
                if plant["related_evidence"] is not None
                else None
            ),
            "decision_value": float(plant["decision_value"]),
            "document_text": next(
                document["text"]
                for document in documents
                if document["doc_id"] == plant["doc_id"]
            ),
        }
        for plant in visible_plants
    ]
    question_oracle.sort(
        key=lambda plant: (
            -_plant_attainable_question_utility(
                plant,
                panel_polarization=panel_polarization,
            ),
            str(plant["doc_id"]),
            str(plant["quote"]),
        )
    )
    answer = (
        {"questions": question_oracle}
        if task == "elicit-ask"
        else {
            "findings": findings_answer,
            "questions": question_oracle,
            "question_count": effective_question_count,
        }
    )
    info = {
        "scenario_id": scenario["scenario_id"],
        "document_count": len(documents),
        "plant_count": len(findings_answer),
        "task_label": task,
        "synthetic": bool(scenario["provenance"]["synthetic"]),
        "template_set": scenario["provenance"]["template_set"],
        "panel_polarization": panel_polarization,
        "question_count": effective_question_count,
        "allow_combined_questions": task == "find",
    }
    return {
        "prompt": [
            {
                "role": "user",
                "content": (
                    render_ask_prompt(
                        documents,
                        scenario["factions"],
                        visible_plants,
                        effective_question_count,
                        panel_polarization=panel_polarization,
                    )
                    if task == "elicit-ask"
                    else render_prompt(
                        documents,
                        factions=scenario["factions"],
                        question_count=effective_question_count,
                    )
                ),
            }
        ],
        "answer": json.dumps(answer, sort_keys=True),
        "info": json.dumps(info, sort_keys=True),
        "example_id": str(scenario["scenario_id"]),
    }


def _canonical_decision_frame(
    plant: Mapping[str, Any], documents: Sequence[Mapping[str, str]]
) -> dict[str, str]:
    """Return the template-authored structured decision reference.

    ``documents`` remains in the private helper signature for compatibility
    with release-analysis code.  The reference itself is deliberately not
    synthesized from token positions: every template authors the five semantic
    slots and generation validates them as part of the scenario answer key.
    """

    del documents
    decision = _parse_decision_frame(plant.get("decision"))
    if decision is None:
        raise ValueError("planted issue is missing its authored decision frame")
    return decision


def _oriented_decision_question(
    question: str, decision: Mapping[str, str], yes_choice: str
) -> str:
    """Make the scored yes-side explicit instead of inferring polarity from prose."""

    outcome_field = (
        "anchor_outcome" if yes_choice == "anchor" else "alternative_outcome"
    )
    stem = question.strip().removesuffix("?").strip()
    if not _decision_stem_expresses_core(stem, decision):
        stem = (
            f"Should {decision['actor']} {decision['action']} when "
            f"{decision['condition']}"
        )
    return f"{stem} (yes selects the {yes_choice} outcome: {decision[outcome_field]})?"


def build_document_view(
    scenario: Mapping[str, Any],
    *,
    docs_count: int | None,
    docs_length: int | None,
    planted_density: float,
    distractor_density: float,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    """Apply difficulty knobs without mutating the canonical scenario."""

    raw_documents = list(scenario["documents"])
    selected_documents = raw_documents[: docs_count or len(raw_documents)]
    selected_doc_ids = {document["doc_id"] for document in selected_documents}

    contradiction_context_is_complete = (
        len(selected_documents) == len(raw_documents)
        and planted_density == 1.0
        and (
            docs_length is None
            or all(len(document["text"]) <= docs_length for document in raw_documents)
        )
    )
    eligible_plants = [
        plant
        for plant in scenario["planted_items"]
        if plant["doc_id"] in selected_doc_ids
        and (plant["type"] != "contradiction" or contradiction_context_is_complete)
    ]
    selected_plants = _density_prefix(eligible_plants, planted_density)
    selected_plant_ids = {plant["plant_id"] for plant in selected_plants}
    omitted_plant_anchors = {
        (plant["doc_id"], plant["anchor_quote"])
        for plant in scenario["planted_items"]
        if plant["doc_id"] in selected_doc_ids
        if plant["plant_id"] not in selected_plant_ids
    }

    selected_distractors = [
        item for item in scenario["distractors"] if item["doc_id"] in selected_doc_ids
    ]
    actor_support_distractors = [
        item for item in selected_distractors if item["reason"] == ACTOR_SUPPORT_REASON
    ]
    required_actor_support = [
        item
        for item in actor_support_distractors
        if any(
            plant["doc_id"] == item["doc_id"]
            and any(
                str(alias).casefold() in str(item["anchor_quote"]).casefold()
                for alias in _plant_actor_aliases(plant)
            )
            for plant in selected_plants
        )
    ]
    visible_neutral_distractors = _density_prefix(
        [
            item
            for item in selected_distractors
            if item["reason"] != ACTOR_SUPPORT_REASON
        ],
        distractor_density,
    )
    # Regression guard: actor support is part of a scored candidate's public
    # evidence, not optional noise. Density still controls only true neutral
    # distractors, while support for omitted plants stays hidden.
    visible_distractors = [*required_actor_support, *visible_neutral_distractors]
    visible_distractor_anchors = {
        (item["doc_id"], item["anchor_quote"]) for item in visible_distractors
    }
    visible_neutral_anchors = {
        (item["doc_id"], item["anchor_quote"]) for item in visible_neutral_distractors
    }
    required_actor_support_anchors = {
        (item["doc_id"], item["anchor_quote"]) for item in required_actor_support
    }
    all_distractor_anchors = {
        (item["doc_id"], item["anchor_quote"]) for item in scenario["distractors"]
    }

    documents: list[dict[str, str]] = []
    for document in selected_documents:
        text = document["text"]
        for doc_id, anchor in sorted(omitted_plant_anchors):
            if doc_id == document["doc_id"]:
                text = text.replace(anchor, "")
        for doc_id, anchor in sorted(
            all_distractor_anchors - visible_distractor_anchors
        ):
            if doc_id == document["doc_id"]:
                text = text.replace(anchor, "")
        text = " ".join(text.split())
        if docs_length is not None and len(text) > docs_length:
            document_plants = [
                plant
                for plant in selected_plants
                if plant["doc_id"] == document["doc_id"]
            ]
            visible_anchors = [plant["anchor_quote"] for plant in document_plants] + [
                anchor
                for doc_id, anchor in sorted(visible_distractor_anchors)
                if doc_id == document["doc_id"]
            ]
            truncated = _truncate_document(text, docs_length, visible_anchors)

            # Optional noise must not crowd a scored actor key out of a bounded
            # excerpt. Prefer the ordinary density view; only discard its
            # neutral spans when doing so preserves more complete candidates.
            support_first_source = _remove_document_anchors(
                text,
                [
                    anchor
                    for doc_id, anchor in sorted(visible_neutral_anchors)
                    if doc_id == document["doc_id"]
                ],
            )
            support_first_source = " ".join(support_first_source.split())
            protected_anchors = [plant["anchor_quote"] for plant in document_plants] + [
                anchor
                for doc_id, anchor in sorted(required_actor_support_anchors)
                if doc_id == document["doc_id"]
            ]
            support_first = _truncate_document(
                support_first_source,
                docs_length,
                protected_anchors,
            )
            standard_complete = sum(
                plant["anchor_quote"] in truncated
                and _plant_actor_aliases_visible(plant, truncated)
                for plant in document_plants
            )
            support_first_complete = sum(
                plant["anchor_quote"] in support_first
                and _plant_actor_aliases_visible(plant, support_first)
                for plant in document_plants
            )
            text = (
                support_first
                if support_first_complete > standard_complete
                else truncated
            )
        documents.append({**document, "text": text})

    visible_text_by_doc = {
        document["doc_id"]: document["text"] for document in documents
    }
    initially_visible_plants = [
        plant
        for plant in selected_plants
        if plant["doc_id"] in visible_text_by_doc
        and plant["anchor_quote"] in visible_text_by_doc[plant["doc_id"]]
    ]
    visible_plants = [
        plant
        for plant in initially_visible_plants
        if _plant_actor_aliases_visible(
            plant,
            visible_text_by_doc[plant["doc_id"]],
        )
    ]
    unobservable_anchors = {
        (plant["doc_id"], plant["anchor_quote"])
        for plant in initially_visible_plants
        if plant not in visible_plants
    }
    if unobservable_anchors:
        # A prefix may preserve an issue anchor but truncate its accepted actor
        # evidence. Remove that now-unscorable issue rather than emitting a
        # visible prompt with a hidden semantic answer-key requirement.
        documents = [
            {
                **document,
                "text": " ".join(
                    _remove_document_anchors(
                        document["text"],
                        [
                            anchor
                            for doc_id, anchor in sorted(unobservable_anchors)
                            if doc_id == document["doc_id"]
                        ],
                    ).split()
                ),
            }
            for document in documents
        ]
    return documents, visible_plants


def _plant_actor_aliases(plant: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the exact actor concepts accepted by the semantic scorer."""

    decision = plant.get("decision")
    canonical = (
        str(decision.get("actor", "")).strip() if isinstance(decision, Mapping) else ""
    )
    decision_aliases = plant.get("decision_aliases")
    raw_aliases = (
        decision_aliases.get("actor", ())
        if isinstance(decision_aliases, Mapping)
        else ()
    )
    aliases = tuple(
        str(alias).strip()
        for alias in raw_aliases
        if isinstance(alias, str) and alias.strip()
    )
    return aliases or ((canonical,) if canonical else ())


def _plant_actor_aliases_visible(plant: Mapping[str, Any], text: str) -> bool:
    """Require every accepted actor spelling to appear in the public document."""

    folded_text = text.casefold()
    aliases = _plant_actor_aliases(plant)
    return bool(aliases) and all(alias.casefold() in folded_text for alias in aliases)


def _remove_document_anchors(text: str, anchors: Sequence[str]) -> str:
    for anchor in anchors:
        text = text.replace(anchor, "")
    return text


def _truncate_document(text: str, limit: int, anchors: Sequence[str]) -> str:
    if len(text) <= limit:
        return text
    truncation_index = _safe_truncation_index(text, limit, anchors)
    return text[:truncation_index].rstrip() + "…"


def render_prompt(
    documents: Sequence[Mapping[str, str]],
    *,
    factions: Sequence[Mapping[str, Any]] = (),
    question_count: int = 1,
) -> str:
    """Render the finding task and its observable weight-zero T2 metric."""

    question_object_label = "question object" + ("" if question_count == 1 else "s")
    lines = [
        "Find material ambiguities, contradictions, and gaps in these policy documents.",
        "Use an exact or close quote from the relevant document for each finding.",
        "Do not flag a passage merely because it discusses a rule; report only a concrete issue.",
        "",
        "Stakeholder factions for the companion question-utility metric:",
    ]
    lines.extend(
        f"[{faction['faction_id']}] {faction['name']}: {faction['summary']}"
        for faction in factions
    )
    lines.extend(["", "Documents:"])
    for document in documents:
        lines.extend(
            [
                "",
                f"[{document['doc_id']}] {document['title']} ({document['style']})",
                document["text"],
            ]
        )
    lines.extend(
        [
            "",
            (
                "Return STRICT JSON only. The extended form below includes the "
                "optional weight-zero companion; a findings-only root is valid:"
            ),
            (
                '{"findings":[{"doc_id":"<document id>","quote":"<minimal passage>","type":"ambiguity|contradiction|gap",'
                '"diagnosis":"<yes/no question naming the unresolved decision>",'
                '"decision":{"actor":"<decision maker>","action":"<decision action>","condition":"<trigger or scope>","anchor_outcome":"<outcome preserving the primary rule>","alternative_outcome":"<clarification, fallback, or conflicting outcome>"},'
                '"related_evidence":null|{"doc_id":"<conflicting document id>","quote":"<conflicting passage>"}}],'
                '"questions":[{"doc_id":"<document id>","quote":"<passage>","type":"ambiguity|contradiction|gap",'
                '"question":"<specific yes/no clarifying question>",'
                '"decision":{"actor":"<decision maker>","action":"<decision action>","condition":"<trigger or scope>","anchor_outcome":"<outcome preserving the primary rule>","alternative_outcome":"<clarification, fallback, or conflicting outcome>"},'
                '"yes_choice":"anchor|alternative",'
                '"related_evidence":null|{"doc_id":"<conflicting document id>","quote":"<conflicting passage>"},'
                '"target_stances":{"<faction id>":"agree|disagree|pass"}}]}'
            ),
            f"The root may also include exactly {question_count} optional {question_object_label} for the logged weight-zero companion metric. Omitting or malformed companion questions never changes the findings reward.",
            "Phrase each question as yes/no: agree means that faction predicts yes, disagree means no, and pass means no position.",
            "Set yes_choice to anchor when yes preserves or applies the primary quoted rule; set it to alternative when yes favors a clarification, fallback, or the second conflicting rule.",
            "For every finding and question, fill all five decision fields from the visible documents. The prose must express that same actor, action, condition, and the outcome designated by yes_choice.",
            'Because findings have no yes_choice field, end each diagnosis with "(yes selects the anchor outcome: <anchor_outcome>)" or "(yes selects the alternative outcome: <alternative_outcome>)".',
            "For contradictions, related_evidence must quote the second conflicting rule. For other finding types it must be null.",
        ]
    )
    return "\n".join(lines)


def render_ask_prompt(
    documents: Sequence[Mapping[str, str]],
    factions: Sequence[Mapping[str, Any]],
    candidate_plants: Sequence[Mapping[str, Any]],
    question_count: int,
    *,
    panel_polarization: float,
) -> str:
    """Render Ask as selection among explicit synthetic decision profiles.

    The profile list exposes exactly the issue-specific weights used with the
    already visible faction vectors. It deliberately omits evidence locations,
    issue labels, relationships, stored stances, and precomputed utilities, so
    the model must still ground and diagnose each candidate in the documents.
    """

    question_label = "question" if question_count == 1 else "questions"
    lines = [
        f"Select and raise exactly {question_count} clarifying {question_label} grounded in these policy documents.",
        "Each question should expose a specific ambiguity, contradiction, or uncovered case that could split the listed stakeholder factions.",
        "Phrase every question as yes/no. Predict every faction's answer: agree means yes, disagree means no, and pass means no position.",
        "Set yes_choice to anchor when yes preserves or applies the primary quoted rule; set it to alternative when yes favors a clarification, fallback, or the second conflicting rule. The stance labels are scored in that declared orientation.",
        "Choose from more candidate issues than the output budget. Candidate order is not a ranking.",
        "For each candidate and faction, compute alternative_preference = sum(faction_value[d] * alternative_tradeoff_weight[d]) / sum(abs(alternative_tradeoff_weight[d])). Scores >= +0.25 favor the alternative, scores <= -0.25 favor the anchor, and scores strictly between those thresholds pass.",
        "Orient target_stances to your submitted yes_choice: alternative keeps those labels; anchor swaps agree and disagree while leaving pass unchanged.",
        "Rank candidates by utility = decision_value * ((1 - panel_polarization) + panel_polarization * disagreement). decision_value = min(1, 4 * (RMS(alternative_preference) * ((max(alternative_preference) - min(alternative_preference)) / 2))^2). disagreement is the mean of (a) normalized three-label Shannon entropy and (b) the fraction of unordered faction pairs with different labels.",
        f"This row uses panel_polarization={panel_polarization:.6g} and pass_threshold={PASS_THRESHOLD:.6g}. Select the {question_count} highest-utility candidate profile(s).",
        "Copy the exact supporting passage into quote and classify its issue type. For contradictions, also copy the second conflicting rule into related_evidence; otherwise related_evidence must be null.",
        "Copy the selected profile's complete decision object into the response. Ground it in the visible documents; the yes/no prose must express that same actor, action, condition, and the outcome designated by yes_choice. Generic questions receive no semantic credit.",
        "",
        "Stakeholder factions:",
    ]
    lines.extend(
        f"[{faction['faction_id']}] {faction['name']}: {faction['summary']}"
        for faction in factions
    )
    lines.extend(
        [
            "",
            "Candidate decision profiles (unordered; one per possible issue):",
            json.dumps(
                _public_candidate_profiles(candidate_plants),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ),
        ]
    )
    lines.extend(["", "Documents:"])
    for document in documents:
        lines.extend(
            [
                "",
                f"[{document['doc_id']}] {document['title']} ({document['style']})",
                document["text"],
            ]
        )
    lines.extend(
        [
            "",
            "Return STRICT JSON only, with this shape:",
            '{"questions":[{"doc_id":"<document id>","quote":"<passage>","type":"ambiguity|contradiction|gap","question":"<specific clarifying question>","decision":{"actor":"<decision maker>","action":"<decision action>","condition":"<trigger or scope>","anchor_outcome":"<outcome preserving the primary rule>","alternative_outcome":"<clarification, fallback, or conflicting outcome>"},"yes_choice":"anchor|alternative","related_evidence":null|{"doc_id":"<conflicting document id>","quote":"<conflicting passage>"},"target_stances":{"<faction id>":"agree|disagree|pass"}}]}',
        ]
    )
    return "\n".join(lines)


def _public_candidate_profiles(
    candidate_plants: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return only the candidate fields intentionally visible in Ask.

    Sorting by the authored decision frame, never by a hidden label or utility,
    prevents list position from becoming the selection answer.
    """

    profiles = [
        {
            "decision": {
                field: str(plant["decision"][field])
                for field in sorted(DECISION_FRAME_FIELDS)
            },
            "alternative_tradeoff_weights": {
                dimension: float(plant["value_weights"][dimension])
                for dimension in VALUE_DIMENSIONS
            },
        }
        for plant in candidate_plants
    ]
    profiles.sort(
        key=lambda profile: json.dumps(
            profile["decision"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    )
    return profiles


async def question_utility(
    completion: list[dict[str, Any]],
    answer: Mapping[str, Any] | str,
    info: Mapping[str, Any] | str,
    parser: ElicitJsonParser,
) -> float:
    """Reward questions that match planted issues and their specific faction split."""

    answer_payload = _parse_mapping_payload(answer)
    info_payload = _parse_mapping_payload(info)
    parsed = parser.parse_answer(completion)
    candidates = parse_candidate_questions(
        parsed,
        allow_findings=bool(info_payload.get("allow_combined_questions", False)),
    )
    if candidates is None:
        return 0.0
    planted = answer_payload.get("questions", [])
    if not isinstance(planted, Sequence) or isinstance(planted, str):
        return 0.0
    return question_utility_score(
        candidates,
        planted,
        panel_polarization=float(info_payload.get("panel_polarization", 0.0)),
        question_count=int(info_payload.get("question_count", 0)),
    )


async def question_format_validity(
    completion: list[dict[str, Any]],
    answer: Mapping[str, Any] | str,
    info: Mapping[str, Any] | str,
    parser: ElicitJsonParser,
) -> float:
    """Metric: exact response schema and requested cardinality are valid."""

    del answer
    info_payload = _parse_mapping_payload(info)
    parsed = parser.parse_answer(completion)
    candidates = parse_candidate_questions(
        parsed,
        allow_findings=bool(info_payload.get("allow_combined_questions", False)),
    )
    question_count = int(info_payload.get("question_count", 0))
    return float(candidates is not None and len(candidates) == question_count)


async def question_top1_selection_accuracy(
    completion: list[dict[str, Any]],
    answer: Mapping[str, Any] | str,
    info: Mapping[str, Any] | str,
    parser: ElicitJsonParser,
) -> float:
    """Metric: exact public evidence selects the highest-utility candidate."""

    answer_payload = _parse_mapping_payload(answer)
    info_payload = _parse_mapping_payload(info)
    parsed = parser.parse_answer(completion)
    candidates = parse_candidate_questions(
        parsed,
        allow_findings=bool(info_payload.get("allow_combined_questions", False)),
    )
    planted = answer_payload.get("questions", [])
    question_count = int(info_payload.get("question_count", 0))
    if (
        candidates is None
        or len(candidates) != question_count
        or question_count <= 0
        or not isinstance(planted, Sequence)
        or isinstance(planted, str)
    ):
        return 0.0
    top_plants = sorted(
        planted,
        key=lambda plant: (
            -_plant_attainable_question_utility(
                plant,
                panel_polarization=float(info_payload.get("panel_polarization", 0.0)),
            ),
            str(plant.get("doc_id", "")),
            str(plant.get("quote", "")),
        ),
    )[:question_count]
    eligible = [
        [_candidate_plant_evidence_match(candidate, plant) for plant in top_plants]
        for candidate in candidates
    ]
    selected, _, _ = _best_question_assignment(
        eligible,
        [[0] * len(top_plants) for _ in candidates],
        [[0] * len(top_plants) for _ in candidates],
    )
    return selected / question_count


async def question_grounding_recall(
    completion: list[dict[str, Any]],
    answer: Mapping[str, Any] | str,
    info: Mapping[str, Any] | str,
    parser: ElicitJsonParser,
) -> float:
    """Metric: selected questions match exact public structured evidence."""

    components = _question_diagnostic_components(completion, answer, info, parser)
    return components[0]


async def question_grounded_stance_recall(
    completion: list[dict[str, Any]],
    answer: Mapping[str, Any] | str,
    info: Mapping[str, Any] | str,
    parser: ElicitJsonParser,
) -> float:
    """Metric: end-to-end stance credit, with ungrounded questions scored zero."""

    components = _question_diagnostic_components(completion, answer, info, parser)
    return components[1]


async def question_evidence_matched_stance_accuracy(
    completion: list[dict[str, Any]],
    answer: Mapping[str, Any] | str,
    info: Mapping[str, Any] | str,
    parser: ElicitJsonParser,
) -> float:
    """Metric: stance accuracy after exact evidence localization.

    Unlike grounded stance recall, this denominator does not require the
    submitted decision/question semantics to pass.  It therefore isolates
    faction-orientation quality after the model has found the right passage.
    """

    components = _question_diagnostic_components(completion, answer, info, parser)
    return components[3]


async def question_evidence_match_recall(
    completion: list[dict[str, Any]],
    answer: Mapping[str, Any] | str,
    info: Mapping[str, Any] | str,
    parser: ElicitJsonParser,
) -> float:
    """Metric: fraction of requested questions with exact evidence matches."""

    components = _question_diagnostic_components(completion, answer, info, parser)
    return components[2]


async def question_conditional_stance_accuracy(
    completion: list[dict[str, Any]],
    answer: Mapping[str, Any] | str,
    info: Mapping[str, Any] | str,
    parser: ElicitJsonParser,
) -> float:
    """Deprecated source-compatible alias for the evidence-matched metric."""

    return await question_evidence_matched_stance_accuracy(
        completion, answer, info, parser
    )


async def question_stance_accuracy(
    completion: list[dict[str, Any]],
    answer: Mapping[str, Any] | str,
    info: Mapping[str, Any] | str,
    parser: ElicitJsonParser,
) -> float:
    """Backward-compatible alias for end-to-end grounded stance recall."""

    return await question_grounded_stance_recall(completion, answer, info, parser)


def _question_diagnostic_components(
    completion: list[dict[str, Any]],
    answer: Mapping[str, Any] | str,
    info: Mapping[str, Any] | str,
    parser: ElicitJsonParser,
) -> tuple[float, float, float, float]:
    answer_payload = _parse_mapping_payload(answer)
    info_payload = _parse_mapping_payload(info)
    parsed = parser.parse_answer(completion)
    candidates = parse_candidate_questions(
        parsed,
        allow_findings=bool(info_payload.get("allow_combined_questions", False)),
    )
    planted = answer_payload.get("questions", [])
    question_count = int(info_payload.get("question_count", 0))
    if (
        candidates is None
        or len(candidates) != question_count
        or question_count <= 0
        or not isinstance(planted, Sequence)
        or isinstance(planted, str)
    ):
        return 0.0, 0.0, 0.0, 0.0
    grounding_edges: list[list[bool]] = []
    evidence_edges: list[list[bool]] = []
    correct_stances: list[list[int]] = []
    stance_opportunities: list[list[int]] = []
    for candidate in candidates:
        candidate_grounding: list[bool] = []
        candidate_evidence: list[bool] = []
        candidate_correct_stances: list[int] = []
        candidate_stance_opportunities: list[int] = []
        for plant in planted:
            evidence_matched = _candidate_plant_evidence_match(candidate, plant)
            grounded = (
                evidence_matched
                and _candidate_plant_grounding(candidate, plant) is not None
            )
            candidate_evidence.append(evidence_matched)
            candidate_grounding.append(grounded)
            expected = _stances_for_yes_choice(
                plant, str(candidate.get("yes_choice", ""))
            )
            submitted = candidate.get("target_stances", {})
            exact_faction_set = isinstance(submitted, Mapping) and set(
                submitted
            ) == set(expected)
            correct_count = sum(
                exact_faction_set and submitted.get(faction_id) == stance
                for faction_id, stance in expected.items()
            )
            candidate_correct_stances.append(correct_count)
            candidate_stance_opportunities.append(len(expected))
        grounding_edges.append(candidate_grounding)
        evidence_edges.append(candidate_evidence)
        correct_stances.append(candidate_correct_stances)
        stance_opportunities.append(candidate_stance_opportunities)

    grounded_count, grounded_correct, _ = _best_question_assignment(
        grounding_edges, correct_stances, stance_opportunities
    )
    evidence_count, evidence_correct, evidence_opportunities = (
        _best_question_assignment(evidence_edges, correct_stances, stance_opportunities)
    )
    faction_count = max(
        (
            len(plant.get("target_stances", {}))
            if isinstance(plant, Mapping)
            and isinstance(plant.get("target_stances"), Mapping)
            else 0
            for plant in planted
        ),
        default=0,
    )
    total_stance_opportunities = question_count * faction_count
    return (
        grounded_count / question_count,
        grounded_correct / total_stance_opportunities
        if total_stance_opportunities
        else 0.0,
        evidence_count / question_count,
        evidence_correct / evidence_opportunities if evidence_opportunities else 0.0,
    )


def _best_question_assignment(
    eligible: Sequence[Sequence[bool]],
    correct_stances: Sequence[Sequence[int]],
    stance_opportunities: Sequence[Sequence[int]],
) -> tuple[int, int, int]:
    """Pair candidates and plants once, prioritizing coverage then stance credit.

    Ask exposes at most three candidate issues, so exhaustive assignment is
    clearer than independently optimizing several weight matrices.  Reusing a
    single deterministic matching prevents coverage and conditional stance
    diagnostics from silently referring to different candidate/plant pairs.
    """

    best_objective = (-1, -1)
    best_pairs: tuple[tuple[int, int], ...] | None = None
    best_result = (0, 0, 0)

    def visit(
        candidate_index: int,
        used_plants: frozenset[int],
        pairs: tuple[tuple[int, int], ...],
        correct: int,
        opportunities: int,
    ) -> None:
        nonlocal best_objective, best_pairs, best_result
        if candidate_index == len(eligible):
            objective = (len(pairs), correct)
            if objective > best_objective or (
                objective == best_objective
                and (best_pairs is None or pairs < best_pairs)
            ):
                best_objective = objective
                best_pairs = pairs
                best_result = (len(pairs), correct, opportunities)
            return

        visit(candidate_index + 1, used_plants, pairs, correct, opportunities)
        for plant_index, is_eligible in enumerate(eligible[candidate_index]):
            if not is_eligible or plant_index in used_plants:
                continue
            visit(
                candidate_index + 1,
                used_plants | {plant_index},
                (*pairs, (candidate_index, plant_index)),
                correct + int(correct_stances[candidate_index][plant_index]),
                opportunities + int(stance_opportunities[candidate_index][plant_index]),
            )

    visit(0, frozenset(), (), 0, 0)
    return best_result


def parse_candidate_questions(
    parsed: Any, *, allow_findings: bool = False
) -> list[dict[str, Any]] | None:
    """Validate the exact T2 response schema, returning None on any error."""

    allowed_roots = (
        ({"questions"}, {"findings", "questions"})
        if allow_findings
        else ({"questions"},)
    )
    if not isinstance(parsed, Mapping) or set(parsed) not in allowed_roots:
        return None
    if "findings" in parsed and _parse_findings_list(parsed["findings"]) is None:
        return None
    return _parse_questions_list(parsed["questions"])


def _parse_questions_list(raw_questions: Any) -> list[dict[str, Any]] | None:
    if not isinstance(raw_questions, list):
        return None
    questions: list[dict[str, Any]] = []
    expected_fields = {
        "doc_id",
        "quote",
        "type",
        "question",
        "decision",
        "yes_choice",
        "related_evidence",
        "target_stances",
    }
    for raw_question in raw_questions:
        if (
            not isinstance(raw_question, Mapping)
            or set(raw_question) != expected_fields
        ):
            return None
        if not all(
            isinstance(raw_question[field], str) and raw_question[field].strip()
            for field in ("doc_id", "quote", "type", "question", "yes_choice")
        ):
            return None
        if raw_question["type"] not in FINDING_TYPES:
            return None
        if raw_question["yes_choice"] not in {"anchor", "alternative"}:
            return None
        decision = _parse_decision_frame(raw_question["decision"])
        if decision is None:
            return None
        related = raw_question["related_evidence"]
        if related is not None and (
            not isinstance(related, Mapping)
            or set(related) != {"doc_id", "quote"}
            or not all(
                isinstance(related[field], str) and related[field].strip()
                for field in ("doc_id", "quote")
            )
        ):
            return None
        target_stances = raw_question["target_stances"]
        if not isinstance(target_stances, Mapping) or not target_stances:
            return None
        if not all(
            isinstance(faction_id, str)
            and bool(faction_id.strip())
            and isinstance(stance, str)
            and stance in STANCE_TO_VOTE
            for faction_id, stance in target_stances.items()
        ):
            return None
        questions.append(
            {
                "doc_id": raw_question["doc_id"],
                "quote": raw_question["quote"],
                "type": raw_question["type"],
                "question": raw_question["question"],
                "decision": decision,
                "yes_choice": raw_question["yes_choice"],
                "related_evidence": dict(related)
                if isinstance(related, Mapping)
                else None,
                "target_stances": dict(target_stances),
            }
        )
    return questions


def _parse_decision_frame(raw_decision: Any) -> dict[str, str] | None:
    """Return a bounded exact decision frame or fail closed."""

    if (
        not isinstance(raw_decision, Mapping)
        or set(raw_decision) != DECISION_FRAME_FIELDS
    ):
        return None
    decision: dict[str, str] = {}
    for field in sorted(DECISION_FRAME_FIELDS):
        value = raw_decision[field]
        if not isinstance(value, str) or not value.strip() or len(value) > 1_000:
            return None
        decision[field] = value.strip()
    return decision


def _accepted_decision_aliases(
    plant: Mapping[str, Any], reference: Mapping[str, str]
) -> dict[str, tuple[str, ...]]:
    """Return validated per-slot concepts, defaulting to the canonical frame.

    Older local rows may not carry explicit aliases.  The fallback preserves
    their exact semantics while 0.6 rows can accept authored source-language
    alternatives without permitting a concept to move into another slot.
    """

    raw_aliases = plant.get("decision_aliases")
    if (
        not isinstance(raw_aliases, Mapping)
        or set(raw_aliases) != DECISION_FRAME_FIELDS
    ):
        return {field: (reference[field],) for field in DECISION_FRAME_FIELDS}
    aliases: dict[str, tuple[str, ...]] = {}
    for field in DECISION_FRAME_FIELDS:
        raw_field_aliases = raw_aliases[field]
        if (
            not isinstance(raw_field_aliases, Sequence)
            or isinstance(raw_field_aliases, str)
            or not raw_field_aliases
        ):
            return {name: (reference[name],) for name in DECISION_FRAME_FIELDS}
        accepted = tuple(
            alias.strip()
            for alias in raw_field_aliases
            if isinstance(alias, str) and alias.strip()
        )
        if len(accepted) != len(raw_field_aliases):
            return {name: (reference[name],) for name in DECISION_FRAME_FIELDS}
        aliases[field] = accepted
    return aliases


def question_utility_score(
    candidates: Sequence[Mapping[str, Any]],
    planted: Sequence[Mapping[str, Any]],
    *,
    panel_polarization: float,
    question_count: int,
) -> float:
    """Return normalized top-K issue selection, semantics, and stance accuracy."""

    if len(candidates) != question_count or question_count <= 0:
        return 0.0
    weights: list[list[float]] = []
    for candidate in candidates:
        candidate_weights = [
            _candidate_plant_question_utility(
                candidate,
                plant,
                panel_polarization=panel_polarization,
            )
            for plant in planted
        ]
        weights.append(candidate_weights)
    attainable = sorted(
        (
            _plant_attainable_question_utility(
                plant,
                panel_polarization=panel_polarization,
            )
            for plant in planted
        ),
        reverse=True,
    )[:question_count]
    denominator = sum(attainable)
    if denominator <= 0:
        return 0.0
    return min(1.0, _maximum_weight_sum(weights) / denominator)


def _candidate_plant_question_utility(
    candidate: Mapping[str, Any],
    plant: Mapping[str, Any],
    *,
    panel_polarization: float,
) -> float:
    semantic_grounding = _candidate_plant_grounding(candidate, plant)
    if semantic_grounding is None:
        return 0.0
    planted_stances = _stances_for_yes_choice(
        plant,
        str(candidate.get("yes_choice", "")),
    )
    candidate_stances = candidate.get("target_stances", {})
    if not isinstance(planted_stances, Mapping) or not planted_stances:
        return 0.0
    if not isinstance(candidate_stances, Mapping) or set(candidate_stances) != set(
        planted_stances
    ):
        return 0.0
    stance_accuracy = (
        sum(
            candidate_stances.get(faction_id) == stance
            for faction_id, stance in planted_stances.items()
        )
        / len(planted_stances)
        if candidate_stances
        else 0.0
    )
    attainable_utility = _plant_attainable_question_utility(
        plant,
        panel_polarization=panel_polarization,
    )
    component_score = (
        QUESTION_GROUNDING_WEIGHT + STANCE_ACCURACY_WEIGHT * stance_accuracy
    )
    return semantic_grounding * component_score * attainable_utility


def _plant_attainable_question_utility(
    plant: Mapping[str, Any],
    *,
    panel_polarization: float,
) -> float:
    """Return disagreement-weighted issue value used to normalize the row maximum."""

    stances = plant.get("target_stances", {})
    if not isinstance(stances, Mapping) or not stances:
        return 0.0
    decision_value = plant.get("decision_value", 1.0)
    if isinstance(decision_value, bool) or not isinstance(decision_value, (int, float)):
        return 0.0
    disagreement = panel_disagreement(stances)
    polarization_weight = (1 - panel_polarization) + panel_polarization * disagreement
    return float(decision_value) * polarization_weight


def _candidate_plant_grounding(
    candidate: Mapping[str, Any],
    plant: Mapping[str, Any],
) -> float | None:
    """Match a generated question to one planted issue using visible evidence."""

    if not _candidate_plant_evidence_match(candidate, plant):
        return None
    candidate_question = str(candidate.get("question", ""))
    if not is_yes_no_question(candidate_question):
        return None
    semantic_score = question_decision_similarity(
        candidate_question,
        plant,
        candidate_decision=candidate.get("decision"),
        yes_choice=str(candidate.get("yes_choice", "")),
    )
    if semantic_score <= 0:
        return None
    return semantic_score


def _candidate_plant_evidence_match(
    candidate: Mapping[str, Any],
    plant: Mapping[str, Any],
) -> bool:
    """Match only the exact public issue evidence, before semantic scoring."""

    if candidate.get("doc_id") != plant.get("doc_id"):
        return False
    if plant.get("type") is not None and candidate.get("type") != plant.get("type"):
        return False
    if not _related_evidence_matches(candidate, plant):
        return False
    candidate_quote = str(candidate.get("quote", ""))
    planted_quote = str(plant.get("quote", ""))
    document_text = str(plant.get("document_text", ""))
    return candidate_quote == planted_quote and candidate_quote in document_text


def question_decision_similarity(
    question: str,
    plant: Mapping[str, Any],
    *,
    candidate_decision: Any = None,
    yes_choice: str | None = None,
) -> float:
    """Match a declared decision frame and require the prose to express it.

    The scorer never compares against a single hidden canonical sentence.
    Instead, it checks five simulator-defined semantic slots against visible
    evidence and the reference decision concepts, then checks that the
    submitted yes/no prose contains the declared core and the outcome named by
    ``yes_choice``. This blocks polarity flips, generic questions, and unrelated
    evidence-word dumps without introducing a one-sentence answer key.
    """

    if not is_yes_no_question(question):
        return 0.0
    submitted = _parse_decision_frame(candidate_decision)
    if submitted is None:
        return 0.0
    reference = _parse_decision_frame(plant.get("decision"))
    if reference is None:
        return 0.0
    aliases = _accepted_decision_aliases(plant, reference)
    if not _decision_frame_matches_reference(
        submitted,
        reference,
        aliases,
        mandatory_condition_tokens=_mandatory_condition_scope_tokens(plant, reference),
    ):
        return 0.0
    evidence_score = _decision_frame_evidence_score(submitted, plant, aliases)
    if evidence_score <= 0:
        return 0.0
    if not _prose_expresses_decision(question, submitted, yes_choice=yes_choice):
        return 0.0
    return evidence_score


def _decision_frame_matches_reference(
    submitted: Mapping[str, str],
    reference: Mapping[str, str],
    aliases: Mapping[str, Sequence[str]],
    *,
    mandatory_condition_tokens: frozenset[str],
) -> bool:
    """Require the submitted frame to recover the simulator's decision concepts.

    Slot prose is not matched verbatim, but actor, action, and condition retain
    their separate meanings. Bidirectional per-slot coverage blocks swapping or
    repeating one evidence phrase across every field, while still permitting
    concise paraphrases through the narrow normalization table.
    """

    submitted_core_tokens = [
        _semantic_tokens(submitted[field]) for field in ("actor", "action", "condition")
    ]
    if len(set(submitted_core_tokens)) != len(submitted_core_tokens):
        return False
    for field in (
        "actor",
        "action",
        "condition",
        "anchor_outcome",
        "alternative_outcome",
    ):
        reference_threshold = 0.5
        submitted_threshold = _SUBMITTED_SLOT_PRECISION_THRESHOLD
        accepted = aliases.get(field, (reference[field],))
        if not any(
            _decision_slot_matches_alias(
                submitted[field],
                alias,
                reference_threshold=reference_threshold,
                submitted_threshold=submitted_threshold,
            )
            for alias in accepted
        ):
            return False
        # Regression guard: a generated scope suffix changes the decision.
        # Whole-slot fuzzy overlap must not let a plausible unscoped condition
        # erase the suffix shared by the source quote and authored reference.
        if field == "condition" and not mandatory_condition_tokens <= _semantic_tokens(
            submitted[field]
        ):
            return False
    return True


def _decision_slot_matches_alias(
    submitted: str,
    alias: str,
    *,
    reference_threshold: float,
    submitted_threshold: float,
) -> bool:
    """Accept omissions and normalized synonyms, but no unauthored concepts."""

    submitted_tokens = _semantic_tokens(submitted)
    alias_tokens = _semantic_tokens(alias)
    return (
        _bidirectional_concept_coverage(
            submitted_tokens,
            alias_tokens,
            reference_threshold=reference_threshold,
            submitted_threshold=submitted_threshold,
        )
        # Regression guard: an otherwise matching slot cannot append a second,
        # unsupported action such as selling records or firing a driver. Any
        # legitimate source wording with new concept tokens needs an explicit
        # authored alias instead of inheriting credit from lexical overlap.
        and submitted_tokens <= alias_tokens
        and _polarity_compatible(submitted, alias)
    )


def _mandatory_condition_scope_tokens(
    plant: Mapping[str, Any], reference: Mapping[str, str]
) -> frozenset[str]:
    """Recover a generated scope suffix shared verbatim with visible evidence."""

    if not isinstance(plant.get("decision_aliases"), Mapping):
        return frozenset()
    normalized_condition = _normalized_text(reference["condition"])
    visible_quote_tokens = _semantic_tokens(str(plant.get("quote", "")))
    for scope in SEMANTIC_SCOPES:
        if scope is None:
            continue
        normalized_scope = _normalized_text(scope)
        scope_tokens = _semantic_tokens(scope)
        if (
            normalized_condition.endswith(normalized_scope)
            and scope_tokens <= visible_quote_tokens
        ):
            return scope_tokens
    return frozenset()


def _bidirectional_concept_coverage(
    submitted: set[str] | frozenset[str],
    reference: set[str] | frozenset[str],
    *,
    reference_threshold: float,
    submitted_threshold: float,
) -> bool:
    if not submitted or not reference:
        return False
    overlap = len(submitted & reference)
    return (
        overlap / len(reference) >= reference_threshold
        and overlap / len(submitted) >= submitted_threshold
    )


def _semantic_tokens(text: str) -> frozenset[str]:
    return frozenset(
        _normalize_semantic_token(token)
        for token in normalized_tokens(text)
        if token not in _SEMANTIC_STOPWORDS
        and token not in _YES_NO_AUXILIARY_TOKENS
        and any(character.isalnum() for character in token)
    )


def _polarity_signature(text: str) -> tuple[str, ...]:
    """Return explicit meaning-reversing markers, retaining multiplicity."""

    normalized = _normalize_negating_phrases(text)
    return tuple(
        "neg" for token in normalized_tokens(normalized) if token in _POLARITY_TOKENS
    )


def _normalize_negating_phrases(text: str) -> str:
    """Map explicit and lexical negators to one deterministic marker token."""

    normalized = _NEGATION_CONTRACTION_PATTERN.sub(" not ", text)
    normalized = _FAIL_TO_PATTERN.sub(" not ", normalized)
    return _NEGATING_VERB_PATTERN.sub(" not ", normalized)


def _polarity_compatible(left: str, right: str) -> bool:
    """Reject a paraphrase that adds, removes, or swaps explicit negation."""

    return _polarity_signature(left) == _polarity_signature(right)


def _normalize_semantic_token(token: str) -> str:
    normalized = _SEMANTIC_EQUIVALENTS.get(token, token)
    if len(normalized) > 4 and normalized.endswith("ies"):
        return normalized[:-3] + "y"
    if (
        len(normalized) > 4
        and normalized.endswith("s")
        and not normalized.endswith(("ss", "us", "is"))
    ):
        return normalized[:-1]
    return normalized


def _evidence_support(candidate: str, evidence: str) -> float:
    candidate_tokens = _semantic_tokens(candidate)
    evidence_tokens = _semantic_tokens(evidence)
    if not candidate_tokens or not evidence_tokens:
        return 0.0
    return len(candidate_tokens & evidence_tokens) / len(candidate_tokens)


def _decision_frame_evidence_score(
    decision: Mapping[str, str],
    plant: Mapping[str, Any],
    aliases: Mapping[str, Sequence[str]],
) -> float:
    primary_text = " ".join(
        (
            str(plant.get("document_text", "")),
            str(plant.get("quote", plant.get("anchor_quote", ""))),
        )
    )
    related = plant.get("related_evidence")
    related_document_text = plant.get("related_document_text")
    related_text = " ".join(
        (
            related_document_text if isinstance(related_document_text, str) else "",
            str(related.get("quote", "")) if isinstance(related, Mapping) else "",
        )
    )
    # Validate the response's own slot text against source passages.  The
    # alias table only defines acceptable concepts; using aliases as the
    # evidence probe would allow a response to append an unsupported negation
    # while inheriting the canonical alias's source support.
    all_evidence = " ".join((primary_text, related_text))
    alternative_evidence = (
        related_text if _semantic_tokens(related_text) else primary_text
    )
    evidence_by_field = {
        "actor": all_evidence,
        "action": all_evidence,
        "condition": all_evidence,
        "anchor_outcome": primary_text,
        "alternative_outcome": alternative_evidence,
    }
    for field, evidence in evidence_by_field.items():
        submitted_threshold = _SUBMITTED_SLOT_PRECISION_THRESHOLD
        matching_aliases = [
            alias
            for alias in aliases.get(field, (decision[field],))
            if _decision_slot_matches_alias(
                decision[field],
                alias,
                reference_threshold=0.5,
                submitted_threshold=submitted_threshold,
            )
        ]
        alias_support = max(
            (_evidence_support(alias, evidence) for alias in matching_aliases),
            default=0.0,
        )
        # Some authored concepts describe what the visible policy fails to
        # specify and therefore have only partial (or no literal) passage
        # support. Whenever the authored concept is observable, require the
        # response's own slot text to retain that evidence support rather than
        # inheriting it from the alias table.
        if alias_support > 0 and _evidence_support(decision[field], evidence) < min(
            0.4, alias_support
        ):
            return 0.0
    if _semantic_tokens(decision["anchor_outcome"]) == _semantic_tokens(
        decision["alternative_outcome"]
    ):
        return 0.0
    return 1.0


def _prose_expresses_decision(
    prose: str,
    decision: Mapping[str, str],
    *,
    yes_choice: str | None,
) -> bool:
    marker_matches = list(_ORIENTATION_MARKER_PATTERN.finditer(prose))
    if len(marker_matches) > 1:
        return False
    declared_choice: str | None = None
    declared_outcome_text = ""
    stem = prose
    if marker_matches:
        marker_match = marker_matches[0]
        declared_choice = marker_match.group(1).casefold()
        stem = prose[: marker_match.start()]
        declared_outcome_text = prose[marker_match.end() :].lstrip(" :")

    if not _decision_stem_expresses_core(stem, decision):
        return False

    selected_choice = yes_choice if yes_choice in {"anchor", "alternative"} else None
    if declared_choice is not None:
        if selected_choice is not None and declared_choice != selected_choice:
            return False
        selected_choice = declared_choice
    if selected_choice is None:
        return False
    opposite = "alternative" if selected_choice == "anchor" else "anchor"
    selected_outcome = decision[f"{selected_choice}_outcome"]
    opposite_outcome = decision[f"{opposite}_outcome"]

    core_text = " ".join(decision[field] for field in ("actor", "action", "condition"))
    stem_polarity = _polarity_signature(stem)
    allowed_stem_polarities = {
        _polarity_signature(core_text),
        _polarity_signature(f"{core_text} {selected_outcome}"),
        _polarity_signature(
            f"{core_text} {decision['anchor_outcome']} "
            f"{decision['alternative_outcome']}"
        ),
    }
    if stem_polarity not in allowed_stem_polarities:
        return False

    outcome_prose = declared_outcome_text if declared_choice is not None else stem
    outcome_tokens = _semantic_tokens(outcome_prose)
    selected_tokens = _semantic_tokens(selected_outcome)
    if not _bidirectional_concept_coverage(
        outcome_tokens,
        selected_tokens,
        reference_threshold=0.5,
        submitted_threshold=0.4 if declared_choice is not None else 0.2,
    ):
        return False
    # The orientation tail is part of the response's semantic contract, not a
    # comment channel. Extra outcome concepts must be explicitly represented by
    # the submitted decision frame instead of riding on a matching phrase.
    if declared_choice is not None and not outcome_tokens <= selected_tokens:
        return False
    if not _polarity_compatible(outcome_prose, selected_outcome):
        return False
    return _semantic_similarity(outcome_prose, selected_outcome) > _semantic_similarity(
        outcome_prose, opposite_outcome
    )


def _semantic_similarity(left: str, right: str) -> float:
    """Return unordered semantic-token F1 for disambiguating outcome prose."""

    left_tokens = _semantic_tokens(left)
    right_tokens = _semantic_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return 2 * len(left_tokens & right_tokens) / (len(left_tokens) + len(right_tokens))


def _decision_stem_expresses_core(stem: str, decision: Mapping[str, str]) -> bool:
    """Require actor, action, and condition in the pre-marker question stem."""

    stem_tokens = _semantic_tokens(stem)
    if len(stem_tokens) < 3:
        return False
    supported_tokens = frozenset().union(
        *(_semantic_tokens(decision[field]) for field in DECISION_FRAME_FIELDS)
    )
    # The structured frame is the response's explicit semantic contract. A
    # question may omit detail, but it may not append an unsupported action and
    # inherit credit from its otherwise valid actor/action/condition. Genuine
    # source-language alternatives belong in the submitted frame and authored
    # alias table, keeping this check deterministic and auditable.
    if not stem_tokens - supported_tokens <= _QUESTION_INFERENCE_TOKENS:
        return False
    for field in ("actor", "action", "condition"):
        concept_tokens = _semantic_tokens(decision[field])
        overlap = len(stem_tokens & concept_tokens)
        if not concept_tokens or overlap < min(2, len(concept_tokens)):
            return False
        if overlap / len(concept_tokens) < 0.4:
            return False
    return _stem_core_polarity_is_slot_consistent(stem, decision)


def _stem_core_polarity_is_slot_consistent(
    stem: str, decision: Mapping[str, str]
) -> bool:
    """Bind each explicit core negator to its nearest semantic slot.

    Comparing only the total number of negators lets a response move ``not``
    from a condition to an action while preserving aggregate polarity.  Core
    fields are already required separately above, so their field-specific
    tokens provide deterministic anchors for assigning each visible negator.
    """

    fields = ("actor", "action", "condition")
    field_tokens = {field: _semantic_tokens(decision[field]) for field in fields}
    position_tokens = tuple(
        _normalize_semantic_token(token)
        for token in normalized_tokens(_normalize_negating_phrases(stem))
    )
    field_positions: dict[str, tuple[int, ...]] = {}
    for field in fields:
        other_tokens = set().union(
            *(field_tokens[other] for other in fields if other != field)
        )
        distinctive = field_tokens[field] - other_tokens
        positions = tuple(
            index for index, token in enumerate(position_tokens) if token in distinctive
        )
        if not positions:
            positions = tuple(
                index
                for index, token in enumerate(position_tokens)
                if token in field_tokens[field]
            )
        if not positions:
            return False
        field_positions[field] = positions

    assigned = dict.fromkeys(fields, 0)
    for index, token in enumerate(position_tokens):
        if token not in _POLARITY_TOKENS:
            continue
        distances = {
            field: min(abs(index - position) for position in positions)
            for field, positions in field_positions.items()
        }
        nearest_distance = min(distances.values())
        nearest_fields = [
            field
            for field, distance in distances.items()
            if distance == nearest_distance
        ]
        if len(nearest_fields) != 1:
            return False
        assigned[nearest_fields[0]] += 1

    return all(
        assigned[field] == len(_polarity_signature(decision[field])) for field in fields
    )


def _stances_for_yes_choice(
    plant: Mapping[str, Any], yes_choice: str
) -> Mapping[str, str]:
    """Orient the composed preference vector to the candidate's declared yes side."""

    alternative = plant.get("alternative_stances", {})
    if not isinstance(alternative, Mapping) or not alternative:
        canonical = plant.get("target_stances", {})
        canonical_choice = str(plant.get("yes_choice", "alternative"))
        if not isinstance(canonical, Mapping) or not canonical:
            return {}
        if yes_choice == canonical_choice:
            return {
                str(faction_id): str(stance) for faction_id, stance in canonical.items()
            }
        if yes_choice not in {"anchor", "alternative"}:
            return {}
        inverse = {"agree": "disagree", "disagree": "agree", "pass": "pass"}
        return {
            str(faction_id): inverse.get(str(stance), "")
            for faction_id, stance in canonical.items()
        }
    if yes_choice == "alternative":
        return {
            str(faction_id): str(stance) for faction_id, stance in alternative.items()
        }
    if yes_choice != "anchor":
        return {}
    inverse = {"agree": "disagree", "disagree": "agree", "pass": "pass"}
    return {
        str(faction_id): inverse.get(str(stance), "")
        for faction_id, stance in alternative.items()
    }


def panel_disagreement(target_stances: Mapping[str, str]) -> float:
    """Combine score-package entropy and pair separation for planted votes."""

    votes: list[int | None] = [
        STANCE_TO_VOTE[stance] for stance in target_stances.values()
    ]
    return float((vote_entropy(votes) + cluster_separation(votes)) / 2)


def _normalized_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(_TOKEN_PATTERN.findall(normalized))


async def finding_f1(
    completion: list[dict[str, Any]],
    answer: Mapping[str, Any] | str,
    parser: ElicitJsonParser,
) -> float:
    """Reward strict finding output by one-to-one planted-item F1."""

    answer_payload = _parse_mapping_payload(answer)
    parsed = parser.parse_answer(completion)
    candidates = parse_candidate_findings(
        parsed,
        question_count=_find_question_count(answer_payload),
    )
    if candidates is None:
        return 0.0
    planted = parse_planted_items(answer_payload.get("findings", []))
    return match_findings(candidates, planted)["f1"]


async def finding_localization_recall(
    completion: list[dict[str, Any]],
    answer: Mapping[str, Any] | str,
    parser: ElicitJsonParser,
) -> float:
    """Metric: fraction of planted spans localized regardless of diagnosis type."""

    result = _scored_findings(completion, answer, parser)
    return float(result.get("localization_recall", 0.0))


async def finding_type_accuracy(
    completion: list[dict[str, Any]],
    answer: Mapping[str, Any] | str,
    parser: ElicitJsonParser,
) -> float:
    """Metric: type accuracy among candidate-localized planted spans."""

    result = _scored_findings(completion, answer, parser)
    return float(result.get("type_accuracy", 0.0))


async def finding_diagnosis_recall(
    completion: list[dict[str, Any]],
    answer: Mapping[str, Any] | str,
    parser: ElicitJsonParser,
) -> float:
    """Metric: recall after localization, type, and diagnosis form checks."""

    result = _scored_findings(completion, answer, parser)
    return float(result.get("diagnosis_recall", 0.0))


async def finding_relation_recall(
    completion: list[dict[str, Any]],
    answer: Mapping[str, Any] | str,
    parser: ElicitJsonParser,
) -> float:
    """Metric: recall after all checks, including contradiction relationships."""

    result = _scored_findings(completion, answer, parser)
    return float(result.get("relation_recall", 0.0))


async def finding_training_reward(
    completion: list[dict[str, Any]],
    answer: Mapping[str, Any] | str,
    parser: ElicitJsonParser,
) -> float:
    """Optional precision-sensitive curriculum reward.

    Each stage uses F1 rather than recall, so unmatched or hedged candidates
    reduce reward at the first stage where they fail. Strict end-to-end F1
    remains the default evaluation reward.
    """

    result = _scored_findings(completion, answer, parser)
    return float(
        0.25
        * sum(
            float(result.get(component, 0.0))
            for component in (
                "localization_f1",
                "type_f1",
                "diagnosis_f1",
                "relation_f1",
            )
        )
    )


def _scored_findings(
    completion: list[dict[str, Any]],
    answer: Mapping[str, Any] | str,
    parser: ElicitJsonParser,
) -> dict[str, float | int]:
    answer_payload = _parse_mapping_payload(answer)
    parsed = parser.parse_answer(completion)
    candidates = parse_candidate_findings(
        parsed,
        question_count=_find_question_count(answer_payload),
    )
    if candidates is None:
        return {}
    planted = parse_planted_items(answer_payload.get("findings", []))
    return match_findings(candidates, planted)


def _parse_mapping_payload(payload: Mapping[str, Any] | str) -> Mapping[str, Any]:
    """Decode a trusted canonical answer/info payload for scoring.

    The 32 KiB limit protects model completions, not environment-authored
    answer keys.  Hidden rows still receive the same depth and node checks, so
    a long source document cannot silently make its own exact answer unscorable.
    """

    try:
        loaded = _load_trusted_json(payload) if isinstance(payload, str) else payload
        _validate_json_shape(loaded)
    except (RecursionError, TypeError, ValueError):
        return {}
    return loaded if isinstance(loaded, Mapping) else {}


def _find_question_count(answer: Mapping[str, Any]) -> int | None:
    """Return the declared Find companion cardinality for 0.6 answer keys."""

    value = answer.get("question_count")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def parse_candidate_findings(
    parsed: Any, *, question_count: int | None = None
) -> list[dict[str, Any]] | None:
    """Validate the primary T1 response independently of its companion metric.

    A Find completion may omit the weight-zero ``questions`` array.  When it is
    present, its own metric validates it separately; malformed companion data
    cannot erase an otherwise valid primary finding score.
    """

    if not isinstance(parsed, Mapping):
        return None
    allowed_roots = (
        ({"findings"}, {"findings", "questions"})
        if question_count is not None
        else ({"findings"},)
    )
    if set(parsed) not in allowed_roots:
        return None
    return _parse_findings_list(parsed["findings"])


def _parse_findings_list(raw_findings: Any) -> list[dict[str, Any]] | None:
    if not isinstance(raw_findings, list):
        return None
    findings: list[dict[str, Any]] = []
    seen_spans: set[tuple[str, tuple[str, ...]]] = set()
    for raw_finding in raw_findings:
        if not isinstance(raw_finding, Mapping) or set(raw_finding) != {
            "doc_id",
            "quote",
            "type",
            "diagnosis",
            "decision",
            "related_evidence",
        }:
            return None
        if not all(
            isinstance(raw_finding[field], str)
            for field in ("doc_id", "quote", "type", "diagnosis")
        ):
            return None
        if not all(
            raw_finding[field].strip() for field in ("doc_id", "quote", "diagnosis")
        ):
            return None
        if raw_finding["type"] not in FINDING_TYPES:
            return None
        decision = _parse_decision_frame(raw_finding["decision"])
        if decision is None:
            return None
        related = raw_finding["related_evidence"]
        if related is not None and (
            not isinstance(related, Mapping)
            or set(related) != {"doc_id", "quote"}
            or not all(
                isinstance(related[field], str) and related[field].strip()
                for field in ("doc_id", "quote")
            )
        ):
            return None
        span_key = (
            raw_finding["doc_id"],
            normalized_tokens(raw_finding["quote"]),
        )
        if span_key in seen_spans:
            return None
        seen_spans.add(span_key)
        findings.append(
            {
                **dict(raw_finding),
                "decision": decision,
                "related_evidence": dict(related)
                if isinstance(related, Mapping)
                else None,
            }
        )
    return findings


def parse_planted_items(
    planted_items: Sequence[Mapping[str, Any]] | str,
) -> list[dict[str, Any]]:
    loaded = (
        _load_trusted_json(planted_items)
        if isinstance(planted_items, str)
        else planted_items
    )
    return [
        {
            "doc_id": str(item["doc_id"]),
            "quote": str(item.get("quote", item.get("anchor_quote", ""))),
            "type": str(item["type"]),
            "diagnosis": str(item.get("diagnosis", item.get("canonical_question", ""))),
            "decision": item.get("decision", item.get("decision_frame")),
            "decision_aliases": item.get("decision_aliases"),
            "related_evidence": item.get("related_evidence"),
            "related_document_text": item.get("related_document_text"),
            "document_text": str(
                item.get(
                    "document_text", item.get("quote", item.get("anchor_quote", ""))
                )
            ),
        }
        for item in loaded
    ]


def match_findings(
    candidates: Sequence[Mapping[str, Any]],
    planted: Sequence[Mapping[str, Any]],
    *,
    quote_overlap_threshold: float = QUOTE_OVERLAP_THRESHOLD,
    plant_coverage_threshold: float = PLANT_COVERAGE_THRESHOLD,
) -> dict[str, float | int]:
    """Match strict diagnosis plus paired evidence and report component scores."""

    localization_adjacency: dict[int, list[tuple[float, int]]] = {
        candidate_index: [] for candidate_index in range(len(candidates))
    }
    typed_adjacency: dict[int, list[tuple[float, int]]] = {
        candidate_index: [] for candidate_index in range(len(candidates))
    }
    diagnosis_adjacency: dict[int, list[tuple[float, int]]] = {
        candidate_index: [] for candidate_index in range(len(candidates))
    }
    adjacency: dict[int, list[tuple[float, int]]] = {
        candidate_index: [] for candidate_index in range(len(candidates))
    }
    for candidate_index, candidate in enumerate(candidates):
        for plant_index, plant in enumerate(planted):
            if candidate.get("doc_id") != plant.get("doc_id"):
                continue
            if not normalized_contiguous_quote(
                candidate.get("quote", ""),
                plant.get("document_text", plant.get("quote", "")),
            ):
                continue
            plant_coverage = normalized_plant_coverage(
                candidate.get("quote", ""), plant.get("quote", "")
            )
            if plant_coverage < plant_coverage_threshold:
                continue
            precision = normalized_quote_precision(
                candidate.get("quote", ""), plant.get("quote", "")
            )
            if precision < QUOTE_PRECISION_THRESHOLD:
                continue
            overlap = normalized_quote_overlap(
                candidate.get("quote", ""), plant.get("quote", "")
            )
            if overlap >= quote_overlap_threshold:
                localization_adjacency[candidate_index].append((overlap, plant_index))
                if candidate.get("type") != plant.get("type"):
                    continue
                typed_adjacency[candidate_index].append((overlap, plant_index))
                if not _finding_diagnosis_matches(candidate, plant):
                    continue
                diagnosis_adjacency[candidate_index].append((overlap, plant_index))
                if not _related_evidence_matches(candidate, plant):
                    continue
                adjacency[candidate_index].append((overlap, plant_index))
    for graph in (
        localization_adjacency,
        typed_adjacency,
        diagnosis_adjacency,
        adjacency,
    ):
        for edges in graph.values():
            edges.sort(key=lambda item: (-item[0], item[1]))

    localization_true_positive = _maximum_cardinality_matches(localization_adjacency)
    typed_true_positive = _maximum_cardinality_matches(typed_adjacency)
    diagnosis_true_positive = _maximum_cardinality_matches(diagnosis_adjacency)
    true_positive = _maximum_cardinality_matches(adjacency)

    false_positive = len(candidates) - true_positive
    false_negative = len(planted) - true_positive
    denominator = 2 * true_positive + false_positive + false_negative
    f1 = 0.0 if denominator == 0 else (2 * true_positive) / denominator
    localization_recall = (
        0.0 if not planted else localization_true_positive / len(planted)
    )
    type_accuracy = (
        0.0
        if localization_true_positive == 0
        else typed_true_positive / localization_true_positive
    )
    planted_count = len(planted)
    candidate_count = len(candidates)
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "f1": f1,
        "localization_recall": localization_recall,
        "type_accuracy": type_accuracy,
        "type_recall": 0.0
        if not planted_count
        else typed_true_positive / planted_count,
        "diagnosis_recall": (
            0.0 if not planted_count else diagnosis_true_positive / planted_count
        ),
        "relation_recall": 0.0 if not planted_count else true_positive / planted_count,
        "localization_f1": _stage_f1(
            localization_true_positive, candidate_count, planted_count
        ),
        "type_f1": _stage_f1(typed_true_positive, candidate_count, planted_count),
        "diagnosis_f1": _stage_f1(
            diagnosis_true_positive, candidate_count, planted_count
        ),
        "relation_f1": _stage_f1(true_positive, candidate_count, planted_count),
    }


def _stage_f1(true_positive: int, candidate_count: int, planted_count: int) -> float:
    """Return a stage F1 that charges every unmatched candidate as a false positive."""

    denominator = candidate_count + planted_count
    return 0.0 if denominator == 0 else 2 * true_positive / denominator


def _maximum_cardinality_matches(
    adjacency: Mapping[int, Sequence[tuple[float, int]]],
) -> int:
    """Return deterministic maximum-cardinality bipartite match count."""

    normalized = {index: list(edges) for index, edges in adjacency.items()}
    for edges in normalized.values():
        edges.sort(key=lambda item: (-item[0], item[1]))

    plant_to_candidate: dict[int, int] = {}

    def augment(candidate_index: int, seen_plants: set[int]) -> bool:
        for _, plant_index in normalized[candidate_index]:
            if plant_index in seen_plants:
                continue
            seen_plants.add(plant_index)
            previous_candidate = plant_to_candidate.get(plant_index)
            if previous_candidate is None or augment(previous_candidate, seen_plants):
                plant_to_candidate[plant_index] = candidate_index
                return True
        return False

    # Deterministic augmenting paths find a maximum-cardinality assignment;
    # overlap-descending adjacency makes stronger quote matches the first
    # preference without sacrificing a valid additional match.
    for candidate_index in normalized:
        augment(candidate_index, set())
    return len(plant_to_candidate)


def _finding_diagnosis_matches(
    candidate: Mapping[str, Any], plant: Mapping[str, Any]
) -> bool:
    diagnosis = str(candidate.get("diagnosis", ""))
    if not plant.get("diagnosis"):
        return True
    return (
        is_yes_no_question(diagnosis)
        and question_decision_similarity(
            diagnosis,
            plant,
            candidate_decision=candidate.get("decision"),
        )
        > 0
    )


def _related_evidence_matches(
    candidate: Mapping[str, Any], plant: Mapping[str, Any]
) -> bool:
    expected = plant.get("related_evidence")
    actual = candidate.get("related_evidence")
    if "related_evidence" not in plant:
        return True
    if plant.get("type") != "contradiction":
        return actual is None and expected is None
    if not isinstance(expected, Mapping) or not isinstance(actual, Mapping):
        return False
    return (
        actual.get("doc_id") == expected.get("doc_id")
        and normalized_contiguous_quote(
            str(actual.get("quote", "")),
            str(plant.get("related_document_text", expected.get("quote", ""))),
        )
        and normalized_quote_overlap(
            str(actual.get("quote", "")), str(expected.get("quote", ""))
        )
        >= QUOTE_OVERLAP_THRESHOLD
        and normalized_plant_coverage(
            str(actual.get("quote", "")), str(expected.get("quote", ""))
        )
        >= PLANT_COVERAGE_THRESHOLD
        and normalized_quote_precision(
            str(actual.get("quote", "")), str(expected.get("quote", ""))
        )
        >= QUOTE_PRECISION_THRESHOLD
    )


def normalized_quote_overlap(left: str, right: str) -> float:
    """Return ordered, longest-contiguous-token F1 after conservative folding."""

    left_tokens = normalized_tokens(left)
    right_tokens = normalized_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    longest = _longest_common_contiguous_span(left_tokens, right_tokens)
    return 2 * longest / (len(left_tokens) + len(right_tokens))


def normalized_plant_coverage(candidate_quote: str, planted_quote: str) -> float:
    """Return the fraction of a planted anchor covered by one ordered span."""

    candidate_tokens = normalized_tokens(candidate_quote)
    planted_tokens = normalized_tokens(planted_quote)
    if not candidate_tokens or not planted_tokens:
        return 0.0
    longest = _longest_common_contiguous_span(candidate_tokens, planted_tokens)
    return longest / len(planted_tokens)


def normalized_quote_precision(candidate_quote: str, planted_quote: str) -> float:
    """Return the fraction of candidate tokens belonging to the planted anchor."""

    candidate_tokens = normalized_tokens(candidate_quote)
    planted_tokens = normalized_tokens(planted_quote)
    if not candidate_tokens or not planted_tokens:
        return 0.0
    longest = _longest_common_contiguous_span(candidate_tokens, planted_tokens)
    return longest / len(candidate_tokens)


def normalized_tokens(text: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return tuple(_TOKEN_PATTERN.findall(normalized))


def normalized_contiguous_quote(quote: str, document_text: str) -> bool:
    """Return whether a normalized quote is an ordered contiguous document span."""

    quote_tokens = normalized_tokens(quote)
    document_tokens = normalized_tokens(document_text)
    if not quote_tokens or len(quote_tokens) > len(document_tokens):
        return False
    width = len(quote_tokens)
    return any(
        document_tokens[index : index + width] == quote_tokens
        for index in range(len(document_tokens) - width + 1)
    )


def _longest_common_contiguous_span(
    left_tokens: Sequence[str], right_tokens: Sequence[str]
) -> int:
    previous = [0] * (len(right_tokens) + 1)
    longest = 0
    for left_token in left_tokens:
        current = [0] * (len(right_tokens) + 1)
        for right_index, right_token in enumerate(right_tokens, start=1):
            if left_token == right_token:
                current[right_index] = previous[right_index - 1] + 1
                longest = max(longest, current[right_index])
        previous = current
    return longest


def _maximum_weight_sum(weights: Sequence[Sequence[float]]) -> float:
    """Return a deterministic maximum-weight bipartite assignment sum."""

    row_count = len(weights)
    column_count = max((len(row) for row in weights), default=0)
    size = max(row_count, column_count)
    if size == 0:
        return 0.0
    costs = [
        [
            -float(weights[row_index][column_index])
            if row_index < row_count and column_index < len(weights[row_index])
            else 0.0
            for column_index in range(size)
        ]
        for row_index in range(size)
    ]

    row_potential = [0.0] * (size + 1)
    column_potential = [0.0] * (size + 1)
    column_match = [0] * (size + 1)
    previous_column = [0] * (size + 1)
    for row in range(1, size + 1):
        column_match[0] = row
        minimum_slack = [float("inf")] * (size + 1)
        used = [False] * (size + 1)
        current_column = 0
        while True:
            used[current_column] = True
            current_row = column_match[current_column]
            delta = float("inf")
            next_column = 0
            for column in range(1, size + 1):
                if used[column]:
                    continue
                reduced_cost = (
                    costs[current_row - 1][column - 1]
                    - row_potential[current_row]
                    - column_potential[column]
                )
                if reduced_cost < minimum_slack[column]:
                    minimum_slack[column] = reduced_cost
                    previous_column[column] = current_column
                if minimum_slack[column] < delta:
                    delta = minimum_slack[column]
                    next_column = column
            for column in range(size + 1):
                if used[column]:
                    row_potential[column_match[column]] += delta
                    column_potential[column] -= delta
                else:
                    minimum_slack[column] -= delta
            current_column = next_column
            if column_match[current_column] == 0:
                break
        while True:
            next_column = previous_column[current_column]
            column_match[current_column] = column_match[next_column]
            current_column = next_column
            if current_column == 0:
                break

    total = 0.0
    for column in range(1, size + 1):
        row = column_match[column]
        if 1 <= row <= row_count and column <= len(weights[row - 1]):
            total += float(weights[row - 1][column - 1])
    return total


def extract_json_object(text: str, *, preferred_key: str) -> Any:
    """Parse one complete JSON object; prose, fences, arrays, and wrappers fail."""

    del preferred_key
    if not isinstance(text, str) or len(text) > MAX_COMPLETION_CHARS:
        raise ValueError("completion exceeds the JSON parser size limit")
    parsed = _load_bounded_json(text.strip())
    if not isinstance(parsed, dict):
        raise ValueError("completion root must be a JSON object")
    return parsed


def _load_bounded_json(text: str) -> Any:
    if len(text) > MAX_COMPLETION_CHARS:
        raise ValueError("JSON payload exceeds the size limit")
    try:
        loaded = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_non_finite_json_constant,
        )
    except (json.JSONDecodeError, RecursionError, _DuplicateJsonKeyError) as error:
        raise ValueError("invalid bounded JSON payload") from error
    _validate_json_shape(loaded)
    return loaded


def _load_trusted_json(text: str) -> Any:
    """Decode environment-authored JSON without the completion byte ceiling."""

    try:
        loaded = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_non_finite_json_constant,
        )
    except (json.JSONDecodeError, RecursionError, _DuplicateJsonKeyError) as error:
        raise ValueError("invalid trusted JSON payload") from error
    _validate_json_shape(loaded)
    return loaded


def _validate_json_shape(value: Any) -> None:
    """Bound decoded container depth and nodes without recursive traversal."""

    stack: list[tuple[Any, int]] = [(value, 1)]
    nodes = 0
    while stack:
        current, depth = stack.pop()
        nodes += 1
        if nodes > MAX_JSON_NODES:
            raise ValueError("JSON payload exceeds the node limit")
        if depth > MAX_JSON_DEPTH:
            raise ValueError("JSON payload exceeds the nesting limit")
        if isinstance(current, Mapping):
            stack.extend((item, depth + 1) for item in current.values())
        elif isinstance(current, list):
            stack.extend((item, depth + 1) for item in current)


def validate_difficulty_args(
    *,
    docs_count: int | None,
    docs_length: int | None,
    planted_density: float,
    distractor_density: float,
    panel_polarization: float,
    question_count: int,
    task: str,
) -> None:
    if task not in VALID_TASKS:
        raise ValueError(f"task must be one of {sorted(VALID_TASKS)}")
    for name, value in (("docs_count", docs_count), ("docs_length", docs_length)):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
        ):
            raise ValueError(f"{name} must be a positive integer or None")
    for name, density, allow_zero in (
        ("planted_density", planted_density, False),
        ("distractor_density", distractor_density, True),
        ("panel_polarization", panel_polarization, False),
    ):
        if (
            isinstance(density, bool)
            or not isinstance(density, (int, float))
            or not math.isfinite(density)
        ):
            raise ValueError(f"{name} must be finite")
        lower_bound_ok = density >= 0 if allow_zero else density > 0
        if not lower_bound_ok or density > 1:
            interval = "[0, 1]" if allow_zero else "(0, 1]"
            raise ValueError(f"{name} must be within {interval}")
    if (
        isinstance(question_count, bool)
        or not isinstance(question_count, int)
        or question_count <= 0
    ):
        raise ValueError("question_count must be a positive integer")


def _density_prefix(items: Sequence[Any], density: float) -> list[Any]:
    if not items or density <= 0:
        return []
    count = min(len(items), max(1, math.ceil(len(items) * density)))
    return list(items[:count])


def _safe_truncation_index(
    text: str, requested_index: int, anchors: Sequence[str]
) -> int:
    """Move a character cut left until it cannot bisect any planted anchor."""

    truncation_index = requested_index
    while True:
        adjusted_index = truncation_index
        for anchor in anchors:
            anchor_start = text.find(anchor)
            anchor_end = anchor_start + len(anchor)
            if anchor_start >= 0 and anchor_start < truncation_index < anchor_end:
                adjusted_index = min(adjusted_index, anchor_start)
        if adjusted_index == truncation_index:
            return truncation_index
        truncation_index = adjusted_index
