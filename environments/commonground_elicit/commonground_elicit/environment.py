"""Verifiers environment for planted document-grounded findings."""

from __future__ import annotations

import json
import math
import os
import re
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import verifiers.v1 as vf
from commonground_scenarios import (
    is_yes_no_question,
    question_fingerprint,
    validate_scenario,
)
from commonground_score import (
    cluster_separation,
    vote_entropy,
)
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
QUOTE_OVERLAP_THRESHOLD = 0.5
PLANT_COVERAGE_THRESHOLD = 0.8
MAX_COMPLETION_CHARS = 32_768
MAX_JSON_STARTS = 64
MAX_JSON_DEPTH = 32
MAX_JSON_NODES = 10_000
VALID_TASKS = frozenset({"find", "elicit-ask"})
STANCE_TO_VOTE = {"agree": 1, "disagree": -1, "pass": 0}
_TOKEN_PATTERN = re.compile(
    r"(?:!=|<=|>=|==)|[!~](?=\s*[^\W_])|-(?=\s*\d)|[^\W_]+|"
    r"[¬≠≤≥=<>±+\N{MINUS SIGN}%$€£¥∉∈∧\N{LOGICAL OR}]",
    flags=re.UNICODE,
)


class ElicitJsonParser:
    """Extract the last task-specific JSON object from a completion."""

    def __init__(self, preferred_key: str = "findings") -> None:
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
            completion,
            self.data.answer,
            ElicitJsonParser("findings"),
        )
        return {"finding_f1": score}

    @vf.metric
    async def companion_metrics(self, trace: vf.Trace) -> Mapping[str, float]:
        if self.data.info["task_label"] != "find":
            return {}
        score = await question_utility(
            [{"role": "assistant", "content": trace.last_reply}],
            self.data.answer,
            self.data.info,
            ElicitJsonParser("findings"),
        )
        return {"question_utility": score}


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
    question_count: int = 3
    split: Literal["eval", "train"] = "eval"


class ElicitHarness(NullHarness):
    """The built-in pure-chat harness, namespaced with this taskset plugin."""


class _CompatibilityRubric:
    """Small adapter for existing local score probes; native runs use ElicitTask."""

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
        else:
            reward = await finding_f1(completion, answer, ElicitJsonParser("findings"))
            metrics["finding_f1"] = reward
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
        self.rubric = _CompatibilityRubric()
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
        }

    def load(self) -> Sequence[ElicitTask]:
        return [
            ElicitTask(
                ElicitTaskData(
                    idx=index,
                    name=str(json.loads(row["info"])["scenario_id"]),
                    prompt=row["prompt"],
                    answer=json.loads(row["answer"]),
                    info=json.loads(row["info"]),
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


def load_environment(
    docs_count: int | None = None,
    docs_length: int | None = None,
    planted_density: float = 1.0,
    distractor_density: float = 1.0,
    data_path: str | os.PathLike[str] | None = None,
    *,
    task: str = "find",
    panel_polarization: float = 1.0,
    question_count: int = 3,
    train_data_path: str | os.PathLike[str] | None = None,
    split: str = "eval",
    **kwargs: Any,
) -> ElicitTaskset:
    """Build the native Verifiers v1 taskset with a compatibility inspection API."""

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
        )
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
            "question": plant["canonical_question"],
            "question_aliases": list(plant["canonical_question_aliases"]),
            "target_stances": dict(plant["target_stances"]),
            "document_text": next(
                document["text"]
                for document in documents
                if document["doc_id"] == plant["doc_id"]
            ),
        }
        for plant in visible_plants
    ]
    answer = (
        {"questions": question_oracle}
        if task == "elicit-ask"
        else {"findings": findings_answer, "questions": question_oracle}
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
                        documents, scenario["factions"], effective_question_count
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

    visible_distractors = _density_prefix(
        [
            item
            for item in scenario["distractors"]
            if item["doc_id"] in selected_doc_ids
        ],
        distractor_density,
    )
    visible_distractor_anchors = {
        (item["doc_id"], item["anchor_quote"]) for item in visible_distractors
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
            truncation_index = _safe_truncation_index(
                text,
                docs_length,
                [
                    plant["anchor_quote"]
                    for plant in selected_plants
                    if plant["doc_id"] == document["doc_id"]
                ]
                + [
                    anchor
                    for doc_id, anchor in sorted(visible_distractor_anchors)
                    if doc_id == document["doc_id"]
                ],
            )
            text = text[:truncation_index].rstrip() + "…"
        documents.append({**document, "text": text})

    visible_text_by_doc = {
        document["doc_id"]: document["text"] for document in documents
    }
    visible_plants = [
        plant
        for plant in selected_plants
        if plant["doc_id"] in visible_text_by_doc
        and plant["anchor_quote"] in visible_text_by_doc[plant["doc_id"]]
    ]
    return documents, visible_plants


def render_prompt(
    documents: Sequence[Mapping[str, str]],
    *,
    factions: Sequence[Mapping[str, Any]] = (),
    question_count: int = 3,
) -> str:
    """Render the finding task and its observable weight-zero T2 metric."""

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
            "Return STRICT JSON only, with this shape:",
            (
                '{"findings":[{"doc_id":"<document id>","quote":"<passage>","type":"ambiguity|contradiction|gap"}],'
                '"questions":[{"doc_id":"<document id>","quote":"<passage>","question":"<specific yes/no clarifying question>",'
                '"target_stances":{"<faction id>":"agree|disagree|pass"}}]}'
            ),
            f"Return exactly {question_count} question objects. The findings determine reward; questions are scored as a logged weight-zero companion metric.",
            "Phrase each question as yes/no: agree means that faction predicts yes, disagree means no, and pass means no position.",
        ]
    )
    return "\n".join(lines)


def render_ask_prompt(
    documents: Sequence[Mapping[str, str]],
    factions: Sequence[Mapping[str, Any]],
    question_count: int,
) -> str:
    """Render the question-raising task without exposing planted stances."""

    lines = [
        f"Raise exactly {question_count} clarifying questions grounded in these policy documents.",
        "Each question should expose a specific ambiguity, contradiction, or uncovered case that could split the listed stakeholder factions.",
        "Phrase every question as yes/no. Predict every faction's answer: agree means yes, disagree means no, and pass means no position.",
        "Generic questions about whether rules should exist are not useful.",
        "",
        "Stakeholder factions:",
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
            "Return STRICT JSON only, with this shape:",
            '{"questions":[{"doc_id":"<document id>","quote":"<passage>","question":"<specific clarifying question>","target_stances":{"<faction id>":"agree|disagree|pass"}}]}',
        ]
    )
    return "\n".join(lines)


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
    expected_fields = {"doc_id", "quote", "question", "target_stances"}
    for raw_question in raw_questions:
        if (
            not isinstance(raw_question, Mapping)
            or set(raw_question) != expected_fields
        ):
            return None
        if not all(
            isinstance(raw_question[field], str) and raw_question[field].strip()
            for field in ("doc_id", "quote", "question")
        ):
            return None
        target_stances = raw_question["target_stances"]
        if not isinstance(target_stances, Mapping) or not target_stances:
            return None
        if not all(
            isinstance(faction_id, str) and faction_id and stance in STANCE_TO_VOTE
            for faction_id, stance in target_stances.items()
        ):
            return None
        questions.append(
            {
                "doc_id": raw_question["doc_id"],
                "quote": raw_question["quote"],
                "question": raw_question["question"],
                "target_stances": dict(target_stances),
            }
        )
    return questions


def question_utility_score(
    candidates: Sequence[Mapping[str, Any]],
    planted: Sequence[Mapping[str, Any]],
    *,
    panel_polarization: float,
    question_count: int,
) -> float:
    """Return optimal one-to-one planted-question utility for exactly K outputs."""

    if len(candidates) != question_count or question_count <= 0:
        return 0.0
    fingerprints = [
        question_fingerprint(str(candidate.get("question", "")))
        for candidate in candidates
    ]
    if len(set(fingerprints)) != len(fingerprints):
        return 0.0
    candidate_question_targets = [
        {
            plant_index
            for plant_index, plant in enumerate(planted)
            if _question_matches_plant(str(candidate.get("question", "")), plant)
        }
        for candidate in candidates
    ]
    for left_index, left_targets in enumerate(candidate_question_targets):
        if any(
            left_targets & right_targets
            for right_targets in candidate_question_targets[left_index + 1 :]
        ):
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
    return _maximum_weight_sum(weights) / question_count


def _candidate_plant_question_utility(
    candidate: Mapping[str, Any],
    plant: Mapping[str, Any],
    *,
    panel_polarization: float,
) -> float:
    text_match = _candidate_plant_text_match(candidate, plant)
    if text_match is None:
        return 0.0
    quote_overlap, question_overlap = text_match
    planted_stances = plant.get("target_stances", {})
    candidate_stances = candidate.get("target_stances", {})
    if not isinstance(planted_stances, Mapping) or set(candidate_stances) != set(
        planted_stances
    ):
        return 0.0
    if dict(candidate_stances) != dict(planted_stances):
        return 0.0
    disagreement = panel_disagreement(planted_stances)
    return quote_overlap * question_overlap * disagreement * panel_polarization


def _candidate_plant_text_match(
    candidate: Mapping[str, Any],
    plant: Mapping[str, Any],
) -> tuple[float, float] | None:
    """Match a candidate to one planted issue before considering stance guesses."""

    if candidate.get("doc_id") != plant.get("doc_id"):
        return None
    candidate_quote = str(candidate.get("quote", ""))
    planted_quote = str(plant.get("quote", ""))
    if candidate_quote != planted_quote:
        return None
    document_text = str(plant.get("document_text", ""))
    if candidate_quote not in document_text:
        return None
    candidate_question = str(candidate.get("question", ""))
    if not is_yes_no_question(candidate_question):
        return None
    if not _question_matches_plant(candidate_question, plant):
        return None
    return 1.0, 1.0


def _question_matches_plant(candidate: str, plant: Mapping[str, Any]) -> bool:
    """Match only a canonical question or an explicitly authored finite alias."""

    candidate_identity = question_fingerprint(candidate)
    allowed = [plant.get("question", ""), *plant.get("question_aliases", ())]
    return candidate_identity in {
        question_fingerprint(str(question)) for question in allowed
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

    parsed = parser.parse_answer(completion)
    candidates = parse_candidate_findings(parsed)
    if candidates is None:
        return 0.0
    answer_payload = _parse_mapping_payload(answer)
    planted = parse_planted_items(answer_payload.get("findings", []))
    return match_findings(candidates, planted)["f1"]


def _parse_mapping_payload(payload: Mapping[str, Any] | str) -> Mapping[str, Any]:
    """Decode a canonical answer/info payload into a mapping for scoring."""

    try:
        loaded = _load_bounded_json(payload) if isinstance(payload, str) else payload
        _validate_json_shape(loaded)
    except (RecursionError, TypeError, ValueError):
        return {}
    return loaded if isinstance(loaded, Mapping) else {}


def parse_candidate_findings(parsed: Any) -> list[dict[str, str]] | None:
    """Validate the exact T1 response schema, returning None on any error."""

    if not isinstance(parsed, Mapping) or set(parsed) not in (
        {"findings"},
        {"findings", "questions"},
    ):
        return None
    return _parse_findings_list(parsed["findings"])


def _parse_findings_list(raw_findings: Any) -> list[dict[str, str]] | None:
    if not isinstance(raw_findings, list):
        return None
    findings: list[dict[str, str]] = []
    for raw_finding in raw_findings:
        if not isinstance(raw_finding, Mapping) or set(raw_finding) != {
            "doc_id",
            "quote",
            "type",
        }:
            return None
        if not all(
            isinstance(raw_finding[field], str) for field in ("doc_id", "quote", "type")
        ):
            return None
        if not raw_finding["doc_id"].strip() or not raw_finding["quote"].strip():
            return None
        if raw_finding["type"] not in FINDING_TYPES:
            return None
        findings.append(dict(raw_finding))
    return findings


def parse_planted_items(
    planted_items: Sequence[Mapping[str, str]] | str,
) -> list[dict[str, str]]:
    loaded = (
        _load_bounded_json(planted_items)
        if isinstance(planted_items, str)
        else planted_items
    )
    return [
        {
            "doc_id": str(item["doc_id"]),
            "quote": str(item.get("quote", item.get("anchor_quote", ""))),
            "type": str(item["type"]),
            "document_text": str(
                item.get(
                    "document_text", item.get("quote", item.get("anchor_quote", ""))
                )
            ),
        }
        for item in loaded
    ]


def match_findings(
    candidates: Sequence[Mapping[str, str]],
    planted: Sequence[Mapping[str, str]],
    *,
    quote_overlap_threshold: float = QUOTE_OVERLAP_THRESHOLD,
    plant_coverage_threshold: float = PLANT_COVERAGE_THRESHOLD,
) -> dict[str, float | int]:
    """Globally match eligible candidates once each and return F1 counts."""

    adjacency: dict[int, list[tuple[float, int]]] = {
        candidate_index: [] for candidate_index in range(len(candidates))
    }
    for candidate_index, candidate in enumerate(candidates):
        for plant_index, plant in enumerate(planted):
            if candidate.get("doc_id") != plant.get("doc_id"):
                continue
            if candidate.get("type") != plant.get("type"):
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
            overlap = normalized_quote_overlap(
                candidate.get("quote", ""), plant.get("quote", "")
            )
            if overlap >= quote_overlap_threshold:
                adjacency[candidate_index].append((overlap, plant_index))
    for edges in adjacency.values():
        edges.sort(key=lambda item: (-item[0], item[1]))

    plant_to_candidate: dict[int, int] = {}

    def augment(candidate_index: int, seen_plants: set[int]) -> bool:
        for _, plant_index in adjacency[candidate_index]:
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
    for candidate_index in range(len(candidates)):
        augment(candidate_index, set())

    true_positive = len(plant_to_candidate)
    false_positive = len(candidates) - true_positive
    false_negative = len(planted) - true_positive
    denominator = 2 * true_positive + false_positive + false_negative
    f1 = 0.0 if denominator == 0 else (2 * true_positive) / denominator
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "f1": f1,
    }


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
    if not isinstance(text, str) or len(text) > MAX_COMPLETION_CHARS:
        raise ValueError("completion exceeds the JSON parser size limit")
    starts = [index for index, character in enumerate(text) if character == "{"]
    if len(starts) > MAX_JSON_STARTS:
        raise ValueError("completion contains too many JSON object starts")
    decoder = json.JSONDecoder()
    last_decodable: Any = None
    last_preferred: dict[str, Any] | None = None
    found_decodable = False
    for index in starts:
        try:
            parsed, _ = decoder.raw_decode(text[index:])
            _validate_json_shape(parsed)
        except (json.JSONDecodeError, RecursionError, ValueError):
            continue
        found_decodable = True
        last_decodable = parsed
        if isinstance(parsed, dict) and isinstance(parsed.get(preferred_key), list):
            last_preferred = parsed
    if last_preferred is not None:
        return last_preferred
    if found_decodable:
        return last_decodable
    raise ValueError("no JSON object found")


def _load_bounded_json(text: str) -> Any:
    if len(text) > MAX_COMPLETION_CHARS:
        raise ValueError("JSON payload exceeds the size limit")
    try:
        loaded = json.loads(text)
    except (json.JSONDecodeError, RecursionError) as error:
        raise ValueError("invalid bounded JSON payload") from error
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
