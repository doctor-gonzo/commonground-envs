"""Verifiers environment for planted document-grounded findings."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
import re
import unicodedata
from typing import Any

import verifiers as vf
from commonground_scenarios import HELDOUT_TEMPLATES, generate_scenario, validate_scenario
from datasets import Dataset


ENV_ID = "commonground-elicit"
DATA_ENV_VAR = "COMMONGROUND_ELICIT_DATA_PATH"
BUNDLED_EVAL_PATH = Path(__file__).resolve().parent / "data" / "eval_synthetic_heldout.jsonl"
FINDING_TYPES = frozenset({"ambiguity", "contradiction", "gap"})
QUOTE_OVERLAP_THRESHOLD = 0.5
DEFAULT_GENERATED_AT = "2026-08-15"
_TOKEN_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)


class ElicitJsonParser(vf.Parser):
    """Extract the last findings JSON object from a completion."""

    def parse(self, text: str) -> dict[str, Any]:
        try:
            parsed = extract_json_object(text, preferred_key="findings")
        except ValueError:
            return {}
        return parsed if isinstance(parsed, dict) else {}


def load_environment(
    docs_count: int | None = None,
    docs_length: int | None = None,
    planted_density: float = 1.0,
    distractor_density: float = 1.0,
    data_path: str | os.PathLike[str] | None = None,
    **kwargs: Any,
) -> vf.SingleTurnEnv:
    """Build the deterministic single-turn planted-finding environment."""

    validate_difficulty_args(
        docs_count=docs_count,
        docs_length=docs_length,
        planted_density=planted_density,
        distractor_density=distractor_density,
    )
    configured_path = data_path or os.environ.get(DATA_ENV_VAR)
    resolved_path = Path(configured_path) if configured_path else BUNDLED_EVAL_PATH
    scenarios = load_scenarios(resolved_path, allow_unbundled_default=configured_path is None)
    candidate_rows = [
        scenario_to_row(
            scenario,
            docs_count=docs_count,
            docs_length=docs_length,
            planted_density=planted_density,
            distractor_density=distractor_density,
        )
        for scenario in scenarios
    ]
    rows = [row for row in candidate_rows if json.loads(row["planted_items"])]
    if not rows:
        raise ValueError("difficulty arguments remove all planted items from the dataset")
    dataset = Dataset.from_list(rows)
    parser = ElicitJsonParser()
    rubric = vf.Rubric(funcs=[finding_f1], weights=[1.0], parser=parser)
    env_args = {
        "docs_count": docs_count,
        "docs_length": docs_length,
        "planted_density": planted_density,
        "distractor_density": distractor_density,
        "data_path": (
            str(resolved_path)
            if configured_path is not None or resolved_path.is_file()
            else None
        ),
    }
    return vf.SingleTurnEnv(
        dataset=dataset,
        eval_dataset=dataset,
        parser=parser,
        rubric=rubric,
        env_id=ENV_ID,
        env_args=env_args,
        **kwargs,
    )


def load_scenarios(path: Path, *, allow_unbundled_default: bool = False) -> list[dict[str, Any]]:
    """Load and validate canonical scenario JSONL.

    Slice 2 can run before the committed split is introduced in slice 4. In
    that narrow case, the default loader generates the held-out templates from
    fixed seeds and an explicit date. Any explicitly configured missing path is
    still an error.
    """

    if not path.is_file():
        if allow_unbundled_default and path == BUNDLED_EVAL_PATH:
            return default_heldout_scenarios()
        raise FileNotFoundError(path)

    scenarios: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            scenario = json.loads(line)
            validate_scenario(scenario)
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            raise ValueError(f"invalid scenario at {path}:{line_number}: {error}") from error
        scenarios.append(scenario)
    if not scenarios:
        raise ValueError(f"no scenarios loaded from {path}")
    return scenarios


def default_heldout_scenarios() -> list[dict[str, Any]]:
    """Generate the temporary pre-bundle held-out set deterministically."""

    return [
        generate_scenario(
            seed=8200 + template_index * 10 + repetition,
            domain_template=template,
            generated_at=DEFAULT_GENERATED_AT,
        )
        for template_index, template in enumerate(HELDOUT_TEMPLATES)
        for repetition in range(2)
    ]


def scenario_to_row(
    scenario: Mapping[str, Any],
    *,
    docs_count: int | None,
    docs_length: int | None,
    planted_density: float,
    distractor_density: float,
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
    answer = [
        {
            "doc_id": plant["doc_id"],
            "quote": plant["anchor_quote"],
            "type": plant["type"],
        }
        for plant in visible_plants
    ]
    info = {
        "scenario_id": scenario["scenario_id"],
        "document_count": len(documents),
        "plant_count": len(answer),
        "synthetic": bool(scenario["provenance"]["synthetic"]),
        "template_set": scenario["provenance"]["template_set"],
    }
    return {
        "prompt": [{"role": "user", "content": render_prompt(documents)}],
        "answer": json.dumps({"findings": answer}, sort_keys=True),
        "planted_items": json.dumps(answer, sort_keys=True),
        "info": json.dumps(info, sort_keys=True),
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
        [item for item in scenario["distractors"] if item["doc_id"] in selected_doc_ids],
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
        for doc_id, anchor in sorted(all_distractor_anchors - visible_distractor_anchors):
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
                ],
            )
            text = text[:truncation_index].rstrip() + "…"
        documents.append({**document, "text": text})

    visible_text_by_doc = {document["doc_id"]: document["text"] for document in documents}
    visible_plants = [
        plant
        for plant in selected_plants
        if plant["doc_id"] in visible_text_by_doc
        and plant["anchor_quote"] in visible_text_by_doc[plant["doc_id"]]
    ]
    return documents, visible_plants


def render_prompt(documents: Sequence[Mapping[str, str]]) -> str:
    """Render only the visible document set and strict response contract."""

    lines = [
        "Find material ambiguities, contradictions, and gaps in these policy documents.",
        "Use an exact or close quote from the relevant document for each finding.",
        "Do not flag a passage merely because it discusses a rule; report only a concrete issue.",
        "",
        "Documents:",
    ]
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
            '{"findings":[{"doc_id":"<document id>","quote":"<passage>","type":"ambiguity|contradiction|gap"}]}',
        ]
    )
    return "\n".join(lines)


async def finding_f1(
    completion: list[dict[str, Any]],
    planted_items: Sequence[Mapping[str, str]] | str,
    parser: ElicitJsonParser,
) -> float:
    """Reward strict finding output by one-to-one planted-item F1."""

    parsed = parser.parse_answer(completion)
    candidates = parse_candidate_findings(parsed)
    if candidates is None:
        return 0.0
    planted = parse_planted_items(planted_items)
    return match_findings(candidates, planted)["f1"]


def parse_candidate_findings(parsed: Any) -> list[dict[str, str]] | None:
    """Validate the exact T1 response schema, returning None on any error."""

    if not isinstance(parsed, Mapping) or set(parsed) != {"findings"}:
        return None
    raw_findings = parsed["findings"]
    if not isinstance(raw_findings, list):
        return None
    findings: list[dict[str, str]] = []
    for raw_finding in raw_findings:
        if not isinstance(raw_finding, Mapping) or set(raw_finding) != {"doc_id", "quote", "type"}:
            return None
        if not all(isinstance(raw_finding[field], str) for field in ("doc_id", "quote", "type")):
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
    loaded = json.loads(planted_items) if isinstance(planted_items, str) else planted_items
    return [
        {
            "doc_id": str(item["doc_id"]),
            "quote": str(item.get("quote", item.get("anchor_quote", ""))),
            "type": str(item["type"]),
        }
        for item in loaded
    ]


def match_findings(
    candidates: Sequence[Mapping[str, str]],
    planted: Sequence[Mapping[str, str]],
    *,
    quote_overlap_threshold: float = QUOTE_OVERLAP_THRESHOLD,
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
            overlap = normalized_quote_overlap(candidate.get("quote", ""), plant.get("quote", ""))
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
    """Return multiset token F1 after Unicode, case, and punctuation folding."""

    left_tokens = normalized_tokens(left)
    right_tokens = normalized_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    shared = sum(
        min(count, right_tokens.get(token, 0))
        for token, count in left_tokens.items()
    )
    return 2 * shared / (sum(left_tokens.values()) + sum(right_tokens.values()))


def normalized_tokens(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    normalized = unicodedata.normalize("NFKC", text).casefold()
    for token in _TOKEN_PATTERN.findall(normalized):
        counts[token] = counts.get(token, 0) + 1
    return counts


def extract_json_object(text: str, *, preferred_key: str) -> Any:
    decoder = json.JSONDecoder()
    last_decodable: Any = None
    last_preferred: dict[str, Any] | None = None
    found_decodable = False
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
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


def validate_difficulty_args(
    *,
    docs_count: int | None,
    docs_length: int | None,
    planted_density: float,
    distractor_density: float,
) -> None:
    for name, value in (("docs_count", docs_count), ("docs_length", docs_length)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
            raise ValueError(f"{name} must be a positive integer or None")
    for name, value, allow_zero in (
        ("planted_density", planted_density, False),
        ("distractor_density", distractor_density, True),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        lower_bound_ok = value >= 0 if allow_zero else value > 0
        if not lower_bound_ok or value > 1:
            interval = "[0, 1]" if allow_zero else "(0, 1]"
            raise ValueError(f"{name} must be within {interval}")


def _density_prefix(items: Sequence[Any], density: float) -> list[Any]:
    if not items or density <= 0:
        return []
    count = min(len(items), max(1, math.ceil(len(items) * density)))
    return list(items[:count])


def _safe_truncation_index(text: str, requested_index: int, anchors: Sequence[str]) -> int:
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
