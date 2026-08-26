from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PREDICT_DATA = (
    ROOT / "environments" / "commonground_predict" / "commonground_predict" / "data"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def statement_texts(rows: list[dict[str, Any]]) -> set[str]:
    return {statement["text"] for row in rows for statement in row["statements"]}


def test_predict_training_and_evaluation_policy_texts_are_disjoint() -> None:
    train = read_jsonl(PREDICT_DATA / "train_synthetic.jsonl")
    evaluation = read_jsonl(PREDICT_DATA / "eval_synthetic.jsonl")

    assert statement_texts(train)
    assert statement_texts(evaluation)
    assert statement_texts(train).isdisjoint(statement_texts(evaluation))
