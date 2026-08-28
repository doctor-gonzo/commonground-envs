"""Generate a lexically and procedurally distinct Predict training split."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from synthetic_core import StatementSpec  # noqa: E402
from synthetic_core import make_snapshot as build_snapshot  # noqa: E402

TRAIN_SEED = 20260829
TRAIN_SNAPSHOT_COUNT = 200
SESSION_INDEX_OFFSET = 1000
GENERATOR_FAMILY = "train-random-mixture-v2"

_TEXTS = [
    (
        "The incident assistant should record the evidence and timestamp behind every proposed severity classification.",
        "evidence",
    ),
    (
        "The recruiting assistant should not infer protected characteristics from names, photographs, schools, or writing style.",
        "fairness",
    ),
    (
        "The procurement assistant should identify the approved supplier record before recommending that an order be placed.",
        "governance",
    ),
    (
        "The translation assistant should preserve legal notices and mark any phrase whose meaning remains uncertain.",
        "transparency",
    ),
    (
        "The security assistant may prepare an access change but an authorized administrator must approve its execution.",
        "oversight",
    ),
    (
        "The scheduling assistant should reveal which constraints caused it to reject a requested meeting time.",
        "transparency",
    ),
    (
        "The knowledge assistant should distinguish retrieved facts from generated suggestions in its response.",
        "evidence",
    ),
    (
        "The claims assistant should route suspected fraud to a qualified investigator rather than making a final accusation.",
        "oversight",
    ),
    (
        "The inventory assistant should confirm warehouse availability before promising a delivery date.",
        "evidence",
    ),
    (
        "The marketing assistant should use only customer segments whose collection purpose covers the proposed campaign.",
        "privacy",
    ),
    (
        "The compliance assistant should preserve the exact source wording when quoting a regulatory obligation.",
        "transparency",
    ),
    (
        "The maintenance assistant should stop automated work when sensor readings conflict with the equipment record.",
        "safety",
    ),
    (
        "The travel assistant should display cancellation conditions before asking an employee to approve a booking.",
        "transparency",
    ),
    (
        "The document assistant should remove hidden comments and tracked changes before preparing a file for external sharing.",
        "privacy",
    ),
    (
        "The benefits assistant should send eligibility exceptions to a designated benefits specialist for a decision.",
        "oversight",
    ),
    (
        "Each production assistant should have a documented owner who can disable its tools and revoke its credentials.",
        "accountability",
    ),
    (
        "Changes to an assistant's system instructions should receive the same risk review as changes to its model or tools.",
        "governance",
    ),
    (
        "A person affected by an automated eligibility recommendation should receive a reason and a meaningful appeal route.",
        "access",
    ),
    (
        "Evaluation sets for an assistant should cover foreseeable misuse as well as its intended workflow.",
        "fairness",
    ),
    (
        "Records created by an assistant should follow the retention schedule of the business process they support.",
        "retention",
    ),
    (
        "The research assistant should attach stable citations for material factual claims and identify inaccessible sources.",
        "evidence",
    ),
    (
        "The forecasting assistant should show the observation window and uncertainty assumptions behind each projection.",
        "transparency",
    ),
    (
        "The customer-success assistant should not promise roadmap work unless the commitment appears in an approved source.",
        "governance",
    ),
    (
        "The payroll assistant should require a second reviewer before changing bank or tax details.",
        "oversight",
    ),
    (
        "The facilities assistant should place occupant safety ahead of energy optimization when its signals disagree.",
        "safety",
    ),
    (
        "The quality assistant should retain failed inspection results instead of replacing them with a later passing run.",
        "accountability",
    ),
    (
        "The learning assistant should label generated examples that have not been reviewed by a subject-matter expert.",
        "transparency",
    ),
    (
        "The records assistant should log who authorized a legal hold and which deletion rules it suspends.",
        "retention",
    ),
    (
        "The vendor-risk assistant should separate verified evidence from questionnaire claims that have not been checked.",
        "evidence",
    ),
    (
        "The localization assistant should request regional review before changing required disclosures or consent language.",
        "access",
    ),
]
TRAIN_STATEMENT_SPECS = [StatementSpec(text, dimension) for text, dimension in _TEXTS]
TRAIN_STATEMENT_BANK = [spec.text for spec in TRAIN_STATEMENT_SPECS]
DIMENSIONS = tuple(sorted({spec.dimension for spec in TRAIN_STATEMENT_SPECS}))


def build_profiles(
    rng: random.Random,
    cluster_count: int,
    statements: list[StatementSpec],
) -> list[dict[str, float]]:
    """Sample continuous profiles instead of the held-out archetype family."""

    del statements
    orientations = rng.sample((-0.85, -0.45, 0.45, 0.85), cluster_count)
    return [
        {
            dimension: max(
                -1.0,
                min(1.0, orientation + rng.uniform(-0.65, 0.65)),
            )
            for dimension in DIMENSIONS
        }
        for orientation in orientations
    ]


def make_snapshot(rng: random.Random, session_index: int) -> dict[str, Any]:
    return build_snapshot(
        rng,
        session_index,
        seed=TRAIN_SEED,
        family=GENERATOR_FAMILY,
        statement_bank=TRAIN_STATEMENT_SPECS,
        profile_builder=build_profiles,
    )


def main() -> None:
    rng = random.Random(TRAIN_SEED)
    output_path = (
        Path(__file__).parents[1]
        / "commonground_predict"
        / "data"
        / "train_synthetic.jsonl"
    )
    snapshots = [
        make_snapshot(rng, SESSION_INDEX_OFFSET + index)
        for index in range(TRAIN_SNAPSHOT_COUNT)
    ]
    output_path.write_text(
        "".join(
            json.dumps(snapshot, separators=(",", ":")) + "\n" for snapshot in snapshots
        ),
        encoding="utf-8",
    )
    print(f"wrote {TRAIN_SNAPSHOT_COUNT} snapshots to {output_path}")


if __name__ == "__main__":
    main()
