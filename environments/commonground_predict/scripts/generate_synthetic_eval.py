"""Generate the semantic held-out split for commonground-predict."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from synthetic_core import (  # noqa: E402
    StatementSpec,
    make_snapshot_with_latent_patterns,
)
from synthetic_core import make_snapshot as build_snapshot  # noqa: E402

SEED = 20260828
SNAPSHOT_COUNT = 100
GENERATOR_FAMILY = "heldout-archetype-threshold-v2"

_TEXTS = [
    (
        "The support assistant should retrieve current prices and rates from the approved pricing system before quoting them to a customer.",
        "evidence",
    ),
    (
        "The support assistant should obtain human approval before offering a credit outside the published service-recovery limits.",
        "oversight",
    ),
    (
        "The coding assistant should reference credentials through the managed secrets service and never place secret values in generated code.",
        "privacy",
    ),
    (
        "The coding assistant should run available checks and clearly label any generated change it could not verify.",
        "evidence",
    ),
    (
        "The communications assistant should omit competitor names from outbound drafts unless an approved comparison brief is provided.",
        "governance",
    ),
    (
        "The HR assistant should cite the controlling policy section and effective date in every employee-policy answer.",
        "transparency",
    ),
    (
        "The HR assistant should route disciplinary and performance recommendations to an authorized HR reviewer before delivery.",
        "oversight",
    ),
    (
        "The sales assistant should ground capability claims in approved product documentation before adding them to a proposal.",
        "evidence",
    ),
    (
        "The finance assistant may draft payment or journal-entry instructions but should not submit them without authorized review.",
        "oversight",
    ),
    (
        "The contract assistant should separate quoted terms from its interpretations and flag every interpretation for legal review.",
        "transparency",
    ),
    (
        "The operations assistant should be allowed to escalate a safety or service risk directly to the on-call human.",
        "safety",
    ),
    (
        "The assistant should verify a data classification before sending work content to any external model or tool.",
        "privacy",
    ),
    (
        "The assistant should include only the minimum customer fields needed to complete the requested task.",
        "privacy",
    ),
    (
        "The analytics assistant should label its data sources, reporting window, and known gaps in every generated analysis.",
        "transparency",
    ),
    (
        "The customer-facing assistant should preserve approved accessibility and localization requirements when rewriting content.",
        "access",
    ),
    (
        "Every deployed AI assistant should have a named business owner, a documented purpose, and a tested rollback path.",
        "accountability",
    ),
    (
        "A material change to an assistant's model, tools, or data access should trigger a fresh risk review before release.",
        "governance",
    ),
    (
        "People affected by a high-impact assistant recommendation should be able to contest it and request human review.",
        "access",
    ),
    (
        "An AI assistant should pass documented evaluations across relevant user groups before its deployment is expanded.",
        "fairness",
    ),
    (
        "Assistant interaction logs should be retained only for a defined audit period and deleted on a published schedule.",
        "retention",
    ),
]
STATEMENT_SPECS = [StatementSpec(text, dimension) for text, dimension in _TEXTS]
STATEMENT_BANK = [spec.text for spec in STATEMENT_SPECS]
DIMENSIONS = tuple(sorted({spec.dimension for spec in STATEMENT_SPECS}))

_ARCHETYPES = (
    {
        "evidence": 0.8,
        "oversight": 0.9,
        "privacy": 0.8,
        "governance": 0.7,
        "transparency": 0.8,
        "safety": 0.9,
        "access": 0.3,
        "accountability": 0.8,
        "fairness": 0.7,
        "retention": 0.5,
    },
    {
        "evidence": 0.2,
        "oversight": -0.8,
        "privacy": -0.3,
        "governance": -0.7,
        "transparency": -0.2,
        "safety": 0.4,
        "access": 0.8,
        "accountability": -0.4,
        "fairness": 0.2,
        "retention": -0.8,
    },
    {
        "evidence": 0.6,
        "oversight": 0.1,
        "privacy": 0.7,
        "governance": 0.3,
        "transparency": 0.9,
        "safety": 0.7,
        "access": 0.9,
        "accountability": 0.4,
        "fairness": 0.9,
        "retention": 0.2,
    },
)


def build_profiles(
    rng: random.Random,
    cluster_count: int,
    statements: list[StatementSpec],
) -> list[dict[str, float]]:
    """Use stable held-out archetypes with per-snapshot semantic jitter."""

    del statements
    order = rng.sample(range(len(_ARCHETYPES)), cluster_count)
    return [
        {
            dimension: max(
                -1.0,
                min(
                    1.0,
                    _ARCHETYPES[index][dimension] + rng.uniform(-0.12, 0.12),
                ),
            )
            for dimension in DIMENSIONS
        }
        for index in order
    ]


def make_snapshot(rng: random.Random, session_index: int) -> dict[str, Any]:
    return build_snapshot(
        rng,
        session_index,
        seed=SEED,
        family=GENERATOR_FAMILY,
        statement_bank=STATEMENT_SPECS,
        profile_builder=build_profiles,
    )


def make_snapshot_with_cluster_patterns(
    rng: random.Random, session_index: int
) -> tuple[dict[str, Any], list[list[int]]]:
    """Compatibility name for the generator-diagnostic replay hook."""

    return make_snapshot_with_latent_patterns(
        rng,
        session_index,
        seed=SEED,
        family=GENERATOR_FAMILY,
        statement_bank=STATEMENT_SPECS,
        profile_builder=build_profiles,
    )


def main() -> None:
    rng = random.Random(SEED)
    output_path = (
        Path(__file__).parents[1]
        / "commonground_predict"
        / "data"
        / "eval_synthetic.jsonl"
    )
    snapshots = [make_snapshot(rng, index) for index in range(SNAPSHOT_COUNT)]
    output_path.write_text(
        "".join(
            json.dumps(snapshot, separators=(",", ":")) + "\n" for snapshot in snapshots
        ),
        encoding="utf-8",
    )
    print(f"wrote {SNAPSHOT_COUNT} snapshots to {output_path}")


if __name__ == "__main__":
    main()
