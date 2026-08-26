"""Generate the bundled synthetic train split for commonground-predict.

Reuses the frozen eval generator's snapshot machinery with a distinct seed,
snapshot count, session-id range, and output file. The eval splits and their
published floors stay untouched; this script only writes
``train_synthetic.jsonl`` for training runs (point ``COMMONGROUND_DATA_PATH``
or the RL config's env data path at it).

Regenerate with: ``python environments/commonground_predict/scripts/generate_synthetic_train.py``
Determinism: same seed -> byte-identical output.
"""

from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path
from typing import Any, Protocol, cast

TRAIN_SEED = 20260815
TRAIN_SNAPSHOT_COUNT = 150
SESSION_INDEX_OFFSET = 1000  # keeps session_ids disjoint from the eval split (0-19)

# Training policies are intentionally text-disjoint from the evaluation bank.
# They exercise the same broad enterprise-governance concepts without making
# exact evaluation statements recoverable from the public training split.
TRAIN_STATEMENT_BANK = [
    "The incident assistant should record the evidence and timestamp behind every proposed severity classification.",
    "The recruiting assistant should not infer protected characteristics from names, photographs, schools, or writing style.",
    "The procurement assistant should identify the approved supplier record before recommending that an order be placed.",
    "The translation assistant should preserve legal notices and mark any phrase whose meaning remains uncertain.",
    "The security assistant may prepare an access change but an authorized administrator must approve its execution.",
    "The scheduling assistant should reveal which constraints caused it to reject a requested meeting time.",
    "The knowledge assistant should distinguish retrieved facts from generated suggestions in its response.",
    "The claims assistant should route suspected fraud to a qualified investigator rather than making a final accusation.",
    "The inventory assistant should confirm warehouse availability before promising a delivery date.",
    "The marketing assistant should use only customer segments whose collection purpose covers the proposed campaign.",
    "The compliance assistant should preserve the exact source wording when quoting a regulatory obligation.",
    "The maintenance assistant should stop automated work when sensor readings conflict with the equipment record.",
    "The travel assistant should display cancellation conditions before asking an employee to approve a booking.",
    "The document assistant should remove hidden comments and tracked changes before preparing a file for external sharing.",
    "The benefits assistant should send eligibility exceptions to a designated benefits specialist for a decision.",
    "Each production assistant should have a documented owner who can disable its tools and revoke its credentials.",
    "Changes to an assistant's system instructions should receive the same risk review as changes to its model or tools.",
    "A person affected by an automated eligibility recommendation should receive a reason and a meaningful appeal route.",
    "Evaluation sets for an assistant should cover foreseeable misuse as well as its intended workflow.",
    "Records created by an assistant should follow the retention schedule of the business process they support.",
    "The research assistant should attach stable citations for material factual claims and identify inaccessible sources.",
    "The forecasting assistant should show the observation window and uncertainty assumptions behind each projection.",
    "The customer-success assistant should not promise roadmap work unless the commitment appears in an approved source.",
    "The payroll assistant should require a second reviewer before changing bank or tax details.",
    "The facilities assistant should place occupant safety ahead of energy optimization when its signals disagree.",
    "The quality assistant should retain failed inspection results instead of replacing them with a later passing run.",
    "The learning assistant should label generated examples that have not been reviewed by a subject-matter expert.",
    "The records assistant should log who authorized a legal hold and which deletion rules it suspends.",
    "The vendor-risk assistant should separate verified evidence from questionnaire claims that have not been checked.",
    "The localization assistant should request regional review before changing required disclosures or consent language.",
]

_EVAL_GENERATOR = Path(__file__).with_name("generate_synthetic_eval.py")


class EvalGenerator(Protocol):
    SEED: int
    STATEMENT_BANK: list[str]

    def make_snapshot(
        self, rng: random.Random, session_index: int
    ) -> dict[str, Any]: ...


def _load_eval_generator() -> EvalGenerator:
    spec = importlib.util.spec_from_file_location(
        "commonground_generate_synthetic_eval", _EVAL_GENERATOR
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {_EVAL_GENERATOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return cast(EvalGenerator, module)


def main() -> None:
    generator = _load_eval_generator()
    # make_snapshot reads the module-level SEED for meta.source labeling; the
    # train split records its own seed there. The separate statement bank is a
    # methodological boundary: eval policy text never appears in training.
    generator.SEED = TRAIN_SEED
    generator.STATEMENT_BANK = TRAIN_STATEMENT_BANK
    rng = random.Random(TRAIN_SEED)
    output_path = (
        Path(__file__).parents[1]
        / "commonground_predict"
        / "data"
        / "train_synthetic.jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    snapshots = [
        generator.make_snapshot(rng, SESSION_INDEX_OFFSET + index)
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
