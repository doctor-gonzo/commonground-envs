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

TRAIN_SEED = 20260815
TRAIN_SNAPSHOT_COUNT = 150
SESSION_INDEX_OFFSET = 1000  # keeps session_ids disjoint from the eval split (0-19)

_EVAL_GENERATOR = Path(__file__).with_name("generate_synthetic_eval.py")


def _load_eval_generator():
    spec = importlib.util.spec_from_file_location(
        "commonground_generate_synthetic_eval", _EVAL_GENERATOR
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    generator = _load_eval_generator()
    # make_snapshot reads the module-level SEED for meta.source labeling; the
    # train split records its own seed there.
    generator.SEED = TRAIN_SEED
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
