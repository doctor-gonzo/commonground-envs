# commonground-elicit dataset card

## Scope and provenance

Both bundled splits contain fictional, synthetically generated policy
scenarios. Every row declares `synthetic: true`, a fixed generator seed, its
template ID and template set, an explicit generation date, and templated
generation mode. The default pipeline is offline and does not call an LLM.

## Planting methodology

Each scenario starts from a committed domain template with public policy
documents and faction descriptions. The generator plants an ambiguity, a
cross-document contradiction, and an uncovered case, then records their raw
document IDs, exact anchor quotes, canonical questions, finite authored alias
lists, and deterministic faction-stance vectors in the hidden answer key. The
canonical questions and aliases document generator intent but are not used as
an exact-string reward gate. Elicit-ask instead matches the exact visible
document anchor, validates yes/no form plus a lexical connection to that
anchor, and separately scores faction-stance accuracy.
Precise, consistent passages are carried as distractors. Prompts contain the
documents and public faction descriptions but not the answer key or priors.

## Split separation

| Split | Scenarios | Template set |
| --- | ---: | --- |
| `train_synthetic.jsonl` | 40 | A (training) |
| `eval_synthetic_heldout.jsonl` | 20 | B (held out) |

The sets use disjoint template IDs, sectors, document styles, and planting
patterns. Training contains ten deterministic semantic contexts for each of
four templates; the context changes the operative scope of every planted
passage and question rather than merely changing presentation order. Held-out
evaluation contains one task from each of twenty authored domains. A
semantic-key release check rejects rows in either split that differ only by
seed, organization name, or document order. Tests regenerate both files
byte-for-byte and verify all forty training keys and all twenty held-out keys
are unique.

## Model-free floors

The committed floor script builds responses from visible documents and faction
descriptions only. It scores those responses against the planted keys in a
separate step. Running `uv run python scripts/compute_elicit_floors.py` prints:

| Task | Baseline | mean reward |
| --- | --- | ---: |
| find | Random visible spans | 0.133 |
| find | Flag vague-sounding spans | 0.195 |
| elicit-ask | Template clarity questions | 0.000 |
| elicit-ask | Randomly targeted questions | 0.000 |

These are deterministic baseline outputs, not model evaluation results.
