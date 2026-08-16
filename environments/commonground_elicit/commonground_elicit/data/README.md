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
lists, and deterministic faction-stance vectors in the hidden answer key.
Precise, consistent passages are carried as distractors. Prompts contain the
documents and public faction descriptions but not the answer key or priors.

## Split separation

`train_synthetic.jsonl` is generated only from template set A.
`eval_synthetic_heldout.jsonl` is generated only from template set B. The sets
use disjoint template IDs, sectors, document styles, and planting patterns.
Tests regenerate both files byte-for-byte and verify that separation.

## Model-free floors

The committed floor script builds responses from visible documents and faction
descriptions only. It scores those responses against the planted keys in a
separate step. Running `uv run python scripts/compute_elicit_floors.py` prints:

| Task | Baseline | mean reward |
| --- | --- | ---: |
| find | Random visible spans | 0.167 |
| find | Flag vague-sounding spans | 0.500 |
| elicit-ask | Template clarity questions | 0.000 |
| elicit-ask | Randomly targeted questions | 0.000 |

These are deterministic baseline outputs, not model evaluation results.
