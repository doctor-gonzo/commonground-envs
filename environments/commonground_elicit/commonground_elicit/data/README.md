# commonground-elicit dataset card

The bundled 0.6.0 candidate contains two deterministic synthetic JSONL splits:

- `train_synthetic.jsonl`: 100 scenarios from four train templates.
- `eval_synthetic_heldout.jsonl`: 100 scenarios from twenty disjoint held-out
  templates.

All organizations, documents, factions, issues, votes, and identifiers are
synthetic. `human_feedback` is `null`. The files contain public answer keys and
are intended for reproducible open training and evaluation.

## Scenario contents

Each scenario includes:

- three to five public organizational documents, including type-neutral
  distractors;
- three to five factions with round-trip-exact general values over access,
  adaptability, continuity, oversight, and safety;
- exactly one ambiguity, contradiction, and gap;
- an authored primary passage for each issue and a five-slot Ask profile;
- an explicit opposing passage for each contradiction;
- stances composed deterministically from faction values, issue trade-off
  weights, the pass threshold, and explicit yes-side orientation.

Faction descriptions are rendered from general values independently of the
current issue answers. Contradiction links are authored in the template and
remapped with opaque document IDs during generation; they are never selected
by lexical overlap.

The canonical question is a readable presentation of each issue. Find scores
visible evidence, issue type, valid yes/no diagnosis form, and paired evidence
for contradictions; it has no submitted decision object or alias table. Ask
exposes the canonical structured profile and requires it to be copied exactly.

## Separation and leakage checks

Generation is deterministic for the checked-in seeds and date. The release
gate byte-compares regenerated files and checks:

- unique exact instance fingerprints within each split and no cross-split
  exact overlap;
- unique policy-semantic identities within and across splits, excluding opaque
  IDs and neutral layout;
- disjoint template and generator-profile labels;
- one word unigram/bigram TF-IDF cross-split nearest-neighbor threshold;
- absence of retired stable document identifiers.

These checks target exact and near-neighbor overlap. They do not remove all
source-aware generator shortcuts or establish independence from the shared
generator recipe.

## Reproduction

```bash
uv run python scripts/generate_elicit_splits.py --output-dir /tmp/elicit-splits
cmp /tmp/elicit-splits/train_synthetic.jsonl train_synthetic.jsonl
cmp /tmp/elicit-splits/eval_synthetic_heldout.jsonl eval_synthetic_heldout.jsonl
```

From the repository root, `bash scripts/local_smoke.sh` performs this comparison
alongside tests, model-free summaries, static checks, and fresh artifact loads.

## Limits

The held-out templates share the same generator, issue taxonomy, value model,
and scoring ontology as training. One public trade-off vector wins Ask in 90 of
100 evaluation rows. Public answers allow memorization. This fixed-corpus
shortcut does not reveal every scored field, but it narrows what a high score
establishes. Claims about transfer, training benefit, or real
organizations require fresh private evaluation from an independently
implemented generator and, for human data, separate consent and governance.
