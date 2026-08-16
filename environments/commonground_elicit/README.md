# commonground-elicit

`commonground-elicit` is a deterministic Verifiers `SingleTurnEnv` for finding
planted ambiguities, contradictions, and gaps in small sets of fictional policy
documents. Its v0 scenarios are synthetic, generated offline from committed
templates, and carry explicit provenance.

The find task accepts strict JSON containing `findings` and `questions`:

```json
{"findings":[{"doc_id":"policy","quote":"ambiguous passage","type":"ambiguity"}],"questions":[{"doc_id":"policy","quote":"ambiguous passage","question":"Should this threshold be made explicit?","target_stances":{"operations":"agree","risk":"disagree","support":"pass"}}]}
```

The reward is one-to-one finding F1 against the planted answer key. A candidate
must match the document, finding type, and normalized quote at the configured
overlap threshold. Extra findings therefore reduce precision. Question utility
is computed beside finding F1 as a logged metric with weight zero.

Set `task="elicit-ask"` for the question-raising split. It requires exactly K
strict-JSON questions with a quote copied from the visible document and a
predicted stance for every listed faction. Utility is credited only when the
raw document ID and quote match the planted anchor, the question equals the
canonical question or a finite generator-authored alias under NFC
normalization, and the complete planted target-stance vector matches.
Duplicates targeting the same planting are rejected even when one uses an
alias. Unlisted paraphrases, composites, and reversals receive zero. This
bounded-recall v0 contract avoids rewarding lexical fragments without a judge
model. Disagreement is the mean of normalized vote entropy and faction-pair separation from
`commonground-score`. Generic divisiveness does not match a planting.
Questions are yes/no propositions: `agree` predicts yes, `disagree` predicts
no, and `pass` means that faction takes no position.

Difficulty arguments are `docs_count`, `docs_length`, `planted_density`,
`distractor_density`, `panel_polarization`, and `question_count`. Generation,
loading, and scoring do not use the network, wall clock, or a judge model. On
the find task, a row with fewer visible plants caps its logged companion
question count to the visible answer-key size.

## Bundled splits and floors

`commonground_elicit/data/train_synthetic.jsonl` uses the committed training
template set. `commonground_elicit/data/eval_synthetic_heldout.jsonl` uses the
disjoint held-out template set. Every scenario is explicitly labeled
`synthetic: true`; generation uses fixed seeds, an explicit provenance date,
and templated offline prose. Regenerate both files with:

```bash
uv run python scripts/generate_elicit_splits.py
```

The generation script is byte-reproducible. Baselines receive only the visible
documents and public faction descriptions; the planted answer key is used only
after generation to score their responses. Running
`uv run python scripts/compute_elicit_floors.py` on the bundled held-out split
prints:

| Task | Baseline | mean reward |
| --- | --- | ---: |
| find | Random visible spans | 0.167 |
| find | Flag vague-sounding spans | 0.500 |
| elicit-ask | Template clarity questions | 0.000 |
| elicit-ask | Randomly targeted questions | 0.000 |

See the bundled data directory's dataset card for the planting and separation
methodology.
