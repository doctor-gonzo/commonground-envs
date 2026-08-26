# commonground-elicit

`commonground-elicit` 0.2.0 is a deterministic native Verifiers v1 taskset for
finding planted ambiguities, contradictions, and gaps in small sets of
fictional policy documents. Its scenarios are synthetic, generated offline
from committed templates, and carry explicit provenance.

The find task accepts strict JSON containing `findings` and `questions`:

```json
{"findings":[{"doc_id":"policy","quote":"ambiguous passage","type":"ambiguity"}],"questions":[{"doc_id":"policy","quote":"ambiguous passage","question":"Should this threshold be made explicit?","target_stances":{"operations":"agree","risk":"disagree","support":"pass"}}]}
```

The reward is global one-to-one finding F1 against the planted answer key. A
candidate quote must first be a normalized ordered contiguous token span of the
claimed visible document. It must then match the document, finding type, and
planted anchor with at least 80% contiguous anchor coverage and a
longest-common-contiguous-token F1 of at least 0.5. This permits a close quote
or modest surrounding context without letting a tiny anchor fragment count as
a full true positive. Semantic operators are preserved during normalization,
so symbolic negation, inequality, sign, and percentage changes cannot inherit
the original quote's grounding. Reordered tokens, fabricated spans,
paraphrases, and actor/object reversals likewise receive no credit. Extra
findings reduce precision. Question utility is computed beside finding F1 as a
logged metric with weight zero.

Set `task="elicit-ask"` for the question-raising split. It requires exactly K
strict-JSON questions with a quote copied from the visible document and a
predicted stance for every listed faction. Utility is credited only when the
raw document ID and quote match the planted anchor, the question equals the
canonical question or a finite generator-authored alias under NFC
normalization, and the complete planted target-stance vector matches.
Duplicates targeting the same planting are rejected even when one uses an
alias. Unlisted paraphrases, composites, and reversals receive zero. This
bounded-recall contract avoids rewarding lexical fragments without a judge
model. Disagreement is the mean of normalized vote entropy and faction-pair
separation from `commonground-score`. Generic divisiveness does not match a
planting.
Questions are yes/no propositions: `agree` predicts yes, `disagree` predicts
no, and `pass` means that faction takes no position.

Difficulty arguments are `docs_count`, `docs_length`, `planted_density`,
`distractor_density`, `panel_polarization`, and `question_count`. Generation,
loading, and scoring do not use the network, wall clock, or a judge model. On
the find task, a row with fewer visible plants caps its logged companion
question count to the visible answer-key size.

Native Verifiers rows use typed `TaskData` with only the model prompt and
scoring-side answer/info fields. The hidden answer carries the planted findings
and question oracle; prompts are rendered only from visible documents and
public faction descriptions. Completion parsing is bounded to 32,768
characters, 64 object starts, depth 32, and 10,000 decoded nodes. Malformed or
over-limit output fails closed to zero reward.

## Configuration

The `split` loader argument selects the bundled rollout/eval file by name:
`"eval"` (the default held-out scenarios) or `"train"`. An explicit
`data_path` or `COMMONGROUND_ELICIT_DATA_PATH` takes precedence over `split`;
`train_data_path` or `COMMONGROUND_ELICIT_TRAIN_DATA_PATH` independently takes
precedence for the environment's training dataset.

Validate the native v1 training taskset without a model:

```bash
uv run validate commonground-elicit --taskset.split train \
  --runtime.type subprocess --rich false
```

The packaged `commonground_elicit/dependency-manifest.txt` records the exact
no-dev resolution used for this 0.2.0 candidate, including the root `uv.lock`
SHA-256, Python scope, and uv generator version. Workspace sources appear as
immutable distribution pins; the manifest is provenance rather than a
cross-platform installer.

For model evaluation, taskset controls live below `env.taskset`:

```bash
uv run eval commonground-elicit --env.taskset.split train -m MODEL --no-push
uv run eval commonground-elicit --env.taskset.task-mode elicit-ask -m MODEL --no-push
```

## Bundled splits and floors

`commonground_elicit/data/train_synthetic.jsonl` contains 40 semantically
unique scenarios across four committed training templates and ten operative
contexts each. `commonground_elicit/data/eval_synthetic_heldout.jsonl`
contains 20 semantically unique scenarios across twenty disjoint held-out
domains. Fingerprints cover document and answer-key semantics while ignoring
seed, organization name, and document order; generation fails on duplicates.
Every row is explicitly labeled `synthetic: true`. Regenerate both files with:

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
| find | Random visible spans | 0.133 |
| find | Flag vague-sounding spans | 0.195 |
| elicit-ask | Template clarity questions | 0.000 |
| elicit-ask | Randomly targeted questions | 0.000 |

See the bundled data directory's dataset card for the planting and separation
methodology.

The 0.1.x model table is intentionally removed: 0.2.0 replaced both the held-out
semantic corpus and the quote-grounding reward, so those historical scores are
not comparable. Fresh baselines must be run on the exact private 0.2.0
candidate before any public performance claim.

The public package contains every planted answer key for reproducibility and
open training. It is not contamination-resistant. A leaderboard or
consequential comparison needs a private server-side evaluation split.

## The commonground family and human socket

`commonground-elicit` and `commonground-predict` are separate Hub IDs in one
program. Predict scores masked-vote inference; elicit scores document-grounded
finding and question raising. They share `commonground-score` and the same
synthetic-versus-human provenance boundary without combining incompatible
rubrics behind a mode flag.

The scenario schema reserves `human_feedback` for a reviewed, consented
Context Engine snapshot. It uses the same strict validator as predict intake:
positional pseudonyms, exact clusters with `k >= 5`, consistent vote statistics,
no held-out labels, and explicit source, rights, schema-version, exporter, and
privacy-review attestations. Automated identifier screening is limited and
never replaces human review. The bundled elicit release uses synthetic
scenarios only.

## Evaluation

After installing the locked workspace dependencies, run the model-free native
v1 validation gate:

```bash
uv run validate commonground-elicit --runtime.type subprocess --rich false
```

Run a model evaluation without uploading results with:

```bash
uv run eval commonground-elicit -m MODEL --no-push
```

The test suite, split generator, rewards, validation, and floor baselines are
hermetic and require no API key.
