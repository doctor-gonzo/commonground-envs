# commonground-elicit

`commonground-elicit` 0.2.5 is a deterministic native Verifiers v1 taskset for
finding planted ambiguities, contradictions, and gaps in small sets of
fictional policy documents. Its scenarios are synthetic, generated offline
from committed templates, and carry explicit provenance.

This environment operationalizes a complementary
[Context Engine](https://contextengine.sh) workflow: identifying unresolved
policy issues and asking questions that reveal meaningful faction differences.
Within an organization, that workflow can turn stakeholder input into auditable
decision records and, through a planned governed export path, preference
datasets that a consenting group could retain, license, or sell for evaluation
or training. Release 0.2.5 is entirely synthetic; Context Engine is a planned
future data source, not its current data source, and the exporter is not yet
implemented.

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
raw document ID and exact quote match a planted anchor. The question must use
strict yes/no form and reuse at least one informative token from its quoted
passage; its wording does not have to match a hidden authored sentence.
One-to-one assignment prevents duplicate questions from claiming the same
planting twice. Half of each matched utility comes from issue grounding and
half from per-faction stance accuracy, so partially correct stance vectors earn
partial credit. The result is scaled by panel polarization and the planting's
mean normalized vote entropy/faction-pair separation from
`commonground-score`. No judge model or network call is used.
Questions are yes/no propositions: `agree` predicts yes, `disagree` predicts
no, and `pass` means that faction takes no position.

This deterministic contract measures grounded issue selection and stance
prediction. Beyond yes/no form and lexical connection to the evidence, it does
not attempt to judge the prose quality or semantic equivalence of an open-ended
question. Use a judge-backed or human-reviewed layer when that distinction is
the research target.

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
no-dev resolution used for this 0.2.5 candidate, including a closure-scoped
resolution SHA-256, Python scope, and uv generator version. Workspace sources
appear as immutable distribution pins; the manifest is provenance rather than
a cross-platform installer.

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

The 0.1.x model table is intentionally removed: 0.2.0 replaced the held-out
semantic corpus and 0.2.2 replaced the question reward, so those historical
scores are not comparable. Fresh baselines must be run on the exact private
0.2.5 candidate before any public performance claim.

The public package contains every planted answer key for reproducibility and
open training. It is not contamination-resistant. A leaderboard or
consequential comparison needs a private server-side evaluation split.

## The commonground family

`commonground-elicit` and `commonground-predict` are separate Hub IDs in one
program. Predict scores masked-vote inference; elicit scores document-grounded
finding and question raising. They share `commonground-score` and the same
synthetic-data provenance conventions without combining incompatible rubrics
behind a mode flag. The bundled 0.2.x scenarios are synthetic. The optional
advanced custom `human_feedback` path uses the same fail-closed validator as
Predict; see
[human-data governance](https://github.com/doctor-gonzo/commonground-envs/blob/master/docs/human-data-governance.md).

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

The package also exposes a genuine legacy `SingleTurnEnv` from
`load_environment()` because Prime CLI 0.6.28 Hosted Evaluations still use the
v0 runner. Native v1 resolves `ElicitTaskset` and its bundled pure-chat
`ElicitHarness` directly from the package exports.

The test suite, split generator, rewards, validation, and floor baselines are
hermetic and require no API key.
