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

Rendered Verifiers rows use only the canonical `prompt`, `answer`, `info`, and
`example_id` columns. The hidden `answer` JSON carries the planted findings and
question oracle; `info` carries the effective question count, panel
polarization, and combined-question mode used by scoring. Prompts are rendered
only from the visible documents and public faction descriptions, never from
either hidden scoring payload.

## Configuration

The `split` loader argument selects the bundled rollout/eval file by name:
`"eval"` (the default held-out scenarios) or `"train"`. An explicit
`data_path` or `COMMONGROUND_ELICIT_DATA_PATH` takes precedence over `split`;
`train_data_path` or `COMMONGROUND_ELICIT_TRAIN_DATA_PATH` independently takes
precedence for the environment's training dataset.

For hosted training against the bundled train split:

```toml
[[env]]
id = "charliethompson/commonground-elicit"

[env.args]
split = "train"
```

## Bundled splits and floors

`commonground_elicit/data/train_synthetic.jsonl` contains 40 scenarios from the
committed training template set. `commonground_elicit/data/eval_synthetic_heldout.jsonl`
contains 20 scenarios from the disjoint held-out template set. Every scenario is explicitly labeled
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
| find | Random visible spans | 0.150 |
| find | Flag vague-sounding spans | 0.500 |
| elicit-ask | Template clarity questions | 0.000 |
| elicit-ask | Randomly targeted questions | 0.000 |

See the bundled data directory's dataset card for the planting and separation
methodology.

Model baselines use the repository's multi-environment sweep with three
rollouts per held-out example. The aggregator emits reward, `finding_f1`, and
`question_utility` mean ± population standard deviation only after every
expected rollout is present; see the
[operator commands](https://github.com/doctor-gonzo/commonground-envs#baseline-sweep).

Recorded 2026-08-16 via `uv run vf-eval commonground-elicit -m <model> -n 20 -r 3 --save-results`
against Prime Inference, aggregated by `scripts/aggregate_baselines.py`:

| Model | finding_f1 (mean ± std) | question_utility (mean) |
| --- | ---: | ---: |
| anthropic/claude-sonnet-4.5 | 0.524 ± 0.205 | 0.000 |
| openai/gpt-4.1-mini | 0.502 ± 0.257 | 0.000 |
| openai/gpt-4.1 | 0.448 ± 0.163 | 0.000 |
| google/gemini-2.5-flash | 0.405 ± 0.268 | 0.000 |
| meta-llama/llama-3.3-70b-instruct | 0.294 ± 0.259 | 0.000 |

Reading: only two of five models clear the "flag vague-sounding spans"
heuristic floor (0.500) — a naive heuristic remains competitive with frontier
models at strict-F1 issue finding (ambiguities, contradictions, and gaps),
which is the headroom this environment exists to measure. Scores also reflect
the strict-compliance design: findings must cite the planted span at the
planted granularity, so recall-heavy smaller models can edge
precision-conservative larger ones. `question_utility` at 0.000 across all
models reflects the deliberately bounded-recall v0 question contract
(canonical/alias matching); in these recorded default find-task runs it is a
logged weight-zero companion metric. In the separate `elicit-ask` task mode it
is the scored reward (weight 1.0) under the same bounded-recall contract.

## The commonground family and human socket

`commonground-elicit` and `commonground-predict` are separate Hub IDs in one
program. Predict scores masked-vote inference; elicit scores document-grounded
finding and question raising. They share `commonground-score` and the same
synthetic-versus-human provenance boundary without combining incompatible
rubrics behind a mode flag.

The scenario schema reserves `human_feedback` for a validated, consented
Context Engine snapshot. On that future path, real votes replace the persona
panel and provenance must be marked human and non-synthetic. The bundled elicit
release uses planted synthetic scenarios only; it makes no real-human-data
claim.

## Evaluation

After installing the locked workspace dependencies, run a local Verifiers
evaluation with:

```bash
uv run vf-eval commonground-elicit
```

The test suite, split generator, rewards, and floor baselines are hermetic and
require no API key.
