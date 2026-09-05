# commonground-predict

`commonground-predict` 0.6.1 is a deterministic Verifiers environment for
probabilistic masked-vote prediction over synthetic stakeholder panels. It has
no judge model: reward is computed locally against eight masked cells per
snapshot. Version 0.6.1 is a documentation-only corrective successor to 0.6.0;
the task implementation, corpus, prompt contract, and scorer are
behavior-identical.

The task is best understood as **semantic-conditioned matrix completion**.
Statement text now causally selects a policy dimension, and latent cluster
profiles determine vote propensity on that dimension. The visible vote matrix
still carries substantial signal: the smoothed neighbor-frequency comparator
earns 0.881 probability reward and 0.900 accuracy. Results therefore do not by themselves establish policy understanding,
human-preference validity, or general collective-preference reasoning.

This environment is associated with [Context Engine](https://contextengine.sh),
which can structure stakeholder statements and votes into auditable preference
maps. A planned governed export path could let consenting individuals and
groups retain, license, or sell derived preference datasets. Version 0.6
contains no live participant data and no live-data exporter.

## Data

Each JSONL snapshot contains ordered policy statements, positional participant
IDs, a visible vote matrix (`1` agree, `-1` disagree, `0` pass, `null` unseen),
masked-cell coordinates, hidden labels, planted clusters, and synthetic
provenance.

| Split | Rows | Generator family | Statement bank | JSONL SHA-256 |
| --- | ---: | --- | ---: | --- |
| `train` | 200 | `train-random-mixture-v2` | 30 training-only statements | `61a512f686f91b5df641ff11ed7376c5a9e01af4ed3909992c24fbdb64e42ce6` |
| `eval` | 100 | `heldout-archetype-threshold-v2` | 20 evaluation-only statements | `f45da41f10567044b14bbb8fcb01d1f11e1fe514035a0beb1374a52adf454d0f` |
| `ce-demo` | 1 | operator-authored synthetic fixture | adapted demo corpus | See `NOTICE` provenance. |

Train and evaluation use different seeds, session IDs, policy text, and
profile-building families. In both synthetic families, each statement has an
explicit semantic dimension; changing that dimension changes its generated
votes. Cluster profiles, thresholding, pass behavior, and participant noise
remain synthetic assumptions rather than estimates of people.

The CE demo fixture is also synthetic. Its immutable Context Engine source,
transformation boundary, hashes, and MPL-2.0 treatment are recorded in
`NOTICE` and its byte-identical Hub-source copy `LICENSES/NOTICE.txt`.

## Prompt and response contract

The prompt contains policy text, the visible matrix, and masked coordinates.
Return exactly one finite, non-negative probability distribution per masked
cell:

```json
{
  "predictions": {
    "0,5": {"agree": 0.7, "disagree": 0.2, "pass": 0.1}
  }
}
```

The normalized keys must exactly equal the masked cells, with no duplicates,
and each mapping must contain
exactly `agree`, `disagree`, and `pass` with a positive total. Values are
normalized before scoring. Bare hard labels are rejected. The argmax label
drives accuracy, with insertion-order-independent ties resolved as agree, then
pass, then disagree. Completion parsing is bounded. Missing, extra,
duplicate-normalized, or malformed output receives zero reward; its diagnostic
Brier loss is 1.0 for non-empty tasks. Numeric strings, booleans, non-finite
values or totals, and integers too large to convert safely are malformed.
Markdown fences, prose wrappers, trailing JSON values, and repeated raw object
keys are rejected rather than recovered.
The complete response must therefore start with `{` and end with `}` without a
Markdown fence or prose wrapper. Provider-native JSON-object mode may be used
as a decoding constraint; it does not change the task or scoring contract.

## Scoring and comparators

- `probability_reward`, reward weight 1.0: `1 - normalized Brier`, so calibrated
  confidence changes the optimized objective.
- `vote_accuracy`, metric only: exact argmax accuracy over masked cells.
- `brier`, metric only: conventional three-class squared error divided by two,
  bounded to `[0,1]`.
- `original_snapshot_visible_prior_brier`, metric only: loss of a visible-matrix
  class prior, held fixed across prompt ablations.
- Brier skill against uniform or the train-split empirical prior: `0` equals the
  named reference, `1` is perfect, and a negative value is worse.

Exact model-free results on the bundled 100-row evaluation split:

| Comparator class | Comparator | probability reward | vote_accuracy | normalized Brier | Brier skill vs uniform | Brier skill vs empirical prior |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| No-input | Uniform probability | 0.667 | 0.590 | 0.333 | 0.000 | -0.008 |
| Train-split no-text | Global empirical class prior | 0.669 | 0.166 | 0.331 | 0.008 | 0.000 |
| Prompt-observable matrix-only | Per-statement visible class frequencies | 0.760 | 0.581 | 0.240 | 0.279 | 0.274 |
| Prompt-observable matrix-only | Smoothed distance-weighted 5-neighbor frequencies | 0.881 | 0.900 | 0.119 | 0.643 | 0.640 |

Uniform already earns `2/3` reward under normalized three-class Brier, so raw
reward must be interpreted against a named reference. The two stronger
prompt-visible references show that matrix structure carries substantial signal.
Prompt-view ablations and a fresh private generator are still required before
claiming that policy text contributes transferable information.

In the behavior-identical exact-artifact 0.6.0 study, Kimi K2 Instruct,
Gemini 2.5 Flash, GPT-4.1, and Qwen3 30B A3B Instruct achieved probability
rewards of 0.761, 0.739, 0.724, and 0.706 respectively across 100 tasks and
five rollouts. All means exceed the 0.667 uniform reference and remain below
the strongest matrix-only comparator. GPT-4.1 prompt-view ablations show that
shuffling statement text did not lower reward, so the study does not establish
use of statement semantics beyond the visible vote matrix. The public records
remain pinned to immutable 0.6.0 artifacts and are not relabeled as 0.6.1
evidence. See the
[0.6.0 evaluation report](https://github.com/doctor-gonzo/commonground-envs/blob/master/docs/evaluation-report-0.6.0.md).

## Usage

```bash
uv run validate commonground-predict --runtime.type subprocess --rich false
uv run eval commonground-predict -m MODEL --no-push
uv run eval commonground-predict --env.taskset.split train -m MODEL --no-push

# Repeat for full, matrix-only, text-only, and shuffled-text.
uv run eval commonground-predict \
  --env.taskset.prompt-mode matrix-only -m MODEL --no-push
```

`split` accepts `eval`, `train`, or `ce-demo`. `data_path` or
`COMMONGROUND_DATA_PATH` overrides the bundled split. `masked_vote_count`
deterministically remasks known votes; `min_cluster_count` filters rows.
`prompt_mode` accepts `full`, `matrix-only`, `text-only`, or `shuffled-text`.
The ablations preserve task IDs and answers: matrix-only withholds policy text,
text-only withholds the vote matrix, and shuffled-text deterministically
misaligns statement text with matrix columns. A repository checkout also
provides the tracked multi-mode
[`configs/eval/predict-ablation.toml`](https://github.com/doctor-gonzo/commonground-envs/blob/master/configs/eval/predict-ablation.toml);
that root-level file is not bundled in the environment package.
Non-synthetic custom rows must pass the separate fail-closed
[human-data governance contract](https://github.com/doctor-gonzo/commonground-envs/blob/master/docs/human-data-governance.md).

Public answer keys make this useful for open training but unsuitable for a
contamination-resistant leaderboard. Use a procedurally fresh private family
for post-training evaluation. Native v1 and the Prime-compatible legacy adapter
share the same data and scoring implementation.
