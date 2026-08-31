# commonground-predict

`commonground-predict` 0.5.0 is a public deterministic Verifiers environment
for probabilistic masked-vote prediction over synthetic stakeholder panels. It
has no judge model: reward is computed locally against eight masked cells per
snapshot. The 0.5 release adds probability-native interpretation and release
reporting; historical 0.4.1 remains valid preliminary evidence for its own
artifact.

The task is best understood as **semantic-conditioned matrix completion**.
Statement text now causally selects a policy dimension, and latent cluster
profiles determine vote propensity on that dimension. The visible vote matrix
still carries substantial signal: a prompt-observable 5-NN comparator scores
0.891. Results therefore do not by themselves establish policy understanding,
human-preference validity, or general collective-preference reasoning.

This environment is associated with [Context Engine](https://contextengine.sh),
which can structure stakeholder statements and votes into auditable preference
maps. A planned governed export path could let consenting individuals and
groups retain, license, or sell derived preference datasets. Version 0.5
contains no live participant data and no live-data exporter.

## Data

Each JSONL snapshot contains ordered policy statements, positional participant
IDs, a visible vote matrix (`1` agree, `-1` disagree, `0` pass, `null` unseen),
masked-cell coordinates, hidden labels, planted clusters, and synthetic
provenance.

| Split | Rows | Generator family | Statement bank |
| --- | ---: | --- | ---: |
| `train` | 200 | `train-random-mixture-v2` | 30 training-only statements |
| `eval` | 100 | `heldout-archetype-threshold-v2` | 20 evaluation-only statements |
| `ce-demo` | 1 | operator-authored synthetic fixture | adapted demo corpus |

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
drives accuracy, with deterministic tie ordering. Completion parsing is
bounded. Missing, extra, duplicate-normalized, or malformed output receives
zero reward; its diagnostic Brier loss is 1.0 for non-empty tasks.

## Scoring and comparators

- `probability_reward`, reward weight 1.0: `1 - normalized Brier`, so calibrated
  confidence changes the optimized objective.
- `vote_accuracy`, metric only: exact argmax accuracy over masked cells.
- `brier`, metric only: conventional three-class squared error divided by two,
  bounded to `[0,1]`.
- Brier skill, comparator report only: improvement over the uniform-probability
  reference, where 0 equals uniform, 1 is perfect, and negative is worse.

Exact model-free results on the bundled 100-row evaluation split:

| Comparator class | Comparator | probability reward | vote_accuracy | normalized Brier | Brier skill vs uniform |
| --- | --- | ---: | ---: | ---: | ---: |
| No-input | Uniform probability | 0.667 | 0.590 | 0.333 | 0.000 |
| Evaluation-corpus visible (transductive) | Global visible class prior | 0.717 | 0.590 | 0.283 | 0.150 |
| Train-split no-text | Global empirical class prior | 0.669 | 0.166 | 0.331 | 0.008 |
| Train-split text-only | Bag-of-words vote probabilities | 0.401 | 0.244 | 0.599 | -0.796 |
| No-input | Always agree | 0.590 | 0.590 | 0.410 | -0.230 |
| Prompt-observable matrix-only | Per-statement visible majority | 0.581 | 0.581 | 0.419 | -0.256 |
| Prompt-observable matrix-only | Per-statement visible class frequencies | 0.760 | 0.581 | 0.240 | 0.279 |
| Prompt-observable matrix-only | Nearest participant (1-NN) | 0.819 | 0.819 | 0.181 | 0.456 |
| Prompt-observable matrix-only | Five-neighbor vote | 0.891 | 0.891 | 0.109 | 0.674 |
| Prompt-observable matrix-only | Five-neighbor vote frequencies | 0.889 | 0.891 | 0.111 | 0.666 |
| Prompt-observable matrix-only | Distance-weighted 5-NN with smoothing | 0.881 | 0.900 | 0.119 | 0.643 |
| Held-out-label diagnostic | Per-snapshot best constant | 0.631 | 0.631 | 0.369 | -0.106 |
| Generator diagnostic | Latent cluster-pattern replay | 0.916 | 0.916 | 0.084 | 0.749 |

Uniform already earns `2/3` probability reward under normalized three-class
Brier, so raw reward must be interpreted against that reference. The
evaluation-corpus prior is explicitly transductive; the clean train-split prior
adds almost no skill, and the train-split text-only model performs poorly across
the held-out generator profile. Strong per-snapshot matrix neighbors show that
visible collaborative structure dominates the current corpus. The last two
rows are diagnostics, not floors: one reads held-out labels and one replays
hidden generator state. Calibrated matrix factorization, item-item/spectral
models, and shuffled-text ablations remain recommended before claiming a
distinct language contribution.

In the exact-artifact 0.5 study, Claude Sonnet 4.5, Gemini 2.5 Flash, GPT-4.1,
and Qwen3 30B A3B Instruct achieved probability rewards of 0.808, 0.773,
0.735, and 0.700 respectively across 100 tasks and five rollouts. All means
exceed the 0.667 uniform reference and all pairwise task-clustered intervals
exclude zero; all remain below the strongest matrix-only comparators. See the
[0.5.0 evaluation report](https://github.com/doctor-gonzo/commonground-envs/blob/master/docs/evaluation-report-0.5.0.md).

## Usage

```bash
uv run validate commonground-predict --runtime.type subprocess --rich false
uv run eval commonground-predict -m MODEL --no-push
uv run eval commonground-predict --env.taskset.split train -m MODEL --no-push
```

`split` accepts `eval`, `train`, or `ce-demo`. `data_path` or
`COMMONGROUND_DATA_PATH` overrides the bundled split. `masked_vote_count`
deterministically remasks known votes; `min_cluster_count` filters rows.
Non-synthetic custom rows must pass the separate fail-closed
[human-data governance contract](https://github.com/doctor-gonzo/commonground-envs/blob/master/docs/human-data-governance.md).

Public answer keys make this useful for open training but unsuitable for a
contamination-resistant leaderboard. Use a procedurally fresh private family
for post-training evaluation. Native v1 and the Prime-compatible legacy adapter
share the same data and scoring implementation.
