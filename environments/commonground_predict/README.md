# commonground-predict

`commonground-predict` 0.4.1 is a deterministic Verifiers environment for
probabilistic masked-vote prediction over synthetic stakeholder panels. It has
no judge model: reward is computed locally against eight masked cells per
snapshot.

The task is best understood as **semantic-conditioned matrix completion**.
Statement text now causally selects a policy dimension, and latent cluster
profiles determine vote propensity on that dimension. The visible vote matrix
still carries substantial signal: a prompt-observable 5-NN comparator scores
0.891. Results therefore do not by themselves establish policy understanding,
human-preference validity, or general collective-preference reasoning.

This environment is associated with [Context Engine](https://contextengine.sh),
which can structure stakeholder statements and votes into auditable preference
maps. A planned governed export path could let consenting individuals and
groups retain, license, or sell derived preference datasets. Version 0.4.1
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

Exact model-free results on the bundled 100-row evaluation split:

| Comparator class | Comparator | one-hot probability reward / vote_accuracy |
| --- | --- | ---: |
| Prompt-observable | Always agree | 0.590 |
| Prompt-observable | Per-statement visible majority | 0.581 |
| Prompt-observable | Nearest participant (1-NN) | 0.819 |
| Prompt-observable | Five-neighbor vote | 0.891 |
| Held-out-label diagnostic | Per-snapshot best constant | 0.631 |
| Generator diagnostic | Latent cluster-pattern replay | 0.916 |

The deterministic comparators emit one-hot forecasts, for which probability
reward equals vote accuracy. The last two rows are not floors: one reads held-out labels and one replays
hidden generator state. The narrow gap between 5-NN and latent replay is a
central limitation and should accompany any model result. Matrix-factorization,
item-item, spectral, text-only, matrix-only, and shuffled-text ablations remain
recommended additions.

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
