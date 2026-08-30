# Common Ground 0.4-series candidate evaluation plan

Status: private-artifact study complete. All gates below passed against source
commit `8d3c4a4fbf55ce46cbabc9774b09f3283dca6e43` and the exact 0.4.1 Hub
artifacts. Public visibility and public evaluation records remain pending; see
the [0.4.1 evaluation report](evaluation-report-0.4.1.md).

The exact publication target is 0.4.1. It differs from the private 0.4.0
candidate only by legal-file placement required by Prime's source-archive
collector; task code, corpora, prompts, and scoring are unchanged.

## Changes that invalidate 0.3 scores

- Predict's primary reward is now calibration-sensitive `1 - normalized Brier`
  and its response key set is exact.
- Elicit's public exact-term/stance clauses are gone, the corpus was
  regenerated, Ask value is answer-conditioned rather than keyword-derived,
  and the split audit is stronger.
- Optional shaped Find reward exists for training, but strict Find F1 remains
  the only release-evaluation objective.

Historical 0.3 model results are not 0.4 baselines.

## Deterministic candidate gates

The candidate must reproduce these comparator facts:

| Environment | Comparator | Expected |
| --- | --- | ---: |
| Predict | one-hot 5-NN probability reward / vote accuracy | 0.891 |
| Elicit Find | vague-span heuristic | 0.140 |
| Elicit Ask | removed 0.3 summary/stance parser | 0.000 |
| Elicit Ask | exact-issue/random-stance component oracle | 0.659 |
| Elicit split audit | maximum token Jaccard | 0.207 (≤ 0.85) |
| Elicit split audit | maximum word-ngram TF-IDF cosine | 0.755 (≤ 0.90) |

The same Elicit parser scores 0.787 on the historical 0.3 corpus, establishing
that its 0.000 candidate score is a real ablation rather than an inert test.

## Private exact-artifact study

Run strict Predict, Find, and Ask on all 100 evaluation tasks with five
rollouts for at least four model families, including an open-weight trainable
model. Require zero recovered/error rollouts. Preserve exact configs, complete
traces, artifact hashes, Hub content hashes, and provider model identifiers.

Analyze the retained runs with:

```bash
uv run python scripts/analyze_release_study.py \
  --root /path/to/release-0.4.1-study \
  --elicit-eval-split environments/commonground_elicit/commonground_elicit/data/eval_synthetic_heldout.jsonl \
  --bootstrap-samples 50000 \
  --seed 20260828 \
  --expected-model-count 4 \
  --expected-task-count 100 \
  --expected-rollouts-per-task 5 \
  --require-no-recoveries
```

Elicit intervals and paired differences must use base-template then variant
resampling. Predict uses paired task resampling. Report the strong 1-NN/5-NN
comparators next to every Predict model result.

## Training-value experiment

No release claim should say that either environment is proven useful for RL
until at least three independent seeds compare:

1. strict Find F1 training;
2. `reward_mode="shaped"` Find curriculum training;
3. an untrained/control checkpoint.

Evaluate all checkpoints with strict reward on a private, independently
implemented generator—not the public answer-bearing rows. Predict training
should additionally compare text+matrix, matrix-only, text-only, and
shuffled-text inputs. Publish learning curves, seed dispersion, compute, and
failure rates whether or not the result is positive.

## Claim boundary

A clean 0.4 study would support “experimental semantic-conditioned matrix
completion” and “experimental structured policy-issue localization and
clarification selection.” It would not establish human-preference validity,
real organizational information gain, or contamination-resistant leaderboard
performance.
