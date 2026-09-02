# Common Ground 0.6.0 evaluation plan

Status: unpublished candidate. Local verification is release-blocking. Hosted
inference, Hub publication, and training studies are separate later stages.

## 1. Freeze the candidate

Before generating model evidence:

1. Commit a clean source tree.
2. Run `bash scripts/local_smoke.sh`.
3. Record the passing source commit and evidence-directory hashes.
4. Build and retain the exact wheels/source archives.
5. Do not change corpora, prompts, scorers, dependencies, or versions after the
   freeze. Any such change creates a new candidate.

The local command must pass:

- byte-identical Elicit split regeneration;
- full tests;
- Ruff check and format check;
- mypy;
- dependency-manifest verification;
- fresh wheel/source builds, installs, legal files, and packaged taskset loads;
- compact Predict and Elicit model-free summaries.

## 2. Release-blocking invariants

### Predict

- exact masked-cell key coverage;
- valid normalized probabilities only;
- proper probability reward preferred for better calibration at equal argmax;
- uniform, empirical-prior, visible-frequency, and smoothed-neighbor references;
- full/matrix-only/text-only/shuffled-text prompt views preserve task IDs and
  answers.

### Elicit

- every contradiction resolves to the template-authored opposing passage;
- faction summaries do not change when current issue trade-offs reverse;
- alternative reversal swaps agree/disagree and preserves pass;
- exact-plus-false-positive shaped Find reward is below exact concise reward;
- opaque IDs remain exact;
- Ask copies only the public canonical profile; Find uses only authored
  per-slot aliases;
- question/diagnosis prose is checked only for yes/no form;
- exact instance, policy-semantic, and TF-IDF split checks pass;
- all packaged tasksets contain 100 train and 100 eval rows.

## 3. Model-free references

Predict should publish the probability-native references produced by
`scripts/compute_floors.py`, including Brier skill against explicit reference
losses.

Elicit should publish only the compact suite from
`scripts/compute_elicit_floors.py`:

- random and longest-sentence Find probes;
- exact Find ceiling;
- uniform-candidate and runner-up Ask references;
- public-profile Ask ceiling;
- top-one tie/margin and issue-class balance.

Historical exploit baselines may be cited from historical reports. They are not
parallel 0.6 release gates.

## 4. Exact-artifact model study

After source and artifacts freeze, evaluate at least four model families on:

- Predict full prompt;
- Predict matrix-only, text-only, and shuffled-text views using the same task
  IDs and answers;
- Elicit Find;
- Elicit Ask.

Use all 100 eval tasks, a fixed sampling configuration, and at least five
rollouts per task. Preserve raw traces and every provider failure. Retries are
allowed only for documented infrastructure failure and must never be selected
by score.

Report:

- primary reward and component metrics;
- task count, rollout count, failure count, and recovery count;
- model/config/artifact hashes;
- Predict task-bootstrap intervals;
- Elicit template-hierarchical intervals;
- paired differences only where task alignment is exact.

No result from 0.5 or an earlier candidate is 0.6 evidence.

## 5. Training utility

Publication as an environment does not require a successful RL result, but
claims of training value do. A training study should use multiple seeds and a
fresh independently implemented private generator for post-training transfer.

Required comparisons:

- untrained checkpoint;
- trained checkpoint selected without access to the private eval;
- prompt/matrix ablations for Predict;
- strict Find evaluation even if shaped Find reward is used for training;
- evidence that reward gains are not formatting-only or spam behavior.

## 6. Publication claims

Supported after local and exact-artifact evidence passes:

- reproducible synthetic environment;
- proper calibrated Predict reward;
- grounded structured Elicit contracts;
- measurable headroom on the fixed public corpus.

Not supported without additional evidence:

- inference of real human preferences;
- real organizational information gain;
- contamination-resistant model ranking;
- broad policy-domain generalization;
- beneficial RL training or private-generator transfer.
