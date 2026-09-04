# Common Ground 0.6.0 evaluation plan

Status: published 0.6.0. Local verification passed, the complete exact-artifact
model study is reported in `evaluation-report-0.6.0.md`, and the Hub
publication is complete. Training studies remain a separate later stage.

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
- Ask copies only the public canonical profile; Find has no decision frame or
  alias table;
- question/diagnosis prose is checked only for yes/no form;
- complete authored Find answers reach reward 1.0;
- contradiction evidence pairs score identically in either direction;
- authored policy sentences remain intact and neutral procedural predicates do
  not repeat within a scenario;
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
- exact Ask answer ceiling;
- top-one tie/margin and issue-class balance.

Historical exploit baselines may be cited from historical reports. They are not
parallel 0.6 release gates.

## 4. Small usefulness pilot

Before the complete study, run a small exact-artifact usefulness pilot. This is
recommended but not release-blocking: it is a cheap check that the hosted
runner works, rewards have usable variation, and failures can be interpreted.

Use two contrasting model families on the same 20 evaluation tasks with two
rollouts each for Predict full, Elicit Find, and Elicit Ask. For one model,
also run Predict matrix-only, text-only, and shuffled-text on those same tasks.
Do not publish these pilot scores as the final 0.6 model comparison.

Use JSON-object decoding when the provider exposes it, and record the resolved
sampling configuration with every run. The task prompts independently state
the raw-JSON envelope and Elicit's auxiliary-first yes/no surface rule; JSON
decoding only prevents Markdown/prose wrappers and does not supply answers.
The baseline aggregator accepts both native trace directories and the nested
`evals/<environment>/<run>` saved-result layout produced when `vf-eval` is
given a wrapper output directory. It also canonicalizes `vf-eval`'s legacy
`question_format_validity` rubric name to the native/reporting name
`question_format_valid`; this is a reporting-only alias for the same recorded
value. For deterministic subset pilots, the hierarchical analyzer accepts the
full evaluation split's template map, requires a label for every evaluated
task, and ignores labels for unevaluated rows. Predict views saved by
`vf-eval` are paired only when their recorded endpoint, sampling, package/source
versions, non-treatment task arguments, task roster, and answer digests match;
`prompt_mode` is the sole excluded treatment field. Saved-result completion
order is intentionally excluded because concurrent `vf-eval` writes can finish
in a different order; statistical pairing is by task ID.

Proceed to the complete study only if:

- every selected trace completes without parser, provider, or harness errors;
- rewards are finite and not uniformly zero or one;
- Predict beats uniform probability reward (`0.667`) and the prompt-view runs
  are aligned to the same task IDs and answers;
- Elicit component metrics distinguish format, grounding, type/relation, and
  stance failures rather than collapsing every failure into one opaque zero;
- a manual review of at least ten non-perfect Find/Ask traces finds genuine
  reasoning or reference-contract errors, not only malformed output;
- no prompt exposes hidden answer fields and no obvious fixed shortcut solves
  the complete response.

If a condition fails, stop and diagnose the smallest failing slice. A pilot is
evidence that the task has measurable model-facing signal; it is not evidence
of real-world usefulness, successful RL training, or transfer.

## 5. Exact-artifact model study

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

## 6. Training utility

Publication as an environment does not require a successful RL result, but
claims of training value do. A training study should use multiple seeds and a
fresh independently implemented private generator for post-training transfer.

Required comparisons:

- untrained checkpoint;
- trained checkpoint selected without access to the private eval;
- prompt/matrix ablations for Predict;
- strict Find evaluation even if shaped Find reward is used for training;
- evidence that reward gains are not formatting-only or spam behavior.

## 7. Publication claims

Supported after the local release evidence passes:

- reproducible synthetic environment;
- proper calibrated Predict reward;
- grounded structured Elicit contracts.

Claims about model performance or measurable headroom on the fixed public
corpus additionally require the exact-artifact model study.

Not supported without additional evidence:

- inference of real human preferences;
- real organizational information gain;
- contamination-resistant model ranking;
- broad policy-domain generalization;
- beneficial RL training or private-generator transfer.

Interpret fixed-corpus Elicit results with the disclosed Ask shortcut in mind:
the same public trade-off vector identifies the Ask winner in 90 of 100
evaluation rows. It does not supply the complete scored response.
