# Common Ground 0.4.1 evaluation report

Status: exact public Hub artifacts, anonymous installation, the full model
study, and all 12 version-pinned public evaluation records are complete.

> **Historical evidence notice (2026-08-30):** Predict 0.4.1 remains a valid
> preliminary artifact. Elicit 0.4.1 is superseded for current benchmark use.
> Five `regional-archives-access` rows selected an unrelated second passage for
> the planted contradiction, and faction summaries used a finite phrase table
> keyed by issue type and target stance. The Elicit Find/Ask scores below remain
> immutable reproducibility records for that artifact; they are not evidence
> for the corrected 0.5 corpus. Elicit 0.5 must be regenerated and rerun before
> new comparative claims are made.

This report records model-free and model-evaluation evidence for the breaking
0.4 redesign. Version 0.4.1 differs from the private 0.4.0 candidate only in
Hub-source legal-file placement; task code, corpora, prompts, and scoring are
unchanged. Historical 0.3 scores are not comparable because both objectives
and the Elicit prompt contract changed.

## Release scope

| Environment | Eval rows | Train rows | Primary reward | Important diagnostics |
| --- | ---: | ---: | --- | --- |
| `commonground-predict` | 100 | 200 | `1 - normalized Brier` | vote accuracy, normalized Brier |
| `commonground-elicit:find` | 100 | 100 | strict `finding_f1` | localization recall, type accuracy, question utility |
| `commonground-elicit:elicit-ask` | 100 | 100 | normalized `question_utility` | top-K selection and faction-stance accuracy |

All bundled rows are synthetic. Answers are public for reproducibility and
open training, so these artifacts are not a contamination-resistant
leaderboard or evidence about real human preferences.

## Model-free comparators

Predict comparators use the committed 100-row evaluation file with eight
masked votes per snapshot:

| Comparator class | Comparator | vote accuracy |
| --- | --- | ---: |
| Prompt-observable | Always agree | 0.590 |
| Prompt-observable | Per-statement visible majority | 0.581 |
| Prompt-observable | Nearest participant (1-NN) | 0.819 |
| Prompt-observable | Five-neighbor vote | 0.891 |
| Held-out-label diagnostic | Per-snapshot best constant | 0.631 |
| Generator diagnostic | Latent cluster-pattern replay | 0.916 |

Only the first four comparators are prompt-observable. The nearest-neighbor
results establish that matrix structure remains a strong signal; model claims
must report them. The one-hot prompt-observable comparators have probability
reward equal to their accuracy under normalized multiclass Brier scoring.

Elicit comparators use only visible prompt information unless marked as a
component oracle:

| Comparator class | Task | Comparator | Mean reward |
| --- | --- | --- | ---: |
| Prompt-observable | Find | Random visible spans | 0.027 |
| Prompt-observable | Find | Flag vague-sounding spans | 0.140 |
| Prompt-observable | Find | Legacy 0.2 document-ID/position codebook | 0.000 |
| Prompt-observable | Ask | Template clarity questions | 0.000 |
| Prompt-observable | Ask | Randomly targeted questions | 0.000 |
| Prompt-observable | Ask | Removed 0.3 summary/stance codebook | 0.000 |
| Component oracle | Ask | Exact top-K issues plus random stances | 0.659 |

The removed-codebook parser scores 0.787 on historical 0.3 and 0.000 on 0.4,
confirming that the public exact-term and stance shortcut was removed. The
component oracle uses hidden issue targets and is an attainability diagnostic,
not a deployable baseline.

## Exact-artifact model study

Four model families, including an open-weight model, each ran all 100 tasks
with five rollouts in all three task modes: 6,000 rollouts total. Every
included rollout completed successfully; none contains recovered retry or
error history.

| Task | Model | Run ID | Reward mean ± rollout SD | 95% clustered bootstrap CI | Diagnostic mean | Zero-reward tasks |
| --- | --- | --- | ---: | ---: | --- | ---: |
| Predict | Claude Sonnet 4.5 | `d29f002d-b57b-428d-8a9b-ed9bced7daec` | 0.809 ± 0.088 | [0.793, 0.825] | accuracy 0.731; Brier 0.191 | 0 |
| Predict | Gemini 2.5 Flash | `734718f9-7cc4-4ce3-b3bc-5fea2d85302a` | 0.771 ± 0.095 | [0.753, 0.789] | accuracy 0.694; Brier 0.229 | 0 |
| Predict | GPT-4.1 | `dc165272-d424-41ce-9379-4350c44cc96e` | 0.733 ± 0.123 | [0.709, 0.755] | accuracy 0.655; Brier 0.267 | 0 |
| Predict | Qwen3 30B A3B Instruct | `618628c9-dfd9-4b1a-95f2-e65e5e2ead1b` | 0.706 ± 0.114 | [0.684, 0.726] | accuracy 0.567; Brier 0.294 | 1 |
| Find | GPT-4.1 | `703f5a57-88ad-436a-be29-db50b26449c4` | 0.160 ± 0.228 | [0.103, 0.224] | localization 0.667; type 0.515 | 45 |
| Find | Gemini 2.5 Flash | `0575acd7-9826-417b-9d43-bd71c0548746` | 0.073 ± 0.134 | [0.045, 0.105] | localization 0.667; type 0.552 | 53 |
| Find | Qwen3 30B A3B Instruct | `c91c9698-8c55-4181-9b7f-abd893132221` | 0.071 ± 0.154 | [0.026, 0.123] | localization 0.479; type 0.531 | 70 |
| Find | Claude Sonnet 4.5 | `6611d15f-a6ac-4806-a315-717ed1d18495` | 0.050 ± 0.128 | [0.024, 0.080] | localization 0.295; type 0.227 | 74 |
| Ask | Claude Sonnet 4.5 | `dd7353ac-d4d5-4cf3-bff8-eadc6c9e7615` | 0.105 ± 0.147 | [0.072, 0.141] | — | 44 |
| Ask | Gemini 2.5 Flash | `3bf07d6d-ec51-4711-9035-9112cbd444d1` | 0.058 ± 0.114 | [0.033, 0.088] | — | 55 |
| Ask | Qwen3 30B A3B Instruct | `7cf229f2-a992-4ca9-814b-67a395f8663f` | 0.032 ± 0.081 | [0.015, 0.052] | — | 69 |
| Ask | GPT-4.1 | `d5456177-12d4-4941-b879-bf97d3ffe4f6` | 0.016 ± 0.062 | [0.006, 0.029] | — | 80 |

Intervals use 50,000 deterministic percentile bootstrap resamples with seed
`20260829`. A task mean first averages its five repeated rollouts. Predict
resamples 100 tasks. Elicit first resamples 20 base templates, then five
variants within each selected template. Paired comparisons reuse the same
hierarchical draw; individual rollouts are not treated as 500 independent
tasks.

Selected paired differences show the principal ranking structure:

| Task | Comparison | Mean difference | 95% paired clustered bootstrap CI |
| --- | --- | ---: | ---: |
| Predict | Sonnet − Gemini | +0.038 | [+0.025, +0.051] |
| Predict | Gemini − GPT-4.1 | +0.038 | [+0.020, +0.056] |
| Predict | GPT-4.1 − Qwen | +0.027 | [+0.002, +0.053] |
| Find | GPT-4.1 − Gemini | +0.087 | [+0.042, +0.136] |
| Find | GPT-4.1 − Qwen | +0.090 | [+0.029, +0.156] |
| Find | GPT-4.1 − Sonnet | +0.111 | [+0.060, +0.167] |
| Ask | Sonnet − Gemini | +0.048 | [+0.004, +0.090] |
| Ask | Gemini − Qwen | +0.026 | [+0.001, +0.055] |
| Ask | Qwen − GPT-4.1 | +0.016 | [-0.004, +0.038] |

The complete 18-comparison table is generated by
`scripts/analyze_release_study.py` and retained with the study evidence.

## Public evaluation records

The retained exact-artifact traces were published as version-pinned Prime
Intellect evaluation records after both 0.4.1 listings passed anonymous wheel
installation and native/legacy task-loading verification. Each record contains
exactly 500 samples (100 tasks with five rollouts) and zero recovered rollouts.

| Task | Model | Public evaluation |
| --- | --- | --- |
| Predict | Claude Sonnet 4.5 | [`uo7mmdlbpsi37fjkvv70031m`](https://app.primeintellect.ai/dashboard/evaluations/uo7mmdlbpsi37fjkvv70031m) |
| Predict | Gemini 2.5 Flash | [`tw0l3zdbf4s8racrp43ea23q`](https://app.primeintellect.ai/dashboard/evaluations/tw0l3zdbf4s8racrp43ea23q) |
| Predict | GPT-4.1 | [`gwnapwuakf3l4mm54objg6lj`](https://app.primeintellect.ai/dashboard/evaluations/gwnapwuakf3l4mm54objg6lj) |
| Predict | Qwen3 30B A3B Instruct | [`g5dn7f6hjoyryq5gscua8zgh`](https://app.primeintellect.ai/dashboard/evaluations/g5dn7f6hjoyryq5gscua8zgh) |
| Find | Claude Sonnet 4.5 | [`icfkp38uvwvd2q60gwcktvbs`](https://app.primeintellect.ai/dashboard/evaluations/icfkp38uvwvd2q60gwcktvbs) |
| Find | Gemini 2.5 Flash | [`vwmu8ffa5s1zzaiymfr88lat`](https://app.primeintellect.ai/dashboard/evaluations/vwmu8ffa5s1zzaiymfr88lat) |
| Find | GPT-4.1 | [`jr6fkw2w90jpyc4cb8nlr6bn`](https://app.primeintellect.ai/dashboard/evaluations/jr6fkw2w90jpyc4cb8nlr6bn) |
| Find | Qwen3 30B A3B Instruct | [`wz6gjqouyekg3d8hxe22rgh0`](https://app.primeintellect.ai/dashboard/evaluations/wz6gjqouyekg3d8hxe22rgh0) |
| Ask | Claude Sonnet 4.5 | [`gc56sz85hv9o5hlt68e83osw`](https://app.primeintellect.ai/dashboard/evaluations/gc56sz85hv9o5hlt68e83osw) |
| Ask | Gemini 2.5 Flash | [`s2gxbmusyj7ib4h3lw8zqqu7`](https://app.primeintellect.ai/dashboard/evaluations/s2gxbmusyj7ib4h3lw8zqqu7) |
| Ask | GPT-4.1 | [`twq2761bsj7sd9rgwcr9df01`](https://app.primeintellect.ai/dashboard/evaluations/twq2761bsj7sd9rgwcr9df01) |
| Ask | Qwen3 30B A3B Instruct | [`ylcylebw5efrxil9ywjne0xa`](https://app.primeintellect.ai/dashboard/evaluations/ylcylebw5efrxil9ywjne0xa) |

## Interpretation

Predict is the stronger release contribution. All four models produce useful
calibrated signal without saturating the task, and all six pairwise model
intervals exclude zero. No model exceeds the prompt-observable 1-NN or 5-NN
comparator. Qwen's mean argmax accuracy also falls below always-agree, while
its probability reward remains higher because calibrated uncertainty receives
partial credit. This supports describing the task as experimental
semantic-conditioned matrix completion, not general collective-preference
reasoning.

Elicit is a difficult, sparse benchmark with substantial headroom. On Find,
only GPT-4.1's mean exceeds the 0.140 vague-span heuristic; the other model
means do not. On Ask, every model mean exceeds the zero-scoring prompt
baselines, but 44–80 of 100 tasks have zero mean reward and the 0.659 component
oracle remains far above every model. The results support evaluation of strict
grounded diagnosis and clarification selection, but do not establish that the
strict rewards are already efficient reinforcement-learning signals. Find's
optional shaped curriculum is therefore a training aid, while strict F1
remains the release-evaluation objective.

## Integrity and operational evidence

- Predict train/eval text, session IDs, seeds, and profile-generator families
  are disjoint; statement dimension changes alter generated votes.
- Elicit train/eval instance identity, canonical visible meaning, policy-issue
  meaning, template/layout labels, and structural signatures are audited.
  Maximum cross-split token Jaccard is 0.207 and maximum word-ngram TF-IDF
  cosine is 0.755, below the committed 0.85 and 0.90 ceilings.
- Exact Elicit answers attain 1.0. Malformed prediction maps, fabricated or
  reordered evidence, incomplete contradiction pairs, duplicate spans, and
  generic questions are regression-tested.
- Both exact public Hub versions passed their integration actions, anonymous
  wheel installation, and native/legacy task loading. The pulled source
  archives retained the nested
  Apache-2.0 text, Predict MPL-2.0 text, and byte-identical provenance notice.

An initial study attempt exhausted the personal inference balance after one
complete run and one partial run. Every affected request returned HTTP 402.
The failed trace tree was moved outside the analysis root and retained as
operational evidence. The one complete, error-free GPT-4.1 Predict run was
kept; the other eleven runs were restarted from scratch after funding. They
were not resumed, and no run was selected or rejected based on its score.

Post-study upload validation then found one Qwen Find episode with a transient
HTTP 502 call followed by a successful SDK retry. Terminal episode fields had
not exposed that call-level history, so the aggregate recovery counter and its
regression coverage were strengthened. The entire 500-rollout Qwen Find run
was quarantined and rerun from the same saved configuration; no individual
episode was resumed or substituted. The final analysis root contains exactly
12 complete runs, and all 6,000 included traces contain exactly one successful
model call with no error history.

## Study protocol and claim boundary

The native Verifiers v1 runs used owner-qualified, version-pinned private Hub
tasksets. Saved configs record `temperature = 0.2`, `max_tokens = 2048`,
`shuffle = false`, ten-way concurrency, the subprocess pure-chat runtime, no
judge model, and zero configured retries. Models were served through Prime
Inference's chat-completions endpoint. Exact configs and per-example traces
are retained outside the repository.

These results support “experimental semantic-conditioned matrix completion”
and “experimental structured policy-issue localization and clarification
selection.” They do not establish human-preference validity, organizational
information gain, contamination-resistant leaderboard performance, or
training value. Training claims still require multiple seeds and transfer to
a fresh private generator family; Predict also needs text/matrix ablations.

## Release attestation

| Field | Value |
| --- | --- |
| Artifact source commit | `8d3c4a4fbf55ce46cbabc9774b09f3283dca6e43` |
| `commonground-score` 0.3.0 wheel/sdist SHA-256 | `8d8b2add5b46b1b46387d875f1ef09704ae90bf90f137dcb9091214d0f3fe428` / `1d78722f02f427af1b631e9b7a844c897cf58c035aefcb24becde81ef9e85b14` |
| `commonground-scenarios` 0.3.0 wheel/sdist SHA-256 | `f673b41ac064ef402374f4fa7a320ee3c40a497550186f7f339a080da1b0f432` / `b44c08701d260c61a9853306465cb7110d1f82100a4b75b9e242dbeb7daf32e6` |
| `commonground-predict` 0.4.1 wheel/Hub content SHA-256 | `8b091a0142b0c161c3e3a8dbc1a8411c8b4253f44923a7cb7a446927c07855c0` / `622ac6555dd1c7fdd2f7cbc2c2975a2b730b706c8289cc87704aef29b3b85aaf` |
| `commonground-elicit` 0.4.1 wheel/Hub content SHA-256 | `8eb5f7b5d6599fb9c0c920ed96a3061041549674146dca52a40c6971a9d59f24` / `07e59501345bf646668e630a6164c52c0e58be0e6af80c51c694f00f1b62fa92` |
| Hub version IDs | Predict `i0idhdqxqmg5oxuif8hxkzes`; Elicit `rbgfk8iccab6x4p8nbg22864` |
| Native evaluation run IDs | 12 run IDs in the exact-artifact model-study table |
| Public evaluation IDs | 12 version-pinned records listed above |
