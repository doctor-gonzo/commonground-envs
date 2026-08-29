# Common Ground 0.3.0 evaluation report

Status: public release complete. Exact Hub artifacts, anonymous installation,
the full model study, and all 12 version-pinned public evaluation records have
been verified.

This report records the model-free and model-evaluation evidence for the
breaking 0.3.0 redesign. It intentionally contains no 0.2.x model scores: both
corpora and both response contracts changed, so those runs are not comparable.

## Post-release audit note

An external audit found that 0.3 faction summaries repeated each planted
issue's decision terms and exact yes/no/pass stance. A new prompt-only parser
that uses only public documents and faction summaries scores **0.787** on Ask,
so the 0.3 Ask results must be interpreted partly as structured extraction,
not latent stance inference. The public 0.3 artifacts and scores remain
historical facts; the unreleased 0.4 candidate removes these clauses and the
same parser scores 0.000. The audit also prompted the hierarchical
template/variant intervals recorded below.

## Release scope

| Environment | Eval rows | Train rows | Primary reward | Important diagnostics |
| --- | ---: | ---: | --- | --- |
| `commonground-predict` | 100 | 200 | `vote_accuracy` | normalized `brier` |
| `commonground-elicit:find` | 100 | 100 | `finding_f1` | localization recall, type accuracy, question utility |
| `commonground-elicit:elicit-ask` | 100 | 100 | normalized `question_utility` | exact top-K attainability |

## Predict comparators

| Comparator class | Comparator | vote_accuracy |
| --- | --- | ---: |
| Prompt-observable | Always agree | 0.590 |
| Prompt-observable | Per-statement visible majority | 0.581 |
| Prompt-observable | Nearest participant (1-NN) | 0.819 |
| Prompt-observable | Five-neighbor vote | 0.891 |
| Held-out-label diagnostic | Per-snapshot best constant | 0.631 |
| Generator diagnostic | Latent cluster-pattern replay | 0.916 |

These exact values come from `scripts/compute_floors.py` on the committed
100-row evaluation file with eight masked cells per snapshot. Only the first
four rows are prompt-observable. The 5-NN result shows that matrix structure
remains the dominant comparator; any model claim must report it.

## Elicit comparators

| Comparator class | Task | Comparator | mean reward |
| --- | --- | --- | ---: |
| Prompt-observable | find | Random visible spans | 0.030 |
| Prompt-observable | find | Flag vague-sounding spans | 0.140 |
| Prompt-observable | find | Legacy 0.2 document-ID/position codebook | 0.000 |
| Prompt-observable | elicit-ask | Template clarity questions | 0.000 |
| Prompt-observable | elicit-ask | Randomly targeted questions | 0.000 |
| Prompt-observable | elicit-ask | 0.3 summary/stance codebook parser | 0.787 |
| Component oracle | elicit-ask | Exact top-K issues + random stances | 0.655 |
| Component oracle | elicit-ask | Exact top-K issues + visible-summary stances | 1.000 |

These values come from `scripts/compute_elicit_floors.py`. The legacy-codebook
regression directly tests the critical 0.2 shortcut. The other prompt baselines
are weak floors, not a complete difficulty ladder. The 0.787 parser establishes
that the public summary clauses were a material shortcut. The component oracles
use hidden top-K issue targets; their gap diagnoses the same stance disclosure.

## Exact-artifact model study

Four model families, including one open-weight model, each ran all 100 tasks
with five rollouts for all three task modes: 6,000 rollouts total. Every rollout
completed successfully and none used recovered retry/error history.

| Task | Model | Reward mean ± rollout SD | 95% clustered bootstrap CI | Diagnostic mean | Zero-reward tasks |
| --- | --- | ---: | ---: | --- | ---: |
| Predict | Claude Sonnet 4.5 | 0.730 ± 0.173 | [0.698, 0.761] | Brier 0.191 | 0 |
| Predict | Gemini 2.5 Flash | 0.694 ± 0.202 | [0.656, 0.730] | Brier 0.226 | 0 |
| Predict | GPT-4.1 | 0.653 ± 0.169 | [0.621, 0.683] | Brier 0.270 | 0 |
| Predict | Qwen3 30B A3B Instruct | 0.572 ± 0.201 | [0.537, 0.607] | Brier 0.286 | 0 |
| Find | Claude Sonnet 4.5 | 0.235 ± 0.307 | [0.158, 0.317] | localization 0.420; type 0.382 | 34 |
| Find | GPT-4.1 | 0.228 ± 0.249 | [0.168, 0.294] | localization 0.603; type 0.543 | 32 |
| Find | Gemini 2.5 Flash | 0.172 ± 0.215 | [0.122, 0.225] | localization 0.711; type 0.545 | 28 |
| Find | Qwen3 30B A3B Instruct | 0.120 ± 0.201 | [0.066, 0.180] | localization 0.483; type 0.675 | 58 |
| Ask | Claude Sonnet 4.5 | 0.296 ± 0.229 | [0.232, 0.361] | — | 15 |
| Ask | Gemini 2.5 Flash | 0.155 ± 0.213 | [0.110, 0.206] | — | 34 |
| Ask | Qwen3 30B A3B Instruct | 0.052 ± 0.112 | [0.026, 0.085] | — | 61 |
| Ask | GPT-4.1 | 0.052 ± 0.120 | [0.024, 0.088] | — | 62 |

Intervals use 20,000 deterministic percentile bootstrap resamples (seed
`20260828`). A task mean first averages its five repeated rollouts. Predict
resamples 100 tasks. Elicit first resamples 20 base templates, then resamples
five variants within each selected template. Paired model comparisons reuse
the same hierarchical draw; individual rollouts are never treated as 500
independent tasks.

Selected paired differences show the main ranking structure:

| Task | Comparison | Mean difference | 95% paired clustered bootstrap CI |
| --- | --- | ---: | ---: |
| Predict | Sonnet − Gemini | +0.037 | [+0.007, +0.067] |
| Predict | Gemini − GPT-4.1 | +0.041 | [+0.006, +0.074] |
| Predict | GPT-4.1 − Qwen | +0.081 | [+0.046, +0.114] |
| Find | Sonnet − GPT-4.1 | +0.007 | [-0.094, +0.111] |
| Find | GPT-4.1 − Gemini | +0.056 | [+0.001, +0.114] |
| Find | Gemini − Qwen | +0.053 | [-0.005, +0.112] |
| Ask | Sonnet − Gemini | +0.141 | [+0.067, +0.216] |
| Ask | Gemini − Qwen | +0.102 | [+0.047, +0.163] |
| Ask | Qwen − GPT-4.1 | +0.000 | [-0.045, +0.041] |

The complete 18-comparison table is generated by
`scripts/analyze_release_study.py`. Predict separates all four models but none
beats the prompt-observable 1-NN (0.819) or 5-NN (0.891) comparators; Qwen's
0.572 also trails the always-agree mean of 0.590. Find is difficult and
non-saturated: Sonnet and GPT-4.1 are statistically unresolved, while the
Qwen mean falls below the 0.140 vague-span heuristic. Ask is compromised by the
0.787 public-summary parser; the model ranking therefore cannot be presented as
evidence of latent faction inference.

These are reference-model results on published synthetic answers, not a
contamination-resistant leaderboard and not evidence of human-preference
validity.

## Public evaluation records

The retained exact-artifact traces were published as version-pinned Prime
Intellect evaluation records after both 0.3.0 listings passed anonymous install
and native/legacy task-loading verification.

| Task | Model | Public evaluation |
| --- | --- | --- |
| Predict | Claude Sonnet 4.5 | [`p6ax4yasiy4chhryja5t4zud`](https://app.primeintellect.ai/dashboard/evaluations/p6ax4yasiy4chhryja5t4zud) |
| Predict | Gemini 2.5 Flash | [`gzz89cwp1oxkj2fq9nipvdew`](https://app.primeintellect.ai/dashboard/evaluations/gzz89cwp1oxkj2fq9nipvdew) |
| Predict | GPT-4.1 | [`id31ny27zqgp4v9md1sfzj8n`](https://app.primeintellect.ai/dashboard/evaluations/id31ny27zqgp4v9md1sfzj8n) |
| Predict | Qwen3 30B A3B Instruct | [`glcbl4dzvxdtjnn0k4tayr35`](https://app.primeintellect.ai/dashboard/evaluations/glcbl4dzvxdtjnn0k4tayr35) |
| Find | Claude Sonnet 4.5 | [`v5yyailapdapy2p8yco5sjvi`](https://app.primeintellect.ai/dashboard/evaluations/v5yyailapdapy2p8yco5sjvi) |
| Find | Gemini 2.5 Flash | [`mlgiauw8hwuoyhokjirg68m7`](https://app.primeintellect.ai/dashboard/evaluations/mlgiauw8hwuoyhokjirg68m7) |
| Find | GPT-4.1 | [`qvtqj5qjlajtxdy2cvqjmric`](https://app.primeintellect.ai/dashboard/evaluations/qvtqj5qjlajtxdy2cvqjmric) |
| Find | Qwen3 30B A3B Instruct | [`y4g3gbcb32zjr2k444ica4kv`](https://app.primeintellect.ai/dashboard/evaluations/y4g3gbcb32zjr2k444ica4kv) |
| Ask | Claude Sonnet 4.5 | [`hy1tvdwhzi7awwsiwlg1dgs6`](https://app.primeintellect.ai/dashboard/evaluations/hy1tvdwhzi7awwsiwlg1dgs6) |
| Ask | Gemini 2.5 Flash | [`vfib3eaor5rv2gnb3kdt1kyf`](https://app.primeintellect.ai/dashboard/evaluations/vfib3eaor5rv2gnb3kdt1kyf) |
| Ask | GPT-4.1 | [`ny6hqx6ayq8iyb743y1yof8f`](https://app.primeintellect.ai/dashboard/evaluations/ny6hqx6ayq8iyb743y1yof8f) |
| Ask | Qwen3 30B A3B Instruct | [`rqxwgvluk42culx09085orqa`](https://app.primeintellect.ai/dashboard/evaluations/rqxwgvluk42culx09085orqa) |

## Integrity evidence

- Predict train/eval text, session IDs, seeds, and generator families are
  disjoint; statement dimension changes alter generated votes.
- Elicit has 100 distinct prompt hashes and answer hashes per split, no exact
  cross-split prompt/answer overlap, disjoint template/layout labels, opaque
  IDs, and varied structural signatures. These 0.3 checks were identity/layout
  sensitive and did not detect the public summary codebook.
- Exact Elicit answers attain 1.0. Type hedging, missing contradiction pairs,
  generic gap labels, broad evidence, duplicate spans, and one-noun questions
  are regression-tested.
- All bundled data is synthetic and public-answer.

## Study protocol and remaining work

The runs used native Verifiers v1 against owner-qualified, version-pinned Hub
tasksets while the exact artifacts were still private. Each saved config records `temperature = 0.2`,
`max_tokens = 2048`, `shuffle = false`, ten-way evaluation concurrency, the
subprocess pure-chat runtime, no judge models, and zero configured retries.
Models were served through the Prime Inference chat-completions endpoint. Full
configs and per-example traces are retained with the release evidence.

The separate Prime Hosted Evaluation canary failed before environment pull
because its managed image retained legacy `connect-python==0.9.0` alongside
the required `connectrpc` distribution. The failure is an upstream runner-image
conflict, not an environment import or scoring failure; native-v1 runs against
the exact Hub versions completed cleanly. The 12 public records above were
subsequently created from the retained traces and were not used to recompute or
select the reported scores.

Useful next comparators are matrix factorization/spectral clustering plus
text-only, matrix-only, and shuffled-text ablations for Predict, and component
oracles plus stronger lexicon/stance-prior baselines for Elicit. Training-value
claims additionally require at least three training seeds and transfer to a
fresh private generator family.

## Release attestation

| Field | Value |
| --- | --- |
| Artifact source commit | `54d19f861edb28336e14038a2f9fff10635ba03f` |
| `commonground-score` 0.2.0 wheel/sdist SHA-256 | `08a85c41f3557927b92d53a78e99b48a071a2e7696d767763211e24028c52ee2` / `6b3631cfffb0d5d589b47b9e63cf3ce2c572f59a267055c0fc988a3a44a75969` |
| `commonground-scenarios` 0.2.0 wheel/sdist SHA-256 | `3c74c95d0c78e5b87d0e3eb57246cac6d461813bb17932493c981f7bf89a788a` / `9c890b889376f664c5d89f5a94b97375130bba1805db06b97c7c22999982e4f1` |
| `commonground-predict` 0.3.0 wheel/Hub content SHA-256 | `3fb5404dc94e57e9ad1d8f2ac349c0036c1df78c4270eee43f20627dc7ea7724` / `ed75b162cec5030c4e4b55d029b0a347b71aeb33a5e8f566a6b12e973da5dc0a` |
| `commonground-elicit` 0.3.0 wheel/Hub content SHA-256 | `6e6594f22539628b5352aa9f420b5f6b9962329ff8ec735723aa586dd8a0bc0e` / `f7d7f70d784245323492e587a96984d94a4456fc45c46f66ca5ddb1c7173c03f` |
| Hub version IDs (validated privately, now public) | Predict `tpu9zthui9zuz6ug94a9pcm9`; Elicit `su5kq9ptdd6nmyv2mveragbm` |
| Native evaluation run IDs and configs | 12 run IDs in the model-study table; exact `config.toml` plus `traces.jsonl` retained |
| Public evaluation IDs | 12 version-pinned records listed above |
