# Common Ground 0.6.0 evaluation report

Status: exact public Hub artifacts, shared PyPI packages, the complete
exact-artifact model study (15 strata, 7,500 rollouts), and
all 12 version-pinned public Prime evaluation records are
complete. Generated 2026-09-03 from the retained study evidence; no number in
this report was transcribed by hand.

Version 0.6.0 is a new benchmark candidate rather than a re-scoring of 0.5.0.
It freezes exact masked-cell Predict scoring with deterministic full,
matrix-only, text-only, and shuffled-text prompt views; replaces Elicit's
inferred contradiction links with authored opposing passages; makes Find
evidence-first with precision-sensitive shaped reward; and represents Ask
decisions as structured, polarity-explicit slots. All four distributions use
the same 0.6.0 release number. No result from 0.5.0 or earlier is 0.6.0
evidence.

## Release scope

| Environment | Eval rows | Train rows | Primary reward | Important diagnostics |
| --- | ---: | ---: | --- | --- |
| `commonground-predict` | 100 | 200 | `1 - normalized Brier` (probability reward) | vote accuracy, normalized Brier, named-reference Brier skill |
| `commonground-elicit:find` | 100 | 100 | strict `finding_f1` | localization, type, diagnosis, relationship recall |
| `commonground-elicit:elicit-ask` | 100 | 100 | normalized `question_utility` | format validity, grounding recall, grounded stance recall, top-1 selection |

All bundled rows are synthetic. Answers are public for reproducibility and
open training, so this release is not a contamination-resistant leaderboard
and does not measure preferences of real people. One fixed-corpus shortcut is
disclosed: one public trade-off vector identifies the Ask winner in 90 of 100
evaluation rows; it does not supply the complete scored response.

## Model-free references

Predict references use the committed 100-row evaluation file with eight masked
votes per snapshot (`scripts/compute_floors.py`, seed 20260831):

| Comparator class | Comparator | probability reward | vote_accuracy | normalized Brier | Brier skill vs uniform | Brier skill vs empirical prior |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| No-input | Uniform probability | 0.667 | 0.590 | 0.333 | 0.000 | -0.008 |
| Train-split no-text | Global empirical class prior | 0.669 | 0.166 | 0.331 | 0.008 | 0.000 |
| Prompt-observable matrix-only | Per-statement visible class frequencies | 0.760 | 0.581 | 0.240 | 0.279 | 0.274 |
| Prompt-observable matrix-only | Smoothed distance-weighted 5-neighbor frequencies | 0.881 | 0.900 | 0.119 | 0.643 | 0.640 |

Elicit references use the compact suite from `scripts/compute_elicit_floors.py`:

| Task | Comparator or diagnostic | mean |
| --- | --- | ---: |
| Find | Random visible sentences | 0.043 |
| Find | Longest visible sentences | 0.000 |
| Find | Exact authored answer (ceiling) | 1.000 |
| Ask | Uniform candidate + exact components | 0.557 |
| Ask | Exact runner-up candidate | 0.398 |
| Ask | Exact Ask answer ceiling | 1.000 |
| Audit | Top-1 tie rate | 0.000 |
| Audit | Minimum top-1 margin | 0.144 |
| Audit | Minimum issue-class share | 0.333 |
| Audit | Maximum issue-class share | 0.333 |

## Exact-artifact model study

Four model families each ran all 100 evaluation tasks with five rollouts in all
three task modes (12 primary runs, 6,000 rollouts). One model
additionally ran the three Predict prompt views on the same task IDs and
answers (3 ablation runs, 1,500 rollouts).

| Task | Model | Run ID | Reward mean ± rollout SD | 95% clustered bootstrap CI | Key diagnostics | Zero-reward tasks | Recovered |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: |
| Predict | Kimi K2 Instruct | `b6f9e043` | 0.761 ± 0.152 | [0.741, 0.780] | accuracy 0.686; Brier 0.239 | 0 | 0 |
| Predict | Gemini 2.5 Flash | `2cbcfe56` | 0.739 ± 0.093 | [0.724, 0.755] | accuracy 0.642; Brier 0.261 | 0 | 0 |
| Predict | GPT-4.1 | `745da7ed` | 0.724 ± 0.117 | [0.703, 0.746] | accuracy 0.642; Brier 0.276 | 0 | 0 |
| Predict | Qwen3 30B A3B Instruct | `0701ff9d` | 0.706 ± 0.110 | [0.684, 0.725] | accuracy 0.565; Brier 0.294 | 1 | 0 |
| Find | GPT-4.1 | `21f39e08` | 0.515 ± 0.272 | [0.440, 0.590] | localization 0.749; type 0.741; diagnosis 0.542; relation 0.510 | 3 | 0 |
| Find | Kimi K2 Instruct | `661a1089` | 0.244 ± 0.189 | [0.202, 0.285] | localization 0.738; type 0.510; diagnosis 0.468; relation 0.453 | 7 | 0 |
| Find | Gemini 2.5 Flash | `a7c3e16e` | 0.230 ± 0.231 | [0.166, 0.297] | localization 0.582; type 0.424; diagnosis 0.334; relation 0.311 | 18 | 1 |
| Find | Qwen3 30B A3B Instruct | `62dcefed` | 0.219 ± 0.255 | [0.159, 0.283] | localization 0.445; type 0.534; diagnosis 0.311; relation 0.281 | 23 | 0 |
| Ask | GPT-4.1 | `60e755db` | 0.294 ± 0.329 | [0.199, 0.401] | format 0.996; grounding 0.724; stance 0.326; top-1 selection 0.214 | 21 | 0 |
| Ask | Kimi K2 Instruct | `2adc57d7` | 0.285 ± 0.356 | [0.191, 0.394] | format 0.964; grounding 0.600; stance 0.309; top-1 selection 0.230 | 24 | 0 |
| Ask | Gemini 2.5 Flash | `61fee42d` | 0.168 ± 0.303 | [0.100, 0.247] | format 0.846; grounding 0.368; stance 0.198; top-1 selection 0.122 | 51 | 7 |
| Ask | Qwen3 30B A3B Instruct | `e3f98ab2` | 0.158 ± 0.295 | [0.081, 0.259] | format 1.000; grounding 0.404; stance 0.156; top-1 selection 0.114 | 52 | 0 |

Intervals use 50,000 deterministic percentile bootstrap resamples with seed
`20260903`. Each task mean first averages five repeated rollouts. Predict
resamples 100 tasks; Elicit resamples 20 base templates and then five variants
within each selected template. Paired comparisons reuse the same draws.

### Predict prompt-view ablations

Ablations pair each prompt view with the full prompt for the same model on the
same task roster and answer digests; the analyzer refuses to pair runs whose
saved endpoint, sampling, artifact version, or answer digests differ.

| Model | Prompt view | Full-prompt mean | View mean | Paired drop (full − view) | 95% paired CI | Tasks |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GPT-4.1 | matrix-only | 0.724 | 0.705 | +0.020 | [0.003, 0.037] | 100 |
| GPT-4.1 | shuffled-text | 0.724 | 0.737 | -0.013 | [-0.024, -0.001] | 100 |
| GPT-4.1 | text-only | 0.724 | 0.694 | +0.030 | [0.003, 0.058] | 100 |

These paired intervals are exploratory and not multiplicity-adjusted.

### Paired model comparisons

The 18 primary pairwise comparisons form one Holm family. Paired
p-values use deterministic cluster-level sign flips and are exploratory rather
than confirmatory.

| Task | Comparison | Mean difference | 95% paired CI | Holm-adjusted p | Holm verdict |
| --- | --- | ---: | ---: | ---: | --- |
| Predict | Kimi K2 Instruct − Qwen3 30B A3B Instruct | +0.055 | [0.033, 0.080] | 0.0004 | moonshotai/kimi-k2-0905 higher after Holm adjustment |
| Predict | Kimi K2 Instruct − GPT-4.1 | +0.036 | [0.016, 0.056] | 0.0120 | moonshotai/kimi-k2-0905 higher after Holm adjustment |
| Predict | Gemini 2.5 Flash − Qwen3 30B A3B Instruct | +0.033 | [0.015, 0.055] | 0.0036 | google/gemini-2.5-flash higher after Holm adjustment |
| Predict | Kimi K2 Instruct − Gemini 2.5 Flash | +0.021 | [0.002, 0.041] | 0.2874 | not significant after Holm adjustment |
| Predict | GPT-4.1 − Qwen3 30B A3B Instruct | +0.019 | [-0.006, 0.046] | 1.0000 | not significant after Holm adjustment |
| Predict | Gemini 2.5 Flash − GPT-4.1 | +0.015 | [-0.005, 0.035] | 1.0000 | not significant after Holm adjustment |
| Find | GPT-4.1 − Qwen3 30B A3B Instruct | +0.296 | [0.225, 0.371] | 0.0004 | openai/gpt-4.1 higher after Holm adjustment |
| Find | GPT-4.1 − Gemini 2.5 Flash | +0.285 | [0.224, 0.347] | 0.0004 | openai/gpt-4.1 higher after Holm adjustment |
| Find | GPT-4.1 − Kimi K2 Instruct | +0.271 | [0.214, 0.329] | 0.0004 | openai/gpt-4.1 higher after Holm adjustment |
| Find | Kimi K2 Instruct − Qwen3 30B A3B Instruct | +0.025 | [-0.040, 0.084] | 1.0000 | not significant after Holm adjustment |
| Find | Kimi K2 Instruct − Gemini 2.5 Flash | +0.014 | [-0.040, 0.064] | 1.0000 | not significant after Holm adjustment |
| Find | Gemini 2.5 Flash − Qwen3 30B A3B Instruct | +0.011 | [-0.050, 0.072] | 1.0000 | not significant after Holm adjustment |
| Ask | GPT-4.1 − Qwen3 30B A3B Instruct | +0.136 | [0.044, 0.231] | 0.0120 | openai/gpt-4.1 higher after Holm adjustment |
| Ask | Kimi K2 Instruct − Qwen3 30B A3B Instruct | +0.127 | [0.029, 0.232] | 0.0378 | moonshotai/kimi-k2-0905 higher after Holm adjustment |
| Ask | GPT-4.1 − Gemini 2.5 Flash | +0.126 | [0.021, 0.240] | 0.1179 | not significant after Holm adjustment |
| Ask | Kimi K2 Instruct − Gemini 2.5 Flash | +0.116 | [0.012, 0.227] | 0.1086 | not significant after Holm adjustment |
| Ask | Gemini 2.5 Flash − Qwen3 30B A3B Instruct | +0.010 | [-0.115, 0.120] | 1.0000 | not significant after Holm adjustment |
| Ask | GPT-4.1 − Kimi K2 Instruct | +0.009 | [-0.076, 0.093] | 1.0000 | not significant after Holm adjustment |

## Excluded model family

Claude Sonnet 4.5 (`anthropic/claude-sonnet-4.5`) was predeclared as the fourth family
and was excluded for a provider decoding failure, not for its scores.
Prime Inference does not honour response_format=json_object for Anthropic models; Sonnet wrapped every JSON object in Markdown fences despite the prompt envelope rule, and a share of Elicit rollouts returned Anthropic refusal-fallback boilerplate instead of a completion. Under the frozen strict raw-JSON contract 0 of 1,500 study rollouts scored above zero. Re-running without JSON mode reproduced the fenced output 15/15.

| Stratum | Rollouts | Markdown-fenced | Provider refusal boilerplate | Raw JSON | Non-zero reward |
| --- | ---: | ---: | ---: | ---: | ---: |
| `predict-full--sonnet45` | 500 | 500 | 0 | 0 | 0 |
| `find--sonnet45` | 500 | 402 | 98 | 0 | 0 |
| `ask--sonnet45` | 500 | 456 | 44 | 0 | 0 |

A 5×1 diagnostic without JSON-object decoding reproduced the fenced output on
15 of 15 rollouts. The excluded runs are retained under
`release-0.6.0-60e6fdd-full-degenerate-sonnet45-20260903T204124Z` and are not published as evaluation records.
Kimi K2 Instruct (`moonshotai/kimi-k2-0905`) replaced it after a 5×1 JSON-mode
diagnostic showed raw-JSON completions on every rollout with non-zero reward
in each task mode; that selection used envelope compliance only. No scorer,
prompt, sampling, or artifact change was made for either model.

## Provider recovery disclosure

8 of 7,500 rollouts (0.107%) were resampled after a
provider-side failure. Every recovery is recorded in a manifest next to the
study evidence, selection never depended on reward, and the original runs are
retained unchanged:

- Find / Gemini 2.5 Flash (run `a7c3e16e`): 1 of 500 rollouts returned a provider-side `has_error` stop condition (task IDs 79). Only those slots were resampled once at concurrency 1; the original run is retained unchanged at `release-0.6.0-60e6fdd-full-provider-failure-20260903T074009Z`.
- Ask / Gemini 2.5 Flash (run `61fee42d`): 7 of 500 rollouts returned a provider-side `has_error` stop condition (task IDs 19, 19, 22, 24, 25, 54, 70). Only those slots were resampled once at concurrency 1; the original run is retained unchanged at `release-0.6.0-60e6fdd-full-provider-failure-ask-20260903T174435Z`.

Public-record tooling reads the manifests, marks the resampled rollouts in each
record's sample metadata, and publishes the recovery count and policy.

## Interpretation

Predict probability reward ranges from 0.706–0.761 across the four
models. Every model has a clustered interval entirely above the
uniform-probability reference (0.667), and all remain below the strongest
prompt-observable matrix-only comparators in the reference table, so the
supported construct is experimental semantic-conditioned matrix completion
rather than general collective-preference reasoning. 3 of 6 Predict
pairwise comparisons remain significant after Holm adjustment. The prompt-view
ablations for GPT-4.1 report paired drops
(matrix-only: full − view = +0.020 [0.003, 0.037]; shuffled-text: full − view = -0.013 [-0.024, -0.001]; text-only: full − view = +0.030 [0.003, 0.058]), where a positive drop means the view scores
below the full prompt. Shuffling which statement text accompanies which statement does not lower reward (the shuffled view scores at least as well as the full prompt), so this study does not demonstrate that the model uses statement semantics beyond the visible vote matrix. Removing the matrix (text-only) and removing the text (matrix-only) both cost reward, so both inputs matter for the score, but the text's contribution is not shown to depend on which statement it is attached to. This limits the semantic-conditioning claim and is the main construct-validity caveat for Predict.

Strict Elicit Find F1 ranges from 0.219–0.515 and Ask utility from
0.158–0.294, against the model-free references in the reference
table (prompt-only Find probes; component-oracle Ask references) and exact
structured ceilings of 1.0. 3 of 6 Find and 2 of 6 Ask pairwise
comparisons remain significant after Holm adjustment. Component diagnostics
separate localization, type, diagnosis, and relationship failures on Find, and
format, grounding, stance, and top-1 selection failures on Ask, rather than
collapsing every failure into one opaque zero.

These results support evaluation claims about measurable headroom on the fixed
public corpus. They do not establish human-preference validity,
contamination-resistant leaderboard performance, organizational information
gain, or beneficial RL training; training claims require multiple seeds and
transfer to a fresh independently implemented private generator.

## Public evaluation records

Each primary run was published as a version-pinned public Prime evaluation and
read back. Every record carries the exact environment/version IDs, the artifact
source commit, model, task mode, sampling configuration, 500 samples, and its
recovery count.

| Task | Model | Public evaluation |
| --- | --- | --- |
| Predict | Kimi K2 Instruct | [`ql3iv94rp7ti3eecroigf8ue`](https://app.primeintellect.ai/dashboard/evaluations/ql3iv94rp7ti3eecroigf8ue) |
| Predict | Gemini 2.5 Flash | [`bk4vj1ht2kipbvmc1efvsowc`](https://app.primeintellect.ai/dashboard/evaluations/bk4vj1ht2kipbvmc1efvsowc) |
| Predict | GPT-4.1 | [`b82u8wrdgvhutoe82ahwd6ft`](https://app.primeintellect.ai/dashboard/evaluations/b82u8wrdgvhutoe82ahwd6ft) |
| Predict | Qwen3 30B A3B Instruct | [`zh0ix44jkujvv7dolsaym1nt`](https://app.primeintellect.ai/dashboard/evaluations/zh0ix44jkujvv7dolsaym1nt) |
| Find | GPT-4.1 | [`opw9n6295a2ple1onbyy6vgb`](https://app.primeintellect.ai/dashboard/evaluations/opw9n6295a2ple1onbyy6vgb) |
| Find | Kimi K2 Instruct | [`l8zux8rg38sbgi9l41zuisv3`](https://app.primeintellect.ai/dashboard/evaluations/l8zux8rg38sbgi9l41zuisv3) |
| Find | Gemini 2.5 Flash | [`o8my6uogmhvmcpil3hxwy2cn`](https://app.primeintellect.ai/dashboard/evaluations/o8my6uogmhvmcpil3hxwy2cn) |
| Find | Qwen3 30B A3B Instruct | [`x8xdxs4i4ss7n2hwfxpucf0x`](https://app.primeintellect.ai/dashboard/evaluations/x8xdxs4i4ss7n2hwfxpucf0x) |
| Ask | GPT-4.1 | [`vd56xn3qdnkk4u8nkxdbam9a`](https://app.primeintellect.ai/dashboard/evaluations/vd56xn3qdnkk4u8nkxdbam9a) |
| Ask | Kimi K2 Instruct | [`qtgo5k674ltgm8ee5owew5yd`](https://app.primeintellect.ai/dashboard/evaluations/qtgo5k674ltgm8ee5owew5yd) |
| Ask | Gemini 2.5 Flash | [`ivwbl2kw6awoyv262k80pgmm`](https://app.primeintellect.ai/dashboard/evaluations/ivwbl2kw6awoyv262k80pgmm) |
| Ask | Qwen3 30B A3B Instruct | [`zo2p834mygmwkwpr4e5gr42h`](https://app.primeintellect.ai/dashboard/evaluations/zo2p834mygmwkwpr4e5gr42h) |

The three Predict prompt-view ablation runs are retained in the study evidence
and reported above; they are not published as separate public records.

## Study protocol and claim boundary

Runs used `vf-eval` from Verifiers 0.3.0 with saved results
(`metadata.json` + `results.jsonl`): `temperature = 0.2`,
`response_format = json_object`, `max_tokens = 2048` (Predict) or `4096`
(Elicit), `shuffle = false`, five-way concurrency, zero configured retries, no
judge model, and the Prime Inference chat-completions endpoint. Controlled
recoveries used concurrency 1 with otherwise identical arguments. Exact
configs, per-example traces, logs, and SHA-256 digests are retained outside the
repository in `study-manifest.json` and `study-evidence.sha256`.

## Release attestation

| Field | Value |
| --- | --- |
| Artifact source commit | `60e6fdd2e8509a25db1c0faa6c2096c000e2b742` |
| `commonground-score` 0.6.0 wheel/sdist SHA-256 | `b2cdaafd68090ecd3ab9b7d37e13d177bc0cda9db8a0b41252c76951242a3de5` / `9c87cd1bb254ea90300005a42798503838d4b5b2e19e3e75beb3a69015b82497` |
| `commonground-scenarios` 0.6.0 wheel/sdist SHA-256 | `46b2709959ef32a472156019eea0544d614d800c120877e768191bb132683ac0` / `9f74c4bb15f1a00fcb5141ded9c2d43d87a152dd01d2aed4f034c6bfbdb0162e` |
| `commonground-predict` 0.6.0 local wheel/Hub content SHA-256 | `228fc77a880266fb038a65368f29dc21cf24e2e9111273b3da1f6ea406fd89e6` / `783914858035810caeca8c790d4041eb4108943ba980f97f2f8f7ffbdd82e012` |
| `commonground-elicit` 0.6.0 local wheel/Hub content SHA-256 | `7c24684a1e70a00fcc7e015b6389ba8b8d76215eb32940adc43ba00e055880fa` / `c4be50694099bd984dfc4591e857306d12d3d17e40b77477624ead8439865197` |
| Hub version IDs | Predict `b9rznmh0i1bmeeb176j7q2gp`; Elicit `yff9k0yxhxm9smvaqhelrs1i` |
| Study run IDs | 15 saved-result run IDs in `study-manifest.json` |
| Public evaluation IDs | 12 version-pinned records listed above |
