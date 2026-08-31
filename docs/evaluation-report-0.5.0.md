# Common Ground 0.5.0 evaluation report

Status: exact public Hub artifacts, shared PyPI packages, and the full
exact-artifact model study are complete. Twelve version-pinned public Prime
evaluation records are the remaining evidence-publication step.

Version 0.5.0 is an immutable correction to Elicit 0.4.1. It replaces inferred
contradiction links with authored relationships, replaces the issue/stance
phrase codebook with issue-independent faction values, makes shaped Find
precision-sensitive, and replaces Ask's hidden vocabulary gate with structured
grounding. All four distributions use the same 0.5.0 release number.

## Release scope

| Environment | Eval rows | Train rows | Primary reward | Important diagnostics |
| --- | ---: | ---: | --- | --- |
| `commonground-predict` | 100 | 200 | `1 - normalized Brier` | vote accuracy, normalized Brier |
| `commonground-elicit:find` | 100 | 100 | strict `finding_f1` | localization, type, diagnosis, relationship, question utility |
| `commonground-elicit:elicit-ask` | 100 | 100 | normalized `question_utility` | format validity, grounding recall, stance accuracy |

All bundled rows are synthetic. Answers are public for reproducibility and
open training, so this release is not a contamination-resistant leaderboard
and does not measure preferences of real people.

## Model-free comparators

Predict comparators use the committed 100-row evaluation file with eight
masked votes per snapshot:

| Comparator class | Comparator | Probability reward | Accuracy | Brier | Brier skill vs uniform |
| --- | --- | ---: | ---: | ---: | ---: |
| No-input | Uniform probability | 0.667 | 0.590 | 0.333 | 0.000 |
| Evaluation-corpus visible (transductive) | Global visible class prior | 0.717 | 0.590 | 0.283 | 0.150 |
| Train-split no-text | Global empirical class prior | 0.669 | 0.166 | 0.331 | 0.008 |
| Train-split text-only | Bag-of-words vote probabilities | 0.401 | 0.244 | 0.599 | -0.796 |
| Prompt-observable matrix-only | Per-statement visible class frequencies | 0.760 | 0.581 | 0.240 | 0.279 |
| Prompt-observable matrix-only | Nearest participant (1-NN) | 0.819 | 0.819 | 0.181 | 0.456 |
| Prompt-observable matrix-only | Five-neighbor vote frequencies | 0.889 | 0.891 | 0.111 | 0.666 |
| Prompt-observable matrix-only | Distance-weighted 5-NN with smoothing | 0.881 | 0.900 | 0.119 | 0.643 |
| Generator diagnostic | Latent cluster-pattern replay | 0.916 | 0.916 | 0.084 | 0.749 |

The full comparator output also includes always-agree, one-hot majority and
five-neighbor decisions, a held-out-label diagnostic, and their explicit
information classes. The strong matrix-only results are a central boundary:
the supported construct is experimental semantic-conditioned matrix
completion, not general collective-preference reasoning.

Elicit comparators use visible prompt information unless explicitly labeled as
a component oracle:

| Comparator class | Task | Comparator | Mean reward |
| --- | --- | --- | ---: |
| Prompt-observable | Find | Random visible spans | 0.080 |
| Prompt-observable | Find | Flag vague-sounding spans | 0.195 |
| Prompt-observable | Find | Legacy 0.2 document-ID/position codebook | 0.000 |
| Prompt-observable | Ask | Template clarity questions | 0.078 |
| Prompt-observable | Ask | Randomly targeted questions | 0.069 |
| Prompt-observable | Ask | Removed 0.3 summary/stance codebook | 0.000 |
| Component oracle | Ask | Exact issues plus removed 0.4 principle parser | 0.000 |
| Source-aware prompt-only | Ask | Public template detector plus removed 0.4 parser | 0.000 |
| Component oracle | Ask | Exact top-K issues plus random stances | 0.670 |
| Component oracle | Ask | Exact top-K issues plus exact stances | 1.000 |

The removed 0.4 parser scoring zero, even when given exact issues, directly
tests the former finite phrase shortcut rather than only the older literal
0.3 format.

## Exact-artifact model study

Four model families each ran all 100 tasks with five rollouts in all three task
modes: 6,000 rollouts total.

| Task | Model | Run ID | Reward mean ± rollout SD | 95% clustered bootstrap CI | Key diagnostics | Zero-reward tasks | Recovered |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: |
| Predict | Claude Sonnet 4.5 | `6b2c2e04-694f-42f7-be0d-0ced22d0f869` | 0.808 ± 0.089 | [0.791, 0.824] | accuracy 0.722; Brier 0.192 | 0 | 0 |
| Predict | Gemini 2.5 Flash | `9895d07a-5adc-4d72-b56c-33b6f5ebf29c` | 0.773 ± 0.096 | [0.755, 0.790] | accuracy 0.694; Brier 0.227 | 0 | 0 |
| Predict | GPT-4.1 | `6a747842-9661-4a84-be03-ae40570edaa8` | 0.735 ± 0.121 | [0.713, 0.757] | accuracy 0.659; Brier 0.265 | 0 | 0 |
| Predict | Qwen3 30B A3B Instruct | `61bb4183-7c01-408b-8f55-a799b7eff2e8` | 0.700 ± 0.113 | [0.678, 0.720] | accuracy 0.552; Brier 0.300 | 1 | 0 |
| Find | GPT-4.1 | `d30bb391-60e9-4ccc-91f6-7dc961ed7a68` | 0.314 ± 0.296 | [0.236, 0.400] | localization 0.525; type 0.465; diagnosis 0.276; relation 0.266 | 31 | 0 |
| Find | Qwen3 30B A3B Instruct | `103d1d72-0836-4377-bb55-7480a243705d` | 0.263 ± 0.238 | [0.199, 0.327] | localization 0.473; type 0.574; diagnosis 0.234; relation 0.221 | 27 | 0 |
| Find | Gemini 2.5 Flash | `9669bc8d-6226-4966-8961-b91f279d356e` | 0.237 ± 0.251 | [0.171, 0.309] | localization 0.615; type 0.476; diagnosis 0.298; relation 0.287 | 26 | 0 |
| Find | Claude Sonnet 4.5 | `8c4b96a6-1cb2-45aa-bec4-245da72f9524` | 0.133 ± 0.235 | [0.076, 0.201] | localization 0.370; type 0.278; diagnosis/relation 0.173 | 54 | 0 |
| Ask | Claude Sonnet 4.5 | `e7c3be02-29ae-4285-8531-6bf073a1a703` | 0.337 ± 0.276 | [0.264, 0.409] | format 0.966; grounding 0.458; stance 0.219 | 22 | 0 |
| Ask | Gemini 2.5 Flash | `908b8c3e-85dd-4942-848f-5899f08cb9fc` | 0.287 ± 0.287 | [0.220, 0.359] | format 1.000; grounding 0.392; stance 0.185 | 23 | 0 |
| Ask | Qwen3 30B A3B Instruct | `5e9270e4-501c-4930-a0ee-00d0a1f01a86` | 0.170 ± 0.236 | [0.108, 0.239] | format 1.000; grounding 0.235; stance 0.106 | 43 | 1 |
| Ask | GPT-4.1 | `a65543a6-aae9-4302-899a-c17d07cad0b4` | 0.110 ± 0.177 | [0.063, 0.163] | format 1.000; grounding 0.157; stance 0.064 | 53 | 0 |

Intervals use 50,000 deterministic percentile bootstrap resamples with seed
`20260830`. Each task mean first averages five repeated rollouts. Predict
resamples 100 tasks. Elicit resamples 20 base templates and then five variants
within each selected template. Paired comparisons reuse the same draws.

All six Predict pairwise intervals exclude zero. Selected Elicit comparisons
show useful but task-specific ordering: GPT-4.1 exceeds Sonnet on Find by 0.181
([0.105, 0.256]); Sonnet exceeds Qwen on Ask by 0.166 ([0.066, 0.268]); and
Sonnet versus Gemini on Ask remains unresolved at +0.050 ([-0.048, 0.142]).
The complete 18-comparison table is generated by
`scripts/analyze_release_study.py` and retained with the study evidence.

## Provider recovery disclosure

Eleven runs contain exactly one successful model call per rollout with no
error history. Qwen Ask task 54, trace
`23a722e1fdff4b50ba9c9d7938eff307`, contains one Prime provider 502 because
an upstream response used an invalid `finish_reason="error"`; the framework
then made one successful call. The final rollout completed and scored 0.0.

Two attempts to replace that run reproduced the same provider-specific
malformed response, including an attempt at concurrency one. Both replacements
were quarantined and the original complete study run was restored. The retained
study therefore reports one recovered rollout out of 6,000 (0.0167%) rather
than selecting a replacement based on score. Public-record tooling accepts
this history only for that exact trace ID and exact 502 signature, uses final
successful-call token usage, and publishes the recovery count and policy.

## Interpretation

Predict is a strong experimental benchmark contribution. Every model exceeds
uniform probability reward, all pairwise intervals separate, and none reaches
the strongest prompt-observable matrix comparators. This gives both meaningful
headroom and a clear non-language baseline.

Elicit now separates complementary capabilities. Find rewards precise
end-to-end diagnosis while its component metrics show where models fail. Ask
has substantially more dynamic range than 0.4.1 without relying on hidden
canonical vocabulary: model means range from 0.110 to 0.337, against simple
prompt-observable baselines of 0.069–0.078 and an exact structured ceiling of
1.0. These results support evaluation claims, not demonstrated reinforcement-
learning benefit or real organizational information gain.

## Public evaluation records

The 12 retained runs are ready for version-pinned public Prime evaluation
records. This section intentionally remains pending until each record has been
uploaded, read back, and checked for the exact environment/version IDs, 500
samples, metrics, source commit, and disclosed recovery count. Historical
0.4.1 records remain attached only to that superseded artifact.

## Integrity and operational evidence

- All 24 Elicit template families author contradiction links explicitly; a
  corpus-wide validator rejects related evidence that resolves to another
  plant or distractor.
- Faction summaries are issue-independent. Stances compose from general
  values, issue alternatives, and explicit yes-side polarity; reversal and
  invariance regressions cover the transform.
- Find's shaped curriculum uses precision-sensitive stage F1 and penalizes
  false-positive spam and overlapping hedges.
- Ask uses structured evidence, type, polarity, relation, and stance fields;
  formatting, grounding, and stance are reported separately.
- Exact split identity, semantic similarity, legal/provenance contents,
  dependency manifests, source archives, wheels, and native/legacy loaders are
  release-gated.

## Study protocol and claim boundary

Saved native Verifiers v1 configs record `temperature = 0.2`,
`max_tokens = 2048`, `shuffle = false`, ten-way concurrency, no judge model,
and zero configured environment/agent retries. Models were served through
Prime Inference's chat-completions endpoint. Exact configs and per-example
traces are retained outside the repository.

Supported claims remain limited to experimental synthetic
semantic-conditioned matrix completion and structured policy-issue
localization/clarification selection. The evidence does not establish human-
preference validity, contamination-resistant leaderboard performance,
organizational information gain, or beneficial RL training. Training claims
require multiple seeds and transfer to a fresh independently implemented
private generator; Predict also needs text/matrix ablations.

## Release attestation

| Field | Value |
| --- | --- |
| Artifact source commit | `83e0b1767e08fa8de0e55a937d34dc0f0af3003b` |
| `commonground-score` 0.5.0 wheel/sdist SHA-256 | `31baee389fed8d275a81baeebfe9649749a0b06aabe3100320bfcd6f28e4689c` / `dc3dacaca71f86400bbb5b3cab5bdf7b6f6d1bc328f26ff81b058b650e20af15` |
| `commonground-scenarios` 0.5.0 wheel/sdist SHA-256 | `2b12d4b5baf90adcefc49e2edbfc0c8cda02368fe65f215fe5f6904013b2cc65` / `d8c8d5211586326084b4723b6c36f0c0d8b6a88ac13ba789dcc9ba56ecbaeca5` |
| `commonground-predict` 0.5.0 wheel/Hub content SHA-256 | `efc8001cb4783d4645560e44a1b82aa96503108109e325a28638cc6e33a09888` / `f07d0fdda58da2a232abf61e18d75368c225bbf076282b1c6877caff8909f93e` |
| `commonground-elicit` 0.5.0 wheel/Hub content SHA-256 | `fd7cac8b92daaeb59bbc2c6f707a62556f1dc9c07cdfc64e0c614cb213443df8` / `8ff9a364be53cfec2dabe785eafe1e5d177365f4a1882771446b65867ab69c8f` |
| Hub version IDs | Predict `kcsr5z1a9rq4ywj7mhswa5zh`; Elicit `sos42d21n3sr7p3zyph66wiz` |
| Native evaluation run IDs | 12 run IDs in the exact-artifact model-study table |
| Public evaluation IDs | Pending exact-record publication and read-back |
