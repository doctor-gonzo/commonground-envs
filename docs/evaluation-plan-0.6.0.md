# Common Ground 0.6.0 evaluation plan

Status: source candidate under local verification. No 0.6.0 model result,
artifact hash, Hub version, or public evaluation record is claimed yet.
Version 0.5.0 remains the current public release and immutable historical
evidence until every gate below passes.

## Release target

| Distribution | Candidate version | Role |
| --- | --- | --- |
| `commonground-score` | 0.6.0 | Probability scoring and Brier-skill helpers |
| `commonground-scenarios` | 0.6.0 | Scenario schema 5, generation, and validation |
| `commonground-predict` | 0.6.0 | Probabilistic masked-vote prediction and prompt ablations |
| `commonground-elicit` | 0.6.0 | Evidence-grounded diagnosis and clarification selection |

All four distributions deliberately share one release number. Environment
packages must pin both shared packages at exactly 0.6.0. The scenario contract
uses `urn:commonground:schema:scenario:5`; revised corpus provenance uses the
v6 template/layout profiles and `value-composition-v2` vote rule.

Version 0.6 is a breaking Elicit benchmark revision. It changes visible
faction information, generated layouts, issue utility, the response contract,
and diagnostic names. No 0.5 Elicit score may be carried forward. Predict keeps
the same core objective but tightens parsing and adds prompt views; its 0.5
scores remain evidence only for the exact 0.5 artifact.

## Release-blocking local evidence

Before packaging, retain machine-readable evidence that:

1. every generated scenario validates and both 100-row Elicit splits
   regenerate byte-for-byte;
2. the v6 training and held-out profile labels are disjoint, all exact and
   semantic split audits pass, and no fixed 0.4/0.5 filler marker remains;
3. every helper-built held-out row has five documents, with separate primary
   issue documents, authored relationship evidence, and a plant-free ledger;
   seeded type-neutral distractors are generated compositionally and document
   sentence counts vary independently rather than exposing an equal-length
   marker. Across every supported distractor density and bounded document
   length, every retained candidate's accepted actor aliases remain visible;
   density filtering affects only true neutral distractors, and a view that
   cannot retain complete actor support must remove both that key and its
   visible anchor;
4. title root, style, anchor position, sentence count, and document-length
   majority-label diagnostics each remain below the release threshold of
   0.45; the combined title-root, normalized anchor-position, sentence-count,
   and word-length one-nearest-neighbor attack is fit with every held-out
   template family excluded and remains below 0.45 macro recall (balanced
   accuracy). The three issue classes must each occupy exactly one third of
   planted anchors, making the reported balanced-accuracy chance reference
   exactly `1/3`. The removed marker baseline scores exactly 0 strict F1 and the
   combined length/position baseline stays below 0.05 strict F1; require the
   genuine prompt-only three-longest-sentences probe to have zero strict F1
   and both localization recall and conditional type accuracy below 0.10; the
   current source-aware
   sector-team marker must have localization recall and F1 below 0.05, with a
   planted/distractor prevalence gap below 0.10. For the retired-marker,
   layout/length/position, longest-sentence, and sector-team locators, replace
   every correctly located span's type, diagnosis, decision frame, and
   relationship with exact gold while preserving all predicted spans and
   false positives. These component-isolating strict-F1 gates must remain
   exactly 0, below 0.25, below 0.10, and below 0.05 respectively, so a
   semantic failure cannot hide a localization shortcut. On the 18
   helper-built five-document families, verify one document in each of five
   roles (three issue primaries, contradiction relationship, plant-free), then
   require a leave-one-template-out five-way structural classifier below 0.30
   balanced accuracy against exact 0.20 class shares/chance. The analogous
   related-versus-other classifier must remain below 0.60 balanced accuracy
   against a 0.50 balanced chance reference, with related-document prevalence
   exactly 0.20;
5. every faction summary exposes the exact signed five-dimensional value
   profile used by stance composition, while summaries remain independent of
   current issue types and answer labels. Ask renders every candidate's exact
   canonical decision frame and signed trade-off weights, plus the composition,
   pass-threshold, orientation, and ranking semantics required to use them,
   without exposing evidence locations, issue types, relationships, accepted
   aliases, stored stances, decision values, or utilities. Every accepted
   target-changing weight mutation must change the rendered Ask prompt, while
   the same mutation leaves Find unchanged, or scenario validation must reject
   it;
6. every stored `decision_value` exactly recomputes from the continuous
   preference trade-off, the default Ask budget is K=1 of three candidates,
   the top-one boundary tie rate is 0, and the minimum normalized margin
   `(U1 - U2) / U1` is at least `0.10`. Publish the mean and minimum raw gaps and
   normalized margins;
7. the exact analytic uniform-random selection expectation scores below `0.90`
   with top-one selection accuracy exactly `1/3`; exact runner-up selection
   scores below `1.0` with zero top-one accuracy; public-profile composition
   reaches `1.0` reward and top-one accuracy; and the exact top-one structured
   answer reaches `1.0` reward and top-one accuracy on every row;
8. a generic question, an unsupported frame, an unsupported semantic clause
   appended to the question stem or orientation tail, a tiny reference-keyword
   fragment, a long evidence-word dump that does not recover the reference, incomplete or
   extra faction keys, and a polarity flip with unchanged prose all receive zero
   semantic credit. Find frames recover the hidden hand-authored per-slot
   concepts; Ask exposes each canonical candidate frame while retaining hidden
   role-specific alias sets for grounded paraphrases. In either task, cited
   passages independently anchor applicable outcomes, and candidate or response
   prose cannot supply evidence for its own frame. The independent two-sided semantic audit in
   `test_elicit_semantic_audit.py` must pass all 23 valid and 31 invalid cases:
   every issue type and each of the five frame slots is identified separately,
   alongside active/passive voice, grounded negation, consistent orientation
   reversal, hidden-reference rewording, actor/action swaps, outcome swaps,
   unsupported exceptions or thresholds, polarity mismatch, and a wrong
   contradiction relationship;
9. unknown issue types, malformed decision objects, stance containers, numeric
   values, duplicate raw JSON keys, wrappers, oversized completion payloads,
   and deeply nested JSON fail closed without exceptions; the completion-size
   limit does not reject a larger trusted generated answer key; Find's primary
   `findings` root is scored independently, while an omitted or malformed
   optional weight-zero companion `questions` field receives zero companion
   credit without erasing valid findings;
10. adding a false positive or overlapping hedge strictly lowers shaped Find
    reward, while strict Find F1 remains the public evaluation objective;
11. Predict preserves the exact masked-cell key set, canonical tie order, and
    probability reward under every prompt mode; malformed numeric values,
    duplicate raw keys, prose/fence wrappers, trailing JSON, or invalid totals
    fail closed; and
12. clean wheels and source archives install all four exact versions together,
    contain the required legal/provenance files, and load all three task modes.

## Model-free evidence

Regenerate the Predict probability-native comparator table. It must include
uniform, current-snapshot visible prior, train-split prior, explicitly
transductive evaluation-corpus prior, per-statement visible frequencies,
one-hot and probability-valued neighbors, smoothed distance-weighted k-NN,
train-split text-only predictions, and clearly labeled held-out/generator
diagnostics. Report probability reward, normalized Brier, accuracy, and Brier
skill relative to both uniform probability and the evaluator-side original-
snapshot visible-matrix class prior. The latter remains fixed across modes and
is unavailable to a text-only agent.

Regenerate the Elicit floor and corpus-audit table. It must include random and
vague spans, the removed filler/type, sector-team, and equal-length markers,
the length/position layout probe, a prompt-only three-longest-sentences probe,
component-oracle strict F1 for each named locator, a combined
title/length/position leave-one-template-out classifier, and five-way
document-role plus binary relationship-document structural probes with class
prevalence and balanced-chance references,
legacy shortcut parsers, simple Ask baselines, the exact analytic uniform-random
selection expectation and top-one accuracy, exact runner-up reward and top-one
accuracy, public-profile composition reward and top-one accuracy, exact issues
with random stances, and the exact structured ceiling. It must also include a deterministic
reconstruction from the candidate generator, templates, seed, and answer
construction, explicitly labeled as a source-aware replay/memorization ceiling
for this unpublished candidate. Prompt-observable, component-oracle, and
source-aware rows must remain separate; replay is not a floor or evidence of
generalization.

The named attacks above are development-time candidate audits, not proof that
all structural shortcuts have been removed. After source and scorer freeze, a
separate reviewer should construct fresh attack examples without reusing these
fixtures. If that independent attack holdout fails a preregistered gate, reopen
the candidate and create a new untouched holdout; do not tune on the failed
holdout and then report it as final evidence.

Do not copy 0.5 table values into 0.6 documentation. The Elicit README and
dataset card may report locally reproduced model-free candidate measurements,
clearly labeled as such; they contain no 0.6 model, training, Hub, or public
evaluation result. Regenerate the floor tables after any corpus or scorer
change and require an exact match before packaging.

## Predict same-model prompt ablations

Use `configs/eval/predict-ablation.toml` to run `full`, `matrix-only`,
`text-only`, and `shuffled-text` over the same 100 evaluation tasks with five
rollouts. Preserve task ordering, masked cells, answers, model, provider,
sampling settings, and concurrency across modes. Run the ablation for every
model used in the primary Predict comparison when budget permits; at minimum,
use one strong hosted model and one open-weight trainable model before making a
claim about the contribution of text.

An explicit `prompt_mode` declares a native 0.6 ablation artifact. Aggregation
requires the saved resolved client identity, sampling settings, concurrency,
matching effective trace-agent settings, exactly one matching successful model
call per episode, ordered task/answer provenance, and raw named
`probability_reward` scores with weight `1.0`. It hashes those comparison inputs
and the ordered task roster into one stable signature that excludes only `prompt_mode`;
analysis rejects a four-mode set unless every signature matches. Historical
full-only artifacts remain readable but cannot stand in for missing ablation
provenance.

Analyze mode differences as paired task-level comparisons. Report each mode's
probability reward, Brier, accuracy, and Brier skill against both uniform and
the evaluator-side original-snapshot visible-matrix climatology, plus paired
differences from `full`. Compute snapshot-prior skill from pooled losses using
the same equal task weights as the primary analysis and recompute the ratio
inside each bootstrap resample; do not average per-task skill ratios. Treat a
zero pooled reference loss as undefined and fail the analysis. A language
contribution is not established merely because `full` beats uniform. The
matrix-only and shuffled-text controls must be included in the interpretation,
and a null or reversed effect must be reported plainly.

## Exact-artifact model study

Push candidates privately only after local, wheel, and archive gates pass. Run
100 tasks with five rollouts for Predict, Elicit Find, and Elicit Ask across at
least four model families, including an open-weight trainable model. Pin exact
Hub versions and preserve complete configs and traces. Reject incomplete runs
or any recovered/error history in the primary study. Preserve failed attempts
in a quarantine directory, and rerun the entire predetermined model/task
stratum from the same frozen configuration; never replace selected episodes or
choose a replacement based on score. The analyzer's `--require-no-recoveries`
gate is mandatory.

Before freezing hosted generation settings, serialize the exact extended Find
and Ask responses for every row with the documented tokenizer. The candidate
must keep the maximum exact response at or below 1,024 `cl100k_base` tokens,
leaving at least half of the 2,048-token study budget for ordinary model
variation. Record observed completion-length distributions during the pilot as
additional descriptive evidence.

For Elicit, report strict reward and clustered component diagnostics:

- Find localization, type, diagnosis, and relationship metrics;
- Ask format validity, exact top-one selection accuracy, and grounding recall;
- Ask end-to-end grounded stance recall; and
- Ask exact-evidence match recall and stance accuracy over the same
  deterministic exact-evidence assignment.

For Predict, report probability reward, Brier, accuracy, and both named Brier
skill references. For
all modes, average repeated rollouts within task before uncertainty analysis.
Resample Elicit templates and then variants; resample Predict tasks.

The 0.6 analysis additionally produces clustered confidence intervals for
diagnostic metrics. Pairwise model comparisons use deterministic
cluster-level sign-flip randomization under a sign-symmetric/exchangeable
paired cluster-effect null; equality of means alone does not justify that
randomization. Before results are inspected, the single global Holm family is
fixed to the 18 primary contrasts from four models across three tasks
(`3 * choose(4, 2)`). Prompt-ablation contrasts and component diagnostics are
excluded from that family and remain descriptive. Their pointwise 95% intervals
are unadjusted, not simultaneous confidence bands. Label all model comparisons
exploratory; an adjusted p-value does not turn this synthetic study into a
general model ranking.

## Publication and claims

After private verification, record the source commit, wheel/sdist hashes, Hub
version IDs and content hashes, exact configs, run IDs, and any disclosed
provider recovery. Manually change visibility only after reviewing those
records. Publish new version-pinned evaluation records and read them back
before calling the evidence public. Do not delete or rewrite 0.5 artifacts;
mark them historical or superseded only after 0.6 is independently usable.

Supported claims remain limited to experimental synthetic
semantic-conditioned matrix completion and structured evidence localization,
issue classification, relationship matching, clarification-target selection,
and synthetic stance prediction. The deterministic scorer does not establish
question usefulness, organizational information gain, human-preference
validity, contamination-resistant leaderboard performance, or beneficial RL
training. Training claims require multiple independent seeds and evaluation on
a fresh, independently implemented private generator. Source-aware replay is
expected to recover the open answer construction and must never be substituted
for that private transfer test.
