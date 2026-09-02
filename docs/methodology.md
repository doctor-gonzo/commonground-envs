# Methodology and benchmark scope

This page describes the unreleased 0.6.0 source candidate. Version 0.5.0
remains the current public artifact; 0.6 model, training, artifact, Hub, and
public-evaluation evidence is prospective until the release plan passes.

## Shared vote statistics

`commonground-score` independently implements the one-proportion and smoothed
two-proportion equations used by Computational Democracy Project Polismath at
immutable commit `5089c6bef9eb1a1e454beb34354fb29dd0a2b6f0`:

https://github.com/compdemocracy/polis/blob/5089c6bef9eb1a1e454beb34354fb29dd0a2b6f0/math/src/polismath/math/stats.clj

Numerical fixtures and edge cases are pinned in tests; no upstream source is
copied. `vote_entropy` and `cluster_separation` are Common Ground metrics, not
Polis implementations. Multiclass Brier is the conventional squared-error sum
divided by two, bounded to `[0,1]`.

## Predict 0.6 construct

Predict contains 200 training and 100 evaluation snapshots. Training and
evaluation have disjoint statement text, session IDs, seeds, and profile
generator families. Each statement has an explicit policy dimension. A
cluster's latent preference for that dimension, plus item bias, pass threshold,
and participant noise, causally determines the vote. A regression test changes
the dimension while preserving statement position and verifies the vote
changes.

This improves on the 0.2 generator, whose vote pattern depended on statement
position rather than text meaning. It does not remove matrix completion as a
major task component: prompt-observable 1-NN and 5-NN score 0.819 and 0.891.
Accordingly, semantic-conditioned matrix completion is the supported claim.
Text-only, matrix-only, shuffled-text, item-item, factorization, and spectral
ablations are recommended before claiming a specific role for language.

The probability-native suite includes uniform, an original-snapshot visible-
matrix class prior, a labeled train-split empirical prior, an explicitly transductive
evaluation-corpus visible prior, per-statement visible frequencies, one-hot and
frequency-valued neighbors, smoothed distance-weighted k-NN, and a train-split
text-only Naive Bayes model. The original-snapshot prior reads only the visible
matrix in one full task; the transductive prior pools visible votes from all
evaluation snapshots and is labeled accordingly. The suite reports Brier skill
against both uniform and that fixed original-snapshot climatology. Uniform
is distribution-free but already earns `2/3` under `1 - normalized Brier`;
the original-snapshot reference reveals improvement over a stronger baseline.
It is evaluator-side and deliberately held fixed across ablations; a text-only
agent does not observe the votes used to construct it. Both environment interfaces emit its raw
reference Brier loss. Release analysis derives skill as the ratio of pooled
model loss to pooled reference loss with equal task weights and recomputes that
ratio inside every bootstrap resample; it derives uniform skill from Brier in
parallel. A zero pooled reference loss makes this skill undefined and fails the
analysis rather than being clipped, dropped, or substituted.
The per-snapshot best-constant comparator reads
held-out labels and is a diagnostic, not a floor. Latent-pattern replay reads
generator state and is a generator diagnostic, not a model-achievable
universal ceiling.

The required response contains exactly one complete three-class probability
map for every masked cell. The primary reward is `1 - normalized Brier`;
argmax accuracy and Brier loss are diagnostics. Missing, extra,
duplicate-normalized, and malformed keys fail the response contract. Numeric
strings, booleans, non-finite values or totals, negative values, and unsafe
magnitudes fail closed. The parser accepts exactly one JSON value: prose,
Markdown fences, trailing values, and repeated raw keys fail closed. Equal
probability scores use a public canonical tie order: agree, then pass, then
disagree.

The 0.6 taskset exposes four prompt views over the same snapshots, task IDs,
masked cells, and answers. `full` shows text and matrix, `matrix-only` replaces
all policy text with one neutral placeholder, `text-only` withholds the matrix,
and `shuffled-text` applies a deterministic session-seeded derangement between
statement text and matrix columns. Same-model comparisons across these views
are ablations of prompt information, not independent benchmark samples.

## Elicit 0.6 construct

Elicit contains 100 training and 100 evaluation scenarios. The splits use
disjoint template/layout-profile labels while sharing one core generator. Each
split enforces unique instance, canonical-prompt, policy-issue, prompt, and
answer hashes, and corresponding cross-split overlap is blocked. A structural
signature records the profile, document and faction counts, issue types,
anchor positions, document lengths, value/stance patterns, and contradiction
relationships. Evaluation uses opaque document and faction IDs, randomized
order, neutral titles/styles, shuffled sentence locations, three to five
factions, and varied stance vectors.

The 18 helper-built held-out families use separate primary documents for each
issue, a separate authored contradiction-related document, and a plant-free
operations ledger. Each document receives an independently sampled number of
seeded, compositionally generated type-neutral administrative distractors
before sentence order is randomized. There is no fixed distractor sentence
pool or equal sentence-count signature. Generated neutral sentences
independently use one or two clauses from the same procedural vocabulary as
authored policy sentences, reducing sentence length itself as a shortcut. The
two independently authored held-out families retain their own layouts. The
floor table reports a source-aware attack that
excludes those predicates, their planted/distractor prevalence, and the
absolute prevalence gap rather than assuming compositional generation removed
lexical source signals. Accepted decision actors must be prompt-visible in the
relevant document; missing authored roles receive classified actor-support
evidence. Actor support is required candidate evidence and survives optional-
distractor density filtering. Bounded document views discard neutral spans
first and remove any issue and visible anchor whose complete actor support
still cannot fit. A separate source-aware probe flags every exact sector-team
phrase: on the candidate split it has `0.000` localization recall and F1, with
planted/distractor marker rates of `0.000`/`0.054`. A genuine prompt-only probe
that selects the three longest visible sentences has `0.000` strict F1,
`0.087` localization recall, and `0.023` conditional type accuracy on the exact
candidate split.

Strict semantic F1 alone can understate a locator shortcut: a probe may select
the right sentence and still fail because its deliberately generic type or
diagnosis is wrong. The 0.6 audit therefore reruns the retired-marker,
layout/length/position, longest-sentence, and sector-team locators as
localization component oracles. It preserves every predicted span and false
positive, but fills type, diagnosis, decision frame, and relationship from the
answer key only when a predicted span passes the scorer's document,
contiguity, coverage, precision, and overlap gates. The resulting strict F1 is
a precision-sensitive measure of localization rather than allowing
downstream semantic failure to mask it. These rows are diagnostics with gold
non-localization components, not deployable prompt-only baselines.

The corpus audit also tests interactions among public structural features. A
frozen one-nearest-neighbor classifier uses neutral title root, normalized
anchor position, sentence count, and word count, with all variants of the test
template family excluded from fitting. It reports macro recall (balanced
accuracy) over one planted-anchor evaluation unit (300 anchors: 100 each for
ambiguity, contradiction, and gap), observed issue-class proportions, and the
exact `1/3` chance reference. For the 18 helper-built five-document families, an
analogous audit uses one document as the evaluation unit (450 documents) and
labels the three issue-primary roles, the separate contradiction-related role,
and the plant-free role. It reports five-way document-role and
binary related-versus-other leave-one-template-out balanced accuracy alongside
the exact 0.20 role shares, 0.20 related-document prevalence, and corresponding
0.20/0.50 balanced-chance references. These are bounded source-development
attacks, not evidence that every prompt-observable classifier is weak.

The candidate source includes the generator, templates, seeds, and answer
construction, so a source-aware deterministic replay can reconstruct the
bundled answers. That row is reported as a source-aware memorization ceiling, not a
prompt-observable floor. Prompt-observable baselines may use only the rendered
task. Neither class establishes transfer to an independently implemented
generator.

The split audit removes opaque identity and neutral layout before hashing
policy propositions, unresolved decisions, evidence relations, and faction
stances. It also measures cross-split token-set Jaccard and word-ngram TF-IDF
cosine. These deterministic lexical audits do not replace an embedding audit
or a fresh independently implemented private generator.

After the candidate source and scorer are frozen, a separate independently
constructed attack holdout is required for final shortcut claims. A failed
holdout reopens the candidate and invalidates that holdout for final reporting;
it must not become another development set through iterative tuning.

Every faction has a general value vector over access, adaptability, continuity,
oversight, and safety. Its prompt-visible description renders the exact signed values
used by the synthetic generator, followed by issue-independent prose; it cannot
inspect planted issues or target stances. Each issue defines a separate value
trade-off and an explicit yes-side (`anchor` or `alternative`).
The generator composes alternative stances from the values, then orients them
to question polarity. Counterfactual tests require polarity reversal to invert
agree/disagree and require issue-trade-off changes to leave prompt-visible summaries
unchanged. A source-aware parser for the removed 0.4 phrase table scores one on
its frozen historical fixture and zero on current data.

Contradiction relationships are explicit authoring data for all 24 template
families. Generation may remap opaque document IDs but must preserve the
authored document/quote pair. It never selects a second passage by lexical
overlap. Validation rejects absent or same-document evidence and any collision
with another planted anchor or distractor.

Find separates evidence localization, type, diagnosis, relationship, and
end-to-end success. A valid result requires a precise contiguous passage,
correct type, a source-authored decision frame, a well-formed yes/no
diagnosis that expresses the same frame, and—only for contradictions—the
authored second passage from another document. The decision frame names the
actor, action, condition, primary-rule outcome, and alternative outcome.
Each planted issue has a prompt-hidden, candidate-bundled, hand-authored
five-slot reference plus complete per-slot accepted-concept lists. Scoring uses
normalized, non-verbatim concept coverage against those concepts: actor,
action, and condition retain their separate roles, while primary and
alternative outcomes remain separately oriented and bound to the cited source
passages where applicable. Candidate question prose is never added to the
evidence used to validate its own frame.
This rejects both underspecified keyword
fragments and long evidence-word dumps. Free-form diagnosis wording is not
required to equal one canonical sentence, but its structured meaning is still
checked against the authored reference concepts.
Unknown issue types and duplicate normalized spans invalidate the complete
response, preventing fail-open type coercion and three-type hedging.
The exact Find primary response root requires `findings` and may include the
requested cardinality of companion `questions`. The latter is logged at weight
zero and validated independently: omission or companion malformation yields a
zero companion metric without invalidating otherwise correct findings.

The release gate includes a separately authored, two-sided semantic audit
rather than testing only generator-provided aliases. Its 23 valid cases cover
all five decision-frame slots across ambiguity, contradiction, and gap, plus
passive voice, grounded negative conditions, consistent yes/no reversal, and
reference-question rewording. Its 31 invalid cases cover every slot and issue
type, actor/action swaps, anchor/alternative swaps, unsupported exceptions and
thresholds, polarity and orientation mismatches, and an incorrect
contradiction relationship. Acceptance and rejection are asserted separately
with issue/slot-specific test IDs; this is deterministic contract coverage,
not evidence that the matcher recognizes unrestricted natural-language
paraphrases.

Ask exposes three unordered candidate profiles and requires the single best
one. Each profile contains its canonical five-field decision and exact signed
alternative-side trade-off weights. The prompt states the faction-preference
composition, pass threshold, yes-side orientation, `decision_value`,
disagreement, and utility-ranking formulas. It omits evidence locations, issue
types, relationships, accepted aliases, stored stances, decision values, and
utilities. The response must provide exact visible evidence, issue type, the
five-field decision frame, yes/no prose, an explicit yes-side orientation,
relationship evidence when applicable, and a complete faction stance map with
exactly the visible faction IDs. Canonical question text and aliases are not
lexical scoring targets. A response may copy the visible canonical frame or use
a source-grounded accepted alias, but passage evidence independently anchors
the issue and applicable outcomes. The profile and response prose cannot
validate their own frame. Generic wording, unsupported semantic clauses, and an
unchanged-prose polarity flip receive no semantic credit.

Utility combines structured grounding, stance accuracy, panel disagreement,
and a continuous issue value derived from the strength and spread of faction
preferences on the public candidate trade-off. With the default K=1 budget, it
is normalized by the best attainable single-item utility, so an exact response
scores one. Release audits report raw and normalized top-one margins, the exact
uniform-random expectation, exact runner-up reward, public-profile composition,
and top-one selection accuracy for each selection diagnostic. Separate metrics
distinguish question form, top-one selection, semantic grounding, end-to-end
grounded stance recall, exact-evidence match recall, and stance accuracy over a
single deterministic assignment of exact-evidence matches. Reporting both
evidence coverage and conditional accuracy prevents a zero-match run from
looking like a meaningful stance result. The evidence-matched accuracy can
remain high when a model finds the right passage and predicts stances but fails
the semantic decision-frame gate. Selecting one of three remains a small
clarification-target selection task, not measured real-world information gain.

Find's default reward remains strict end-to-end F1. Optional shaped mode
averages localization, type, diagnosis, and relation-evidence F1 scores. Every
unmatched candidate is a false positive at every stage, so exact-plus-spam and
overlapping hedges score below a concise exact response. It must not be used for
reported strict evaluation scores. Multi-seed shaped-versus-strict learning and
transfer to a fresh private generator remain unproven.

No judge model is called. The deterministic rewards establish conformance to
these structured synthetic contracts, not prose quality, real organizational
usefulness, or human-deliberation validity.

## Evaluation design

`configs/eval/baseline-sweep.toml` specifies 100 tasks and five rollouts. The
100 tasks provide structural coverage; repeated rollouts only estimate sampling
variation. Model comparisons should preserve per-task pairing. Elicit analyses
resample base templates and then variants within selected templates; Predict
resamples tasks. Reports must not treat all 500 observations as independent
and should include source commit, artifact hashes, environment
content hash, exact config, provider/model identifiers, sampling settings,
retry/error fields, and per-example traces.

The 0.6 primary study admits no recovered/error histories. A failed provider
attempt is retained separately, then the complete predetermined model/task
stratum is rerun from the frozen configuration. Individual episodes are never
selectively replaced, and replacement decisions cannot depend on score. Exact
extended answer serialization is gated at no more than 1,024 `cl100k_base`
tokens against a 2,048-token generation budget; pilot completion-length
distributions are reported separately.

The 0.6 analyzer also applies the same clustering to diagnostic metrics. Paired
model comparisons use cluster-level sign-flip randomization under a sign-
symmetric/exchangeable paired cluster-effect null; equality of means alone is
not sufficient. With four models and three primary tasks, the predeclared global
Holm family contains exactly 18 pairwise contrasts. Prompt-ablation contrasts
and component diagnostics are excluded. Their 95% percentile intervals are
pointwise and unadjusted, not simultaneous confidence bands. All comparisons
remain exploratory rather than confirmatory. At least four model families,
including open-weight trainable models, should be
run before comparative claims. Training-utility claims require multiple
independent training seeds and evaluation on a procedurally fresh or private
generator family.

## Public-answer and human-data limits

All bundled data is synthetic and every evaluation key is included in the
candidate source for reproducibility and open training if released. Same-row
evaluation after training is not a valid generalization test. The
operator-authored Predict demo fixture is also synthetic and open-answer.

Non-synthetic custom input is a separate governed path. Automated validation
can enforce metadata and structural requirements but cannot prove consent,
privacy, or redistribution authority. See `docs/human-data-governance.md`.

## Schema identity

Scenario 0.6 data uses Draft 2020-12 schema identifier
`urn:commonground:schema:scenario:5`. Consumers should call
`commonground_scenarios.load_scenario_schema()` rather than dereferencing a
website. Version 5 adds authored structured decision frames to the scenario
contract; explicit contradiction relationships introduced earlier remain
required. Provenance labels the revised corpus
`train-template-layout-profile-v6` or `heldout-template-layout-profile-v6`, and
changes `persona_panel.vote_rule` from `value-composition-v1` to
`value-composition-v2`.
