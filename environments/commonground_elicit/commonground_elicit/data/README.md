# commonground-elicit dataset card

This card describes the unreleased 0.6.0 candidate corpus. Version 0.5.0
remains the current public artifact; no 0.6 model, training, Hub, or public
evaluation evidence is claimed yet.

## Scope and provenance

Both bundled splits are fictional and synthetic. Every row records a seed,
template ID/set, template/layout profile, fixed generation date, and generation
mode. The offline generator calls no model or network service.

## Planting and visible structure

Each scenario contains an ambiguity, a cross-document contradiction, and an
uncovered case. The hidden key records exact primary evidence, issue type, a
yes/no authoring question, the yes-side polarity, value trade-off, composed
faction stances, decision value, and explicitly authored second evidence for a
contradiction.

Prompt-visible document and faction IDs are opaque. Titles, styles, document
order, sentence order, faction order, faction count, and stance patterns vary
by row. Public faction summaries print the exact signed values used by the
synthetic generator—access, adaptability, continuity, oversight, and
safety—then render them as issue-independent prose. The generator derives issue
stances compositionally from those values and explicit alternatives; it does
not render a phrase selected by hidden issue type or stance.

Every contradiction template authors a stable second document and anchor.
Generation remaps document IDs while preserving that relationship. Validation
rejects a related passage that is absent, same-document, another planted
anchor, or a distractor.

The 18 helper-built held-out families each contain five documents: three
primary issue documents, a separate contradiction-related document, and a
plant-free operations ledger. Each document receives an independently sampled
number of seeded, compositionally generated type-neutral administrative
distractors before layout randomization. There is no fixed distractor sentence
pool or equal sentence-count signature. Neutral spans independently use one or
two clauses from the same procedural-component vocabulary as authored spans,
which balances sentence lengths without introducing a second marker. Accepted
decision actors are prompt-visible in the relevant document; missing authored
roles receive classified actor-support evidence instead of stamping one
sector-team phrase onto planted anchors. Actor support is required candidate
evidence, not optional distractor noise: density filtering affects only true
neutral distractors, and bounded document views remove an issue and its anchor
if they cannot preserve every accepted actor alias. The release audit reports
both source-marker rates and source-aware predicate-exclusion and sector-team
attacks. The two
independently authored held-out families keep their own document layouts.

## Split separation

| Split | Rows | Template set | Variants per template | Template/layout profile | SHA-256 |
| --- | ---: | --- | ---: | --- | --- |
| `train_synthetic.jsonl` | 100 | 4 training templates | 25 | `train-template-layout-profile-v6` | `04487e9a5ccb7642aab30b2d60e02c2c894b941781a150e6e208faeb5242b90d` |
| `eval_synthetic_heldout.jsonl` | 100 | 20 held-out templates | 5 | `heldout-template-layout-profile-v6` | `25a63e6f548c1a630cd074921c1e5142f14e4057e433097369bd888fd537ebe6` |

Generation distinguishes instance, canonical-prompt, and policy-issue
fingerprints and blocks cross-split overlap. It also enforces token-Jaccard and
word-ngram TF-IDF nearest-neighbor thresholds, profile-label separation, and
the absence of legacy held-out document IDs. Structural signatures separately
capture layout, counts, value/stance patterns, and evidence relationships.
Both profiles share one core generator. The committed files regenerate
byte-for-byte. Rows conform to `urn:commonground:schema:scenario:5` and declare
`persona_panel.vote_rule: value-composition-v2`.

## Find and Ask targets

Find requires localized primary evidence, a single issue type, a five-field
decision frame, a consistent yes/no diagnosis, and authored paired evidence for
contradictions. Its gold frame and accepted aliases remain in the answer key.
Ask requires exactly one of three visible candidate issues plus the same
decision frame, explicit yes-side orientation, relationship evidence, and
complete faction stance predictions. Ask renders one unordered public profile
per candidate containing its canonical five-slot decision and exact signed
alternative-side trade-off weights, as well as the pass threshold, composition,
orientation, and ranking semantics needed to select it. Evidence locations,
issue labels, relationships, accepted aliases, stored stances, decision values,
and utilities remain hidden. Free-form question prose is not required to equal
a canonical question. Normalized non-verbatim matching keeps actor, action,
condition, and the two outcomes in their authored roles. The candidate profile
and response question are never evidence for their own frame; cited passages
independently anchor the issue and applicable outcomes.
Unknown issue types and malformed decision objects fail the complete response
contract closed. Ask reward is normalized by the highest attainable
single-item continuous preference-trade-off utility on that row.

## Model-free comparators

The 0.6 candidate suite includes prompt-observable probes for the removed
filler/type and equal-length markers, document length and sentence position,
legacy shortcuts, and simple Ask strategies. It separately reports component
oracles and selection diagnostics that read specified hidden fields. Neither
class should be confused with a deterministic reconstruction using the
candidate source's generator, templates, seed, and answer construction; that
source-aware replay
is a memorization ceiling, not a prompt-observable floor.

On the completed local candidate, every prompt-observable strict Find shortcut
scores `0.000`; the layout probe reaches `0.200` localization recall and
`0.130` conditional type accuracy. The prompt-only three-longest-sentences
probe has `0.087` localization recall and `0.023` conditional type accuracy.
Excluding the shared procedural predicate
pool scores `0.000` for strict Find, localization recall, and conditional type
accuracy; planted anchors and distractors both carry those predicates at rate
`1.000`. Selecting every source-visible sector-team marker also scores `0.000`
strict Find, localization recall, and localization F1. The marker occurs in
`0.000` of planted anchors and `0.054` of distractors. Exact uniform-random
selection has expected reward `0.557` and top-one accuracy `0.333`; exact
runner-up selection has reward `0.398` and top-one accuracy `0.000`. Public
profile composition reaches `1.000` reward and top-one accuracy. Exact top-one
selection with random and exact stances scores `0.655` and `1.000`. The top-one
tie rate is `0.000`, with mean/minimum raw gaps `0.133`/`0.023` and
mean/minimum normalized margins `0.602`/`0.144`. The source-aware replay reaches
`1.000` on both tasks, as expected for a deterministic open generator. These
values must reproduce from the exact release artifact; see the environment
README for the full provenance-separated table. The answer keys are bundled in
the candidate source and would become public with publication,
so consequential transfer claims require an independently implemented private
generator.

The answer keys are bundled in the candidate source. If published, this corpus
is for reproducibility, open training, and experimental evaluation—not a
contamination-resistant leaderboard.
