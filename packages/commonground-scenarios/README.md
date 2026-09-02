# commonground-scenarios

Deterministic, offline scenario generation for the Common Ground environment
family. Each generated scenario contains a fictional organization, policy
documents, planted ambiguities/contradictions/gaps, precise distractors, a
seeded persona panel, and provenance.

Every canonical clarifying question is phrased as a yes/no proposition and
declares whether “yes” selects its primary `anchor` or an `alternative`.
`agree` predicts yes, `disagree` predicts no, and `pass` means the faction takes
no position. Stances are composed from issue-independent faction values and an
issue-specific value trade-off, then oriented to that declared polarity.
Canonical wording and aliases remain authoring metadata, not hidden lexical
requirements in environment scoring.

The default generator uses committed templates only. An operator may inject a
prose-polishing callback, but no model client or network path is built into the
package, and validation requires every planted anchor to survive polishing.

Training and held-out template registries and v6 template/layout-profile labels
are disjoint. The 0.6 candidate contains 100 training and 100 held-out rows.
The generator assigns opaque document/faction IDs, varies visible layout and
faction count/order, prints round-trip-exact faction values in prompt-visible
summaries, and records question polarity, continuous preference-trade-off
value, and explicitly authored paired evidence for every contradiction. Each
planted issue also carries a hand-authored five-slot decision reference plus
bounded, role-specific aliases whose first entry is the canonical phrase. Find accepts
only those aliases under the documented text normalization. Ask renders the
canonical five-slot frame and exact signed trade-off weights for every unordered
candidate while retaining aliases, evidence keys, issue labels, relationships,
stored stances, decision values, and utilities as answer metadata. Validation
rejects authored relationships that collide with another planted issue or
distractor.

Split generation distinguishes exact-instance and policy-issue fingerprints;
it also runs one word-ngram TF-IDF cross-split neighbor audit. Generated
scenarios are canonical JSON and regenerate byte-for-byte from the same
template, seed, and explicit generation date.

The 18 helper-built held-out families separate each primary issue, authored
contradiction relationship, and plant-free ledger into five documents. Seeded
type-neutral administrative distractors are generated compositionally in
independently sampled counts before sentence order, title, and style
randomization. There is no fixed distractor pool or equal sentence-count
signature. Neutral spans independently use one or two clauses from the same
seeded procedural-component distribution as authored rules, so predicate
vocabulary alone is not an exclusive distractor marker. Clause ordering still
leaves a source-aware first-clause localization shortcut, which limits what
localization scores demonstrate. Historical corpus audits report the exclusion
attack, the planted/distractor predicate-rate gap, a source-aware
sector-team-marker attack, residual layout-to-label signal, and top-one utility
ties for the default K=1 Ask budget. Every accepted actor is prompt-visible in
the relevant document; when the authored rule does not name it, classified
actor-support evidence does. Actor support is required evidence and is retained
independently of optional neutral-distractor density; bounded views remove an
unobservable issue and its anchor rather than exposing an unreachable key.
This avoids placing one sector-team marker on every planted anchor.
Scoped rows preserve the complete generated scope in every condition alias.

The bundled scenarios are synthetic. The optional `human_feedback`
field supports advanced custom input and delegates to the shared fail-closed
validator. It is not used by the bundled data; see
[human-data governance](https://github.com/doctor-gonzo/commonground-envs/blob/master/docs/human-data-governance.md)
for its separate review requirements.

The packaged Draft 2020-12 schema has the stable identifier
`urn:commonground:schema:scenario:5`, requires
`persona_panel.vote_rule: value-composition-v2`, and is available through
`load_scenario_schema()`.
