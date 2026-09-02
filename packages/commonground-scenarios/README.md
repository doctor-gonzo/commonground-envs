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
Canonical question wording remains authoring metadata, not a hidden lexical
requirement in environment scoring.

The default generator uses committed templates only. An operator may inject a
prose-polishing callback, but no model client or network path is built into the
package, and validation requires every planted anchor to survive polishing.

Training and held-out template registries and v6 template/layout-profile labels
are disjoint. The 0.6 candidate contains 100 training and 100 held-out rows.
The generator assigns opaque document/faction IDs, varies visible layout and
faction count/order, prints round-trip-exact faction values in prompt-visible
summaries, and records question polarity, continuous preference-trade-off
value, and explicitly authored paired evidence for every contradiction. Each
planted issue also carries a hand-authored five-slot decision reference for
Ask. Find intentionally omits that reference and scores visible evidence, issue
type, valid yes/no diagnosis form, and contradiction relationships. Ask renders
the canonical five-slot frame and exact signed trade-off weights for every
unordered candidate while retaining evidence keys, issue labels, relationships,
stored stances, decision values, and utilities as answer metadata. Validation
rejects authored relationships that collide with another planted issue or
distractor.

Split generation distinguishes exact-instance and policy-issue fingerprints;
it also runs one word-ngram TF-IDF cross-split neighbor audit. Generated
scenarios are canonical JSON and regenerate byte-for-byte from the same
template, seed, and explicit generation date.

The 18 helper-built held-out families separate each primary issue, authored
contradiction relationship, and plant-free ledger into five documents. Authored
policy sentences remain intact. Seeded type-neutral administrative distractors
are composed separately before sentence order, title, and style randomization.
Each procedural predicate is used at most once within a scenario, preventing
filler from inventing repeated cross-document responsibilities. Balanced filler
counts and seeded sentence shuffling keep issue position from becoming a fixed
first-clause rule. Historical reports retain earlier shortcut audits; the active
gate keeps compact prompt-only probes and a top-one utility diagnostic for the
default K=1 Ask budget. Bounded views retain only issues whose authored anchor
remains visible.

The bundled scenarios are synthetic. The optional `human_feedback`
field supports advanced custom input and delegates to the shared fail-closed
validator. It is not used by the bundled data; see
[human-data governance](https://github.com/doctor-gonzo/commonground-envs/blob/master/docs/human-data-governance.md)
for its separate review requirements.

The packaged Draft 2020-12 schema has the stable identifier
`urn:commonground:schema:scenario:5`, requires
`persona_panel.vote_rule: value-composition-v2`, and is available through
`load_scenario_schema()`.
