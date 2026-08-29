# commonground-scenarios

Deterministic, offline scenario generation for the Common Ground environment
family. Each generated scenario contains a fictional organization, policy
documents, planted ambiguities/contradictions/gaps, precise distractors, a
seeded persona panel, and provenance.

Every canonical clarifying question is phrased as a yes/no proposition. Its
`target_stances` have an explicit interpretation: `agree` predicts yes,
`disagree` predicts no, and `pass` means the faction takes no position.
Templates may also author a finite `canonical_question_aliases` allowlist.
Aliases obey the same yes/no form and are kept distinct across the answer key.

The default generator uses committed templates only. An operator may inject a
prose-polishing callback, but no model client or network path is built into the
package, and validation requires every planted anchor to survive polishing.

Training and held-out template registries and template/layout-profile labels are
disjoint. The canonical release contains 100 training and 100 held-out rows.
The generator assigns opaque document/faction IDs, varies visible layout and
faction count/order, and records decision terms, answer-conditioned value, and
paired evidence for contradictions. Split generation distinguishes exact
instance, canonical prompt, and policy-issue fingerprints; it also runs
token-Jaccard and word-ngram TF-IDF cross-split neighbor audits. Generated
scenarios are canonical JSON and regenerate byte-for-byte from the same
template, seed, and explicit generation date.

The bundled scenarios are synthetic. The optional `human_feedback`
field supports advanced custom input and delegates to the shared fail-closed
validator. It is not used by the bundled data; see
[human-data governance](https://github.com/doctor-gonzo/commonground-envs/blob/master/docs/human-data-governance.md)
for its separate review requirements.

The packaged Draft 2020-12 schema has the stable identifier
urn:commonground:schema:scenario:2 and is available through
load_scenario_schema.
