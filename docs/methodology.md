# Methodology and benchmark scope

## Vote statistics

commonground-score independently implements the one-proportion and smoothed
two-proportion equations used by Polis for comment and group comparison
statistics. The immutable upstream reference is Computational Democracy
Project Polismath at commit 5089c6bef9eb1a1e454beb34354fb29dd0a2b6f0:

https://github.com/compdemocracy/polis/blob/5089c6bef9eb1a1e454beb34354fb29dd0a2b6f0/math/src/polismath/math/stats.clj

The functions prop_test and two_prop_test implement those equations in
independent Python code. The package tests pin numerical fixtures and edge-case
behavior. No Polis source file is copied into this Apache-licensed package.

The broader Polis method is described in:

https://www.e-revistes.uji.es/index.php/recerca/article/view/5516/6558

vote_entropy and cluster_separation are Common Ground metrics. They are not
claimed to be Polis implementations: entropy measures normalized
agree/disagree/pass uncertainty, while separation measures the fraction of
observed faction pairs taking different stances.

## Predict scope

The 0.2.0 predict evaluation is session-heldout and uses a policy-text bank
disjoint from its synthetic training bank. It tests inference of masked votes
from visible votes and statement text under a planted-cluster generator. It
does not establish performance on human deliberation or on a private,
contamination-resistant benchmark.

The bundled evaluation labels are deliberately public. The CE demo fixture is
also synthetic and open-answer; its immutable source, transformation boundary,
hashes, and MPL-2.0 treatment are recorded in the predict package NOTICE.

## Elicit scope

The 0.2.0 elicit corpus contains 40 unique synthetic training tasks and 20
unique synthetic held-out tasks. Semantic fingerprints include document text,
document identity/style, planted anchors and types, canonical questions,
aliases, and faction stances while ignoring seed, organization name, and
document order. Generation fails on a duplicate fingerprint.

Finding reward is document-grounded: a candidate quote must be a normalized,
ordered, contiguous span in its claimed visible document before it can be
matched to an answer-key anchor. Matching then uses longest-common-contiguous
token overlap, requires at least 80% contiguous coverage of the planted anchor,
and retains the 0.5 symmetric overlap-F1 floor. Semantic operators are tokens,
so normalization does not erase negation, inequality, sign, or percentage
changes. Question reward remains deterministic but no longer depends on hidden
authored wording. A candidate must name the planted document, copy its exact
visible anchor, use strict yes/no form, and reuse at least one informative
token from that quote. Global one-to-one assignment prevents duplicate claims.
Half of matched utility comes from issue grounding and half from per-faction
stance accuracy, then the result is scaled by the planted faction disagreement
and configured panel polarization. Neither path calls a judge model.
Consequently, Elicit-ask measures grounded issue selection and stance
prediction; it validates question form and lexical grounding but does not
semantically grade prose quality.

This is a reproducible synthetic policy-reasoning environment, not evidence of
human deliberation validity. Because the public package contains planted answer
keys, consequential comparisons require a private server-side split. The 0.2.0
corpus and 0.2.2 question reward invalidate earlier elicit model baselines;
fresh private-candidate baselines are required before public performance
claims.

## Schema identity

The scenario JSON Schema uses the stable identifier
urn:commonground:schema:scenario:1. Consumers should load the packaged schema
through commonground_scenarios.load_scenario_schema rather than dereferencing a
website.
