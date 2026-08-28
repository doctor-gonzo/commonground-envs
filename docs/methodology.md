# Methodology and benchmark scope

## Shared vote statistics

`commonground-score` independently implements the one-proportion and smoothed
two-proportion equations used by Computational Democracy Project Polismath at
immutable commit `5089c6bef9eb1a1e454beb34354fb29dd0a2b6f0`:

https://github.com/compdemocracy/polis/blob/5089c6bef9eb1a1e454beb34354fb29dd0a2b6f0/math/src/polismath/math/stats.clj

Numerical fixtures and edge cases are pinned in tests; no upstream source is
copied. `vote_entropy` and `cluster_separation` are Common Ground metrics, not
Polis implementations. Multiclass Brier is the conventional squared-error sum
divided by two, bounded to `[0,1]`.

## Predict 0.3.0 construct

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

The per-snapshot best-constant comparator reads held-out labels and is a
diagnostic, not a floor. Latent-pattern replay reads generator state and is a
generator diagnostic, not a model-achievable universal ceiling.

## Elicit 0.3.0 construct

Elicit contains 100 training and 100 evaluation scenarios. The splits use
disjoint generator-family labels. Each split enforces unique prompt-only and
answer-key hashes, and cross-split exact overlap is blocked. A structural
signature records generator family, document and faction counts, issue types,
anchor positions, document lengths, stance multisets, and contradiction-pair
presence. Evaluation uses opaque document and faction IDs, randomized order,
neutral randomized titles/styles, shuffled sentence locations, three to five
factions, and varied stance vectors. A regression baseline that implements the
0.2 document-ID/position codebook scores zero. Row-specific decision tendencies
in public faction summaries keep the varied stance targets model-inferable; an
exact-issue component oracle recovers all stances from those summaries.

The train/held-out family labels select disjoint template registries and some
different ID behavior, but both still share the same core scenario-generation
implementation. This is stronger than the 0.2 structural reuse but does not
satisfy the longer-term goal of several independently implemented generator
families or an embedding-based cross-split similarity audit.

Find separates evidence localization, type accuracy, and end-to-end diagnosis.
A valid result requires a precise contiguous passage, correct type, a yes/no
diagnosis covering the issue's decision terms, and—only for contradictions—a
second contiguous passage from another document. Duplicate normalized spans
invalidate a response, preventing three-type hedging.

Ask exposes three candidate issues and requires the best two. Questions must
be grounded in exact visible evidence, use yes/no form, and cover at least two
decision terms. Utility combines grounding, stance accuracy, disagreement, and
an issue decision-value weight. It is normalized by the best attainable K-item
sum, so an exact response scores one. This is genuine but small top-K selection;
8–12 candidate issues, answer-conditioned faction updates, and measured
information gain remain longer-term work.

No judge model is called. The deterministic rewards establish conformance to
these structured synthetic contracts, not prose quality, real organizational
usefulness, or human-deliberation validity.

## Evaluation design

`configs/eval/baseline-sweep.toml` specifies 100 tasks and five rollouts. The
100 tasks provide structural coverage; repeated rollouts only estimate sampling
variation. Model comparisons should preserve per-task pairing and use a paired
task or task-cluster bootstrap rather than treating all 500 observations as
independent. Reports should include source commit, artifact hashes, environment
content hash, exact config, provider/model identifiers, sampling settings,
retry/error fields, and per-example traces.

At least four model families, including open-weight trainable models, should be
run before comparative claims. Training-utility claims require multiple
independent training seeds and evaluation on a procedurally fresh or private
generator family.

## Public-answer and human-data limits

All bundled data is synthetic and every public evaluation key is included for
reproducibility and open training. Same-row evaluation after training is not a
valid generalization test. The operator-authored Predict demo fixture is also
synthetic and open-answer.

Non-synthetic custom input is a separate governed path. Automated validation
can enforce metadata and structural requirements but cannot prove consent,
privacy, or redistribution authority. See `docs/human-data-governance.md`.

## Schema identity

Scenario 0.3 data uses Draft 2020-12 schema identifier
`urn:commonground:schema:scenario:2`. Consumers should call
`commonground_scenarios.load_scenario_schema()` rather than dereferencing a
website.
