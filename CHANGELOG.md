# Changelog

All notable changes to this project are documented in this file.

## 0.2.3 - 2026-08-27

- Reissue both Hub environments with closure-scoped dependency provenance so
  an unrelated workspace member's version bump cannot mutate an already
  published sibling artifact.
- Align the inspectable Predict source README and dependency manifest with the
  audited repository state before public visibility.

## 0.2.2 - 2026-08-26

- Replace Elicit-ask's unreachable hidden canonical-sentence gate with a
  prompt-visible contract: exact document/quote grounding, strict yes/no
  form, and at least one informative word shared with the quoted passage.
- Split each matched question's utility evenly between issue grounding and
  per-faction stance accuracy, retain disagreement/polarization scaling, and
  use global one-to-one assignment so duplicates earn no second match.
- Add the observed baseline regression: a valid grounded paraphrase with
  partially correct stances now receives partial credit, while changing hidden
  canonical wording cannot change its score.

## 0.2.1 - 2026-08-26

- Restore real legacy `SingleTurnEnv` adapters from `load_environment()` so
  the current Prime Hosted Evaluation runner can execute both packages, while
  preserving the exported native-v1 tasksets used by local evaluation and
  Hosted Training.
- Add `load_taskset()` as the explicit direct-construction helper and bundle a
  pure-chat `NullHarness` with Predict so native-v1 runs do not fall back to the
  bash/code-agent harness.
- Extend source and clean-wheel regression gates across both runner contracts,
  including exact legacy loader calls, taskset/harness plugin resolution, and
  Predict hidden-label non-leakage.

## 0.2.0 - 2026-08-26

- Migrate both environments from deprecated Verifiers v0 to native v1 tasksets,
  rewards, metrics, harnesses, and trace scoring on `verifiers==0.3.0`.
- Redesign the elicit held-out corpus as twenty unique authored semantic tasks
  and require every submitted finding quote to be an ordered contiguous span of
  its claimed visible document with substantial planted-anchor coverage and no
  normalization of meaning-changing symbolic operators.
- Add bounded, fail-closed completion parsing and adversarial reward regressions
  for oversized, deeply nested, fabricated, reordered, and meaning-changing
  output.
- Separate predict training and evaluation policy-text banks, strictly validate
  all snapshot dimensions, masks, labels, votes, clusters, and metadata, and
  remove hidden labels from auxiliary task state.
- Centralize a strict human-snapshot contract with positional pseudonyms,
  k-anonymous cluster partitions, recomputed counts, identifier screening, and
  consent, rights, source, schema, exporter, and privacy-review attestations;
  apply it to every non-synthetic Predict input path before masking. Bound
  snapshot sizes and reject unverified PCA-derived centers and comment metrics.
- Prepare corrected shared packages as 0.1.1 and exactly pin them from the
  0.2.0 environments.
- Include complete license files in every artifact and preserve the Context
  Engine demo fixture's MPL-2.0 notice, immutable source, and hashes.
- Remove colliding top-level project files from wheels, update the lock to
  patched dependencies, ignore local Prime/output trees, and strengthen archive
  and release verification.
- Mark public evaluation answers and the synthetic/human-validity boundary
  explicitly. Historical elicit baselines are not comparable to 0.2.0 and must
  be rerun before a public performance claim.

## 0.1.2 - 2026-08-25

- Make env packages standalone-installable for Hub actions by moving workspace
  sources to the root; add Hub tags metadata.

## 0.1.1 - 2026-08-25

- Add a `split` env-arg for bundled data selection.

## 0.1.0 - 2026-08-16

- Release `commonground-predict`, a deterministic masked-vote prediction
  environment for fictional enterprise AI-deployment policies, and
  `commonground-elicit`, a document-grounded policy-problem finding and
  question-raising environment.
- Include reproducible model-free floor tooling, synthetic train/eval splits,
  shared scoring utilities, and seeded planted-scenario generation.
- Label every bundled data split as synthetic; v0 contains no verified
  real-human data.
- Known limit: `commonground-elicit` question raising uses bounded-recall
  matching against generator-authored questions and stance vectors. Sensible
  unlisted paraphrases can receive zero utility, so the find task remains the
  v0 headline.
