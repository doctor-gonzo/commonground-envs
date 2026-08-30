# Changelog

All notable changes to this project are documented in this file.

## 0.5.0 - Unreleased

- Replace heuristic contradiction-passage discovery with explicit authored
  document/quote relationships for all 24 Elicit template families. Regenerate
  both corpora and add corpus-wide validation that opposing evidence cannot be
  another planted issue or distractor. This corrects the five affected
  `regional-archives-access` evaluation rows in 0.4.1.
- Replace the finite `(issue type, target stance) -> phrase` renderer with
  issue-independent faction value descriptions. Compose issue stances from
  general latent values and explicit alternatives, record question polarity,
  and add polarity-reversal and summary-invariance regressions.
- Make shaped Find training precision-sensitive by averaging stage F1 scores
  instead of stage recalls. Unmatched and overlapping hedge candidates now
  lower reward; concise exact answers strictly beat exact-plus-spam answers.
- Replace Ask's hidden canonical-token gate with a structured target containing
  evidence, type, yes-side polarity, contradiction relationship, and faction
  stances. Add separate format, grounding, stance, diagnosis, and relationship
  diagnostics so semantic failures are distinguishable from formatting errors.
- Add uniform, empirical-prior, visible-frequency, smoothed neighbor-frequency,
  distance-weighted k-NN, and train-split text-only probability comparators for
  Predict. Report normalized Brier and Brier skill relative to uniform alongside
  probability reward and accuracy.
- Bump both Hub environments to 0.5.0, both shared packages to 0.4.0, and the
  scenario schema to `urn:commonground:schema:scenario:3`. A fresh immutable
  artifact study and new Elicit evaluation records are required before release.

## 0.4.1 - 2026-08-30

- Include byte-identical legal files under each environment's nested
  `LICENSES/` directory so Prime CLI source archives retain Apache-2.0 text and
  Predict's provenance notice. Add a regression model of the Prime 0.6.28 and
  0.6.29 archive collector. Task code, corpora, and scoring are unchanged from
  the private 0.4.0 candidate.
- Complete the exact-private-artifact study across four model families and
  three task modes: 6,000 clean rollouts, zero recovered/error history, and
  paired task or template-hierarchical bootstrap intervals. Document the
  sparse Elicit rewards and model-free comparator boundary without claiming
  demonstrated training value.

## 0.4.0 - 2026-08-29 (private candidate)

- Remove Elicit's public issue-signature and exact-stance clauses. Faction
  summaries now expose indirect policy principles, and the prompt-only 0.3
  summary-codebook baseline falls from 0.787 on the historical corpus to 0.000.
- Separate exact-instance, canonical-prompt, and policy-issue fingerprints;
  block cross-split semantic overlap and add token-Jaccard and word-ngram
  TF-IDF nearest-neighbor audits. Rename Elicit split provenance as
  template/layout profiles rather than independent generator families.
- Replace the hidden Ask keyword-value table with simulated answer coverage
  combined with faction disagreement. Add optional staged Find training reward
  while retaining strict end-to-end F1 as the default evaluation objective.
- Make Predict's primary reward `1 - normalized Brier`, retain argmax accuracy
  and Brier as diagnostics, and reject missing, extra, duplicate-normalized, or
  malformed prediction maps.
- Add hierarchical Elicit uncertainty analysis that resamples 20 base
  templates and then their five variants; record corrected historical 0.3
  intervals and the measured 0.3 leakage baseline.
- Bump both Hub environments to 0.4.0 and both shared packages to 0.3.0. A new
  exact-artifact model study is required before the candidate is published.

## 0.3.0 - 2026-08-28

- Replace both public evaluation corpora with 100-row, breaking benchmark
  revisions and expand their synthetic training splits to 200 Predict rows and
  100 Elicit rows.
- Make Predict votes depend on explicit statement dimensions and use disjoint
  training/evaluation profile-generator families; require three-class
  probabilities, normalize multiclass Brier to `[0,1]`, and publish 1-NN/5-NN
  alongside clearly labeled held-out and generator diagnostics.
- Remove Elicit's fixed document/faction codebook with opaque identifiers,
  randomized visible structure, varying faction counts, and varied stance
  patterns; add prompt, answer, structural, and cross-split integrity audits.
- Redesign Find around precise evidence, semantic diagnosis, paired
  contradiction evidence, duplicate-span rejection, and separate localization
  and type metrics.
- Redesign Ask as normalized top-two selection from three candidate issues,
  strengthen the semantic decision-term gate, and make an exact response attain
  1.0 under every polarization setting.
- Bump `commonground-score` and `commonground-scenarios` to 0.2.0 for the
  breaking score and scenario-schema contracts.
- Complete the exact-private-artifact 100-task/five-rollout study across four
  model families and three task modes (6,000 clean rollouts); add deterministic
  paired task-cluster bootstrap intervals and preserve complete run provenance.

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
