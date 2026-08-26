# Changelog

All notable changes to this project are documented in this file.

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
