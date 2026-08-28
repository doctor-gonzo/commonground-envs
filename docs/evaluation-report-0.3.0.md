# Common Ground 0.3.0 evaluation report

Status: release candidate; hosted model results pending.

This report records the model-free evidence for the breaking 0.3.0 redesign.
It intentionally contains no 0.2.x model scores: both corpora and both response
contracts changed, so those runs are not comparable.

## Candidate scope

| Environment | Eval rows | Train rows | Primary reward | Important diagnostics |
| --- | ---: | ---: | --- | --- |
| `commonground-predict` | 100 | 200 | `vote_accuracy` | normalized `brier` |
| `commonground-elicit:find` | 100 | 100 | `finding_f1` | localization recall, type accuracy, question utility |
| `commonground-elicit:elicit-ask` | 100 | 100 | normalized `question_utility` | exact top-K attainability |

## Predict comparators

| Comparator class | Comparator | vote_accuracy |
| --- | --- | ---: |
| Prompt-observable | Always agree | 0.590 |
| Prompt-observable | Per-statement visible majority | 0.581 |
| Prompt-observable | Nearest participant (1-NN) | 0.819 |
| Prompt-observable | Five-neighbor vote | 0.891 |
| Held-out-label diagnostic | Per-snapshot best constant | 0.631 |
| Generator diagnostic | Latent cluster-pattern replay | 0.916 |

These exact values come from `scripts/compute_floors.py` on the committed
100-row evaluation file with eight masked cells per snapshot. Only the first
four rows are prompt-observable. The 5-NN result shows that matrix structure
remains the dominant comparator; any model claim must report it.

## Elicit comparators

| Comparator class | Task | Comparator | mean reward |
| --- | --- | --- | ---: |
| Prompt-observable | find | Random visible spans | 0.030 |
| Prompt-observable | find | Flag vague-sounding spans | 0.140 |
| Prompt-observable | find | Legacy 0.2 document-ID/position codebook | 0.000 |
| Prompt-observable | elicit-ask | Template clarity questions | 0.000 |
| Prompt-observable | elicit-ask | Randomly targeted questions | 0.000 |
| Component oracle | elicit-ask | Exact top-K issues + random stances | 0.655 |
| Component oracle | elicit-ask | Exact top-K issues + visible-summary stances | 1.000 |

These values come from `scripts/compute_elicit_floors.py`. The legacy-codebook
regression directly tests the critical 0.2 shortcut. The other prompt baselines
are weak floors, not a complete difficulty ladder. The component oracles use
hidden top-K issue targets; their gap isolates stance inference and confirms
that public faction summaries support exact stance recovery.

## Integrity evidence

- Predict train/eval text, session IDs, seeds, and generator families are
  disjoint; statement dimension changes alter generated votes.
- Elicit has 100 distinct prompt hashes and answer hashes per split, no exact
  cross-split prompt/answer overlap, disjoint generator families, opaque IDs,
  and varied structural signatures.
- Exact Elicit answers attain 1.0. Type hedging, missing contradiction pairs,
  generic gap labels, broad evidence, duplicate spans, and one-noun questions
  are regression-tested.
- All bundled data is synthetic and public-answer.

## Required hosted study before publication claims

Run the committed 100-task/five-rollout configuration for Predict, Find, and
Ask across at least four model families, including an open-weight model. Record
source commit, wheel and Hub hashes, full config/provider parameters,
per-example traces, retries/errors, and public evaluation IDs. Report paired
task-level differences or task-cluster bootstrap intervals; do not divide a
500-rollout standard deviation by `sqrt(500)` as though rollouts were
independent tasks.

Useful next comparators are matrix factorization/spectral clustering plus
text-only, matrix-only, and shuffled-text ablations for Predict, and component
oracles plus stronger lexicon/stance-prior baselines for Elicit. Training-value
claims additionally require at least three training seeds and transfer to a
fresh private generator family.

## Release attestation fields

Fill these only after building and privately validating the exact artifacts:

| Field | Value |
| --- | --- |
| Source commit | pending |
| `commonground-score` 0.2.0 wheel/sdist SHA-256 | pending |
| `commonground-scenarios` 0.2.0 wheel/sdist SHA-256 | pending |
| `commonground-predict` 0.3.0 wheel/Hub content SHA-256 | pending |
| `commonground-elicit` 0.3.0 wheel/Hub content SHA-256 | pending |
| Hosted evaluation IDs and configs | pending |
