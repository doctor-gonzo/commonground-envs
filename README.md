# commonground-envs

Two deterministic Verifiers environments for structured reasoning over
synthetic group data:

- `commonground-predict` predicts masked group votes as calibrated
  agree/disagree/pass probabilities.
- `commonground-elicit` finds consequential ambiguities, contradictions, and
  gaps in organizational documents, or selects the highest-value clarification
  candidate and predicts faction stances.

Version 0.6.0 is the current public Hub release. Its exact-artifact model
study, artifact hashes, and version-pinned public evaluation records are in
the [0.6.0 evaluation report](docs/evaluation-report-0.6.0.md).

All bundled rows are synthetic and all release answer keys are public. The
environments are suitable for reproducible open training and evaluation, not a
contamination-resistant leaderboard or evidence about real human preferences.

![Common Ground Predict and Elicit use synthetic data in the current release; reviewed human data is shown only as a future option.](docs/assets/commonground-envs-overview.png)

## Packages

| Distribution | Version | Responsibility |
| --- | ---: | --- |
| `commonground-score` | 0.6.0 | Shared vote statistics and proper probability scoring. |
| `commonground-scenarios` | 0.6.0 | Shared synthetic scenario model, generation, schema, and validation. |
| `commonground-predict` | 0.6.0 | Thin Predict Taskset/Harness and legacy adapter. |
| `commonground-elicit` | 0.6.0 | Thin Find/Ask Taskset/Harness and legacy adapter. |

The environments expose native Verifiers v1 plugins and legacy
`SingleTurnEnv` adapters backed by the same rows and scorers.

## 0.6 architecture

- Predict accepts exactly one probability triple for every masked cell and no
  other keys. Its primary reward is `1 - normalized multiclass Brier`; accuracy,
  Brier, and Brier skill are diagnostics.
- Predict includes only four probability-native references: uniform, empirical
  prior, visible frequency, and smoothed neighbor frequency. Four fixed prompt
  views isolate text and matrix contributions.
- Each Elicit issue authors one primary passage and—only for contradictions—one
  explicit opposing passage. Ask additionally exposes one five-slot decision
  profile. No relationship is inferred through token overlap.
- Find scores source grounding, issue type, and required opposing evidence.
  Diagnosis prose is checked only for yes/no form; it has no hidden vocabulary
  or structured-decision answer key.
- Ask publishes unordered decision profiles and the value-composition rule.
  It scores candidate selection, exact evidence, an exact copied profile,
  explicit `yes_choice`, and faction stances. Question prose is presentation.
- Opaque identifiers and JSON keys are exact. Ask decision strings and evidence
  normalize only Unicode compatibility, case, and whitespace. Find grounding
  uses a bounded contiguous-span match against visible source text and accepts
  a contradiction's two passages in either order.
- Faction descriptions come from general value vectors that are independent of
  the current issue labels. Stances are composed from those values and reverse
  consistently when the yes-side alternative reverses.
- Find's optional shaped reward is precision-sensitive; adding unmatched
  findings lowers the score.
- Split auditing retains exact instance identity, one policy-semantic identity,
  and one word-ngram TF-IDF near-neighbor check.
- One local command regenerates splits and summaries, runs tests/static checks,
  builds fresh artifacts, loads packaged tasksets, and records concise hashes.

The normative Elicit response contract is in
[environments/commonground_elicit/README.md](environments/commonground_elicit/README.md).
Method definitions and scientific boundaries are in
[docs/methodology.md](docs/methodology.md).

## Local verification

Run the complete candidate gate from a clean committed worktree:

```bash
bash scripts/local_smoke.sh
```

It does not call Prime, run hosted inference, push artifacts, or publish
anything. Evidence is written to a printed directory under `/tmp`.

Individual development commands remain available:

```bash
uv sync --all-packages --locked
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run python scripts/check_dependency_manifests.py --check
uv run python scripts/check_release_wheel.py
```

## Model-free summaries

```bash
uv run python scripts/compute_floors.py \
  environments/commonground_predict/commonground_predict/data/eval_synthetic.jsonl \
  --masked-vote-count 8 \
  --train-split environments/commonground_predict/commonground_predict/data/train_synthetic.jsonl

uv run python scripts/compute_elicit_floors.py
```

Predict reports probability-native references. Elicit reports two prompt-only
Find probes, a uniform Ask candidate reference, a runner-up diagnostic, exact
answer ceilings, class balance, and the minimum top-one margin. Historical
exploit baselines remain in historical release reports rather than the active
gate.

One fixed-corpus shortcut remains disclosed: one public trade-off vector
identifies the Ask winner in 90 of 100 evaluation rows. It does not supply the
evidence, type, orientation, or faction predictions, but it narrows what model
selection results can demonstrate.

## Evaluation status

The 0.6.0 release passed the local artifact gate and release checklist, and its
frozen exact-artifact model study is published in the
[0.6.0 evaluation report](docs/evaluation-report-0.6.0.md), following the
[0.6 evaluation plan](docs/evaluation-plan-0.6.0.md). Genuinely fresh transfer
evidence is still needed for generalization or training claims. Historical
results remain available in the
[0.5.0 evaluation report](docs/evaluation-report-0.5.0.md) and earlier reports,
but are not 0.6 evidence.

## Context Engine and human data

This benchmark family is associated with
[Context Engine](https://contextengine.sh), an open-source deliberation system
for collecting statements and votes and mapping agreement and disagreement.
A future separately governed exporter could allow consenting individuals and
groups to retain, license, or sell preference datasets from their sessions.
No live-human exporter is included here. Non-synthetic custom inputs remain
subject to the separate fail-closed rules in
[docs/human-data-governance.md](docs/human-data-governance.md).

## Release and license

The public 0.6.0 Hub listings are:

- [`charliethompson/commonground-predict`](https://app.primeintellect.ai/dashboard/environments/charliethompson/commonground-predict)
- [`charliethompson/commonground-elicit`](https://app.primeintellect.ai/dashboard/environments/charliethompson/commonground-elicit)

Exact 0.6.0 Hub version IDs are Predict `b9rznmh0i1bmeeb176j7q2gp` and
Elicit `yff9k0yxhxm9smvaqhelrs1i`; content hashes are in the evaluation
report. See the [public release checklist](docs/public-release-checklist.md).

Repository code is [Apache-2.0](LICENSE). Predict's adapted synthetic Context
Engine demo fixture remains MPL-2.0 and is covered by its packaged notice.
