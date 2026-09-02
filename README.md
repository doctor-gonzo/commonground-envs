# commonground-envs

Two open, deterministic Verifiers environments for structured reasoning over
synthetic stakeholder data:

- `commonground-predict` requires calibrated probabilities for masked votes in
  a partially visible stakeholder-by-policy matrix.
- `commonground-elicit` requires grounded diagnosis of policy ambiguities,
  contradictions, and gaps, or selection of the single most valuable
  clarification question from three candidate issues (default K=1 of 3).

Version 0.5.0 is the current public release. This repository contains the
0.6.0 release candidate; its model and training-utility studies, artifact
hashes, Hub versions, and public evaluation records are still pending. Predict
0.4.1 remains a historical
preliminary artifact; Elicit 0.4.1 is retained only for reproducibility because
five evaluation rows used an incorrect contradiction relationship and its
public faction prose encoded stances through a finite phrase table.

Version 0.6 tightens those contracts. Elicit requires a source-anchored,
authored-concept five-field decision frame and explicit yes-side orientation,
exposes the exact faction value profile used by the synthetic generator,
replaces answer-count
question value with a continuous preference-trade-off value, and audits
selection ties and prompt-visible layout shortcuts. The helper-built held-out
families now use separate relationship documents and seeded type-neutral,
compositionally generated distractors with independently varied document
lengths. Predict adds deterministic full, matrix-only,
text-only, and shuffled-text prompt views and fails closed on malformed numeric
values. Public answer keys make every release unsuitable for
contamination-resistant leaderboards.

This is the reinforcement-learning environment family associated with
[Context Engine](https://contextengine.sh), an open-source deliberation system
for collecting statements and votes, mapping agreement and disagreement, and
turning the results into auditable organizational decisions. A future governed
export path could allow consenting individuals and groups to retain, license,
or sell preference datasets derived from their sessions. No such exporter is
included here. Every bundled row is synthetic, including one explicitly
synthetic, operator-authored demo fixture.

![Common Ground Predict and Elicit use synthetic data in the current release; reviewed human data is shown only as a future option.](docs/assets/commonground-envs-overview.png)

## Packages

| Path | Distribution | Purpose |
| --- | --- | --- |
| `packages/commonground-score` | `commonground-score` 0.6.0 | Vote statistics, calibrated rewards, and named-reference Brier skill. |
| `packages/commonground-scenarios` | `commonground-scenarios` 0.6.0 | Offline scenario generation, schema, and validation. |
| `environments/commonground_predict` | `commonground-predict` 0.6.0 | Probabilistic masked-vote prediction. |
| `environments/commonground_elicit` | `commonground-elicit` 0.6.0 | Structured finding diagnosis and top-one clarification. |

Both environments export native Verifiers v1 `Taskset`/`Harness` plugins and a
real legacy `SingleTurnEnv` adapter for the current Prime Hosted Evaluation
runner. The same rows and scoring functions back both interfaces.

## Prime Intellect Hub

- [`charliethompson/commonground-predict@0.5.0`](https://app.primeintellect.ai/dashboard/environments/charliethompson/commonground-predict)
- [`charliethompson/commonground-elicit@0.5.0`](https://app.primeintellect.ai/dashboard/environments/charliethompson/commonground-elicit)

Install the exact reviewed versions with:

```bash
prime env install charliethompson/commonground-predict@0.5.0
prime env install charliethompson/commonground-elicit@0.5.0
```

## What the 0.6 candidate changes

Predict still optimizes `1 - normalized Brier`; argmax accuracy and Brier remain
diagnostics, and the exact masked-cell key set is mandatory. Version 0.6 fixes
the public argmax tie order at agree, pass, then disagree; numeric strings,
booleans, non-finite values, unsafe magnitudes, and non-finite totals fail
closed. A checked-in ablation config runs the same task IDs and answers under
full, matrix-only, text-only, and deterministic shuffled-text prompts. Standard
matrix comparators remain strong, so the supported construct is
semantic-conditioned matrix completion—not general collective-preference
reasoning.

Elicit retains authored contradiction relationships and precision-sensitive
shaped Find reward. Find and Ask require a decision object naming the actor,
action, condition, primary-rule outcome, and alternative outcome. Find keeps
the authored gold frame and its role-specific accepted aliases in the bundled
answer key. Ask instead renders an unordered profile for every candidate with
that candidate's canonical five-field decision and exact signed trade-off
weights, together with the pass threshold, stance-composition rule, and ranking
formula. Ask does not expose evidence locations, issue types, relationships,
accepted alias sets, stored stance labels, decision values, or utilities.
Primary and contradiction outcomes remain bound to source passages, candidate
prose is never evidence for its own frame, and accepted concepts cannot migrate
between roles. The yes/no prose must express the same decision and its declared
orientation; generic, polarity-inconsistent, or unsupported-clause-appended
questions receive no semantic credit. Every faction summary exposes the exact
signed value vector used to compose synthetic stances. Ask separately reports
top-one selection accuracy, end-to-end grounded stance recall, exact-evidence
match recall, and stance accuracy over that same deterministic evidence
assignment. These are deterministic synthetic conformance measures, not
judgments of real-world question usefulness.

Structural release evidence no longer relies on ordinary semantic F1 alone.
Named locator attacks are rescored after exact gold type/diagnosis/relationship
components are supplied only to correctly located spans, while every false
positive remains charged. A combined title/length/position classifier and
helper-document role/relationship classifiers are fit leave-one-template-family
out and report balanced accuracy beside exact class shares and chance
references. These are bounded candidate audits; final shortcut claims still
require a fresh attack holdout constructed after source and scorer freeze.

See the [0.6.0 evaluation plan](docs/evaluation-plan-0.6.0.md), historical
[0.5.0 evaluation report](docs/evaluation-report-0.5.0.md), historical
[0.5.0 evaluation plan](docs/evaluation-plan-0.5.0.md), historical
[0.4.1 evaluation report](docs/evaluation-report-0.4.1.md), historical
[0.3.0 evaluation report](docs/evaluation-report-0.3.0.md),
[0.4-series evaluation plan](docs/evaluation-plan-0.4.0.md), and
[methodology](docs/methodology.md) for exact definitions and limits.

## Setup and verification

```bash
uv sync --all-packages --locked
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run python scripts/check_dependency_manifests.py --check
uv run python scripts/check_release_wheel.py
```

Each environment artifact bundles an exact, hashed no-dev dependency manifest
generated from `uv.lock`. Workspace sources appear as immutable distribution
pins. The manifest is provenance, not a cross-platform installation lock.
Python is intentionally constrained to `>=3.12,<3.13` for the currently tested
Prime/Verifiers runtime; broader interpreter support has not been attested.

## Comparator and model sweep

The checked-in native-v1 config runs 100 tasks with five rollouts. Run it for
at least four model families, overriding the taskset for both Elicit modes:

```bash
for MODEL in <models>; do
  uv run eval @ configs/eval/baseline-sweep.toml -m "$MODEL" --no-push --rich false
  uv run eval @ configs/eval/baseline-sweep.toml \
    --env.taskset.id commonground-elicit \
    --env.taskset.task-mode find \
    -m "$MODEL" --no-push --rich false
  uv run eval @ configs/eval/baseline-sweep.toml \
    --env.taskset.id commonground-elicit \
    --env.taskset.task-mode elicit-ask \
    -m "$MODEL" --no-push --rich false
done
```

Render only complete saved runs with:

```bash
uv run python scripts/aggregate_baselines.py
uv run python scripts/aggregate_baselines.py --csv /path/to/baselines.csv
```

The historical 0.5.0 exact-artifact study completed all 6,000 planned rollouts
across Claude Sonnet 4.5, Gemini 2.5 Flash, GPT-4.1, and Qwen3 30B A3B Instruct.
Predict probability rewards ranged from 0.700 to 0.808, strict Find F1 from
0.133 to 0.314, and Ask utility from 0.110 to 0.337. All Predict means exceed
the 0.667 uniform-probability reward, while prompt-observable matrix baselines
remain stronger. One Qwen Ask rollout (1/6,000) contains a disclosed provider
502 followed by a successful internal retry; the same failure reproduced under
serial execution, and no score-based replacement was made. Those values must
not be presented as 0.6 evidence. Exact 0.5 intervals,
diagnostics, run IDs, hashes, and limitations are in the
[0.5.0 evaluation report](docs/evaluation-report-0.5.0.md).

Reproduce the task-level Predict and template-hierarchical Elicit analysis with:

```bash
uv run python scripts/analyze_release_study.py \
  --root /path/to/study \
  --elicit-eval-split environments/commonground_elicit/commonground_elicit/data/eval_synthetic_heldout.jsonl \
  --bootstrap-samples 50000 \
  --seed 20260830 \
  --expected-model-count 4 \
  --expected-task-count 100 \
  --expected-rollouts-per-task 5
```

Repeated rollouts estimate sampling variation, not independent task coverage.
Elicit intervals resample base templates and then variants; Predict intervals
resample tasks. The 0.6 analyzer also produces clustered intervals for component
diagnostics and exploratory paired cluster sign-flip tests with one global Holm
adjustment. Versions 0.3–0.5 Elicit scores are historical and must not be used
as 0.6 evidence because the corpus, visible decision contract, and scoring
diagnostics changed. The candidate remains unpublished until its regenerated
floors, exact-artifact study, and publication evidence pass the
[0.6.0 evaluation plan](docs/evaluation-plan-0.6.0.md).

Model-free evidence must keep provenance classes separate. Prompt-observable
baselines use only the task prompt. A deterministic replay using the candidate
source's generator, templates, seed, and answer construction is instead a
source-aware memorization ceiling, not a benchmark floor or evidence of
generalization.
Consequential transfer still requires an independently implemented private
generator.

## Provenance, scope, and license

All evaluation answers are bundled in the candidate source and would be public
for reproducibility and open training if released. Consequential comparison
requires a fresh private or server-side generator family. Non-synthetic custom
snapshots are accepted only through the
separate fail-closed contract in
[human-data governance](docs/human-data-governance.md); automated validation
cannot establish consent or legal authority.

Repository code is [Apache-2.0](LICENSE). Predict's adapted synthetic Context
Engine demo fixture remains MPL-2.0; its packaged `NOTICE` records the immutable
source, transformation boundary, hashes, and license treatment. Byte-identical
legal files also live under each environment's `LICENSES/` directory because
current Prime CLI source archives omit top-level legal files. Follow the
[public release checklist](docs/public-release-checklist.md) before publishing.
