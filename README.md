# commonground-envs

Two open, deterministic Verifiers environments for structured reasoning over
synthetic stakeholder data:

- `commonground-predict` requires calibrated probabilities for masked votes in
  a partially visible stakeholder-by-policy matrix.
- `commonground-elicit` requires grounded diagnosis of policy ambiguities,
  contradictions, and gaps, or selection of the two most valuable
  clarification questions from three candidate issues.

Version 0.4.0 is an unreleased, review-driven candidate. It removes a
prompt-visible 0.3 Elicit stance codebook, makes calibrated probabilities the
Predict objective, strengthens semantic split audits, and adds optional Find
training shaping. Keep it private until the exact candidate passes the full
model study. Public answer keys make every release unsuitable for
contamination-resistant leaderboards.

This is the reinforcement-learning environment family associated with
[Context Engine](https://contextengine.sh), an open-source deliberation system
for collecting statements and votes, mapping agreement and disagreement, and
turning the results into auditable organizational decisions. A future governed
export path could allow consenting individuals and groups to retain, license,
or sell preference datasets derived from their sessions. No such exporter is
included here. Every bundled row is synthetic, apart from one explicitly
synthetic, operator-authored demo fixture.

![Common Ground Predict and Elicit use synthetic data in the current release; reviewed human data is shown only as a future option.](docs/assets/commonground-envs-overview.png)

## Packages

| Path | Distribution | Purpose |
| --- | --- | --- |
| `packages/commonground-score` | `commonground-score` 0.3.0 | Vote statistics and calibrated-reward helpers. |
| `packages/commonground-scenarios` | `commonground-scenarios` 0.3.0 | Offline scenario generation, schema, and validation. |
| `environments/commonground_predict` | `commonground-predict` 0.4.0 | Probabilistic masked-vote prediction. |
| `environments/commonground_elicit` | `commonground-elicit` 0.4.0 | Structured finding diagnosis and top-K clarification. |

Both environments export native Verifiers v1 `Taskset`/`Harness` plugins and a
real legacy `SingleTurnEnv` adapter for the current Prime Hosted Evaluation
runner. The same rows and scoring functions back both interfaces.

## What 0.4.0 changes

Predict now optimizes `1 - normalized Brier`; argmax accuracy and Brier remain
diagnostics. The exact masked-cell key set is mandatory. Standard
collaborative-filtering comparators remain strong—one-hot 5-NN scores 0.891—so
the environment is semantic-conditioned matrix completion, not evidence of
general collective-preference reasoning.

Elicit faction summaries now state indirect policy principles instead of exact
issue terms and yes/no/pass answers. A parser exploiting the old 0.3 clauses
scores 0.787 on the historical corpus and 0.000 on 0.4. Split audits separately
track instance identity, canonical visible meaning, policy-issue meaning,
token Jaccard, and word-ngram TF-IDF similarity. Ask value derives from
simulated answer coverage and disagreement, not policy keywords. Find keeps
strict F1 for evaluation and offers an explicit staged training reward.

See the historical [0.3.0 evaluation report](docs/evaluation-report-0.3.0.md),
the [0.4.0 evaluation plan](docs/evaluation-plan-0.4.0.md), and
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

The historical 0.3.0 release study completed all 6,000 planned rollouts across Claude
Sonnet 4.5, Gemini 2.5 Flash, GPT-4.1, and Qwen3 30B A3B Instruct with no
recovered rollouts. Predict rewards ranged from 0.572 to 0.730, Find from 0.120
to 0.235, and Ask from 0.052 to 0.296. The exact clustered intervals,
pairwise comparisons, configs, artifact hashes, and limitations are in the
[0.3.0 evaluation report](docs/evaluation-report-0.3.0.md).

Reproduce the task-level Predict and template-hierarchical Elicit analysis with:

```bash
uv run python scripts/analyze_release_study.py \
  --root /path/to/study \
  --elicit-eval-split environments/commonground_elicit/commonground_elicit/data/eval_synthetic_heldout.jsonl \
  --bootstrap-samples 50000 \
  --seed 20260828 \
  --expected-model-count 4 \
  --expected-task-count 100 \
  --expected-rollouts-per-task 5 \
  --require-no-recoveries
```

Repeated rollouts estimate sampling variation, not independent task coverage.
Elicit intervals resample base templates and then variants; Predict intervals
resample tasks. Version 0.3 model scores are historical and must not be used as
0.4 evidence because both corpora/objectives changed.

## Provenance, scope, and license

All bundled evaluation answers are published for reproducibility and open
training. Consequential comparison requires a fresh private or server-side
generator family. Non-synthetic custom snapshots are accepted only through the
separate fail-closed contract in
[human-data governance](docs/human-data-governance.md); automated validation
cannot establish consent or legal authority.

Repository code is [Apache-2.0](LICENSE). Predict's adapted synthetic Context
Engine demo fixture remains MPL-2.0; its packaged `NOTICE` records the immutable
source, transformation boundary, hashes, and license treatment. Follow the
[public release checklist](docs/public-release-checklist.md) before publishing.
