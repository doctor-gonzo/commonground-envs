# commonground-envs

Two open, deterministic Verifiers environments for structured reasoning over
synthetic stakeholder data:

- `commonground-predict` requires calibrated probabilities for masked votes in
  a partially visible stakeholder-by-policy matrix.
- `commonground-elicit` requires grounded diagnosis of policy ambiguities,
  contradictions, and gaps, or selection of the two most valuable
  clarification questions from three candidate issues.

Version 0.3.0 is a breaking benchmark-design release. It replaces the 0.2.x
evaluation corpora and response contracts after an independent review found
structural shortcuts in Elicit and a non-semantic vote generator in Predict.
The current release is suitable for open training and experimental evaluation;
its public answer keys make it unsuitable for contamination-resistant
leaderboards.

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
| `packages/commonground-score` | `commonground-score` 0.2.0 | Vote statistics and normalized scoring helpers. |
| `packages/commonground-scenarios` | `commonground-scenarios` 0.2.0 | Offline scenario generation, schema, and validation. |
| `environments/commonground_predict` | `commonground-predict` 0.3.0 | Probabilistic masked-vote prediction. |
| `environments/commonground_elicit` | `commonground-elicit` 0.3.0 | Structured finding diagnosis and top-K clarification. |

Both environments export native Verifiers v1 `Taskset`/`Harness` plugins and a
real legacy `SingleTurnEnv` adapter for the current Prime Hosted Evaluation
runner. The same rows and scoring functions back both interfaces.

## What 0.3.0 changes

Predict now generates votes from explicit statement dimensions and latent
cluster preference profiles. Training and evaluation use disjoint text banks,
seeds, session IDs, and generator families. The response contract requires a
three-class probability distribution per cell, and Brier is normalized to
`[0,1]`. Standard collaborative-filtering comparators remain strong—5-NN
scores 0.891—so the environment should be described as semantic-conditioned
matrix completion, not evidence of general collective-preference reasoning.

Elicit now uses opaque randomized document/faction IDs, varied order, titles,
styles, sentence positions, faction counts, and issue-specific stance vectors.
Train and evaluation contain 100 rows each with disjoint generator families
and separate prompt, answer, and structural audits. Find requires a concise
diagnosis and paired second-document evidence for contradictions. Ask selects
two of three issues, requires at least two decision terms, and normalizes reward
by the row's attainable maximum. The exact 0.2 ID/position shortcut scores zero.

See the [0.3.0 evaluation report](docs/evaluation-report-0.3.0.md) and
[methodology](docs/methodology.md) for exact comparator definitions and limits.

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

The 0.3.0 release study completed all 6,000 planned rollouts across Claude
Sonnet 4.5, Gemini 2.5 Flash, GPT-4.1, and Qwen3 30B A3B Instruct with no
recovered rollouts. Predict rewards ranged from 0.572 to 0.730, Find from 0.120
to 0.235, and Ask from 0.052 to 0.296. The exact task-cluster intervals,
pairwise comparisons, configs, artifact hashes, and limitations are in the
[0.3.0 evaluation report](docs/evaluation-report-0.3.0.md).

Reproduce the independent-task analysis with:

```bash
uv run python scripts/analyze_release_study.py \
  --root /path/to/study \
  --bootstrap-samples 50000 \
  --seed 20260828 \
  --expected-model-count 4 \
  --expected-task-count 100 \
  --expected-rollouts-per-task 5 \
  --require-no-recoveries
```

Repeated rollouts estimate sampling variation, not independent task coverage.
Comparative claims should use per-task paired or task-cluster bootstrap
intervals and disclose model/provider settings, retries, environment hashes,
and per-example traces.

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
