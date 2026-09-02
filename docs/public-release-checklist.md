# Public release checklist

Run this checklist from a clean checkout. A green Prime Hub action is necessary
but does not replace the local, artifact, legal, and benchmark-integrity gates.

## 1. Candidate source

- Confirm the intended commit and a clean tracked working tree.
- Confirm .prime and outputs directories are ignored and absent from release
  archives.
- Search current files and history for credentials, private keys, personal
  identifiers, absolute local paths, temporary sockets, and raw model outputs.
- Confirm every bundled row declares synthetic true unless the full human-data
  governance process has been completed.

## 2. Determinism and benchmark integrity

Run:

    uv sync --all-packages --locked
    uv run python environments/commonground_predict/scripts/generate_synthetic_eval.py
    uv run python environments/commonground_predict/scripts/generate_synthetic_train.py
    uv run python scripts/generate_elicit_splits.py
    uv run python scripts/compute_floors.py environments/commonground_predict/commonground_predict/data/eval_synthetic.jsonl --masked-vote-count 8
    uv run python scripts/compute_elicit_floors.py
    uv run pytest -q tests/test_corpus_integrity.py
    uv run pytest -q

The committed generated files must remain byte-identical, and both floor tables
must exactly match their environment READMEs. Elicit generation must report 100
unique training prompts and answers and 100 unique evaluation prompts and
answers, distinct template/layout profiles, canonical semantic fingerprints,
and bounded cross-split lexical/TF-IDF neighbors. The grounded-quote,
evidence-precision, contradiction-pair, gap-diagnosis, wrong-document,
type-hedging, malformed JSON, depth, size, and
duplicate-span tests must pass. Elicit regressions must prove that the default
K=1 budget requires selection from three candidates and an exact top-one answer
reaches 1.0. Find must accept a findings-only root and independently score an
optional requested-cardinality `questions` companion at weight zero; companion
omission or malformation must not erase valid findings. Verbatim canonical-
question wording must not be required or receive special lexical credit. Find
must recover its hidden gold frame and role-specific accepted concepts. Ask
must expose each canonical candidate decision and exact signed trade-off
weights, while keeping accepted aliases, key metadata, labels, and utility
hidden. Cited passages independently anchor the issue and applicable outcomes,
and candidate profiles or response prose cannot validate their own frame.
Generic questions, unsupported clauses appended to either visible-question
segment, and unchanged-prose polarity flips must receive zero. Unknown issue
types and malformed stance/decision types fail closed, question-polarity
reversal inverts stances, and every accepted target-changing weight mutation
changes the Ask prompt (while Find is unchanged) or validation rejects it. At
every supported distractor density and bounded document length, retained keys
must have every accepted actor alias visible; density changes only true neutral
noise, and an unobservable issue and its anchor must be removed. All 24
contradiction relationships resolve to authored evidence, and related evidence
never aliases another plant or distractor. Faction summaries must expose the
exact signed values used by `value-composition-v2`. The fixed-marker,
sector-team-marker, equal-length, length/position, and prompt-only
longest-sentence probes must be rerun with exact non-localization components so
semantic failure cannot hide locator precision/recall. All five single-feature
layout-label audits and the combined title/length/anchor-position
leave-one-template-out balanced classifier must pass with explicit issue-class
shares and the `1/3` chance reference. On helper-built rows, the five-way
document-role and binary relationship-document leave-one-template-out probes
must pass with exact role/prevalence and 0.20/0.50 balanced-chance references.
The exact analytic uniform-random expectation/accuracy, runner-up
reward/accuracy, public-profile composition reward/accuracy, exact-top-one
reward/accuracy, top-one tie rate, and mean/minimum raw and normalized margins
must satisfy the thresholds in the 0.6 evaluation plan. The floor report
must separate
prompt-observable rows from the source-aware deterministic replay/memorization
ceiling. Adding a false positive or overlapping hedge must lower shaped Find
reward. Predict must retain 200/100 rows, disjoint policy text and generator
families, causal statement dimensions,
calibration-sensitive primary reward, exact response key sets, canonical ties,
fail-closed numeric parsing, four same-task prompt modes, and the
probability-native comparator/Brier-skill table.
The model-completion size bound must not make a larger trusted, generated
answer key unscorable; trusted answers retain independent depth and node
bounds.

Once source and scorer are frozen, retain a separately constructed attack
holdout that was not used to tune the candidate. A failed holdout reopens the
candidate; do not tune against it and continue calling it independent.

The public distributions contain evaluation answers. Never describe a public
score as contamination-resistant. Use a private server-side split for a
leaderboard or consequential comparison, and require an independently
implemented private generator before making transfer claims.

Run native Verifiers validation explicitly in the local subprocess runtime:

    uv run validate commonground-predict --runtime.type subprocess --rich false
    uv run validate commonground-elicit --runtime.type subprocess --rich false
    uv run validate commonground-elicit --taskset.task-mode elicit-ask --runtime.type subprocess --rich false

## 3. Static and dependency gates

Run:

    uv lock --check
    uv run ruff check .
    uv run ruff format --check .
    uv run mypy
    uv run python scripts/check_dependency_manifests.py --check
    uv run python scripts/check_release_wheel.py

The artifact checker builds and inspects all four wheels and source archives,
then installs the wheels together in a fresh Python 3.12 environment under the
two exact dependency manifests. It imports all four packages and loads Predict,
Elicit find, and Elicit ask from the installed artifacts rather than the source
tree.

Audit the exact locked environment with pip-audit or an equivalent scanner. Do
not publish with a known affected dependency in the install closure.
After an intentional lock or uv-generator update, regenerate both manifests
with `uv run python scripts/check_dependency_manifests.py --write`, review the
resolved-version diff, and rerun `--check`. Do not hand-edit the manifests.

## 4. Package order

Publish immutable shared packages first:

1. commonground-score 0.6.0
2. commonground-scenarios 0.6.0
3. commonground-predict 0.6.0
4. commonground-elicit 0.6.0

The environment packages exactly pin the shared versions. Do not reuse a
published version number.

## 5. Exact artifact verification

For every wheel, sdist, and Hub source archive:

- verify package name, version, Python requirement, repository URL, tags, and
  exact dependency pins;
- verify legal text is present in every artifact; current Prime CLI source
  archives omit root files other than README, pyproject, and Python modules, so
  both environments must carry Apache-2.0 under nested `LICENSES/`, and Predict
  must also carry its MPL-2.0 text and byte-identical provenance notice there;
- verify the packaged dependency manifest matches the candidate `uv.lock`
  closure-scoped resolution SHA-256, Python scope, pinned uv version, and exact
  no-dev resolution;
- verify no top-level pyproject.toml is installed;
- install into a fresh Python 3.12 environment;
- import the package, resolve its native Verifiers v1 taskset and pure-chat
  harness, and load its legacy Hosted Evaluation adapter;
- run every test included in the extracted source archive;
- compare generated artifact hashes with the candidate recorded for release.

## 6. Private Hub verification

Push the new versions privately first. On the exact private artifacts:

- require the managed Hub action to pass;
- pull and repeat clean install, import, taskset load, and shipped-test checks;
- run a small native-v1 server evaluation and a standalone hosted legacy
  evaluation for predict and both elicit task modes;
- run the full 100-task/five-rollout sweep on at least four model families for
  Predict and both Elicit modes; preserve exact configs and per-example traces;
- require zero recovered/error histories; quarantine failed attempts and rerun
  the complete frozen model/task stratum rather than selecting episodes;
- verify exact extended Elicit answers remain at most 1,024 `cl100k_base`
  tokens under the 2,048-token generation cap, then report observed pilot
  completion lengths;
- run the same-model Predict `full`, `matrix-only`, `text-only`, and
  `shuffled-text` prompt ablations with paired task IDs and answers;
- analyze paired task differences for Predict and template-hierarchical
  differences for Elicit, clustered component diagnostics, exploratory
  cluster-level sign-flip tests under the declared sign-symmetry/exchangeability
  assumption, and the predeclared 18-comparison global Holm family; keep
  component and prompt-ablation intervals explicitly pointwise/unadjusted;
- review the rendered README and license/provenance notice.

For the 0.6.0 study, generate the attested analysis directly from retained
native-v1 traces:

    uv run python scripts/analyze_release_study.py \
      --root /path/to/release-study \
      --elicit-eval-split environments/commonground_elicit/commonground_elicit/data/eval_synthetic_heldout.jsonl \
      --diagnostics-csv /path/to/diagnostics.csv \
      --pairwise-csv /path/to/pairwise.csv \
      --json /path/to/release-analysis.json \
      --bootstrap-samples 50000 \
      --seed RELEASE_SEED \
      --expected-model-count 4 \
      --expected-task-count 100 \
      --expected-rollouts-per-task 5 \
      --expected-pairwise-family-size 18 \
      --require-no-recoveries

Only then should an operator manually make a listing public. Publishing and
visibility changes are never performed by the automated local release checks.
Create an annotated source tag and formal GitHub release only after the final
source commit, artifact hashes, and Hub identifiers are recorded.
