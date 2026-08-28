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
must exactly match their environment READMEs. Elicit generation must report 40
unique training semantic tasks and 20 unique evaluation semantic tasks, with
disjoint template families. The grounded-quote, wrong-document, semantic-
operator, malformed JSON, depth, size, duplicate-assignment, and short-fragment
tests must pass. The Elicit-ask regression must prove that hidden canonical
wording cannot alter a grounded paraphrase's score. Predict training and
evaluation policy texts must remain disjoint.

The public distributions contain evaluation answers. Never describe a public
score as contamination-resistant. Use a private server-side split for a
leaderboard or consequential comparison.

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

1. commonground-score 0.1.1
2. commonground-scenarios 0.1.1
3. commonground-predict 0.2.4
4. commonground-elicit 0.2.4

The environment packages exactly pin the shared versions. Do not reuse a
published version number.

## 5. Exact artifact verification

For every wheel, sdist, and Hub source archive:

- verify package name, version, Python requirement, repository URL, tags, and
  exact dependency pins;
- verify LICENSE is present; predict must also contain its MPL-2.0 text and
  NOTICE;
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
- rerun and publish elicit model baselines because its corpus changed in 0.2.0
  and its question reward changed in 0.2.2;
- review the rendered README and license/provenance notice.

Only then should an operator manually make a listing public. Publishing and
visibility changes are never performed by the automated local release checks.
