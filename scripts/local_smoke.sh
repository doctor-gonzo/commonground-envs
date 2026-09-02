#!/usr/bin/env bash
# One local release verification path. It never publishes or runs hosted inference.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ -n "$(git status --porcelain)" ]]; then
  echo "release verification requires a clean committed worktree" >&2
  exit 1
fi

evidence_dir="$(mktemp -d /tmp/commonground-release-verify.XXXXXX)"
generated_dir="$evidence_dir/elicit-splits"
mkdir -p "$generated_dir"

source_commit="$(git rev-parse HEAD)"
printf '%s\n' "$source_commit" >"$evidence_dir/source-commit.txt"

echo "== sync locked workspace =="
uv sync --all-packages --locked

echo "== regenerate and compare Elicit splits =="
uv run python scripts/generate_elicit_splits.py --output-dir "$generated_dir"
cmp "$generated_dir/train_synthetic.jsonl" \
  environments/commonground_elicit/commonground_elicit/data/train_synthetic.jsonl
cmp "$generated_dir/eval_synthetic_heldout.jsonl" \
  environments/commonground_elicit/commonground_elicit/data/eval_synthetic_heldout.jsonl

echo "== model-free summaries =="
uv run python scripts/compute_floors.py \
  environments/commonground_predict/commonground_predict/data/eval_synthetic.jsonl \
  --masked-vote-count 8 \
  --train-split environments/commonground_predict/commonground_predict/data/train_synthetic.jsonl \
  --seed 20260831 \
  --output-json "$evidence_dir/predict-floors.json" \
  | tee "$evidence_dir/predict-floors.md"
uv run python scripts/compute_elicit_floors.py \
  environments/commonground_elicit/commonground_elicit/data/eval_synthetic_heldout.jsonl \
  | tee "$evidence_dir/elicit-floors.md"

echo "== tests and static gates =="
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run python scripts/check_dependency_manifests.py --check

echo "== fresh release artifacts and taskset loads =="
uv run python scripts/check_release_wheel.py \
  --output-dir "$evidence_dir/artifacts" \
  | tee "$evidence_dir/release-artifacts.txt"

shasum -a 256 \
  uv.lock \
  pyproject.toml \
  environments/commonground_predict/commonground_predict/data/eval_synthetic.jsonl \
  environments/commonground_predict/commonground_predict/data/train_synthetic.jsonl \
  environments/commonground_elicit/commonground_elicit/data/eval_synthetic_heldout.jsonl \
  environments/commonground_elicit/commonground_elicit/data/train_synthetic.jsonl \
  >"$evidence_dir/source-inputs.sha256"

(
  cd "$evidence_dir"
  shasum -a 256 \
    artifacts/* \
    elicit-floors.md \
    predict-floors.json \
    predict-floors.md \
    release-artifacts.txt \
    source-commit.txt \
    source-inputs.sha256 \
    >evidence.sha256
)

[[ "$(git rev-parse HEAD)" == "$source_commit" ]]
if [[ -n "$(git status --porcelain)" ]]; then
  echo "release verification changed the committed source tree" >&2
  exit 1
fi

echo "LOCAL RELEASE VERIFICATION PASS"
echo "Evidence: $evidence_dir"
