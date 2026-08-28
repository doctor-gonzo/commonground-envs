#!/usr/bin/env bash
# Local smoke battery for the commonground environments.
#
# ONE-TIME AUTHENTICATION (both options keep the key out of shell history):
#   prime login
#   prime config set-api-key
#
# Then run this script from the repository root:
#   bash scripts/local_smoke.sh
#
# Never pass an API key as a command argument or commit it to a file in this repo.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "== 1/6 sync exact workspace =="
uv sync --all-packages --locked

echo "== 2/6 pytest (no model or Hub network) =="
uv run pytest -q

echo "== 3/6 native v1 taskset validation (no model or Hub network) =="
uv run validate commonground-predict --runtime.type subprocess --rich false
uv run validate commonground-elicit --runtime.type subprocess --rich false
uv run validate commonground-elicit \
  --taskset.task-mode elicit-ask \
  --runtime.type subprocess \
  --rich false

if ! command -v prime >/dev/null 2>&1; then
  echo "Prime CLI is required for live evals; install it, then run 'prime login'." >&2
  exit 1
fi
if ! prime --plain whoami >/dev/null 2>&1; then
  echo "Prime authentication is required; run 'prime login' or 'prime config set-api-key'." >&2
  exit 1
fi

echo "== 4/6 native v1 eval commonground-predict (Prime Inference, ~cents) =="
uv run eval commonground-predict -n 1 -r 1 --no-push --rich false

echo "== 5/6 native v1 eval commonground-elicit find =="
uv run eval commonground-elicit -n 1 -r 1 --no-push --rich false

echo "== 6/6 native v1 eval commonground-elicit ask =="
uv run eval commonground-elicit \
  --env.taskset.task-mode elicit-ask \
  -n 1 -r 1 --no-push --rich false

echo "Done. Keep the candidate private until the public release checklist passes."
