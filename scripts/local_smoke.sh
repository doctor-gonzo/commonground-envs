#!/usr/bin/env bash
# Local smoke battery for the commonground environments.
#
# ONE-TIME KEY SETUP (paste your Prime Intellect key into this command, run from any terminal):
#   mkdir -p ~/.config/prime && printf 'export PRIME_API_KEY="PASTE-KEY-HERE"\n' > ~/.config/prime/env && chmod 600 ~/.config/prime/env
#
# Then run this script from anywhere:
#   bash ~/Desktop/xoCortex/projects/commonground-envs/scripts/local_smoke.sh
#
# The key lives only in ~/.config/prime/env (outside every git repo, mode 600).
# Never commit or paste the key anywhere else.

set -euo pipefail

if [[ ! -f "$HOME/.config/prime/env" ]]; then
  echo "Missing $HOME/.config/prime/env — run the one-time key setup command in this script's header." >&2
  exit 1
fi
source "$HOME/.config/prime/env"
export OPENAI_API_KEY="$PRIME_API_KEY"
export OPENAI_BASE_URL="https://api.pinference.ai/api/v1"

cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "== 1/3 pytest (no network) =="
uv run pytest -q

echo "== 2/3 vf-eval commonground-predict (Prime Inference, ~cents) =="
uv run vf-eval commonground-predict

echo "== 3/3 vf-eval commonground-elicit (watch: does any model score >0 on elicit-ask?) =="
uv run vf-eval commonground-elicit

echo "Done. Baselines next (still private): see runbook Step 4 (prime eval run ... --skip-upload)."
