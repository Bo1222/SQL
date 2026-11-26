#!/usr/bin/env bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# E1-03: repair continuations from 150 selected real failure states (n=4 each).
set -eo pipefail
source "$ROOT/scripts/env.sh"
cd "$ROOT"
export PYTHONPATH="$ROOT:$PYTHONPATH"

python3 -u -m e1_alignment.collect_b \
  --limit-states "${STATES:-150}" --n "${N:-4}" --seed "${SEED:-0}" \
  --states-per-batch "${STATES_PER_BATCH:-4}" \
  --fingerprint reports/m6/e1_alignment/runtime_fingerprint.json "$@"
