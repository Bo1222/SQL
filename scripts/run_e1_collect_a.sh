#!/usr/bin/env bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# E1-02: autonomous E1 rollouts from the frozen SFT policy.
# 600 RL-Train questions x 4 samples, temperature 1.0, frozen fingerprint.
set -eo pipefail
source "$ROOT/scripts/env.sh"
cd "$ROOT"
export PYTHONPATH="$ROOT:$PYTHONPATH"

python3 -u -m e1_alignment.collect_a \
  --limit "${LIMIT:-600}" --n 4 --questions-per-batch "${QPB:-8}" --seed "${SEED:-0}" \
  --fingerprint reports/m6/e1_alignment/runtime_fingerprint.json "$@"
