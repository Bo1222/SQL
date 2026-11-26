#!/usr/bin/env bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# E1-04: gold-AST-assisted demonstrations for questions unresolved by A/B.
# Gold SQL never enters the user prompt or a tool observation.
set -eo pipefail
source "$ROOT/scripts/env.sh"
cd "$ROOT"
export PYTHONPATH="$ROOT:$PYTHONPATH"

python3 -u -m e1_alignment.collect_c \
  --limit "${LIMIT:-400}" --seed "${SEED:-0}" \
  --exclude-from \
    artifacts/m6/e1_alignment/e1_self_rollouts.jsonl \
    artifacts/m6/e1_alignment/e1_repair_rollouts.jsonl \
  --only-from \
    artifacts/m6/e1_alignment/e1_self_rollouts.raw.jsonl \
    artifacts/m6/e1_alignment/e1_failure_prefixes.jsonl \
  --fingerprint reports/m6/e1_alignment/runtime_fingerprint.json "$@"
