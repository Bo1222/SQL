#!/usr/bin/env bash
# E1-03: repair continuations from 150 selected real failure states (n=4 each).
set -eo pipefail
source /home/ma-user/work/sqlrl/scripts/env.sh
cd /home/ma-user/work/sqlrl
export PYTHONPATH=/home/ma-user/work/sqlrl:$PYTHONPATH

python3 -u -m e1_alignment.collect_b \
  --limit-states "${STATES:-150}" --n "${N:-4}" --seed "${SEED:-0}" \
  --states-per-batch "${STATES_PER_BATCH:-4}" \
  --fingerprint reports/m6/e1_alignment/runtime_fingerprint.json "$@"
