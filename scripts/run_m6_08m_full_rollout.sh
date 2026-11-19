#!/usr/bin/env bash
# M6-08m full student-only n4+E1.1 rollout.
# Usage: bash scripts/run_m6_08m_full_rollout.sh [fresh|resume]
set -eo pipefail
cd /home/ma-user/work/sqlrl
MODE="${1:-fresh}"
EXTRA=()
if [ "$MODE" = "resume" ]; then
  EXTRA+=(--resume)
  if [ "${ALLOW_RUNNER_CHANGE:-0}" = "1" ]; then
    EXTRA+=(--allow-runner-change)
  fi
else
  EXTRA+=(--overwrite)
fi
python3 -u scripts/run_student_rollouts.py \
  --tasks artifacts/m6/e1_student_only/tasks4190.jsonl \
  --out-dir reports/m6/e1_student_only/p3_full_n4 \
  --fingerprint reports/m6/e1_student_only/runtime_fingerprint_e1_1_full.json \
  --retrieval-version E1.1 \
  --arm full_n4 --mode raw --num-samples 4 --temperature 0.8 --top-p 0.95 \
  --cards 0,1,2,3 --num-workers 4 \
  "${EXTRA[@]}"
