#!/usr/bin/env bash
# M5: paired retrieval-backend comparison with a frozen SFT adapter (no RL).
set -eo pipefail
source /home/ma-user/work/sqlrl/scripts/env.sh
cd /home/ma-user/work/sqlrl
N="${N:-100}"
for backend in legacy unified; do
  echo "=== E-$backend n=$N ==="
  python3 -u eval/evaluate_agent.py \
    --data data/processed/bird_agent/val.jsonl \
    --limit "$N" --backend "$backend" --engine vllm \
    --model artifacts/checkpoints/step_600_merged \
    --out "reports/m5/e_${backend}_val${N}.json" \
    --trace-dir "reports/m5/traces_${backend}"
done
python3 eval/paired_statistics.py \
  --a "reports/m5/e_legacy_val${N}.jsonl" \
  --b "reports/m5/e_unified_val${N}.jsonl" \
  --out "reports/m5/e0_vs_e1_paired.json"
