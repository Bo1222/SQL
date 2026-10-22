#!/usr/bin/env bash
set -eo pipefail

source /home/ma-user/work/verl_quickstart/scripts/env.sh
cd /home/ma-user/work/sqlrl
mkdir -p logs reports data/processed/spider

TS=$(date +%Y%m%d_%H%M%S)
LOG="logs/spider_base200_${TS}.log"

{
  echo "===== Spider dev 200 stratified baseline ====="
  echo "timestamp: $(date -Is)"
  echo "host: $(hostname)"
  echo "data: data/processed/spider/test.jsonl"
  echo "subset: data/processed/spider/test_200.jsonl"
  echo "model: /home/ma-user/work/env_verify/models/Qwen2.5-3B-Instruct"
  echo "batch: 4, max_new_tokens: 260, temperature: 0"
} | tee "$LOG"

echo "[step 1] build deterministic stratified subset (seed=0, proportional by db_id)" | tee -a "$LOG"
python3 -u data/sample_stratified.py \
  --data data/processed/spider/test.jsonl \
  --limit 200 \
  --seed 0 \
  --out data/processed/spider/test_200.jsonl \
  --manifest reports/spider_test200_manifest.json 2>&1 | tee -a "$LOG"

echo "[step 2] zero-shot generation + execution scoring (progress every 20)" | tee -a "$LOG"
python3 -u eval/eval_base.py \
  --data data/processed/spider/test_200.jsonl \
  --model /home/ma-user/work/env_verify/models/Qwen2.5-3B-Instruct \
  --limit 200 \
  --batch-size 4 \
  --max-new-tokens 260 \
  --log-every 20 \
  --out reports/spider_base200.json 2>&1 | tee -a "$LOG"

echo "[step 3] recompute audit metrics from saved raw outputs" | tee -a "$LOG"
python3 -u eval/recompute.py \
  --records reports/spider_base200.jsonl \
  --out reports/spider_base200_recomputed.json 2>&1 | tee -a "$LOG"

echo "[done] $(date -Is) log=$LOG" | tee -a "$LOG"
