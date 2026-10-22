#!/usr/bin/env bash
set -eo pipefail

source /home/ma-user/work/verl_quickstart/scripts/env.sh
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
cd /home/ma-user/work/sqlrl
mkdir -p logs reports checkpoints

LOG="logs/grpo_run.log"
: >"$LOG"
{
  echo "===== GRPO (SFT-initialized) ====="
  echo "timestamp: $(date -Is)"
  echo "host: $(hostname)"
  echo "sft adapter: checkpoints/sft_lora"
  echo "rl data: data/processed/spider/rl.jsonl"
  echo "eval: val_200 + test_200"
} | tee -a "$LOG"

echo "[step 1] GRPO training" | tee -a "$LOG"
python3 -u train/grpo.py \
  --data data/processed/spider/rl.jsonl \
  --model /home/ma-user/work/env_verify/models/Qwen2.5-3B-Instruct \
  --sft-adapter checkpoints/sft_lora \
  --out checkpoints/grpo \
  --steps 200 --prompts-per-step 4 --n 8 \
  --temperature 1.0 --max-new-tokens 200 \
  --lr 1e-5 --kl-coef 0.001 \
  --log-every 1 --save-every 100 2>&1 | tee -a "$LOG"

echo "[step 2] evaluate on val_200" | tee -a "$LOG"
python3 -u eval/eval_base.py \
  --data data/processed/spider/val_200.jsonl \
  --model /home/ma-user/work/env_verify/models/Qwen2.5-3B-Instruct \
  --adapter checkpoints/grpo \
  --limit 200 --batch-size 4 --max-new-tokens 260 --log-every 20 \
  --out reports/grpo_val200.json 2>&1 | tee -a "$LOG"

echo "[step 3] evaluate on test_200" | tee -a "$LOG"
python3 -u eval/eval_base.py \
  --data data/processed/spider/test_200.jsonl \
  --model /home/ma-user/work/env_verify/models/Qwen2.5-3B-Instruct \
  --adapter checkpoints/grpo \
  --limit 200 --batch-size 4 --max-new-tokens 260 --log-every 20 \
  --out reports/grpo_test200.json 2>&1 | tee -a "$LOG"

echo "[step 4] recompute audit metrics" | tee -a "$LOG"
python3 -u eval/recompute.py \
  --records reports/grpo_val200.jsonl \
  --out reports/grpo_val200_recomputed.json 2>&1 | tee -a "$LOG"
python3 -u eval/recompute.py \
  --records reports/grpo_test200.jsonl \
  --out reports/grpo_test200_recomputed.json 2>&1 | tee -a "$LOG"

echo "[done] $(date -Is) log=$LOG" | tee -a "$LOG"
