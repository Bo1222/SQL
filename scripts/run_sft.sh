#!/usr/bin/env bash
set -eo pipefail

source /home/ma-user/work/verl_quickstart/scripts/env.sh
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
cd /home/ma-user/work/sqlrl
mkdir -p logs reports checkpoints data/processed/spider

LOG="logs/sft_run.log"
: >"$LOG"
{
  echo "===== SFT + val + pass@8 ====="
  echo "timestamp: $(date -Is)"
  echo "host: $(hostname)"
  echo "sft data: data/processed/spider/sft.jsonl"
  echo "model: /home/ma-user/work/env_verify/models/Qwen2.5-3B-Instruct"
} | tee -a "$LOG"

echo "[step 1] build val_200 and rl_100 subsets" | tee -a "$LOG"
python3 -u data/sample_stratified.py \
  --data data/processed/spider/val.jsonl --limit 200 --seed 0 \
  --out data/processed/spider/val_200.jsonl \
  --manifest reports/spider_val200_manifest.json 2>&1 | tee -a "$LOG"
python3 -u data/sample_stratified.py \
  --data data/processed/spider/rl.jsonl --limit 100 --seed 0 \
  --out data/processed/spider/rl_100.jsonl \
  --manifest reports/spider_rl100_manifest.json 2>&1 | tee -a "$LOG"

echo "[step 2] LoRA SFT, 1 epoch" | tee -a "$LOG"
python3 -u train/sft_lora.py \
  --data data/processed/spider/sft.jsonl \
  --model /home/ma-user/work/env_verify/models/Qwen2.5-3B-Instruct \
  --out checkpoints/sft_lora \
  --batch-size 4 --grad-accum 4 --lr 1e-4 --epochs 1 --max-len 2048 \
  --log-every 10 --save-every 100 2>&1 | tee -a "$LOG"

echo "[step 3] evaluate SFT on val_200" | tee -a "$LOG"
python3 -u eval/eval_base.py \
  --data data/processed/spider/val_200.jsonl \
  --model /home/ma-user/work/env_verify/models/Qwen2.5-3B-Instruct \
  --adapter checkpoints/sft_lora \
  --limit 200 --batch-size 4 --max-new-tokens 260 --log-every 20 \
  --out reports/sft_val200.json 2>&1 | tee -a "$LOG"

echo "[step 4] pass@8 on rl_100" | tee -a "$LOG"
python3 -u eval/pass_at_k.py \
  --data data/processed/spider/rl_100.jsonl \
  --model /home/ma-user/work/env_verify/models/Qwen2.5-3B-Instruct \
  --adapter checkpoints/sft_lora \
  --limit 100 --n 8 --temperature 1.0 --batch-size 2 \
  --max-new-tokens 200 --log-every 10 \
  --out reports/pass8_rl100.json 2>&1 | tee -a "$LOG"

echo "[done] $(date -Is) log=$LOG" | tee -a "$LOG"
