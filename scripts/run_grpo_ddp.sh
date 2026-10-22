#!/usr/bin/env bash
set -eo pipefail

GRPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ "${SQLRL_ENV_LOADED:-}" != "$GRPO_ROOT" ]; then
  exec python3 "$GRPO_ROOT/train/project_env.py" -- bash "$GRPO_ROOT/scripts/run_grpo_ddp.sh" "$@"
fi

# Preserve the requested mode before env.sh overwrites WANDB_MODE=disabled.
GRPO_WANDB_MODE="${WANDB_MODE:-offline}"
source /home/ma-user/work/verl_quickstart/scripts/env.sh
export WANDB_MODE="$GRPO_WANDB_MODE"
export WANDB_PROJECT="${WANDB_PROJECT:-sqlrl}"
export WANDB_NAME="${WANDB_NAME:-grpo-ddp-$(date +%Y%m%d_%H%M%S)}"
# env.sh pins ASCEND_RT_VISIBLE_DEVICES=0; override explicitly for multi-card.
export ASCEND_RT_VISIBLE_DEVICES="${CARDS:-0,1,2,3}"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
cd "$GRPO_ROOT"

NPROC="${NPROC:-4}"
if ! [[ "$NPROC" =~ ^[1-9][0-9]*$ ]]; then
  echo "NPROC must be a positive integer" >&2
  exit 2
fi
IFS=',' read -r -a GRPO_CARDS <<< "$ASCEND_RT_VISIBLE_DEVICES"
if [ "$NPROC" -gt "${#GRPO_CARDS[@]}" ]; then
  echo "NPROC=$NPROC exceeds visible card count=${#GRPO_CARDS[@]}" >&2
  exit 2
fi
mkdir -p logs reports
if [ "$NPROC" -gt 4 ]; then
  echo "[warn] prompts-per-step=4; ranks beyond 4 will idle (parameters are unchanged)."
fi

LOG="logs/grpo_ddp_run.log"
: >"$LOG"
{
  echo "===== GRPO DDP (SFT-initialized) ====="
  echo "timestamp: $(date -Is)"
  echo "host: $(hostname)"
  echo "world_size: $NPROC"
  echo "cards: $ASCEND_RT_VISIBLE_DEVICES"
  echo "sft adapter: checkpoints/sft_lora"
  echo "rl data: data/processed/spider/rl.jsonl"
  echo "wandb: mode=$WANDB_MODE project=$WANDB_PROJECT name=$WANDB_NAME"
  echo "params: steps=200 prompts_per_step=4 n=8 lr=1e-5 kl=0.001"
} | tee -a "$LOG"

echo "[step 1] GRPO training (torchrun/HCCL)" | tee -a "$LOG"
torchrun --nnodes=1 --nproc_per_node="$NPROC" --master_addr=127.0.0.1 --master_port="${MASTER_PORT:-29511}" -- \
  train/grpo_ddp.py \
  --data data/processed/spider/rl.jsonl \
  --model /home/ma-user/work/env_verify/models/Qwen2.5-3B-Instruct \
  --sft-adapter checkpoints/sft_lora \
  --out checkpoints/grpo_ddp \
  --steps 200 --prompts-per-step 4 --n 8 \
  --temperature 1.0 --max-new-tokens 200 \
  --lr 1e-5 --kl-coef 0.001 \
  --log-every 1 --save-every 100 \
  --wandb-mode "$WANDB_MODE" --wandb-project "$WANDB_PROJECT" --wandb-name "$WANDB_NAME" \
  2>&1 | tee -a "$LOG"

echo "[step 2] evaluate on val_200" | tee -a "$LOG"
python3 -u eval/eval_base.py \
  --data data/processed/spider/val_200.jsonl \
  --model /home/ma-user/work/env_verify/models/Qwen2.5-3B-Instruct \
  --adapter checkpoints/grpo_ddp \
  --limit 200 --batch-size 4 --max-new-tokens 260 --log-every 20 \
  --out reports/grpo_ddp_val200.json 2>&1 | tee -a "$LOG"

echo "[step 3] evaluate on test_200" | tee -a "$LOG"
python3 -u eval/eval_base.py \
  --data data/processed/spider/test_200.jsonl \
  --model /home/ma-user/work/env_verify/models/Qwen2.5-3B-Instruct \
  --adapter checkpoints/grpo_ddp \
  --limit 200 --batch-size 4 --max-new-tokens 260 --log-every 20 \
  --out reports/grpo_ddp_test200.json 2>&1 | tee -a "$LOG"

echo "[step 4] recompute audit metrics" | tee -a "$LOG"
python3 -u eval/recompute.py --records reports/grpo_ddp_val200.jsonl \
  --out reports/grpo_ddp_val200_recomputed.json 2>&1 | tee -a "$LOG"
python3 -u eval/recompute.py --records reports/grpo_ddp_test200.jsonl \
  --out reports/grpo_ddp_test200_recomputed.json 2>&1 | tee -a "$LOG"

echo "[done] $(date -Is) log=$LOG" | tee -a "$LOG"
