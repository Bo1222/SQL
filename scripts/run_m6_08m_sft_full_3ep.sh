#!/usr/bin/env bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# M6-08m 3B full-parameter SFT (FSDP, 4 NPUs).
# Usage:
#   SMOKE=10 bash scripts/run_m6_08m_sft_full_3ep.sh   # 10-step smoke, separate out dir
#   bash scripts/run_m6_08m_sft_full_3ep.sh            # 3 epochs
set -eo pipefail
cd "$ROOT"
CALLER_WANDB_MODE="${WANDB_MODE:-}"
source "$ROOT/scripts/env.sh"
# env.sh forces WANDB_MODE=disabled; load the project .env afterwards so the
# requested online/offline mode and API key are exported to rank 0.
if [ -f .env ]; then
  set -a
  source .env
  set +a
fi
if [ -n "$CALLER_WANDB_MODE" ]; then
  export WANDB_MODE="$CALLER_WANDB_MODE"
fi
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

DATA="${DATA:-artifacts/m6/e1_student_only/sft_student_full_prefer.jsonl}"
LR="${LR:-2e-5}"
WD="${WD:-0.01}"
SMOKE="${SMOKE:-0}"
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-0}"
SAVE_TRAINING_STATE="${SAVE_TRAINING_STATE:-1}"
CHECKPOINT_EVERY_EPOCH="${CHECKPOINT_EVERY_EPOCH:-1}"
WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_PROJECT="${WANDB_PROJECT:-sqlrl}"
RUN_NAME="${RUN_NAME:-student_sft_full_3ep}"
COMMON=(--wandb-mode "$WANDB_MODE" --wandb-project "$WANDB_PROJECT"
        --wandb-run-name "$RUN_NAME" --save-every-steps "$SAVE_EVERY_STEPS")
if [ "$SAVE_TRAINING_STATE" = "1" ]; then
  COMMON+=(--save-training-state)
else
  COMMON+=(--no-save-training-state)
fi
if [ "$CHECKPOINT_EVERY_EPOCH" = "1" ]; then
  COMMON+=(--checkpoint-every-epoch)
else
  COMMON+=(--no-checkpoint-every-epoch)
fi
if [ "$SMOKE" != "0" ]; then
  OUT="${OUT:-outputs/m6_08m/student_sft_full_smoke}"
  EXTRA=(--smoke-steps "$SMOKE" --overwrite)
else
  OUT="${OUT:-outputs/m6_08m/student_sft_full_3ep}"
  EXTRA=()
fi
torchrun --nnodes=1 --nproc_per_node=4 \
  --master_addr=127.0.0.1 --master_port="${MASTER_PORT:-29501}" \
  -m e1_alignment.student_sft_full \
  --data "$DATA" \
  --out "$OUT" \
  --model artifacts/checkpoints/step_600_merged \
  --epochs 3 --grad-accum 2 --lr "$LR" --weight-decay "$WD" \
  --max-context 6144 --seed 0 "${COMMON[@]}" "${EXTRA[@]}"
