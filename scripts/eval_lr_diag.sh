#!/usr/bin/env bash
# Merge one LR-diagnostic LoRA checkpoint and evaluate dev200 with compact observation.
# Usage: bash scripts/eval_lr_diag.sh <tag> [step]
set -eo pipefail
source /home/ma-user/work/verl_quickstart/scripts/env.sh
cd /home/ma-user/work/sqlrl
export ASCEND_RT_VISIBLE_DEVICES="${CARDS:-0}"
export TOKENIZERS_PARALLELISM=false
unset PYTORCH_NPU_ALLOC_CONF

TAG="$1"
STEP="${2:-5}"
ROOT=/home/ma-user/work/sqlrl
BASE=$ROOT/artifacts/checkpoints/step_600_merged
ADAPTER=$ROOT/artifacts/checkpoints/lr_diag/$TAG/global_step_$STEP/actor/lora_adapter
MERGED=$ROOT/artifacts/checkpoints/merged_lr_diag_${TAG}_step${STEP}
[ -d "$ADAPTER" ] || { echo "missing adapter: $ADAPTER" >&2; exit 2; }
rm -rf "$MERGED"
python3 scripts/merge_adapter.py --base "$BASE" --adapter "$ADAPTER" \
  --tokenizer-from "$BASE" --out "$MERGED"
python3 -u eval/evaluate_agent.py \
  --data artifacts/datasets/dev200.jsonl --limit 0 --backend unified --engine vllm \
  --model "$MERGED" --out "reports/m6/lr_diag_${TAG}_step${STEP}.json" \
  --max-rounds 5 --max-observation-tokens 1024 --observation-budget 1024 \
  --observation-mode compact
