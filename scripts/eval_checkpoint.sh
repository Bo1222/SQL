#!/usr/bin/env bash
# Merge one veRL checkpoint's LoRA adapter and evaluate it on dev200 (E1, greedy).
#
# IMPORTANT: the Stage-1 actor was trained on top of the merged SFT weights
# (model.path=artifacts/checkpoints/step_600_merged), so the adapter must be merged
# back onto THAT base, not onto the original base model.
#
# Usage: bash scripts/eval_checkpoint.sh <checkpoint_dir> <tag> [data] [base]
set -eo pipefail
source /home/ma-user/work/sqlrl/scripts/env.sh
cd /home/ma-user/work/sqlrl

CKPT="$1"
TAG="$2"
DATA="${3:-artifacts/datasets/dev200.jsonl}"
BASE="${4:-artifacts/checkpoints/step_600_merged}"
[ -n "$CKPT" ] || { echo "usage: eval_checkpoint.sh <checkpoint_dir> <tag> [data] [base]" >&2; exit 2; }
ADAPTER="$CKPT/actor/lora_adapter"
[ -d "$ADAPTER" ] || { echo "no lora_adapter under $CKPT" >&2; exit 2; }

MERGED="artifacts/checkpoints/merged_${TAG}"
rm -rf "$MERGED"
python3 scripts/merge_adapter.py --base "$BASE" \
  --adapter "$ADAPTER" --tokenizer-from "$BASE" \
  --out "$MERGED"

python3 -u eval/evaluate_agent.py \
  --data "$DATA" --limit 0 --backend unified --engine vllm \
  --model "$MERGED" \
  --out "reports/m6/dev200_${TAG}.json" \
  --max-rounds 5 --max-observation-tokens 1024 --observation-budget 1024
