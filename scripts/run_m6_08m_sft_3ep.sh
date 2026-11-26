#!/usr/bin/env bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# M6-08m student-only 3-epoch SFT from the built dataset.
# Usage: DATA=... OUT=... bash scripts/run_m6_08m_sft_3ep.sh
set -eo pipefail
cd "$ROOT"
source "$ROOT/scripts/env.sh"
export ASCEND_RT_VISIBLE_DEVICES="${CARDS:-0}"
DATA="${DATA:-artifacts/m6/e1_student_only/sft_student_full_prefer.jsonl}"
OUT="${OUT:-outputs/m6_08m/student_sft_3ep}"
python3 -u -m e1_alignment.student_sft_train \
  --data "$DATA" \
  --out "$OUT" \
  --model artifacts/checkpoints/step_600_merged \
  --epochs 3 --grad-accum 8 --lr 5e-5 \
  --max-context 6144 --save-every 50 --max-seconds 0 \
  --device npu:0 --seed 0 --require-trainable
