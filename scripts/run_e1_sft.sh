#!/usr/bin/env bash
# E1-06: segment-masked LoRA SFT V2 from the frozen SFT capability.
# Budget (实测 2.4 s/micro-step at ~1k tokens): 1 epoch over <=1,200 examples
# with grad_accum=8 is about 40-60 min, under the 1 h training gate.
set -eo pipefail
source /home/ma-user/work/sqlrl/scripts/env.sh
cd /home/ma-user/work/sqlrl
export PYTHONPATH=/home/ma-user/work/sqlrl:$PYTHONPATH

python3 -u -m e1_alignment.sft_train \
  --data artifacts/m6/e1_alignment/e1_alignment_sft.jsonl \
  --model artifacts/checkpoints/step_600_merged \
  --out outputs/e1_alignment/e1_alignment_adapter_v2 \
  --epochs "${EPOCHS:-1}" --grad-accum "${GRAD_ACCUM:-8}" --lr "${LR:-1e-5}" \
  --max-context 6144 --save-every "${SAVE_EVERY:-50}" --seed 0 "$@"
