#!/usr/bin/env bash
set -eo pipefail
cd /home/ma-user/work/sqlrl
source scripts/env.sh
export PYTHONPATH=/home/ma-user/work/sqlrl:${PYTHONPATH:-}
python -c 'from e1_alignment.rebuild.contract import verify_contract; verify_contract()'
exec python -u -m e1_alignment.sft_train \
  --data artifacts/m6/sft_rebuild/train.jsonl \
  --quality-manifest artifacts/m6/sft_rebuild/frozen_manifest.json \
  --model artifacts/checkpoints/step_600_merged \
  --out outputs/sft_rebuild/adapter \
  --lr 2e-6 --epochs 1 --grad-accum 8 --max-seconds 3300 --max-context 6144 --save-every 0 --seed 0
