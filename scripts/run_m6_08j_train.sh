#!/usr/bin/env bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set -eo pipefail
cd "$ROOT"
source "$ROOT/scripts/env.sh"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
python -c 'from e1_alignment.rebuild.contract import verify_contract; verify_contract()'
exec python -u -m e1_alignment.sft_train \
  --data artifacts/m6/sft_rebuild/train.jsonl \
  --quality-manifest artifacts/m6/sft_rebuild/frozen_manifest.json \
  --model artifacts/checkpoints/step_600_merged \
  --out outputs/sft_rebuild/adapter \
  --lr 2e-6 --epochs 1 --grad-accum 8 --max-seconds 3300 --max-context 6144 --save-every 0 --seed 0
