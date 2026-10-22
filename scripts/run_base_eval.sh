#!/usr/bin/env bash
set -euo pipefail
source /home/ma-user/work/verl_quickstart/scripts/env.sh
cd /home/ma-user/work/sqlrl

ROOT=${ROOT:-data/raw/dev_extract/dev_20240627}
MODEL=${MODEL:-/home/ma-user/work/env_verify/models/Qwen2.5-3B-Instruct}
LIMIT=${LIMIT:-200}

python3 -u data/build_bird.py --root "$ROOT" --out data/processed --limit "$LIMIT"
python3 -u data/functional_check.py \
  --data data/processed/dev_200.jsonl \
  --tokenizer "$MODEL" \
  --out reports/functional_check_200.json
python3 -u eval/eval_base.py \
  --data data/processed/dev_200.jsonl \
  --model "$MODEL" \
  --limit "$LIMIT" \
  --batch-size 4 \
  --out reports/base_eval_200.json
