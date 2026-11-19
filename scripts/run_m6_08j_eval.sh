#!/usr/bin/env bash
# Each arm must be invoked via rebuild.budget; candidate requires final completed epoch.
set -eo pipefail
cd /home/ma-user/work/sqlrl
source scripts/env.sh
export PYTHONPATH=/home/ma-user/work/sqlrl:${PYTHONPATH:-}
python -c 'from e1_alignment.rebuild.contract import verify_contract; verify_contract()'
STAGE=$1
MODEL=artifacts/checkpoints/step_600_merged
DATA=artifacts/m6/sft_rebuild/monitor64.jsonl
if [ "$STAGE" != monitor_sft ]; then
  python - <<'PY'
import json
from pathlib import Path
s=json.loads(Path('outputs/sft_rebuild/adapter/train_summary.json').read_text())
assert s['completed'] is True and s['stop_reason'] is None, 'incomplete final checkpoint'
PY
  MODEL=outputs/sft_rebuild/merged
  if [ ! -f "$MODEL/config.json" ]; then
    python scripts/merge_adapter.py --base artifacts/checkpoints/step_600_merged --adapter outputs/sft_rebuild/adapter --out "$MODEL"
  fi
fi
if [ "$STAGE" = dev200_candidate ]; then
  python - <<'PY'
import json
from pathlib import Path
assert json.loads(Path('reports/m6/sft_rebuild/monitor_gate.json').read_text())['pass'] is True
PY
  DATA=artifacts/datasets/dev200.jsonl
elif [ "$STAGE" != monitor_sft ] && [ "$STAGE" != monitor_candidate ]; then
  echo "unknown evaluation stage" >&2
  exit 1
fi
OUT=reports/m6/sft_rebuild/$STAGE.json
[ ! -e "$OUT" ] && [ ! -e "${OUT%.json}.jsonl" ]
python -u eval/evaluate_agent.py --data "$DATA" --limit 0 --backend unified --engine vllm --model "$MODEL" \
  --max-rounds 5 --max-observation-tokens 1024 --observation-budget 1024 --observation-mode compact \
  --max-context 6144 --max-model-len 6144 --out "$OUT"
