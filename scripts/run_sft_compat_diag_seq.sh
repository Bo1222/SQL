#!/usr/bin/env bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Sequential B/C arms on the single visible NPU (this session only exposes one).
# A is already complete as reports/m6/diag_sft_A.json.
set -o pipefail
cd "$ROOT"
mkdir -p reports/m6 logs
export CARDS=0
source "$ROOT/scripts/env.sh"

run_arm() {
  local name=$1
  shift
  python3 -u eval/evaluate_agent.py \
    --data artifacts/datasets/dev200.jsonl --limit 0 --backend unified --engine vllm \
    --model artifacts/checkpoints/step_600_merged \
    --out "reports/m6/diag_sft_${name}.json" \
    --max-rounds 5 --max-observation-tokens 1024 --observation-budget 1024 \
    "$@" > "logs/diag_sft_${name}.log" 2>&1
  echo "${name} exit=$?"
}

run_arm B --observation-mode compact
run_arm C --observation-mode compact --force-first-schema-lookup
echo "done"
