#!/usr/bin/env bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# SFT-compatibility diagnostic: can the frozen SFT checkpoint support RL gains?
#
# A: legacy observation encoder, autonomous lookup  (current Stage-1 runtime)
# B: compact observation encoder (drop retrieval_metadata), autonomous lookup
# C: compact encoder + forced first schema_lookup
#
# All arms use the same frozen SFT checkpoint and dev200, greedy, 1 card per arm.
set -o pipefail
cd "$ROOT"
mkdir -p reports/m6/diag logs

run_arm() {
  local card=$1 name=$2 mode=$3 forced=$4
  (
    export CARDS="$card"
    source "$ROOT/scripts/env.sh"
    local extra=()
    if [ "$forced" = "1" ]; then
      extra+=(--force-first-schema-lookup)
    fi
    python3 -u eval/evaluate_agent.py \
      --data artifacts/datasets/dev200.jsonl --limit 0 --backend unified --engine vllm \
      --model artifacts/checkpoints/step_600_merged \
      --out "reports/m6/diag_sft_${name}.json" \
      --max-rounds 5 --max-observation-tokens 1024 --observation-budget 1024 \
      --observation-mode "$mode" "${extra[@]}"
  ) > "logs/diag_sft_${name}.log" 2>&1
  echo "${name} exit=$?"
}

run_arm 0 A legacy 0 &
p0=$!
run_arm 1 B compact 0 &
p1=$!
run_arm 2 C compact 1 &
p2=$!

wait "$p0" "$p1" "$p2"
echo "done"
