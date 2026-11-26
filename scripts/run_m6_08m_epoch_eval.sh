#!/usr/bin/env bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# M6-08m shared-config evaluation for one SFT epoch checkpoint.
#
# Usage:
#   bash scripts/run_m6_08m_epoch_eval.sh \
#     outputs/m6_08m/student_sft_full_3ep/epoch_1/hf_model \
#     reports/m6/e1_student_only/sft_full_epoch1 \
#     artifacts/datasets/dev200.jsonl
#
# The same retrieval/prompt/observation/context configuration is used for
# base-model and epoch comparisons; only the model path changes.
set -eo pipefail
cd "$ROOT"
source "$ROOT/scripts/env.sh"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

MODEL="${1:?usage: $0 MODEL_DIR OUT_PREFIX [DATA]}"
OUT_PREFIX="${2:?usage: $0 MODEL_DIR OUT_PREFIX [DATA]}"
DATA="${3:-artifacts/datasets/dev200.jsonl}"
NUM_SHARDS="${NUM_SHARDS:-4}"

if [ ! -f "$MODEL/config.json" ]; then
  echo "model dir has no config.json: $MODEL" >&2
  exit 1
fi
if [ ! -f "$DATA" ]; then
  echo "data file missing: $DATA" >&2
  exit 1
fi
EXPECTED_N=$(python -c 'import sys; print(sum(1 for line in open(sys.argv[1], encoding="utf-8") if line.strip()))' "$DATA")
OUT_DIR="$(dirname "$OUT_PREFIX")"
BASE="$(basename "$OUT_PREFIX")"
SHARD_DIR="${OUT_DIR}/${BASE}"
mkdir -p "$SHARD_DIR" logs/m6_08m_eval

pids=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  (
    export ASCEND_RT_VISIBLE_DEVICES="$i"
    python -u eval/evaluate_agent.py \
      --data "$DATA" --limit 0 \
      --shard-index "$i" --num-shards "$NUM_SHARDS" \
      --backend unified --retrieval-version E1.1 --engine vllm \
      --model "$MODEL" \
      --prompt-mode query --observation-mode compact \
      --max-rounds 5 --max-tools 4 \
      --max-schema-lookups 2 --max-sql-executes 2 \
      --max-observation-tokens 1024 --observation-budget 1024 \
      --max-context 8192 --max-model-len 8192 \
      --out "${SHARD_DIR}/shard_${i}.json" \
      --trace-dir "${SHARD_DIR}/trace_${i}" \
      > "logs/m6_08m_eval/${BASE}_shard_${i}.log" 2>&1
  ) &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
if [ "$status" -ne 0 ]; then
  echo "at least one eval shard failed; see logs/m6_08m_eval/${BASE}_shard_*.log" >&2
  exit 1
fi

SHARD_ARGS=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  SHARD_ARGS+=(--shard "${SHARD_DIR}/shard_${i}.json")
done
python scripts/merge_eval_shards.py \
  "${SHARD_ARGS[@]}" \
  --out "${OUT_PREFIX}.json" \
  --records-jsonl "${OUT_PREFIX}.jsonl" \
  --expected-n "$EXPECTED_N"
