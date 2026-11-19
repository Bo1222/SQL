#!/usr/bin/env bash
# Qwen3-4B-Instruct-2507 zero-shot probe on the frozen E1 dev200 contract.
# Usage: bash scripts/run_qwen3_4b_eval.sh smoke|full
set -o pipefail
source /home/ma-user/work/sqlrl/scripts/env.sh
cd /home/ma-user/work/sqlrl
export PYTHONPATH=/home/ma-user/work/sqlrl:${PYTHONPATH:-}

MODE=${1:-smoke}
MODEL=${MODEL:-/home/ma-user/work/env_verify/models/Qwen3-4B-Instruct-2507-ms}
OUTDIR=${OUTDIR:-reports/m6/qwen3_4b_zero_shot}
GPU_MEM=${GPU_MEM:-0.55}
NPU_ID=${NPU_ID:-2}
CHAT_TEMPLATE=${CHAT_TEMPLATE:-1}
STOP_AFTER_ACTION=${STOP_AFTER_ACTION:-1}
TAG=${TAG:-chat}
mkdir -p "$OUTDIR"

for f in config.json tokenizer.json tokenizer_config.json \
         model-00001-of-00003.safetensors model-00002-of-00003.safetensors \
         model-00003-of-00003.safetensors; do
  if [ ! -s "$MODEL/$f" ]; then
    echo "missing model file: $MODEL/$f" >&2
    exit 2
  fi
done

case "$MODE" in
  smoke)
    LIMIT=2
    OUT="$OUTDIR/smoke_${TAG}.json"
    TRACE="$OUTDIR/smoke_${TAG}_traces"
    ;;
  full)
    LIMIT=0
    OUT="$OUTDIR/dev200_zero_shot_${TAG}.json"
    TRACE="$OUTDIR/traces_${TAG}"
    ;;
  *)
    echo "usage: $0 smoke|full" >&2
    exit 2
    ;;
esac

COMMON=(--data artifacts/datasets/dev200.jsonl --limit "$LIMIT" --backend unified --engine vllm
  --max-rounds 5 --max-observation-tokens 1024 --observation-budget 1024
  --observation-mode compact --max-context 6144 --max-model-len 6144
  --gpu-memory-utilization "$GPU_MEM")
[ "$CHAT_TEMPLATE" = "1" ] && COMMON+=(--chat-template)
[ "$STOP_AFTER_ACTION" = "1" ] && COMMON+=(--stop-after-action)

echo "[qwen3-4b] mode=$MODE model=$MODEL limit=$LIMIT gpu_mem=$GPU_MEM" | tee "$OUT.log"
: > "$OUTDIR/hbm_samples.log"

python3 -u eval/evaluate_agent.py "${COMMON[@]}" --model "$MODEL" \
  --out "$OUT" --trace-dir "$TRACE" >> "$OUT.log" 2>&1 &
PID=$!
(
  while kill -0 "$PID" 2>/dev/null; do
    pct=$(npu-smi info -t usages -i "$NPU_ID" 2>/dev/null | awk '/HBM Usage Rate/{print $NF; exit}')
    if [ -n "${pct:-}" ]; then
      echo "$(date +%s) $pct" >> "$OUTDIR/hbm_samples.log"
    fi
    sleep 2
  done
) &
MON=$!

wait "$PID"
RC=$?
kill "$MON" 2>/dev/null || true
wait "$MON" 2>/dev/null || true

if [ "$RC" -eq 0 ]; then
  echo "[qwen3-4b] done rc=$RC out=$OUT"
else
  echo "[qwen3-4b] FAILED rc=$RC; see $OUT.log" >&2
fi
exit "$RC"
