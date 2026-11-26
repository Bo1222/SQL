#!/usr/bin/env bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# M6-08l L2: merge the Qwen3-4B LoRA SFT adapter and run the raw E1 dev200 contract.
# Usage: bash scripts/run_qwen3_4b_sft_eval.sh check|full
set -o pipefail
source "$ROOT/scripts/env.sh"
cd "$ROOT"
export PYTHONPATH="$ROOT/.codex-deps/m6-08k:$ROOT:${PYTHONPATH:-}"

MODE=${1:-check}
ADAPTER=${ADAPTER:-outputs/qwen3_4b_sft/e1_lora_v1}
MERGED=${MERGED:-outputs/qwen3_4b_sft/e1_lora_v1_merged}
BASE=${BASE:-${SQLRL_MODEL_ROOT:-$HOME/work/env_verify/models}/Qwen3-4B-Instruct-2507-ms}
OUTDIR=${OUTDIR:-reports/m6/qwen3_4b_sft}
GPU_MEM=${GPU_MEM:-0.55}
mkdir -p "$OUTDIR"

if [ ! -f "$MERGED/config.json" ]; then
  python3 scripts/merge_adapter.py --base "$BASE" --adapter "$ADAPTER" --out "$MERGED"
fi

case "$MODE" in
  check) LIMIT=50; OUT="$OUTDIR/check50_sft.json"; TRACE="$OUTDIR/check50_traces" ;;
  full)  LIMIT=0;  OUT="$OUTDIR/dev200_sft.json"; TRACE="$OUTDIR/traces_sft" ;;
  *) echo "usage: $0 check|full" >&2; exit 2 ;;
esac

COMMON=(--data artifacts/datasets/dev200.jsonl --limit "$LIMIT" --backend unified --engine vllm
  --model "$MERGED" --max-rounds 5 --max-observation-tokens 1024 --observation-budget 1024
  --observation-mode compact --max-context 6144 --max-model-len 6144
  --gpu-memory-utilization "$GPU_MEM")
python3 -u eval/evaluate_agent.py "${COMMON[@]}" --out "$OUT" --trace-dir "$TRACE" 2>&1 | tee "$OUT.log"
