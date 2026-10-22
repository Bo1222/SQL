#!/usr/bin/env bash
# Explicit stage selection. Never chains a full training/evaluation run implicitly.
set -euo pipefail
AGENT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ "${SQLRL_ENV_LOADED:-}" != "$AGENT_ROOT" ]; then
  exec python3 "$AGENT_ROOT/train/project_env.py" -- bash "$AGENT_ROOT/scripts/run_bird_agent.sh" "$@"
fi
AGENT_WANDB_MODE="${WANDB_MODE:-offline}"
if [ -f /home/ma-user/work/verl_quickstart/scripts/env.sh ]; then
  set +u
  source /home/ma-user/work/verl_quickstart/scripts/env.sh
  set -u
fi
export WANDB_MODE="$AGENT_WANDB_MODE"
export ASCEND_RT_VISIBLE_DEVICES="${CARDS:-0,1,2,3}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
cd "$AGENT_ROOT"
AGENT_STAGE="${1:-help}"
if [ "$#" -gt 0 ]; then shift; fi
AGENT_DATA="${AGENT_DATA:-data/processed/bird_agent}"
AGENT_MODEL="${AGENT_MODEL:-/home/ma-user/work/env_verify/models/Qwen2.5-3B-Instruct}"
AGENT_SEED="${SEED:-0}"
case "$AGENT_STAGE" in
  prepare)
    exec python3 -u -m data.prepare_bird_agent --download-filtered --validate-gold "$@" ;;
  sql-sft)
    exec python3 -u -m train.agent_sft --data "$AGENT_DATA/sft.jsonl" --model "$AGENT_MODEL" \
      --adapter checkpoints/sft_lora --out checkpoints/bird_sql_sft "$@" ;;
  demonstrations)
    exec python3 -u -m agent.build_sft --data "$AGENT_DATA/sft.jsonl" --model "$AGENT_MODEL" \
      --source synthetic --out "$AGENT_DATA/synthetic.jsonl" "$@" ;;
  combine)
    exec python3 -u -m data.combine_agent_sft --synthetic "$AGENT_DATA/synthetic.jsonl" \
      --on-policy "$AGENT_DATA/on_policy.jsonl" --out "$AGENT_DATA/tool_sft.jsonl" "$@" ;;
  tool-sft)
    exec python3 -u -m train.agent_sft --data "$AGENT_DATA/tool_sft.jsonl" --model "$AGENT_MODEL" \
      --adapter checkpoints/bird_sql_sft --out checkpoints/agent_sft "$@" ;;
  grpo)
    AGENT_NPROC="${NPROC:-4}"
    IFS=',' read -r -a AGENT_CARDS <<< "$ASCEND_RT_VISIBLE_DEVICES"
    if ! [[ "$AGENT_NPROC" =~ ^[1-9][0-9]*$ ]] || [ "$AGENT_NPROC" -gt "${#AGENT_CARDS[@]}" ]; then
      echo "NPROC must be positive and not exceed the visible card count" >&2; exit 2
    fi
    exec torchrun --nnodes=1 --nproc_per_node="$AGENT_NPROC" --master_addr=127.0.0.1 \
      --master_port="${MASTER_PORT:-29531}" --module -- train.agent_grpo_ddp \
      --data "$AGENT_DATA/rl.jsonl" --val-data "$AGENT_DATA/val.jsonl" --model "$AGENT_MODEL" \
      --adapter checkpoints/agent_sft --pilot-report reports/agent_pilot.jsonl.summary.json \
      --out "checkpoints/agent_grpo_seed$AGENT_SEED" --seed "$AGENT_SEED" "$@" ;;
  pilot)
    exec python3 -u -m eval.pilot_agent --data "$AGENT_DATA/rl.jsonl" --model "$AGENT_MODEL" \
      --adapter checkpoints/agent_sft --out reports/agent_pilot.jsonl "$@" ;;
  eval)
    exec python3 -u -m eval.eval_agent --data "$AGENT_DATA/val.jsonl" --model "$AGENT_MODEL" \
      --adapter checkpoints/agent_sft --out reports/agent_sft_val.json "$@" ;;
  compare)
    exec python3 -u -m eval.compare_agent "$@" ;;
  smoke)
    exec python3 -B -m pytest tests/test_agent_pipeline.py -q "$@" ;;
  *)
    echo "Usage: bash scripts/run_bird_agent.sh {prepare|sql-sft|demonstrations|combine|tool-sft|pilot|grpo|eval|compare|smoke} [arguments]"
    echo "Each stage is explicit. See docs/agent_pipeline.md."
    [ "$AGENT_STAGE" = help ] ;;
esac
