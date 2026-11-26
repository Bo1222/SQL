#!/usr/bin/env bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# GRPO / DAPO launcher; explicit A typed contract and visible-device count.
#
# This script is intentionally dry-run by default.  RL must not start until the
# SFT final test (test1534) gate passes and the user explicitly authorises the
# run.  Use:
#
#   bash scripts/run_m6_08m_rl.sh                 # print the exact command
#   RUN=1 bash scripts/run_m6_08m_rl.sh           # execute (still gated)
#
# Environment:
#   ALGO=grpo|dapo           algorithm preset
#   MODEL=...                full HF checkpoint (no adapter)
#   TRAIN/VAL=...            named-runtime parquet built by data/build_rl_dataset.py
#   RUNTIME_CONTRACT=...     default A-e1.2-typed-v1; legacy is opt-in
#   CARDS=0                 logical devices visible inside the container
#   STEPS=...                optimizer steps (use 1 for a smoke)
#   PROMPTS=8                prompts per update
#   N=4                      rollouts per prompt
#   LR=1e-6                  actor LR
#   MINI_BATCH=8             ppo_mini_batch_size in prompts
#   CLIP_LOW/CLIP_HIGH       asymmetric clip (DAPO: 0.2/0.28)
#   KL_COEF                  KL coefficient (GRPO 0.001, DAPO 0.0)
#   DRY_RUN=1|0              default 1
#   REQUIRE_SFT_GATE=1|0     default 1 when RUN=1
#   TEST1534_BASE/SFT/PAIRED=...  final-test reports used by the gate
#   ENABLE_FILTER_GROUPS=1   requires project-local TRAINER=dapo_ds
set -eo pipefail
cd "$ROOT"

CALLER_WANDB_MODE="${WANDB_MODE:-}"
source "$ROOT/scripts/env.sh"
if [ -f .env ]; then
  set -a
  source .env
  set +a
fi
if [ -n "$CALLER_WANDB_MODE" ]; then
  export WANDB_MODE="$CALLER_WANDB_MODE"
fi
if [ -n "${CPU_THREADS:-}" ]; then
  export OMP_NUM_THREADS="$CPU_THREADS"
  export MKL_NUM_THREADS="$CPU_THREADS"
fi
export ASCEND_RT_VISIBLE_DEVICES="${CARDS:-0}"
IFS=',' read -r -a SQLRL_CARD_IDS <<< "$ASCEND_RT_VISIBLE_DEVICES"
NPU_COUNT="${NPU_COUNT:-${#SQLRL_CARD_IDS[@]}}"
if [ "$NPU_COUNT" -eq 1 ]; then
  DEFAULT_FSDP_CPU_POLICY=True
else
  DEFAULT_FSDP_CPU_POLICY=False
fi
FSDP_CPU_OFFLOAD_POLICY="${FSDP_CPU_OFFLOAD_POLICY:-$DEFAULT_FSDP_CPU_POLICY}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0

ALGO="${ALGO:-grpo}"
TRAINER="${TRAINER:-ppo}"
MODEL="${MODEL:-outputs/m6_08m/student_sft_full_lowlr_5e6/epoch_2/hf_model}"
RUNTIME_CONTRACT="${RUNTIME_CONTRACT:-A-e1.2-typed-v1}"
if [ "$RUNTIME_CONTRACT" = "legacy" ]; then
  TRAIN="${TRAIN:-artifacts/datasets/rl_e1_1.parquet}"
  VAL="${VAL:-artifacts/datasets/dev200_rl_e1_1.parquet}"
  RETRIEVAL_VERSION=E1.1
  OBSERVATION_MODE=compact
  REWARD_FUNCTION=compute_score
  REWARD_PATH=reward/bird_ex_reward.py
  DEFAULT_MODEL_LEN=6144
else
  case "$RUNTIME_CONTRACT" in
    A-e1.2-typed-v1)
      DEFAULT_TRAIN="artifacts/datasets/rl_A_typed_v1.parquet"
      DEFAULT_VAL="artifacts/datasets/dev200_A_typed_v1.parquet"
      ;;
    A-e1.2-typed-v3)
      DEFAULT_TRAIN="artifacts/datasets/rl_A_typed_v3_reviewed.parquet"
      DEFAULT_VAL="artifacts/datasets/dev200_A_typed_v3.parquet"
      ;;
    *)
      echo "unknown RUNTIME_CONTRACT=$RUNTIME_CONTRACT" >&2
      exit 2
      ;;
  esac
  TRAIN="${TRAIN:-$DEFAULT_TRAIN}"
  VAL="${VAL:-$DEFAULT_VAL}"
  RETRIEVAL_VERSION=E1.2
  OBSERVATION_MODE=compact_v2
  REWARD_FUNCTION=compute_score_with_details
  REWARD_PATH=reward/typed_rl_reward.py
  DEFAULT_MODEL_LEN=8192
fi
STEPS="${STEPS:-1}"
EPOCHS="${EPOCHS:-1}"
PROMPTS="${PROMPTS:-8}"
TARGET_GROUPS="${TARGET_GROUPS:-$PROMPTS}"
MAX_GEN_BATCHES="${MAX_GEN_BATCHES:-3}"
N="${N:-4}"
AGENT_WORKERS="${AGENT_WORKERS:-}"
if [ -z "$AGENT_WORKERS" ]; then
  AGENT_WORKERS=8
  while [ "$AGENT_WORKERS" -gt 1 ] && [ $(((PROMPTS * N) % AGENT_WORKERS)) -ne 0 ]; do
    AGENT_WORKERS=$((AGENT_WORKERS - 1))
  done
fi
LR="${LR:-1e-6}"
DATA_SEED="${DATA_SEED:-42}"
MINI_BATCH="${MINI_BATCH:-8}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-2048}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-4096}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$DEFAULT_MODEL_LEN}"
MAX_ACTION_TOKENS="${MAX_ACTION_TOKENS:-512}"
MAX_OBSERVATION_TOKENS="${MAX_OBSERVATION_TOKENS:-1024}"
MAX_ROUNDS="${MAX_ROUNDS:-5}"
MAX_TOOLS="${MAX_TOOLS:-4}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.30}"
CALCULATE_LOG_PROBS="${CALCULATE_LOG_PROBS:-0}"
SAVE_FREQ="${SAVE_FREQ:-10}"
MAX_CKPT_KEEP="${MAX_CKPT_KEEP:-2}"
# Save model+extra by default to limit disk usage. Set SAVE_OPTIMIZER=1
# only for a dedicated resume run.
SAVE_OPTIMIZER="${SAVE_OPTIMIZER:-0}"
if [ "$SAVE_OPTIMIZER" = "1" ]; then
  SAVE_CONTENTS="${SAVE_CONTENTS:-model,optimizer,extra}"
else
  SAVE_CONTENTS="${SAVE_CONTENTS:-model,extra}"
fi
DRY_RUN="${DRY_RUN:-1}"
if [ "${RUN:-0}" = "1" ]; then
  DRY_RUN=0
fi
REQUIRE_SFT_GATE="${REQUIRE_SFT_GATE:-$([ "$DRY_RUN" = "0" ] && echo 1 || echo 0)}"
ENABLE_FILTER_GROUPS="${ENABLE_FILTER_GROUPS:-0}"
WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_PROJECT="${WANDB_PROJECT:-sqlrl}"
RUN_NAME="${RUN_NAME:-m6_08m_rl_${ALGO}_step${STEPS}}"
OUT="${OUT:-outputs/m6_08m/rl_${ALGO}_$(date +%Y%m%d_%H%M%S)}"

if [ "$ALGO" != "grpo" ] && [ "$ALGO" != "dapo" ]; then
  echo "ALGO must be grpo or dapo, got $ALGO" >&2
  exit 2
fi
if [ "$TRAINER" != "ppo" ] && [ "$TRAINER" != "dapo_ds" ]; then
  echo "TRAINER must be ppo or dapo_ds, got $TRAINER" >&2
  exit 2
fi
if [ $((MAX_PROMPT_LEN + MAX_RESPONSE_LEN)) -gt "$MAX_MODEL_LEN" ]; then
  echo "max_prompt_length + max_response_length must be <= max_model_len" >&2
  exit 2
fi
if [ "$N" -lt 2 ]; then
  echo "GRPO requires N>=2 rollouts per prompt" >&2
  exit 2
fi
if [ "$AGENT_WORKERS" -lt 1 ] || [ $(((PROMPTS * N) % AGENT_WORKERS)) -ne 0 ]; then
  echo "PROMPTS*N must be divisible by AGENT_WORKERS for this veRL AgentLoop" >&2
  exit 2
fi
if [ "$NPU_COUNT" -lt 1 ] || [ "$NPU_COUNT" -ne "${#SQLRL_CARD_IDS[@]}" ]; then
  echo "NPU_COUNT must match CARDS" >&2
  exit 2
fi
if [ $((PROMPTS % NPU_COUNT)) -ne 0 ] || [ $((MINI_BATCH % NPU_COUNT)) -ne 0 ]; then
  echo "PROMPTS and MINI_BATCH must be divisible by NPU_COUNT" >&2
  exit 2
fi
if [ "$TRAINER" = "dapo_ds" ]; then
  if [ "$ENABLE_FILTER_GROUPS" != "1" ]; then
    echo "TRAINER=dapo_ds requires ENABLE_FILTER_GROUPS=1" >&2
    exit 2
  fi
  if [ "$TARGET_GROUPS" -ne "$PROMPTS" ]; then
    echo "TARGET_GROUPS must equal PROMPTS (target groups per update)" >&2
    exit 2
  fi
  if [ "$MAX_GEN_BATCHES" -lt 1 ]; then
    echo "MAX_GEN_BATCHES must be >= 1" >&2
    exit 2
  fi
elif [ "$ENABLE_FILTER_GROUPS" = "1" ]; then
  echo "ENABLE_FILTER_GROUPS=1 is only allowed with TRAINER=dapo_ds" >&2
  exit 2
fi

if [ "$CALCULATE_LOG_PROBS" = "1" ]; then
  CALC_LOG_PROBS_BOOL=True
else
  CALC_LOG_PROBS_BOOL=False
fi

if [ "$ALGO" = "dapo" ]; then
  CLIP_LOW="${CLIP_LOW:-0.2}"
  CLIP_HIGH="${CLIP_HIGH:-0.28}"
  KL_COEF="${KL_COEF:-0.0}"
  USE_KL_LOSS="${USE_KL_LOSS:-False}"
  ADV_NORM="${ADV_NORM:-True}"
  LOSS_AGG="${LOSS_AGG:-token-mean}"
  REWARD_MANAGER="${REWARD_MANAGER:-dapo}"
  # DAPORewardManager penalises the full response length, including tool
  # observations.  In a multi-turn agent those tokens are not policy actions,
  # so keep overlong shaping off until an action-token-length implementation
  # is verified.
  OVERLONG_BUFFER="${OVERLONG_BUFFER:-0}"
else
  CLIP_LOW="${CLIP_LOW:-0.2}"
  CLIP_HIGH="${CLIP_HIGH:-0.2}"
  KL_COEF="${KL_COEF:-0.001}"
  USE_KL_LOSS="${USE_KL_LOSS:-True}"
  ADV_NORM="${ADV_NORM:-True}"
  LOSS_AGG="${LOSS_AGG:-token-mean}"
  REWARD_MANAGER="${REWARD_MANAGER:-naive}"
  OVERLONG_BUFFER="${OVERLONG_BUFFER:-0}"
fi

if [ ! -f "$MODEL/config.json" ]; then
  echo "model checkpoint missing config.json: $MODEL" >&2
  exit 2
fi
if [ ! -f "$TRAIN" ] || [ ! -f "$VAL" ]; then
  echo "missing parquet: TRAIN=$TRAIN VAL=$VAL" >&2
  exit 2
fi

mkdir -p "$OUT" "$(dirname "$OUT.readiness.log")" logs/m6_08m_rl
TEST1534_BASE="${TEST1534_BASE:-reports/m6/e1_student_only/test1534_base_e1_1.json}"
TEST1534_SFT="${TEST1534_SFT:-reports/m6/e1_student_only/test1534_sft_lowlr_788.json}"
TEST1534_PAIRED="${TEST1534_PAIRED:-reports/m6/e1_student_only/test1534_paired_lowlr_788.json}"
READINESS_ARGS=(--model "$MODEL" --train-parquet "$TRAIN" --val-parquet "$VAL"
                --test1534-base "$TEST1534_BASE" --test1534-sft "$TEST1534_SFT"
                --test1534-paired "$TEST1534_PAIRED"
                --out "$OUT/readiness.json")
if [ "$RUNTIME_CONTRACT" != "legacy" ]; then
  READINESS_ARGS+=(--runtime-contract "$RUNTIME_CONTRACT")
fi
if [ "$REQUIRE_SFT_GATE" = "1" ]; then
  READINESS_ARGS+=(--require-sft-gate)
fi
if [ "$TRAINER" = "dapo_ds" ]; then
  READINESS_ARGS+=(--require-dynamic-sampling)
fi
set +e
python3 -u scripts/check_rl_readiness.py "${READINESS_ARGS[@]}" > "$OUT.readiness.log" 2>&1
READINESS_RC=$?
set -e
if [ "$READINESS_RC" -ne 0 ]; then
  echo "RL readiness check failed (rc=$READINESS_RC); see $OUT.readiness.log" >&2
  tail -n 40 "$OUT.readiness.log" >&2 || true
  if [ "$DRY_RUN" = "0" ]; then
    exit 2
  fi
fi
if [ "$TRAINER" = "dapo_ds" ]; then
  ENTRY_MODULE=train.main_dapo
elif [ "$RUNTIME_CONTRACT" != "legacy" ]; then
  ENTRY_MODULE=train.main_grpo
else
  ENTRY_MODULE=verl.trainer.main_ppo
fi
CMD=(python3 -u -m "$ENTRY_MODULE"
  algorithm.adv_estimator=grpo
  algorithm.norm_adv_by_std_in_grpo="$ADV_NORM"
  algorithm.use_kl_in_reward=False
  algorithm.kl_ctrl.kl_coef=0.0

  data.train_files="$TRAIN"
  data.val_files="$VAL"
  data.train_batch_size="$PROMPTS"
  data.max_prompt_length="$MAX_PROMPT_LEN"
  data.max_response_length="$MAX_RESPONSE_LEN"
  data.truncation=error
  data.filter_overlong_prompts=True
  data.return_raw_chat=True
  data.shuffle=True
  data.seed="$DATA_SEED"

  reward.reward_manager.name="$REWARD_MANAGER"
  reward.custom_reward_function.path="$PWD/$REWARD_PATH"
  reward.custom_reward_function.name="$REWARD_FUNCTION"

  actor_rollout_ref.model.path="$MODEL"
  actor_rollout_ref.model.lora_rank=0
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  actor_rollout_ref.model.use_remove_padding=False

  actor_rollout_ref.actor.strategy=fsdp2
  "actor_rollout_ref.actor.checkpoint.save_contents=[${SAVE_CONTENTS}]"
  actor_rollout_ref.actor.fsdp_config.model_dtype=fp32
  actor_rollout_ref.actor.fsdp_config.param_offload=True
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
  actor_rollout_ref.actor.fsdp_config.offload_policy="$FSDP_CPU_OFFLOAD_POLICY"
  actor_rollout_ref.actor.fsdp_config.forward_prefetch=False
  actor_rollout_ref.actor.fsdp_config.use_orig_params=True
  actor_rollout_ref.actor.optim.lr="$LR"
  actor_rollout_ref.actor.optim.weight_decay=0.0
  actor_rollout_ref.actor.optim.lr_scheduler_type=constant
  actor_rollout_ref.actor.ppo_mini_batch_size="$MINI_BATCH"
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
  actor_rollout_ref.actor.use_dynamic_bsz=False
  actor_rollout_ref.actor.grad_clip=1.0
  actor_rollout_ref.actor.use_kl_loss="$USE_KL_LOSS"
  actor_rollout_ref.actor.kl_loss_coef="$KL_COEF"
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.actor.clip_ratio_low="$CLIP_LOW"
  actor_rollout_ref.actor.clip_ratio_high="$CLIP_HIGH"
  actor_rollout_ref.actor.loss_agg_mode="$LOSS_AGG"
  actor_rollout_ref.actor.entropy_coeff=0.0

  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.ref.fsdp_config.model_dtype=fp32
  actor_rollout_ref.ref.fsdp_config.param_offload=True

  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.mode=async
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEM_UTIL"
  actor_rollout_ref.rollout.n="$N"
  actor_rollout_ref.rollout.temperature=1.0
  actor_rollout_ref.rollout.top_p=1.0
  actor_rollout_ref.rollout.top_k=-1
  actor_rollout_ref.rollout.enforce_eager=True
  actor_rollout_ref.rollout.enable_chunked_prefill=False
  actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN"
  actor_rollout_ref.rollout.max_num_batched_tokens="$MAX_MODEL_LEN"
  actor_rollout_ref.rollout.max_num_seqs=16
  actor_rollout_ref.rollout.calculate_log_probs="$CALC_LOG_PROBS_BOOL"
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.rollout.agent.default_agent_loop=bird_sql_agent
  actor_rollout_ref.rollout.agent.num_workers="$AGENT_WORKERS"
  actor_rollout_ref.rollout.agent.agent_loop_config_path="$PWD/configs/agent/agent_loops.yaml"

  +bird_agent.catalog_dir="$PWD/artifacts/catalog"
  +bird_agent.retrieval_version="$RETRIEVAL_VERSION"
  +bird_agent.max_rounds="$MAX_ROUNDS"
  +bird_agent.max_tools="$MAX_TOOLS"
  +bird_agent.max_schema_lookups=2
  +bird_agent.max_sql_executes=2
  +bird_agent.max_invalid_actions=2
  +bird_agent.max_observation_tokens="$MAX_OBSERVATION_TOKENS"
  +bird_agent.observation_mode="$OBSERVATION_MODE"
  +bird_agent.max_action_tokens="$MAX_ACTION_TOKENS"
  +bird_agent.max_context="$MAX_MODEL_LEN"
  +bird_agent.tool_timeout=10
  +bird_agent.preview_rows=3
  +bird_agent.use_chat_template=false

  trainer.device=npu
  trainer.n_gpus_per_node="$NPU_COUNT"
  trainer.nnodes=1
  trainer.logger='[console,wandb]'
  trainer.project_name="$WANDB_PROJECT"
  trainer.experiment_name="$RUN_NAME"
  trainer.default_local_dir="$OUT"
  trainer.resume_mode=disable
  trainer.val_before_train=False
  trainer.test_freq=-1
  trainer.save_freq="$SAVE_FREQ"
  trainer.max_actor_ckpt_to_keep="$MAX_CKPT_KEEP"
  trainer.total_epochs="$EPOCHS"
  trainer.total_training_steps="$STEPS"
)
if [ "$RUNTIME_CONTRACT" != "legacy" ]; then
  CMD+=("+bird_agent.runtime_contract=$RUNTIME_CONTRACT"
        "+reward_filter.max_gen_batches=$MAX_GEN_BATCHES")
fi
if [ -n "${CPU_THREADS:-}" ]; then
  CMD+=("+ray_kwargs.ray_init.runtime_env.env_vars.OMP_NUM_THREADS='$CPU_THREADS'"
        "+ray_kwargs.ray_init.runtime_env.env_vars.MKL_NUM_THREADS='$CPU_THREADS'")
fi

if [ "$TRAINER" = "dapo_ds" ]; then
  CMD+=(
    "+algorithm.filter_groups.enable=True"
    "+algorithm.filter_groups.metric=seq_final_reward"
    "+algorithm.filter_groups.max_num_gen_batches=$MAX_GEN_BATCHES"
    "+dapo.target_groups=$TARGET_GROUPS"
  )
fi
if [ "$ALGO" = "dapo" ]; then
  CMD+=(actor_rollout_ref.actor.clip_ratio_c=10.0)
fi
if [ "$ALGO" = "dapo" ] && [ "$OVERLONG_BUFFER" -gt 0 ]; then
  CMD+=(
    +reward.reward_kwargs.max_resp_len="$MAX_RESPONSE_LEN"
    +reward.reward_kwargs.overlong_buffer_cfg.enable=True
    +reward.reward_kwargs.overlong_buffer_cfg.len="$OVERLONG_BUFFER"
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0
    +reward.reward_kwargs.overlong_buffer_cfg.log=False
  )
fi

python3 - "$OUT/command.json" "${CMD[@]}" <<'PY'
import json
import os
import sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps(sys.argv[2:], indent=2) + "\n")
Path(sys.argv[1]).with_name("environment.json").write_text(json.dumps({
    k: os.environ.get(k) for k in ("ASCEND_RT_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "WANDB_MODE")
}, indent=2) + "\n")
PY

mkdir -p "$OUT" logs/m6_08m_rl
export SQLRL_TRACE_DIR="$OUT/traces"
LOG="logs/m6_08m_rl/${RUN_NAME}.log"
echo "ALGO=$ALGO TRAINER=$TRAINER MODEL=$MODEL STEPS=$STEPS PROMPTS=$PROMPTS N=$N LR=$LR" | tee -a "$LOG"
echo "TRAIN=$TRAIN VAL=$VAL OUT=$OUT WANDB_MODE=$WANDB_MODE TARGET_GROUPS=$TARGET_GROUPS MAX_GEN_BATCHES=$MAX_GEN_BATCHES" | tee -a "$LOG"
echo "clip=$CLIP_LOW/$CLIP_HIGH kl=$KL_COEF loss_agg=$LOSS_AGG reward_manager=$REWARD_MANAGER" | tee -a "$LOG"
if [ "$USE_KL_LOSS" = "True" ]; then
  echo "[warn] use_kl_loss=True creates a separate frozen ref model; 3B full-FT memory may not fit alongside vLLM. Smoke before a long run." | tee -a "$LOG"
fi

if [ "$DRY_RUN" = "1" ]; then
  printf '%q ' "${CMD[@]}" | tee -a "$LOG"
  printf '\n' | tee -a "$LOG"
  echo "[dry-run] set RUN=1 to execute (requires test1534 SFT gate)" | tee -a "$LOG"
  exit 0
fi

"${CMD[@]}" 2>&1 | tee -a "$LOG"
