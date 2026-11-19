#!/usr/bin/env bash
# M6-01: 5-step throughput calibration at 8 prompts x n=4, with rollout tracing for coverage.
set -eo pipefail
source /home/ma-user/work/verl_quickstart/scripts/env.sh
cd /home/ma-user/work/sqlrl
export ASCEND_RT_VISIBLE_DEVICES="${CARDS:-0}"
export PYTHONPATH=/home/ma-user/work/sqlrl:$PYTHONPATH
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export WANDB_MODE=disabled
export SQLRL_TRACE_DIR=/home/ma-user/work/sqlrl/reports/m6/traces_calib
unset PYTORCH_NPU_ALLOC_CONF

ROOT=/home/ma-user/work/sqlrl
MODEL=/home/ma-user/work/env_verify/models/Qwen2.5-3B-Instruct
ADAPTER=$ROOT/checkpoints/bird_agent_full_sft_20261026/seed_0/step_600
rm -rf "$SQLRL_TRACE_DIR"

python3 -u -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=$ROOT/artifacts/datasets/rl_full.parquet \
  data.val_files=$ROOT/artifacts/datasets/rl_full.parquet \
  data.train_batch_size=8 \
  data.max_prompt_length=2048 \
  data.max_response_length=2048 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  reward.custom_reward_function.path=$ROOT/reward/bird_ex_reward.py \
  reward.custom_reward_function.name=compute_score \
  actor_rollout_ref.model.path=$MODEL \
  actor_rollout_ref.model.lora_rank=16 \
  actor_rollout_ref.model.lora_alpha=32 \
  actor_rollout_ref.model.lora_adapter_path=$ADAPTER \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.ref.use_torch_compile=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.max_model_len=6144 \
  actor_rollout_ref.rollout.max_num_batched_tokens=6144 \
  actor_rollout_ref.rollout.max_num_seqs=16 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.agent.default_agent_loop=bird_sql_agent \
  actor_rollout_ref.rollout.agent.agent_loop_config_path=$ROOT/configs/agent/agent_loops.yaml \
  +bird_agent.catalog_dir=$ROOT/artifacts/catalog \
  +bird_agent.max_rounds=5 \
  +bird_agent.max_tools=4 \
  +bird_agent.max_schema_lookups=2 \
  +bird_agent.max_sql_executes=2 \
  +bird_agent.max_observation_tokens=1024 \
  +bird_agent.max_action_tokens=512 \
  +bird_agent.max_context=6144 \
  +bird_agent.tool_timeout=10 \
  +bird_agent.preview_rows=3 \
  +bird_agent.use_chat_template=false \
  trainer.device=npu \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.logger=console \
  trainer.project_name=sqlrl \
  trainer.experiment_name=m6_calib \
  trainer.default_local_dir=$ROOT/artifacts/checkpoints/m6_calib \
  trainer.val_before_train=False \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.total_epochs=1 \
  trainer.total_training_steps=5 \
  "$@"

