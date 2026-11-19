#!/usr/bin/env bash
# M0: single-step GRPO with Qwen2.5-3B + step_600 LoRA on one Ascend 910B4.
set -eo pipefail
source /home/ma-user/work/verl_quickstart/scripts/env.sh
cd /home/ma-user/work/sqlrl
export ASCEND_RT_VISIBLE_DEVICES="${CARDS:-0}"
unset PYTORCH_NPU_ALLOC_CONF
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export WANDB_MODE=disabled

ROOT=/home/ma-user/work/sqlrl
MODEL=/home/ma-user/work/env_verify/models/Qwen2.5-3B-Instruct
ADAPTER=$ROOT/checkpoints/bird_agent_full_sft_20261026/seed_0/step_600

python3 -u -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=$ROOT/artifacts/datasets/m0_smoke.parquet \
  data.val_files=$ROOT/artifacts/datasets/m0_smoke.parquet \
  data.train_batch_size=4 \
  data.max_prompt_length=2048 \
  data.max_response_length=512 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  custom_reward_function.path=$ROOT/reward/m0_smoke_reward.py \
  custom_reward_function.name=compute_score \
  actor_rollout_ref.model.path=$MODEL \
  actor_rollout_ref.model.lora_rank=16 \
  actor_rollout_ref.model.lora_alpha=32 \
  actor_rollout_ref.model.lora_adapter_path=$ADAPTER \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.optim.lr=1e-5 \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
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
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.max_model_len=2560 \
  actor_rollout_ref.rollout.max_num_batched_tokens=2560 \
  actor_rollout_ref.rollout.max_num_seqs=8 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  trainer.device=npu \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.logger=console \
  trainer.project_name=sqlrl \
  trainer.experiment_name=m0_smoke \
  trainer.default_local_dir=$ROOT/artifacts/checkpoints/m0_smoke \
  trainer.val_before_train=False \
  trainer.save_freq=1 \
  trainer.test_freq=-1 \
  trainer.total_epochs=1 \
  trainer.total_training_steps=1 \
  "$@"
