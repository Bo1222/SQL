"""DAPO dynamic-sampling trainer for the project's async AgentLoop.

This is a project-local subclass; it does not modify the installed veRL
package.  The update path is the current ``RayPPOTrainer`` path (FSDP2 actor,
async vLLM-Ascend rollout, checkpoint manager weight sync).  The only changed
part is the sampling loop: generate/reward/filter mixed terminal groups and
accumulate until the target is reached or the generation budget is exhausted.
"""
from __future__ import annotations

import time
import uuid
from collections import defaultdict
from pprint import pprint

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_variance_proxy_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.reward import extract_reward
from verl.utils.metric import reduce_metrics
from verl.utils.tracking import Tracking

from train.dynamic_sampling import (
    count_infra_groups,
    ds_metrics,
    select_group_indices,
)


class RayDAPOTrainer(RayPPOTrainer):
    """RayPPOTrainer with bounded DAPO dynamic sampling.

    ``algorithm.filter_groups.max_num_gen_batches`` is the maximum number of
    generation batches per optimizer step.  The target number of mixed groups
    is ``data.train_batch_size``.  Underfilled attempts are logged and skipped;
    three consecutive underfilled attempts stop the run.
    """

    def _max_gen_batches(self) -> int:
        refill = self.config.get("reward_filter", None)
        if refill is not None:
            return max(1, int(refill.get("max_gen_batches", 3)))
        cfg = self.config.algorithm.get("filter_groups", None)
        value = 3 if cfg is None else int(cfg.get("max_num_gen_batches", 3))
        return max(1, value)

    def _requires_mixed_groups(self):
        return True

    def _generate_one_batch(self, batch_dict, *, epoch: int):
        n = int(self.config.actor_rollout_ref.rollout.n)
        # Mirror RayPPOTrainer.fit(): finalize any previous async actor/rollout
        # call before starting a new generation.  This is a no-op when the
        # worker group does not expose the method (the FSDP path here).
        if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
            self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
        batch: DataProto = DataProto.from_single_dict(batch_dict)
        batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
        batch.non_tensor_batch["uid"] = np.array(
            [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
        )
        gen_batch = self._get_gen_batch(batch)
        gen_batch.meta_info["global_steps"] = self.global_steps
        gen_batch_output = gen_batch.repeat(repeat_times=n, interleave=True)
        if self.config.get("bird_agent", {}).get("runtime_contract", ""):
            # Pass trace identity locally; do not require a site-package patch
            # merely to recover which update produced a trajectory.
            infos = gen_batch_output.non_tensor_batch.get("extra_info", [{}] * len(gen_batch_output))
            gen_batch_output.non_tensor_batch["_trajectory_info"] = np.array([
                {"step": self.global_steps, "sample_index": info.get("index", info.get("question_id")),
                 "rollout_n": i % n, "validate": False} for i, info in enumerate(infos)
            ], dtype=object)
        gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
        timing = gen_batch_output.meta_info.pop("timing", None)
        batch = batch.repeat(repeat_times=n, interleave=True)
        batch = batch.union(gen_batch_output)
        if "response_mask" not in batch.batch:
            batch.batch["response_mask"] = compute_response_mask(batch)

        reward_tensor, reward_extra_info = extract_reward(batch)
        batch.batch["token_level_scores"] = reward_tensor
        if reward_extra_info:
            batch.non_tensor_batch.update(
                {key: np.array(value) for key, value in reward_extra_info.items()}
            )
        return batch, timing

    def _train_on_batch(self, batch: DataProto, *, epoch: int, timing_raw: dict):
        metrics = {}
        n = int(self.config.actor_rollout_ref.rollout.n)
        target_groups = int(self.config.data.train_batch_size)
        expected = target_groups * n

        self._balance_batch(batch, metrics=metrics)
        batch.meta_info["global_token_num"] = torch.sum(
            batch.batch["attention_mask"], dim=-1
        ).tolist()

        with _timer("old_log_prob", timing_raw):
            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
            entropys = old_log_prob.batch["entropys"]
            response_masks = batch.batch["response_mask"]
            actor_cfg = self.config.actor_rollout_ref.actor
            entropy_agg = agg_loss(
                loss_mat=entropys,
                loss_mask=response_masks,
                loss_agg_mode=actor_cfg.loss_agg_mode,
                loss_scale_factor=getattr(actor_cfg, "loss_scale_factor", 1.0),
            )
            metrics["actor/entropy"] = entropy_agg.detach().item()
            metrics["perf/mfu/actor_infer"] = old_log_prob_mfu
            old_log_prob.batch.pop("entropys")
            batch = batch.union(old_log_prob)

        if self.use_reference_policy:
            with _timer("ref", timing_raw):
                ref_log_prob = self._compute_ref_log_prob(batch)
                batch = batch.union(ref_log_prob)
            mask = batch.batch["response_mask"].float()
            difference = (batch.batch["old_log_probs"].float() - batch.batch["ref_log_prob"].float()).abs()
            metrics["policy/action_logprob_abs_delta_from_ref"] = float(
                (difference * mask).sum().item() / max(float(mask.sum().item()), 1.0))
            metrics["policy/action_logprob_max_abs_delta_from_ref"] = float((difference * mask).max().item())
        if "rollout_log_probs" in batch.batch:
            mask = batch.batch["response_mask"].float()
            difference = (batch.batch["old_log_probs"].float() - batch.batch["rollout_log_probs"].float()).abs()
            metrics["policy/rollout_action_logprob_abs_delta"] = float(
                (difference * mask).sum().item() / max(float(mask.sum().item()), 1.0))

        if self.use_critic:
            with _timer("values", timing_raw):
                values = self._compute_values(batch)
                batch = batch.union(values)

        with _timer("adv", timing_raw):
            if self.config.algorithm.use_kl_in_reward:
                batch, kl_metrics = apply_kl_penalty(
                    batch, kl_ctrl=self.kl_ctrl_in_reward,
                    kl_penalty=self.config.algorithm.kl_penalty,
                )
                metrics.update(kl_metrics)
            else:
                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

            norm_adv_by_std = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
            batch = compute_advantage(
                batch,
                adv_estimator=self.config.algorithm.adv_estimator,
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
                num_repeat=n,
                norm_adv_by_std_in_grpo=norm_adv_by_std,
                config=self.config.algorithm,
            )

        if self.use_critic:
            with _timer("update_critic", timing_raw):
                critic_output = self._update_critic(batch)
            metrics.update(reduce_metrics(critic_output.meta_info["metrics"]))

        if self.config.trainer.critic_warmup <= self.global_steps:
            with _timer("update_actor", timing_raw):
                actor_output = self._update_actor(batch)

            is_last_step = self.global_steps >= self.total_training_steps
            if self.config.trainer.save_freq > 0 and (
                is_last_step or self.global_steps % self.config.trainer.save_freq == 0
            ):
                with _timer("save_checkpoint", timing_raw):
                    self._save_checkpoint()

            with _timer("update_weights", timing_raw):
                self.checkpoint_manager.update_weights(self.global_steps)
            metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))

        if self.config.trainer.test_freq > 0 and (
            self.global_steps >= self.total_training_steps
            or self.global_steps % self.config.trainer.test_freq == 0
        ):
            with _timer("testing", timing_raw):
                metrics.update(self._validate())

        return batch, metrics

    def fit(self):
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()
        self.checkpoint_manager.update_weights(self.global_steps)
        self.max_steps_duration = 0

        if self.config.trainer.get("val_before_train", False):
            val_metrics = self._validate()
            if val_metrics:
                logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        self.global_steps += 1
        progress_bar = tqdm(
            total=self.total_training_steps, initial=0, desc="Training Progress"
        )
        max_gen_batches = self._max_gen_batches()
        target_groups = int(self.config.data.train_batch_size)
        n = int(self.config.actor_rollout_ref.rollout.n)
        expected = target_groups * n
        underfilled_total = 0
        consecutive_underfilled = 0

        for epoch in range(self.config.trainer.total_epochs):
            data_iter = iter(self.train_dataloader)
            while self.global_steps <= self.total_training_steps:
                step_start = time.monotonic()
                timing_raw: dict[str, float] = {}
                all_batches: list[DataProto] = []
                selection = None
                generated_tokens = 0
                trajectory_tokens = 0
                infra_groups = 0
                num_gen_batches = 0
                dataset_exhausted = False

                for _ in range(max_gen_batches):
                    try:
                        batch_dict = next(data_iter)
                    except StopIteration:
                        dataset_exhausted = True
                        break
                    num_gen_batches += 1
                    with _timer("gen", timing_raw):
                        batch, gen_timing = self._generate_one_batch(batch_dict, epoch=epoch)
                    if gen_timing:
                        timing_raw.update(gen_timing)
                    # Observations and prompts are context, not policy output.
                    generated_tokens += int(batch.batch["response_mask"].sum().item())
                    trajectory_tokens += int(batch.batch["attention_mask"].sum().item())
                    all_batches.append(batch)

                    combined = DataProto.concat(all_batches) if len(all_batches) > 1 else all_batches[0]
                    uids = combined.non_tensor_batch["uid"].tolist()
                    rewards = combined.batch["token_level_scores"].sum(dim=-1).tolist()
                    available = combined.non_tensor_batch.get("reward_available")
                    require_typed = bool(self.config.get("bird_agent", {}).get("runtime_contract", ""))
                    if require_typed and available is None:
                        raise RuntimeError("typed training requires structured reward_available metadata")
                    available = ([True] * len(uids) if available is None else list(available))
                    # Tool/database failures can prevent final submission before
                    # the scorer runs. They also invalidate the entire n-group.
                    fields = combined.non_tensor_batch.get("tool_extra_fields")
                    if require_typed and fields is not None:
                        if len(fields) != len(uids):
                            raise RuntimeError("tool availability metadata is not aligned with trajectories")
                        available = [bool(a) and not (isinstance(f, dict) and f.get("infrastructure_error"))
                                     for a, f in zip(available, fields)]
                    selection = select_group_indices(
                        uids,
                        rewards,
                        target_groups=target_groups,
                        max_groups=target_groups * max_gen_batches,
                        reward_available=available,
                        require_mixed=self._requires_mixed_groups(),
                    )
                    infra_groups = count_infra_groups(
                        uids, combined.non_tensor_batch.get("tool_extra_fields")
                    )
                    if len(selection.indices) >= expected:
                        break

                if selection is None and dataset_exhausted:
                    break
                if selection is None or len(selection.indices) < expected:
                    underfilled_total += 1
                    consecutive_underfilled += 1
                    metrics = {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                        "ds/underfilled_batches": underfilled_total,
                        "ds/num_gen_batches": num_gen_batches,
                        "ds/generated_tokens": generated_tokens,
                        "ds/infra_groups": infra_groups,
                    }
                    if selection is not None:
                        metrics.update(ds_metrics(
                            selection,
                            num_gen_batches=num_gen_batches,
                            generated_tokens=generated_tokens,
                            infra_groups=infra_groups,
                            optimizer_steps=self.global_steps,
                            underfilled_batches=underfilled_total,
                        ))
                    logger.log(data=metrics, step=self.global_steps)
                    print(f"[DAPO] underfilled at step {self.global_steps}: "
                          f"{num_gen_batches} gen batches, "
                          f"{0 if selection is None else len(selection.indices)}/{expected} trajectories")
                    if dataset_exhausted:
                        break
                    if consecutive_underfilled >= 3:
                        raise RuntimeError(
                            "DAPO stopped after 3 consecutive underfilled batches; "
                            "increase max_num_gen_batches or inspect mixed-group rate"
                        )
                    continue

                consecutive_underfilled = 0
                combined = DataProto.concat(all_batches) if len(all_batches) > 1 else all_batches[0]
                selected_indices = selection.indices[:expected]
                batch = combined[selected_indices]
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                padding_metrics = {}
                if self.config.get("bird_agent", {}).get("runtime_contract", ""):
                    from train.padding import compact_padding
                    padding_metrics = compact_padding(batch)

                # Keep vLLM replicas awake across the consecutive generation
                # batches used by dynamic sampling.  Sleeping here would force
                # the next generate_sequences call to run against a slept
                # engine, which has produced NPU aivec/DDR faults.  Sleep once,
                # after sampling is complete and before actor/ref computation.
                self.checkpoint_manager.sleep_replicas()

                is_last_step = self.global_steps >= self.total_training_steps
                batch, metrics = self._train_on_batch(batch, epoch=epoch, timing_raw=timing_raw)
                metrics.update(padding_metrics)
                metrics.update(ds_metrics(
                    selection,
                    num_gen_batches=num_gen_batches,
                    generated_tokens=generated_tokens,
                    infra_groups=infra_groups,
                    optimizer_steps=self.global_steps,
                    underfilled_batches=underfilled_total,
                ))
                metrics.update({
                    "training/global_step": self.global_steps,
                    "training/epoch": epoch,
                    "ds/underfilled_batches": underfilled_total,
                    "sampling/require_mixed_groups": int(self._requires_mixed_groups()),
                    "sampling/trajectory_tokens": trajectory_tokens,
                    "sampling/generated_action_tokens": generated_tokens,
                })
                for label, source in (("generated", combined), ("selected", batch)):
                    infos = source.non_tensor_batch.get("extra_info")
                    if infos is not None:
                        metrics[f"sampling/unique_questions_{label}"] = len({
                            info.get("question_id") for info in infos if isinstance(info, dict)
                        } - {None})
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                if padding_metrics:
                    from train.padding import budget_clipping_metrics
                    metrics.update(budget_clipping_metrics(batch,
                        self.config.data.get("max_prompt_length", 2048),
                        self.config.data.get("max_response_length", 4096)))
                timing_raw["step"] = time.monotonic() - step_start
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(
                    batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                metrics.update(compute_variance_proxy_metrics(
                    batch=batch, gradient_norm=metrics.get("actor/grad_norm")))
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    progress_bar.close()
                    pprint("DAPO training completed" if self._requires_mixed_groups() else "GRPO training completed")
                    return

        progress_bar.close()
        if self.global_steps <= self.total_training_steps:
            raise RuntimeError(
                f"DAPO ended before target steps: {self.global_steps - 1}/"
                f"{self.total_training_steps}"
            )


class _timer:
    """Small monotonic timer that accumulates into a dict."""

    def __init__(self, name: str, timing: dict):
        self.name = name
        self.timing = timing

    def __enter__(self):
        self.start = time.monotonic()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.timing[self.name] = self.timing.get(self.name, 0.0) + (
            time.monotonic() - self.start
        )
        return False
