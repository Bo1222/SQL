"""GRPO with bounded refill of unavailable-reward groups, without DS.

Legitimate all-zero and all-correct groups remain in vanilla GRPO. Only
unavailable grading/tool groups are removed before advantage computation.
"""
import hydra
import ray

from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.trainer.main_ppo import TaskRunner, run_ppo
from verl.utils.device import auto_set_device
from train.dapo_ray_trainer import RayDAPOTrainer


class RewardAwareGRPOTrainer(RayDAPOTrainer):
    def _requires_mixed_groups(self):
        return False


@ray.remote(num_cpus=1)
class GRPOTaskRunner(TaskRunner):
    def run(self, config):
        import verl.trainer.main_ppo as main_ppo
        main_ppo.RayPPOTrainer = RewardAwareGRPOTrainer
        return super().run(config)


@hydra.main(config_path="pkg://verl.trainer.config", config_name="ppo_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    run_ppo(migrate_legacy_reward_impl(config), task_runner_class=GRPOTaskRunner)


if __name__ == "__main__":
    main()
