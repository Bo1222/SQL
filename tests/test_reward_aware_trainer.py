"""Exercise the real trainer control loop without a model or accelerator."""
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import torch
from verl import DataProto

import train.dapo_ray_trainer as module
from train.main_grpo import RewardAwareGRPOTrainer


def test_generation_passes_trace_identity_and_preserves_reward_metadata():
    trainer=object.__new__(RewardAwareGRPOTrainer)
    trainer.config=OmegaConf.create({'actor_rollout_ref':{'rollout':{'n':2,'temperature':1.0}},
                                   'bird_agent':{'runtime_contract':'A-e1.2-typed-v1'}})
    trainer.global_steps=2
    trainer.actor_rollout_wg=SimpleNamespace()
    trainer._get_gen_batch=lambda b:b

    def generate(batch):
        info=batch.non_tensor_batch['_trajectory_info'].tolist()
        assert [r['step'] for r in info]==[2,2]
        assert [r['sample_index'] for r in info]==['q','q']
        assert [r['rollout_n'] for r in info]==[0,1]
        return DataProto.from_dict(tensors={'attention_mask':torch.ones((2,2)),
            'response_mask':torch.ones((2,1)), 'rm_scores':torch.tensor([[1.],[0.]])},
            non_tensors={'reward_available':np.array([True,False])},
            meta_info={'reward_extra_keys':['reward_available']})

    trainer.async_rollout_manager=SimpleNamespace(generate_sequences=generate)
    batch,_=trainer._generate_one_batch({'prompts':torch.tensor([[1]]),
        'extra_info':np.array([{'question_id':'q'}],dtype=object)},epoch=0)
    assert batch.non_tensor_batch['reward_available'].tolist()==[True,False]
    assert batch.batch['token_level_scores'].tolist()==[[1.],[0.]]
    assert len(set(batch.non_tensor_batch['uid'].tolist()))==1


def test_real_fit_refills_unavailable_groups_and_restarts_each_epoch(monkeypatch):
    logs, trained = [], []
    monkeypatch.setattr(module, "Tracking", lambda **kwargs: SimpleNamespace(
        log=lambda data, step: logs.append((data, step))))
    for name in ("compute_data_metrics", "compute_throughout_metrics", "compute_timing_metrics",
                 "compute_variance_proxy_metrics"):
        monkeypatch.setattr(module, name, lambda **kwargs: {})
    trainer = object.__new__(RewardAwareGRPOTrainer)
    trainer.config = OmegaConf.create({"trainer": {"project_name":"test", "experiment_name":"test",
        "logger":["console"], "total_epochs":2, "val_before_train":False},
        "data":{"train_batch_size":1}, "actor_rollout_ref":{"rollout":{"n":2,"temperature":1.0}},
        "bird_agent":{"runtime_contract":"A-e1.2-typed-v1"},
        "reward_filter":{"max_gen_batches":2}})
    trainer.total_training_steps = 2
    trainer.train_dataloader = ["bad", "good"]
    trainer.use_critic = False
    trainer._load_checkpoint = lambda: None
    trainer.checkpoint_manager = SimpleNamespace(update_weights=lambda step: None, sleep_replicas=lambda: None)
    trainer.resource_pool_manager = SimpleNamespace(get_n_gpus=lambda: 1)
    trainer.actor_rollout_wg = SimpleNamespace()

    def generate(which, *, epoch):
        proto = DataProto.from_dict(tensors={
            "input_ids":torch.tensor([[1,2], [1,2]]), "responses":torch.tensor([[2],[2]]),
            "attention_mask":torch.ones((2,2),dtype=torch.long),
            "response_mask":torch.ones((2,1),dtype=torch.long),
            "token_level_scores":torch.tensor([[1.],[0.]] if which=="bad" else [[0.],[1.]])},
            non_tensors={"uid":np.array([which,which],dtype=object),
                         "reward_available":np.array([True, which!="bad"]),
                         "tool_extra_fields":np.array([{},{}],dtype=object)})
        return proto, {}

    def train(batch, *, epoch, timing_raw):
        assert batch.non_tensor_batch["uid"].tolist() == ["good", "good"]
        assert batch.non_tensor_batch["reward_available"].all()
        trained.append(epoch)
        return batch, {"actor/grad_norm":1.0}

    trainer._generate_one_batch, trainer._train_on_batch = generate, train
    trainer.fit()
    assert trained == [0,1]
    assert len(logs) == 2
    for metrics, step in logs:
        assert metrics["ds/reward_unavailable_groups"] == 1
        assert metrics["ds/num_gen_batches"] == 2
        assert metrics["sampling/require_mixed_groups"] == 0
        assert metrics["sampling/generated_action_tokens"] == 4
        assert metrics["sampling/trajectory_tokens"] == 8
