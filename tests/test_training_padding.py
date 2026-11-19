import numpy as np
import pytest
import torch

from verl import DataProto
from train.padding import budget_clipping_metrics, compact_padding


def test_compaction_preserves_action_likelihood_rewards_and_positions():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(42)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=64, attention_dropout=0, pad_token_id=0)).eval()
    ids = torch.tensor([[0,0,0,2,3,4,5,0,0,0], [0,0,6,7,8,9,10,11,0,0]])
    attention = (ids != 0).long()
    positions = (attention.cumsum(-1)-1).clamp(min=0)
    response_mask = torch.tensor([[1,1,0,0,0], [1,0,1,0,0]])
    scores = torch.tensor([[0.,1.,0.,0.,0.], [0.,0.,1.,0.,0.]])
    batch = DataProto.from_dict(tensors={"input_ids":ids, "attention_mask":attention,
        "position_ids":positions, "prompts":ids[:,:5], "responses":ids[:,5:],
        "response_mask":response_mask, "token_level_scores":scores},
        non_tensors={"uid":np.array(["q", "q"], dtype=object)})
    with torch.no_grad():
        logits = model(ids, attention_mask=attention, position_ids=positions).logits[:,4:9]
        original = logits.log_softmax(-1).gather(-1, ids[:,5:,None]).squeeze(-1)
    metric = compact_padding(batch)
    assert metric["sampling/removed_padding_columns"] == 4
    assert batch.batch["responses"].shape == (2,3)
    assert torch.equal(batch.batch["token_level_scores"].sum(-1), scores.sum(-1))
    assert torch.equal(batch.batch["response_mask"], response_mask[:,:3])
    # The longest item now fills the compact width; it has not hit a budget.
    assert all(value == 0 for value in budget_clipping_metrics(batch, 5, 5).values())
    with torch.no_grad():
        tensors=batch.batch
        logits=model(tensors["input_ids"], attention_mask=tensors["attention_mask"],
                     position_ids=tensors["position_ids"]).logits[:,2:5]
        compact=logits.log_softmax(-1).gather(-1, tensors["responses"][...,None]).squeeze(-1)
    mask=response_mask[:,:3].bool()
    torch.testing.assert_close(compact[mask], original[:,:3][mask], rtol=1e-5, atol=1e-6)


def test_compaction_rejects_interleaved_padding():
    batch=DataProto.from_dict(tensors={"input_ids":torch.tensor([[0,1,2,0]]),
        "responses":torch.tensor([[2,0]]), "attention_mask":torch.tensor([[1,0,1,0]])})
    with pytest.raises(ValueError, match="left-padded"):
        compact_padding(batch)
