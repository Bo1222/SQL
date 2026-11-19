import asyncio
from types import SimpleNamespace

import pytest

from agent.bird_agent_loop import BirdSQLAgentLoop
from agent.sql_episode import EpisodeConfig


class CharTokenizer:
    """Reversible character tokenizer; enough for the loop contract test."""

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(int(token)) for token in ids)


class FakeServer:
    def __init__(self, replies):
        self.replies = list(replies)
        self.budgets = []

    async def generate(self, request_id, prompt_ids, sampling_params, image_data=None, video_data=None):
        self.budgets.append(sampling_params["max_tokens"])
        text = self.replies.pop(0)
        ids = [ord(char) for char in text]
        return SimpleNamespace(token_ids=ids, log_probs=[-0.1] * len(ids))


class DummySchemaTool:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, arguments):
        return {"status": "ok", "observation": "table: items\ncolumn: id"}


def _loop(replies, *, max_rounds=2, max_tools=4, max_schema_lookups=2):
    loop = object.__new__(BirdSQLAgentLoop)
    loop.ep_cfg = EpisodeConfig(
        max_rounds=max_rounds, max_tools=max_tools, max_schema_lookups=max_schema_lookups,
        max_sql_executes=2, max_invalid_actions=2,
        max_observation_tokens=64, max_action_tokens=64, max_context=4096,
        observation_mode="compact",
    )
    loop.catalog_dir = "artifacts/catalog"
    loop.retrieval_version = "E1.1"
    loop.response_length = 1024
    loop.enable_value_lookup = False
    loop.use_chat_template = False
    loop.trace_dir = ""
    loop.tokenizer = CharTokenizer()
    loop.server_manager = FakeServer(replies)
    loop.rollout_config = SimpleNamespace(response_length=1024, max_model_len=4096)
    return loop


def test_agent_loop_caps_action_and_aligns_observation_logprobs(monkeypatch):
    monkeypatch.setattr("agent.bird_agent_loop._load_store", lambda *args: None)
    monkeypatch.setattr("agent.bird_agent_loop.SchemaLookupTool", DummySchemaTool)
    loop = _loop([
        '<schema_lookup>{"query":"id"}</schema_lookup>',
        "<answer>SELECT 1</answer>",
    ])
    output = asyncio.run(loop.run(
        sampling_params={"temperature": 1.0, "top_p": 1.0},
        raw_prompt=[{"role": "user", "content": "question"}],
        extra_info={
            "db_id": "address",
            "db_path": "/tmp/does-not-exist.sqlite",
            "question_id": "q1",
            "retrieval_version": "E1.1",
        },
    ))
    assert output.response_mask[0] == 1
    assert 0 in output.response_mask
    assert output.response_mask[-1] == 1
    assert len(output.response_logprobs) == len(output.response_ids)
    observation_positions = [i for i, mask in enumerate(output.response_mask) if mask == 0]
    assert observation_positions
    assert all(output.response_logprobs[i] == 0.0 for i in observation_positions)
    # One action turn plus one observation turn; both are capped at 64.
    assert loop.server_manager.budgets == [64, 64]
    assert output.num_turns == 2


def test_agent_loop_terminates_on_total_tool_budget(monkeypatch):
    monkeypatch.setattr("agent.bird_agent_loop._load_store", lambda *args: None)
    monkeypatch.setattr("agent.bird_agent_loop.SchemaLookupTool", DummySchemaTool)
    lookup = '<schema_lookup>{"query":"id"}</schema_lookup>'
    loop = _loop([lookup, lookup, lookup], max_rounds=5, max_tools=2,
                 max_schema_lookups=10)
    output = asyncio.run(loop.run(
        sampling_params={"temperature": 1.0, "top_p": 1.0},
        raw_prompt=[{"role": "user", "content": "question"}],
        extra_info={"db_id": "address", "db_path": "/tmp/x.sqlite",
                    "retrieval_version": "E1.1"},
    ))
    assert output.num_turns == 3
    assert output.extra_fields["finish_reason"] == "tool_limit"


def test_agent_loop_rejects_retrieval_version_mismatch(monkeypatch):
    monkeypatch.setattr("agent.bird_agent_loop._load_store", lambda *args: None)
    monkeypatch.setattr("agent.bird_agent_loop.SchemaLookupTool", DummySchemaTool)
    loop = _loop(["<answer>SELECT 1</answer>"])
    with pytest.raises(ValueError, match="retrieval version mismatch"):
        asyncio.run(loop.run(
            sampling_params={"temperature": 1.0, "top_p": 1.0},
            raw_prompt=[{"role": "user", "content": "question"}],
            extra_info={"db_id": "address", "db_path": "/tmp/x.sqlite",
                        "retrieval_version": "E1"},
        ))


def test_sampling_budget_overrides_rollout_default():
    loop = object.__new__(BirdSQLAgentLoop)
    params = {"temperature": 1.0, "top_p": 1.0, "max_new_tokens": 4096}
    capped = loop._sampling_with_budget(params, 512)
    assert capped["max_tokens"] == 512
    assert "max_new_tokens" not in capped
    # The caller's dict must not be mutated; the same sampling params are
    # reused for every turn of the episode.
    assert params["max_new_tokens"] == 4096
    assert "max_tokens" not in params


def test_sampling_budget_supports_object_params():
    loop = object.__new__(BirdSQLAgentLoop)
    params = SimpleNamespace(temperature=1.0, max_tokens=4096)
    capped = loop._sampling_with_budget(params, 128)
    assert capped.max_tokens == 128
    assert params.max_tokens == 4096
