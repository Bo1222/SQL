import asyncio
import json
import sqlite3
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.runtime_contract import CONTRACT, CONTRACT_ID, FINGERPRINT, config, load_store, prompt
from agent.sql_episode import run_episode
from retrieval.catalog_store import build_store
from reward.bird_ex_reward import compute_score_with_details, score_sql
from tools.sql_execute import SQLExecuteTool, run_sql


@pytest.fixture
def tiny_db(tmp_path):
    db = tmp_path / "tiny" / "tiny.sqlite"
    db.parent.mkdir()
    with sqlite3.connect(db) as con:
        con.execute("CREATE TABLE parent(id INTEGER PRIMARY KEY)")
        con.execute("CREATE TABLE child(id INT, parent_id INT REFERENCES parent)")
        con.execute("INSERT INTO parent VALUES(1)")
    build_store(db, tmp_path / "catalog")
    return db, tmp_path / "catalog"


def test_typed_rewards_separate_gold_failure_from_candidate_failure(tiny_db):
    db, _ = tiny_db
    good = compute_score_with_details("bird", "<answer>SELECT id FROM parent</answer>",
                                     "SELECT id FROM parent", {"db_path": str(db), "score_mode": "bird_set"})
    assert good["score"] == 1 and good["reward_available"]
    bad = compute_score_with_details("bird", "<answer>SELECT absent FROM parent</answer>",
                                    "SELECT id FROM parent", {"db_path": str(db)})
    assert bad["score"] == 0 and bad["reward_available"] and bad["failure_origin"] == "candidate"
    unavailable = compute_score_with_details("bird", "<answer>SELECT id FROM parent</answer>",
                                            "SELECT absent FROM parent", {"db_path": str(db)})
    assert unavailable["score"] == 0 and not unavailable["reward_available"]
    assert unavailable["failure_origin"] == "gold"
    missing = compute_score_with_details("bird", "nothing", "SELECT id FROM parent", {"db_path": str(db)})
    assert missing["reward_available"] and missing["score"] == 0


def test_batched_reward_schema_is_stable_for_missing_and_failed_answers(tiny_db):
    from reward.typed_rl_reward import DEFAULTS, compute_score_with_details as fixed_grade
    db, _ = tiny_db
    outputs = []
    for answer, gold in [('<answer>SELECT id FROM parent</answer>', 'SELECT id FROM parent'),
                         ('no answer', 'SELECT id FROM parent'),
                         ('<answer>SELECT absent FROM parent</answer>', 'SELECT id FROM parent'),
                         ('<answer>SELECT id FROM parent</answer>', 'SELECT absent FROM parent')]:
        extra={'db_path':str(db),'score_mode':'bird_set'}
        actual=fixed_grade('bird',answer,gold,extra)
        original=compute_score_with_details('bird',answer,gold,extra)
        assert actual.keys()==DEFAULTS.keys()
        assert all(actual[k]==v for k,v in original.items())
        outputs.append(actual)
    # This is the indexing performed by the installed veRL collector.
    batched={key:[r[key] for r in outputs] for key in outputs[0]}
    assert batched['reward_available']==[True,True,True,False]


def test_typed_database_failure_and_watchdog_stage(tiny_db):
    db, _ = tiny_db
    bad = score_sql(db.parent / "missing.sqlite", "SELECT 1", "SELECT 1", typed=True)
    assert not bad["reward_available"] and bad["failure_origin"] == "database"
    for stage, available in [("gold", False), ("candidate", True), ("database", False)]:
        exc = subprocess.TimeoutExpired("worker", 35, stderr=("SQLRL_STAGE=" + stage + "\n").encode())
        with patch("tools.sql_execute.subprocess.run", side_effect=exc):
            result = run_sql(db, "SELECT 1", report_stage=True)
        assert result["reward_available"] is available
        assert result["failure_origin"] == stage


def test_typed_candidate_memory_is_a_policy_failure_not_unavailable(tiny_db):
    db, _ = tiny_db
    payload = {"ok": False, "infrastructure_error": True, "failure_origin": "candidate",
               "reward_available": True, "error": "MemoryError"}
    completed = SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="SQLRL_STAGE=candidate\n")
    with patch("tools.sql_execute.subprocess.run", return_value=completed):
        scored = compute_score_with_details("bird", "<answer>SELECT 1</answer>", "SELECT 1", {"db_path": str(db)})
        observation = SQLExecuteTool(db, failure_policy="typed-v1")("SELECT 1")
    assert scored["reward_available"] and scored["score"] == 0
    assert "failure_origin" not in observation


def test_shared_runtime_resolves_fk_and_is_catalog_only(tiny_db):
    db, catalog = tiny_db
    store = load_store(catalog, "tiny", db)
    assert store.relationships[0]["to_column"] == "id"
    row = {"question": "How many parents?", "evidence": "", "gold_sql": "GOLD_SENTINEL"}
    text = prompt(row, store)
    assert "GOLD_SENTINEL" not in text and "CREATE TABLE" not in text
    assert config().max_response_tokens == 4096
    assert config().observation_mode == "compact_v2"
    assert len(FINGERPRINT) == 64


def test_offline_runtime_honors_total_response_budget():
    class Tokenizer:
        def encode(self, text, add_special_tokens=False): return list(text.encode())
        def decode(self, ids, skip_special_tokens=True): return bytes(ids).decode()
    cfg = config()
    cfg.max_response_tokens = 4
    budgets = []
    def generate(ids, remaining):
        budgets.append(remaining)
        return list(b"abcd")
    state = run_episode({"question": "q"}, Tokenizer(), generate, None, None, cfg, prompt="p")
    assert budgets == [4]
    assert state["finish_reason"] == "generation_error"
    assert sum(len(s["ids"]) for s in state["segments"] if s["kind"] != "prompt") <= 4


@pytest.mark.parametrize("replies", [
    ['<schema_lookup>{"query":"parent id"}</schema_lookup>',
     '<sql_execute>SELECT id FROM parent</sql_execute>', '<answer>SELECT id FROM parent</answer>'],
    ['bad action', '<sql_execute>SELECT absent FROM parent</sql_execute>',
     '<answer>SELECT id FROM parent</answer>'],
])
def test_typed_async_and_offline_match_prompts_observations_and_actions(tiny_db, replies):
    from agent.bird_agent_loop import BirdSQLAgentLoop
    from tests.test_bird_agent_loop_contract import CharTokenizer, FakeServer
    from tools.schema_lookup import SchemaLookupTool

    db, catalog = tiny_db
    store = load_store(catalog, "tiny", db)
    row = {"question": "Find parent ids", "evidence": "", "db_id": "tiny", "db_path": str(db)}
    text = prompt(row, store)
    tokenizer = CharTokenizer()
    pending = iter(replies)
    seen_ids, seen_budgets = [], []

    def generate(ids, budget):
        seen_ids.append(list(ids)); seen_budgets.append(budget)
        return tokenizer.encode(next(pending))

    state = run_episode(row, tokenizer, generate,
                        SchemaLookupTool(store, token_budget=1024, retrieval_version="E1.2",
                                         count_tokens=lambda s: len(tokenizer.encode(s))),
                        SQLExecuteTool(db, failure_policy="typed-v1"), config(), prompt=text)
    loop = object.__new__(BirdSQLAgentLoop)
    loop.ep_cfg = config()
    loop.catalog_dir, loop.retrieval_version = str(catalog), "E1.2"
    loop.runtime_contract, loop.response_length = CONTRACT_ID, 4096
    loop.enable_value_lookup = loop.use_chat_template = False
    loop.trace_dir, loop.tokenizer = "", tokenizer

    class RecordingServer(FakeServer):
        async def generate(self, request_id, prompt_ids, sampling_params, **kwargs):
            assert list(prompt_ids) == seen_ids[len(self.budgets)]
            return await super().generate(request_id, prompt_ids, sampling_params, **kwargs)

    loop.server_manager = RecordingServer(replies)
    extra = dict(row, question_id="q", runtime_contract=CONTRACT_ID,
                 runtime_fingerprint=FINGERPRINT, score_mode="bird_set", retrieval_version="E1.2")
    output = asyncio.run(loop.run(sampling_params={"temperature": 1.0},
        raw_prompt=[{"role": "user", "content": text}], extra_info=extra))
    assert output.prompt_ids == tokenizer.encode(text)
    assert output.response_ids == [i for s in state["segments"][1:] for i in s["ids"]]
    assert output.response_mask == [int(s["kind"] == "action")
                                   for s in state["segments"][1:] for _ in s["ids"]]
    assert output.extra_fields["finish_reason"] == state["finish_reason"] == "answer_submitted"
    assert loop.server_manager.budgets == seen_budgets
    assert all(p == 0 for p, m in zip(output.response_logprobs, output.response_mask) if m == 0)


def test_verl_reward_manager_preserves_unavailable_for_whole_group_filter(tiny_db):
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from verl import DataProto
    from verl.experimental.reward_loop.reward_manager.naive import NaiveRewardManager
    from tests.test_bird_agent_loop_contract import CharTokenizer
    from train.dynamic_sampling import select_group_indices

    db, _ = tiny_db
    tokenizer = CharTokenizer()
    ids = tokenizer.encode('<answer>SELECT id FROM parent</answer>')

    async def exercise():
        manager = NaiveRewardManager(OmegaConf.create({}), tokenizer, compute_score_with_details)
        results = []
        for i in range(8):
            # A Gold failure would create a spurious [1,1,1,0] mixed group
            # if its unavailable state were discarded by the real manager.
            gold = 'SELECT absent FROM parent' if i == 3 else 'SELECT id FROM parent'
            data = DataProto.from_dict(tensors={
                'responses': torch.tensor([ids]), 'attention_mask': torch.ones((1, len(ids)), dtype=torch.long)},
                non_tensors={'data_source': np.array(['bird_sql'], dtype=object),
                    'reward_model': np.array([{'ground_truth': gold}], dtype=object),
                    'extra_info': np.array([{'db_path': str(db), 'score_mode': 'bird_set'}], dtype=object)})
            results.append(await manager.run_single(data))
        return results

    results = asyncio.run(exercise())
    assert [r['reward_score'] for r in results[:4]] == [1, 1, 1, 0]
    assert results[3]['reward_extra_info']['reward_available'] is False
    selected = select_group_indices(['bad'] * 4 + ['good'] * 4,
        [r['reward_score'] for r in results],
        reward_available=[r['reward_extra_info']['reward_available'] for r in results],
        target_groups=1, max_groups=2, require_mixed=False)
    assert selected.indices == [4, 5, 6, 7]
    assert selected.unavailable_groups == 1


def test_offline_cli_uses_runtime_catalog_order_not_metadata_order(tiny_db, monkeypatch, tmp_path):
    from eval import evaluate_agent
    from tests.test_bird_agent_loop_contract import CharTokenizer
    from transformers import AutoTokenizer
    db, catalog = tiny_db
    store = load_store(catalog, "tiny", db)
    metadata = tmp_path / 'metadata.json'
    metadata.write_text(json.dumps({'tables':[{'name':name} for name in reversed(store.table_names)]}))
    row = {'id':'q', 'db_id':'tiny', 'db_path':str(db), 'metadata_path':str(metadata),
           'question':'Find parent ids', 'evidence':'', 'gold_sql':'SELECT id FROM parent'}
    data = tmp_path / 'data.jsonl'
    data.write_text(json.dumps(row)+'\n')
    tokenizer = CharTokenizer()
    tokenizer.pad_token_id = 0
    seen = []

    def generate(ids, budget):
        seen.append(tokenizer.decode(ids))
        return tokenizer.encode('<answer>SELECT id FROM parent</answer>')

    monkeypatch.setattr(AutoTokenizer, 'from_pretrained', lambda *a, **kw: tokenizer)
    monkeypatch.setattr(evaluate_agent, 'HFEngine', lambda *a, **kw: generate)
    monkeypatch.setattr('sys.argv', ['evaluate_agent', '--engine','hf','--model','fixture',
        '--data',str(data),'--catalog-dir',str(catalog),'--runtime-contract',CONTRACT_ID,
        '--out',str(tmp_path/'result.json')])
    evaluate_agent.main()
    assert seen == [prompt(row, store)]
