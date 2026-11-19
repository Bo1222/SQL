"""M6-08j semantic data boundaries and training-math regressions (CPU)."""
import copy
import json
import sqlite3
from pathlib import Path

import pytest

from e1_alignment.rebuild.ast_check import extract, AST_RULE_VERSION, PARSER_VERSION
from e1_alignment.rebuild.visibility import coverage
from e1_alignment.rebuild.quality import (REVIEW_FIELDS, content_hash, review_errors,
                                         revised, versioned)
from e1_alignment.rebuild import RULE_VERSION
from e1_alignment.rebuild.freeze import quota_errors, select

SCHEMA = {"cars": {"id": "INT", "brand": "TEXT"}, "price": {"id": "INT", "price": "REAL"},
          "a": {"id": "INT", "x": "INT"}, "b": {"id": "INT", "a_id": "INT", "y": "INT"}}


def obs(body, **extra):
    return {"kind": "observation", "text": "<observation>" + json.dumps(
        {"status": "ok", "observation": body, **extra}) + "</observation>\n"}


def test_car_price_column_and_metadata_cannot_prove_visible_coverage():
    req = extract("SELECT price.price FROM cars JOIN price ON cars.id=price.id", schema=SCHEMA)
    assert req["ok"] and {"price.id", "price.price"} <= set(req["columns"])
    text = "Tables:\n- cars\n  - id INT\n- price\n  - id INT\nRelationships:\n- cars.id = price.id   [sqlite_fk]"
    segments = [obs(text, retrieval_metadata={"columns": ["price.price"]})]
    got = coverage(req, segments, 1, SCHEMA)
    assert not got["ok"] and got["missing"]["columns"] == ["price.price"]
    segments[0] = obs(text.replace("Relationships:", "  - price REAL\nRelationships:"))
    assert coverage(req, segments, 1, SCHEMA)["ok"]


def test_later_observation_never_supports_earlier_sql():
    req = extract("SELECT x FROM a", schema=SCHEMA)
    segments = [{"kind": "action"}, obs("Tables:\n- a\n  - x INT\nRelationships:")]
    assert not coverage(req, segments, 0, SCHEMA)["ok"]
    assert coverage(req, segments, 2, SCHEMA)["ok"]


@pytest.mark.parametrize("sql,columns", [
    ("SELECT x AS z FROM a ORDER BY z", {"a.x"}),
    ("WITH c AS (SELECT id AS k FROM a) SELECT c.k FROM c", {"a.id"}),
    ("SELECT q.x FROM (SELECT x FROM a) q", {"a.x"}),
    ("SELECT * FROM a", {"a.id", "a.x"}),
    ("SELECT COUNT(*) FROM a", set()),
    ('SELECT "P"."price" FROM "price" AS "P"', {"price.price"}),
    ("SELECT id FROM a JOIN b USING(id)", {"a.id", "b.id"}),
    ("SELECT x FROM a NATURAL JOIN b", {"a.x", "a.id", "b.id"}),
    ("SELECT a.x FROM a, b WHERE a.id=b.a_id", {"a.x", "a.id", "b.a_id"}),
    ("SELECT a.x FROM a WHERE EXISTS(SELECT 1 FROM b a WHERE a.y>0)", {"a.x", "b.y"}),
    ("SELECT a.id FROM a WHERE EXISTS(SELECT 1 FROM b WHERE b.a_id=a.id)", {"a.id", "b.a_id"}),
])
def test_scopes_aliases_stars(sql, columns):
    req = extract(sql, schema=SCHEMA)
    assert req["ok"], req["reasons"]
    assert set(req["columns"]) == columns


def test_self_join_keeps_instances():
    req = extract("SELECT t.id FROM a t JOIN a u ON t.id=u.x", schema=SCHEMA)
    assert req["ok"]
    j = req["joins"][0]
    assert j["left"]["table"] == j["right"]["table"] == "a"
    assert j["left"]["instance"] != j["right"]["instance"]


@pytest.mark.parametrize("sql", ["SELECT id FROM a,b", "SELECT missing FROM a", "SELECT FROM",
                                "SELECT x FROM a; SELECT id FROM b", "DELETE FROM a",
                                "SELECT a.x FROM a JOIN b ON a.id>b.id"])
def test_ambiguous_invalid_unsupported_are_quarantined(sql):
    assert not extract(sql, schema=SCHEMA)["ok"]


def reviewed_row():
    row = versioned({"id": "train_q", "db_id": "a", "question": "x?", "evidence": "", "gold_sql": "SELECT x FROM a",
                     "db_path": "a.sqlite", "input_ids": [1, 2], "loss_mask": [0, 1]}, "fixture", 1)
    row["program_check"] = {"ok": True, "content_sha256": row["content_sha256"],
                             "rule_version": RULE_VERSION, "parser_version": PARSER_VERSION, "ast_rule_version": AST_RULE_VERSION, "ast": {"gold": extract("SELECT x FROM a", schema=SCHEMA)}}
    review = {"trajectory_id": row["trajectory_id"], "content_sha256": content_hash(row),
              "status": "pass", "rule_version": RULE_VERSION, "reviewer": "test-fixture-only",
              "reviewed_at": "test", "method": "individual_assistant_review", "review_seconds": 1}
    for key in REVIEW_FIELDS:
        review[key] = {"verdict": "pass", "notes": "test fixture"}
    return row, review


def test_review_required_and_modification_invalidates_review():
    row, review = reviewed_row()
    assert not review_errors(row, review)
    assert review_errors(row, None)
    pending = {**review, "status": "pending"}
    assert review_errors(row, pending)
    row["input_ids"].append(3)
    assert "stale_or_wrong_review" in review_errors(row, review)


def test_name_match_join_needs_independent_semantic_verification():
    row, review = reviewed_row()
    row["program_check"]["ast"]["gold"] = extract("SELECT a.x FROM a JOIN b ON a.id=b.id", schema=SCHEMA)
    assert "join_semantics_not_individually_verified" in review_errors(row, review)


def test_revision_clears_old_review_and_limits_to_one():
    row, review = reviewed_row()
    row["semantic_review"] = review
    new = revised(row, row, "metadata rebuild")
    assert new["parent_version_id"] == row["trajectory_id"] and "semantic_review" not in new
    assert "stale_or_wrong_review" in review_errors(new, review)
    with pytest.raises(ValueError, match="one_revision_limit"):
        revised(new, new, "again")
    with pytest.raises(ValueError, match="cannot_modify_task"):
        revised(row, {**row, "gold_sql": "SELECT 1"}, "make it pass")


def test_selector_cannot_fill_quotas_with_unreviewed_records():
    row, _ = reviewed_row()
    chosen, report = select([row] * 500, {})
    assert not chosen and report["eligible"] == 0


def test_empty_set_never_passes_dataset_gate():
    assert "size_outside_400_800" in quota_errors([])


def test_token_accumulation_matches_equivalent_batch_complete_and_partial():
    import torch
    from e1_alignment.rebuild.training_math import backward_window
    torch.manual_seed(42)
    model = torch.nn.Linear(3, 5, bias=False).double()
    reference = copy.deepcopy(model)
    # Unequal lengths and masks; final window is shorter than accumulation=8.
    batches = []
    for n in [2, 4, 6, 3, 8, 5, 3, 4, 7, 2, 3]:
        labels = torch.arange(n)[None] % 5
        labels[:, 0] = -100
        if n > 4:
            labels[:, 2] = -100
        batches.append({"x": torch.randn(1, n, 3, dtype=torch.float64), "labels": labels})
    def loss_fn(m, b):
        logits = m(b["x"])[:, :-1].reshape(-1, 5)
        return torch.nn.functional.cross_entropy(logits, b["labels"][:, 1:].reshape(-1), ignore_index=-100)
    opt = torch.optim.SGD(model.parameters(), lr=.03)
    ref_opt = torch.optim.SGD(reference.parameters(), lr=.03)
    for window in (batches[:8], batches[8:]):
        backward_window(model, window, loss_fn)
        logits = torch.cat([reference(b["x"])[:, :-1].reshape(-1, 5) for b in window])
        labels = torch.cat([b["labels"][:, 1:].reshape(-1) for b in window])
        torch.nn.functional.cross_entropy(logits, labels, ignore_index=-100).backward()
        torch.testing.assert_close(model.weight.grad, reference.weight.grad, rtol=1e-12, atol=1e-12)
        opt.step(); ref_opt.step()
        opt.zero_grad(); ref_opt.zero_grad()
        torch.testing.assert_close(model.weight, reference.weight, rtol=1e-12, atol=1e-12)


def test_masked_predictions_have_zero_loss_gradient():
    import torch
    logits = torch.randn(1, 4, 5, requires_grad=True)
    labels = torch.tensor([[-100, 2, -100, 1]])
    loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 5), labels[:, 1:].reshape(-1), ignore_index=-100)
    loss.backward()
    assert not torch.count_nonzero(logits.grad[:, 1])
    assert not torch.count_nonzero(logits.grad[:, 3])
    assert torch.count_nonzero(logits.grad[:, 0])


def test_adapter_init_and_merge_preserve_base_logits():
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(0)
    base = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                                       num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2))
    base.eval()
    ids = torch.tensor([[1, 2, 3]])
    expected = base(ids).logits.detach()
    adapted = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, lora_dropout=.05,
                                            target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    adapted.eval()
    torch.testing.assert_close(adapted(ids).logits, expected, rtol=0, atol=0)
    merged = adapted.merge_and_unload()
    torch.testing.assert_close(merged(ids).logits, expected, rtol=0, atol=0)


def test_program_check_blocks_gold_mislabel_monitor_leak_and_failed_prefix(monkeypatch, tmp_path):
    from e1_alignment.rebuild.quality import program_check
    db = tmp_path / "x.sqlite"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE a(x INT)")
        conn.execute("INSERT INTO a VALUES(1)")
    row = {"id": "q", "db_id": "db", "question": "x?", "evidence": "", "gold_sql": "SELECT x FROM a", "db_path": str(db)}
    record = versioned({**row, "trajectory_source": "e1_gold_assisted", "gold_used_for_construction": False,
                        "autonomous_repair": True, "success_type": "autonomous_repair", "final_correct": True,
                        "final_sql": "SELECT x FROM a", "segments": [], "input_ids": [], "loss_mask": [],
                        "runtime_fingerprint_sha256": "fp"}, "test", 1)
    check = program_check(record, {"q": row}, {"q"}, "fp", score=False)
    assert {"monitor_leakage", "gold_source_flag_mismatch", "gold_assisted_marked_autonomous"} <= set(check["reasons"])
    check = program_check(record, {}, set(), "fp", score=False, forbidden_dbs={"db"})
    assert {"not_in_rl_train", "evaluation_database_leakage"} <= set(check["reasons"])
    record["trajectory_source"] = "e1_repair_continuation"
    record["prefix_action_segments"] = 1
    record["failure_state"] = {"state_id": "s"}
    check = program_check(record, {"q": row}, set(), "fp", score=False, prefix_index={})
    assert "failure_prefix_changed_or_missing" in check["reasons"]


def test_monitor_full_set_and_stop_thresholds():
    from e1_alignment.rebuild.assess import monitor_gate
    fixed = [{"id": str(i)} for i in range(64)]
    control = [{"id": str(i), "correct": i < 32} for i in range(64)]
    candidate = [{"id": str(i), "correct": i < 28} for i in range(64)]
    assert not monitor_gate(control, candidate, fixed)["pass"]
    with pytest.raises(ValueError, match="incomplete"):
        monitor_gate(control, candidate[:-1], fixed)
    candidate = copy.deepcopy(control)
    for r in candidate[:4]:
        r["error_type"] = "no_such_column"
    assert not monitor_gate(control, candidate, fixed)["pass"]


def test_first_candidate_includes_direct_answer():
    from e1_alignment.rebuild.assess import first_candidate
    record = {"segments": [{"kind": "action", "text": "<answer>SELECT 1</answer>"}]}
    assert first_candidate(record) == ("answer", "SELECT 1")


def test_frozen_manifest_rejects_stale_rules_and_hash_changes(tmp_path):
    from e1_alignment.rebuild.freeze import verify_frozen
    from e1_alignment.common import sha256_file
    data, reviews, monitor = [tmp_path / f for f in ("data.jsonl", "reviews.jsonl", "monitor.jsonl")]
    for f in (data, reviews, monitor):
        f.write_text("")
    manifest = {"format": "sqlrl-m6-08j-frozen-v1", "rule_version": RULE_VERSION,
                "parser_version": "30.30.0", "ast_rule_version": AST_RULE_VERSION,
                "dataset_sha256": sha256_file(data), "reviews_path": str(reviews),
                "reviews_sha256": sha256_file(reviews), "monitor_path": str(monitor),
                "monitor_sha256": sha256_file(monitor), "record_hashes": {}}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({**manifest, "rule_version": "old"}))
    with pytest.raises(ValueError, match="version_mismatch"):
        verify_frozen(data, path)
    path.write_text(json.dumps(manifest))
    reviews.write_text("modified")
    with pytest.raises(ValueError, match="hash_mismatch"):
        verify_frozen(data, path)


def test_budget_cannot_exceed_stage_or_total_caps(tmp_path):
    from e1_alignment.rebuild.budget import reserve, finish
    ledger = tmp_path / "budget.json"
    with pytest.raises(ValueError, match="55_minutes"):
        reserve(ledger, "train", 3301, 1, ["echo"])
    job = reserve(ledger, "A", 100, 800, ["echo"])
    with pytest.raises(ValueError, match="previous_job"):
        reserve(ledger, "B", 100, 600, ["echo"])
    finish(ledger, job, 10, 0, False)
    with pytest.raises(ValueError, match="budget_exhausted"):
        reserve(ledger, "A", 100, 1, ["echo"])
    with pytest.raises(ValueError, match="budget_exhausted"):
        reserve(ledger, "B", 14400, 600, ["echo"])


def test_gold_construction_starts_with_question_and_reruns_targeted_lookup():
    from types import SimpleNamespace
    from e1_alignment.rebuild.collect import strict_gold_attempt
    class Tokenizer:
        def encode(self, s, **kwargs): return list(s.encode())
        def decode(self, ids, **kwargs): return bytes(ids).decode()
    calls = []
    class Runtime:
        tokenizer = Tokenizer()
        contract = SimpleNamespace(max_schema_lookups=2, max_observation_tokens=10000, observation_mode="compact")
        def schema_tool(self, db):
            def lookup(args):
                calls.append(args)
                return {"status": "ok", "observation": ("Tables:\n- a\n  - id INT\nRelationships:" if len(calls) == 1 else "Tables:\n- a\n  - x INT\nRelationships:")}
            return lookup
        def sql_tool(self, path): return lambda sql: {"status": "ok", "columns": ["x"], "rows_preview": [[1]]}
    import unittest.mock
    row = {"question": "What is x?", "evidence": "Use a.", "gold_sql": "SELECT x FROM a", "db_path": "fake", "db_id": "a"}
    with unittest.mock.patch("e1_alignment.rebuild.collect.sqlite_schema", return_value={"a": {"x": "INT"}}), \
         unittest.mock.patch("e1_alignment.rebuild.collect.extract", return_value=extract(row["gold_sql"], schema=SCHEMA)):
        selected, detail = strict_gold_attempt(row, Runtime())
    assert selected and calls[0] == {"query": "What is x? Use a."}
    assert len(calls) == 2 and "x" in calls[1]["query"] and calls[1] != calls[0]


def test_initial_source_prefix_tokens_are_preserved(monkeypatch):
    from e1_alignment.rebuild.collect import source_sft_prefixes
    from e1_alignment.common import canonical_hash
    real = {"id": "q", "final_sql": "SELECT 2", "final_correct": False, "tool_events": [{"action": "sql_execute"}],
            "segments": [{"kind": "prompt", "text": "q", "ids": [1]},
                         {"kind": "action", "text": "bad", "ids": [2, 999]},
                         {"kind": "observation", "text": "error", "ids": [3]}]}
    canonical = {"question_id": "q", "prefix_actions": "bad", "prefix_tool_calls": 1,
                 "source_trace": {"trace_final_sql": "SELECT 2"}, "segments": [{"kind": "action", "text": "bad", "ids": [2]},
                                                    {"kind": "observation", "text": "error", "ids": [3]}]}
    monkeypatch.setattr("e1_alignment.rebuild.collect.read_jsonl", lambda _: [real])
    monkeypatch.setattr("e1_alignment.rebuild.collect.candidate_states", lambda *args: [canonical])
    monkeypatch.setattr("e1_alignment.rebuild.collect.select_candidates", lambda c, *args: c)
    selected = source_sft_prefixes({}, None)
    assert selected[0]["segments"][0]["ids"] == [2, 999]


def test_ast_covers_group_having_and_rejects_computed_implicit_join():
    req = extract("SELECT id, SUM(x) AS total FROM a GROUP BY id HAVING total>2 ORDER BY total", schema=SCHEMA)
    assert req["ok"] and set(req["columns"]) == {"a.id", "a.x"}
    assert not extract("SELECT a.x FROM a,b WHERE lower(a.id)=lower(b.id)", schema=SCHEMA)["ok"]


def test_in_subquery_records_semijoin_without_changing_its_semantics():
    req = extract("SELECT x FROM a WHERE id IN(SELECT a_id FROM b WHERE y>1)", schema=SCHEMA)
    assert req["ok"], req["reasons"]
    assert req["joins"][0]["operator"] == "IN"
    assert {req["joins"][0]["left"]["table"], req["joins"][0]["right"]["table"]} == {"a", "b"}
    assert set(req["columns"]) == {"a.x", "a.id", "b.a_id", "b.y"}


def test_observation_cannot_be_directly_edited_during_revision():
    row, _ = reviewed_row()
    row['segments'] = [{'kind': 'observation', 'text': 'real', 'ids': [1]}]
    modified = {**row, 'segments': [{'kind': 'observation', 'text': 'invented', 'ids': [2]}]}
    with pytest.raises(ValueError, match='requires_real_acquisition_receipt'):
        revised(row, modified, 'inject missing column')


def test_executable_car_gold_still_rejected_when_price_missing_from_visible_text(tmp_path):
    from tools.sql_worker import run
    db = tmp_path / 'cars.sqlite'
    with sqlite3.connect(db) as conn:
        conn.execute('CREATE TABLE cars(ID INT, brand TEXT)')
        conn.execute('CREATE TABLE price(ID INT, price REAL)')
        conn.execute("INSERT INTO cars VALUES(1,'a')")
        conn.execute('INSERT INTO price VALUES(1,20)')
    sql = 'SELECT price.price FROM cars JOIN price ON cars.ID=price.ID'
    result = run({'db_path':str(db), 'operation':'score', 'sql':sql, 'gold_sql':sql})
    assert result['ok'] and result['correct']
    req = extract(sql, str(db))
    assert {'price.id', 'price.price'} <= set(req['columns'])
    segment = obs('Tables:\n- cars\n  - ID INT\n- price\n  - ID INT\nRelationships:\n- cars.ID = price.ID   [sqlite_fk]')
    result = coverage(req, [segment], 1, SCHEMA)
    assert not result['ok'] and result['missing']['columns'] == ['price.price']


def test_alphabetical_gold_table_lookup_is_not_a_valid_rebuild_demo(tmp_path):
    from e1_alignment.rebuild.quality import program_check
    db = tmp_path/'a.sqlite'
    with sqlite3.connect(db) as conn:
        conn.execute('CREATE TABLE a(x INT)')
    row = {'id':'q','db_id':'a','db_path':str(db),'question':'x?','gold_sql':'SELECT x FROM a','evidence':''}
    record = versioned({**row,'trajectory_source':'e1_gold_assisted','gold_used_for_construction':True,
                        'autonomous_repair':False,'success_type':'gold_assisted','final_sql':row['gold_sql'],
                        'segments':[{'kind':'action','action':'schema_lookup','arguments':{'table':'a'},'ids':[],'loss_mask':[]}],
                        'input_ids':[],'loss_mask':[]}, 'fixture',1)
    got = program_check(record, {'q':row}, set(), None, score=False)
    assert 'gold_first_lookup_not_question_evidence' in got['reasons']


def test_orphan_observation_cannot_inject_schema_into_training():
    from e1_alignment.rebuild.quality import sequence_errors
    real = [{'kind':'prompt'}, {'kind':'action','action':'schema_lookup','tool_event_index':0},
            {'kind':'observation','tool_event_index':0}, {'kind':'action','action':'answer'}]
    assert not sequence_errors(real, [{}])
    injected = real[:1] + [{'kind':'observation','text':'price.price magically visible'}] + real[1:]
    assert 'orphan_or_unexpected_segment' in sequence_errors(injected, [{}])
    assert 'tool_event_count_mismatch' in sequence_errors(real, [{}, {}])


def test_failure_manifest_requires_a_real_policy_origin(monkeypatch):
    from e1_alignment.rebuild.audit import verified_prefixes
    prefix = {'state_id':'s', 'question_id':'q', 'prefix_tool_calls':1,
              'source_trace': {'dir':'source_sft','trace_final_sql':'SELECT 1'},
              'tool_events':[{'action':'sql_execute','arguments':'SELECT bad','result':{'status':'error'}}]}
    monkeypatch.setattr('e1_alignment.rebuild.audit.read_jsonl', lambda p: [prefix] if str(p)=='prefix' else [])
    valid, rejected = verified_prefixes(['prefix'])
    assert not valid and rejected[0]['reason'] == 'failure_origin_not_verified'


def test_watchdog_receipt_required_and_cleanup_is_budgeted(tmp_path, monkeypatch):
    from e1_alignment.rebuild.budget import active_budget, reserve
    monkeypatch.delenv('SQLRL_REBUILD_BUDGET_JOB', raising=False)
    monkeypatch.delenv('SQLRL_REBUILD_BUDGET_LEDGER', raising=False)
    with pytest.raises(ValueError, match='watchdog'):
        active_budget('train')
    path=tmp_path/'budget.json'
    job=reserve(path,'train',3300,1,['test'])
    state=json.loads(path.read_text())
    assert state['jobs'][job]['reserved_s'] == 3330
    monkeypatch.setenv('SQLRL_REBUILD_BUDGET_JOB',str(job))
    monkeypatch.setenv('SQLRL_REBUILD_BUDGET_LEDGER',str(path))
    assert 0 < active_budget('train') <= 3300
    with pytest.raises(ValueError, match='not_active'):
        active_budget('A')


def test_training_split_rechecks_block_validation_db_question_changes_and_holdout():
    from e1_alignment.rebuild.freeze import split_errors
    train = {'q':{'id':'q','question':'original','gold_sql':'SELECT 1','evidence':'','db_id':'train','db_path':'x'}}
    assert not split_errors([train['q']],train,set(),{'val'})
    assert 'not_in_rl_train' in split_errors([{'id':'val_question'}],train,set(),{'val'})
    assert 'train_identity_mismatch:question' in split_errors([{**train['q'],'question':'rewritten'}],train,set(),{'val'})
    assert 'monitor_leakage' in split_errors([train['q']],train,{'q'},{'val'})
    assert 'evaluation_database_leakage' in split_errors([train['q']],train,set(),{'train'})


def test_quoted_identifiers_with_spaces_and_unsupported_dotted_names():
    schema={'sale price':{'gross value':'REAL'}}
    req=extract('SELECT "gross value" FROM "sale price"',schema=schema)
    assert req['ok'] and req['columns']==['sale price.gross value']
    seg=obs('Tables:\n- sale price\n  - gross value REAL\nRelationships:')
    assert coverage(req,[seg],1,schema)['ok']
    assert not extract('SELECT "x.y" FROM a',schema={'a':{'x.y':'INT'}})['ok']


def test_aggregate_cte_join_is_never_misreported_as_a_raw_fk_equality():
    req=extract('WITH c AS (SELECT SUM(x) AS total FROM a) SELECT c.total FROM c JOIN b ON c.total=b.y',schema=SCHEMA)
    assert not req['ok'] and any('computed_join_output_lineage' in r for r in req['reasons'])
    req=extract('WITH c AS (SELECT id AS k FROM a) SELECT c.k FROM c JOIN b ON c.k=b.a_id',schema=SCHEMA)
    assert req['ok']
    assert req['joins'][0]['left']['table']=='a'


def test_supervised_execute_rechecks_original_literals_even_if_legacy_metadata_says_ex_one(tmp_path):
    from e1_alignment.rebuild.quality import program_check
    db=tmp_path/'literals.sqlite'
    with sqlite3.connect(db) as conn:
        conn.execute('CREATE TABLE a(x INT)')
    row={'id':'q','question':'exact literal','evidence':'','db_id':'a','db_path':str(db),'gold_sql':"SELECT 'x  y'"}
    wrong="SELECT 'x y'"
    rec=versioned({**row,'trajectory_source':'e1_self_rollout','gold_used_for_construction':False,'final_correct':True,
                   'final_sql':row['gold_sql'],'segments':[{'kind':'action','action':'sql_execute','arguments':wrong,'loss_mask':[1],
                   'ids':[1],'text':f'<sql_execute>{wrong}</sql_execute>','offline_ex':True}], 'input_ids':[1],'loss_mask':[1]},'fixture',1)
    check=program_check(rec,{'q':row},set(),None,score=True)
    assert check['full_score']['correct'] is True
    assert 'supervised_sql_full_score_failed' in check['reasons']


def test_b_requires_a_real_failed_sql_not_lookup_only_or_already_correct_prefix():
    from e1_alignment.rebuild.quality import failure_prefix_check
    r={'prefix_action_segments':1,'segments':[{'kind':'action','action':'schema_lookup'}]}
    assert not failure_prefix_check(r)['ok']
    r={'prefix_action_segments':1,'db_path':'x','gold_sql':'SELECT 1',
       'segments':[{'kind':'action','action':'sql_execute','arguments':'SELECT 1','tool_event_index':0}],
       'tool_events':[{'result':{'status':'ok'}}]}
    assert failure_prefix_check(r,lambda *args:{'ok':True,'correct':True})['reason']=='failure_prefix_latest_sql_already_correct'
    assert failure_prefix_check(r,lambda *args:{'ok':True,'correct':False})['ok']
    r['tool_events'][0]['result']={'status':'error','message':'no such column: x'}
    assert failure_prefix_check(r)['ok']
