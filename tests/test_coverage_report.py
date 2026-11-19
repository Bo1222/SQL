"""Coverage accounting tests for M6-01 (no model, no training)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.coverage_report import _aggregate, _step_rows, parse_train_log  # noqa: E402


def make_record(step, qid, db, action=100, obs=200, prompt=300, lookups=1):
    return {"step": step, "question_id": qid, "db_id": db, "prompt_tokens": prompt,
            "action_tokens": action, "observation_tokens": obs, "turns": 3,
            "tool_calls": 2, "schema_lookups": lookups, "finish_reason": "answer_submitted"}


def test_step_rows_count_prompt_draws_not_trajectories():
    # q1 is sampled once per step but rolled out twice in step 0 (n=2 duplication)
    records = [
        make_record(0, "q1", "db1"), make_record(0, "q1", "db1"),
        make_record(1, "q2", "db2"), make_record(1, "q3", "db2"),
    ]
    rows = _step_rows(records, {})
    assert rows[0]["trajectories"] == 2 and rows[0]["unique_prompts"] == 1
    assert rows[0]["cumulative_prompt_draws"] == 1
    assert rows[1]["cumulative_prompt_draws"] == 3
    assert rows[1]["cumulative_unique"] == 3
    assert rows[1]["cumulative_prompt_repeat_rate"] == 0.0


def test_aggregate_excludes_warmup_step():
    records = [
        make_record(1, "q1", "db1", lookups=0), make_record(1, "q1", "db1", lookups=0),
        make_record(2, "q2", "db2"), make_record(2, "q3", "db3"),
    ]
    rows = _step_rows(records, {})
    steady = _aggregate(records, rows, warmup=1)
    assert steady["steps"] == 1
    assert steady["trajectories"] == 2
    assert steady["prompt_draws"] == 2
    assert steady["unique_prompts"] == 2
    assert steady["dbs"] == 2
    assert steady["lookup_rate"] == 1.0
    assert steady["protocol_errors"] == 0


def test_protocol_errors_counted():
    records = [
        dict(make_record(2, "q1", "db1"), finish_reason="invalid_action"),
        make_record(2, "q2", "db2"),
    ]
    rows = _step_rows(records, {})
    steady = _aggregate(records, rows, warmup=1)
    assert steady["protocol_errors"] == 1


def test_parse_train_log_extracts_step_metrics(tmp_path):
    log = tmp_path / "train.log"
    line = ("\x1b[36m(TaskRunner)\x1b[0m step:1 - global_seqlen/min:5584 - "
            "timing_s/step:102.28 - timing_s/gen:55.96 - perf/total_num_tokens:5584 - "
            "training/global_step:1 - critic/score/mean:0.0")
    log.write_text(line + "\n", encoding="utf-8")
    steps = parse_train_log(log)
    assert steps[1]["timing_s/step"] == 102.28
    assert steps[1]["perf/total_num_tokens"] == 5584
    assert steps[1]["critic/score/mean"] == 0.0
