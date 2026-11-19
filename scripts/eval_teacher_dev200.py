"""Run a strong teacher model on dev200 under the project's SQL agent harness.

The configured ANTHROPIC_* endpoint in this environment is DeepSeek's
Anthropic-compatible API (not Anthropic Claude).  The script records the actual
endpoint host and model; it never prints the auth token.

Arms
----
tool        same three-action protocol, prompt, tools and budgets as the 3B eval
fullschema  one-shot SQL with the complete catalog schema in the prompt (no tools)
oracle      one-shot SQL with Gold-selected table/columns only (privileged, diagnostic)

Scoring is reported both as the historical project-strict contract and as the
official BIRD set-only contract.  The teacher outputs are diagnostic only and
are never written into SFT/RL data.
"""
import argparse
import hashlib
import json
import math
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import anthropic

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.protocol import build_prompt, parse_action
from agent.sql_episode import EpisodeConfig, encode_observation
from eval.harness_audit import AuditTokenizer
from eval.sql_dependencies import sqlite_requirements
from retrieval.catalog_store import CatalogStore
from reward.bird_ex_reward import FENCE_RE, extract_final_sql
from tools.schema_lookup import SchemaLookupTool
from tools.sql_execute import SQLExecuteTool, run_sql

TOKENIZER_DIR = ROOT / "artifacts/checkpoints/step_600_merged"
CATALOG_DIR = ROOT / "artifacts/catalog"
DATA = ROOT / "artifacts/datasets/dev200.jsonl"
BASELINE_3B = ROOT / "reports/m6/e1_student_only/sft_lowlr_788.jsonl"

_TS = threading.local()
_WRITE_LOCK = threading.Lock()

ONESHOT_INSTRUCTION = """You are a read-only SQLite database agent.
Write exactly one SQLite SELECT or WITH query that answers the question.
Use only tables and columns from the schema below.
Return only <answer>YOUR SQL</answer> and nothing else.

### Database schema
{schema}

### Question
{question}

### Evidence
{evidence}

### Answer
"""

DIRECT_CATALOG_INSTRUCTION = """You are a read-only SQLite database agent.
Answer the question with exactly one SQLite SELECT or WITH query.
You have no tools in this ablation. Only the table names are available;
infer the columns from their names. Return only <answer>YOUR SQL</answer>
and nothing else.

### Table catalog
{catalog}

### Question
{question}

### Evidence
{evidence}

### Answer
"""

DIRECT_QUESTION_INSTRUCTION = """You are a SQLite expert.
Answer the question with exactly one SQLite SELECT or WITH query.
No database schema is provided in this ablation; use your prior knowledge.
Return only <answer>YOUR SQL</answer> and nothing else.

### Question
{question}

### Evidence
{evidence}

### Answer
"""


def direct_catalog_prompt(row, store):
    return DIRECT_CATALOG_INSTRUCTION.format(
        catalog="\n".join(store.table_names),
        question=row["question"],
        evidence=row.get("evidence") or "(not supplied)",
    )


def direct_question_prompt(row):
    return DIRECT_QUESTION_INSTRUCTION.format(
        question=row["question"],
        evidence=row.get("evidence") or "(not supplied)",
    )


def get_tokenizer():
    if not hasattr(_TS, "tokenizer"):
        _TS.tokenizer = AuditTokenizer(TOKENIZER_DIR)
    return _TS.tokenizer


def get_client():
    if not hasattr(_TS, "client"):
        _TS.client = anthropic.Anthropic(
            base_url=os.environ["ANTHROPIC_BASE_URL"],
            auth_token=os.environ["ANTHROPIC_AUTH_TOKEN"],
            max_retries=5,
            timeout=180,
        )
    return _TS.client


def complete(model, messages, max_tokens):
    """One teacher completion, with a small explicit retry loop."""
    last = None
    for attempt in range(8):
        try:
            resp = get_client().messages.create(
                model=model, max_tokens=max_tokens, temperature=0.0, messages=messages
            )
            text = "".join(
                getattr(block, "text", "") for block in resp.content
                if getattr(block, "type", "") == "text"
            )
            if not text:
                text = "".join(getattr(block, "text", "") for block in resp.content)
            usage = getattr(resp, "usage", None)
            return {
                "text": text,
                "stop_reason": getattr(resp, "stop_reason", None),
                "model": getattr(resp, "model", model),
                "input_tokens": getattr(usage, "input_tokens", None),
                "output_tokens": getattr(usage, "output_tokens", None),
            }
        except Exception as exc:  # rate limit / proxy / transient
            last = exc
            time.sleep(min(2 ** attempt, 20))
    raise RuntimeError(f"teacher API failed after retries: {type(last).__name__}: {last}")


def _unwrap_sql_body(body):
    """Accept a JSON-wrapped SQL body from a teacher (diagnostic leniency only)."""
    body = body.strip()
    if not body.startswith("{"):
        return body
    try:
        obj = json.loads(body)
    except json.JSONDecodeError:
        return body
    if isinstance(obj, dict):
        for key in ("query", "sql", "answer"):
            if isinstance(obj.get(key), str) and obj[key].strip():
                return obj[key].strip()
    return body


def _normalise_teacher_tags(text):
    """Turn JSON-wrapped sql_execute/answer bodies into the raw protocol body."""
    def replace(match):
        tag, body = match.group(1), match.group(2)
        if tag in ("sql_execute", "answer"):
            body = _unwrap_sql_body(body)
        return f"<{tag}>{body}</{tag}>"

    return re.sub(r"<(schema_lookup|sql_execute|answer)>(.*?)</\1>", replace, text,
                  flags=re.S | re.I)


def parse_action_lenient(text):
    """Strict parser first; then tolerate fences / extra prose / JSON SQL bodies."""
    try:
        return parse_action(text), "strict"
    except (ValueError, TypeError):
        pass
    cleaned = text.strip()
    fence = re.search(r"```(?:sql|json)?\s*(.*?)```", cleaned, re.S | re.I)
    if fence:
        cleaned = fence.group(1).strip()
    cleaned = _normalise_teacher_tags(cleaned)
    try:
        return parse_action(cleaned), "fenced"
    except (ValueError, TypeError):
        pass
    match = re.search(r"<(schema_lookup|sql_execute|answer)>.*?</\1>", text, re.S | re.I)
    if match:
        candidate = _normalise_teacher_tags(match.group(0))
        try:
            return parse_action(candidate), "extracted"
        except (ValueError, TypeError):
            pass
    raise ValueError("teacher output is not one complete action")


def score_record(row, final_sql):
    if not final_sql:
        return {
            "correct": False, "correct_set": False, "valid": False,
            "error": "missing_answer", "error_type": "missing_answer",
            "infrastructure_error": False,
        }
    result = run_sql(
        row["db_path"], final_sql, operation="score_diagnostic", timeout=30,
        gold_sql=row["gold_sql"], score_mode="project_strict",
    )
    return {
        "correct": bool(result.get("project_strict_correct")),
        "correct_set": bool(result.get("bird_set_only_correct")),
        "valid": bool(result.get("ok")),
        "error": result.get("error"),
        "error_type": result.get("error_type") or ("none" if result.get("ok") else "other"),
        "infrastructure_error": bool(result.get("infrastructure_error")),
        "gold_rows": result.get("gold_rows"),
        "pred_rows": result.get("pred_rows"),
    }


def resolve_null_relationship_endpoints(store):
    """SQLite FK rows may omit the parent column when it references the PK.

    The catalog stores that as ``null``; rendering it produces the misleading
    ``Customers.GEOID = Demog.None`` observation.  Resolve it to the parent
    primary key before the teacher sees the schema.  This is a diagnostic-side
    fix so the teacher is not penalised by a known harness defect.
    """
    primary_keys = {}
    for table in store.schema["tables"]:
        columns = [c["column"] for c in store.catalog
                   if c["table"] == table["name"] and c.get("primary_key")]
        if len(columns) == 1:
            primary_keys[table["name"]] = columns[0]
    fixed = []
    for relationship in store.relationships:
        relationship = dict(relationship)
        if not relationship.get("from_column") and relationship["from_table"] in primary_keys:
            relationship["from_column"] = primary_keys[relationship["from_table"]]
        if not relationship.get("to_column") and relationship["to_table"] in primary_keys:
            relationship["to_column"] = primary_keys[relationship["to_table"]]
        fixed.append(relationship)
    store.relationships = fixed
    store._adjacency = None


def full_schema_context(store):
    lines = [f"Database: {store.manifest.get('db_id', '')}", "Tables:"]
    for table in store.schema["tables"]:
        lines.append(f"- {table['name']}")
        for column in store.catalog:
            if column["table"] != table["name"]:
                continue
            desc = (column.get("description") or "")[:200]
            values = (column.get("value_description") or "")[:160]
            pk = " [PK]" if column.get("primary_key") else ""
            line = f"  - {column['column']} {column.get('data_type', '')}{pk}"
            if desc:
                line += f" -- {desc}"
            if values:
                line += f" | values: {values}"
            lines.append(line)
    lines.append("Relationships:")
    for rel in store.relationships:
        lines.append(
            f"- {rel['from_table']}.{rel['from_column']} = "
            f"{rel['to_table']}.{rel['to_column']} [{rel.get('source', '')}]"
        )
    return "\n".join(lines)


def oracle_schema_context(requirements, store):
    selected = set(requirements["columns"])
    lines = ["Gold-selected database context (diagnostic only):"]
    for table in requirements["tables"]:
        lines.append(f"Table {json.dumps(table, ensure_ascii=False)}")
        for record in store.catalog:
            key = f"{record['table']}.{record['column']}".lower()
            if key not in selected:
                continue
            item = {k: record[k] for k in (
                "column", "data_type", "primary_key", "description", "value_description"
            ) if record.get(k) not in (None, "", False)}
            lines.append("  " + json.dumps(item, ensure_ascii=False))
    lines.append("Declared database relationships:")
    for rel in store.relationships:
        if rel["from_table"].lower() in requirements["tables"] and \
                rel["to_table"].lower() in requirements["tables"]:
            lines.append(json.dumps(rel, ensure_ascii=False))
    return "\n".join(lines).replace("<", "\\u003c").replace(">", "\\u003e")


def run_tool_arm(row, store, model, config, max_tokens, lenient_parse):
    tokenizer = get_tokenizer()
    prompt = build_prompt(
        row["question"], row.get("evidence", ""), "\n".join(store.table_names), config.max_tools
    )
    schema_tool = SchemaLookupTool(
        store, token_budget=config.max_observation_tokens,
        count_tokens=lambda text: len(tokenizer.encode(text)),
        enable_value_lookup=False, retrieval_version="E1.1",
    )
    sql_tool = SQLExecuteTool(row["db_path"], timeout=config.tool_timeout,
                              preview_rows=config.preview_rows)
    messages = [{"role": "user", "content": prompt}]
    transcript = prompt
    tool_calls = schema_lookups = sql_executes = invalid = turns = 0
    final_sql = None
    finish_reason = "turn_limit"
    usages = []
    parse_modes = []
    invalid_events = []
    tool_events = []
    for _ in range(config.max_rounds):
        if len(tokenizer.encode(transcript)) > config.max_context:
            finish_reason = "generation_error"
            break
        turns += 1
        out = complete(model, messages, max_tokens)
        usages.append({k: out.get(k) for k in ("input_tokens", "output_tokens", "stop_reason")})
        text = out["text"]
        messages.append({"role": "assistant", "content": text})
        transcript += text
        try:
            action, mode = parse_action_lenient(text) if lenient_parse else (parse_action(text), "strict")
        except (ValueError, TypeError) as exc:
            invalid += 1
            invalid_events.append(str(exc)[:300])
            if invalid >= config.max_invalid_actions:
                finish_reason = "invalid_action"
                break
            observation = {"status": "error", "error_type": "invalid_action",
                           "message": str(exc)[:300]}
            _, observation_text = encode_observation(
                tokenizer, observation, config.max_observation_tokens,
                mode=config.observation_mode)
            messages.append({"role": "user", "content": observation_text})
            transcript += observation_text
            continue
        parse_modes.append(mode)
        if action.name == "answer":
            final_sql = action.arguments
            finish_reason = "answer_submitted"
            break
        if action.name == "schema_lookup" and schema_lookups >= config.max_schema_lookups:
            payload = {"status": "error", "error_type": "budget",
                       "message": "schema_lookup budget exhausted"}
        elif action.name == "sql_execute" and sql_executes >= config.max_sql_executes:
            payload = {"status": "error", "error_type": "budget",
                       "message": "sql_execute budget exhausted"}
        elif tool_calls >= config.max_tools:
            payload = {"status": "error", "error_type": "budget",
                       "message": "tool budget exhausted"}
            finish_reason = "tool_limit"
        else:
            tool_calls += 1
            if action.name == "schema_lookup":
                schema_lookups += 1
                payload = schema_tool(action.arguments)
            else:
                sql_executes += 1
                payload = sql_tool(action.arguments)
        tool_events.append({
            "action": action.name, "arguments": action.arguments,
            "result_status": payload.get("status"),
            "result_error_type": payload.get("error_type"),
            "observation_chars": len(json.dumps(payload, ensure_ascii=False, default=str)),
        })
        _, observation_text = encode_observation(
            tokenizer, payload, config.max_observation_tokens, mode=config.observation_mode)
        messages.append({"role": "user", "content": observation_text})
        transcript += observation_text
        if payload.get("failure_origin") == "infrastructure":
            finish_reason = "infrastructure_error"
            break
        if finish_reason == "tool_limit":
            break
    scored = score_record(row, final_sql)
    return {
        "arm": "tool", "id": row["id"], "db_id": row["db_id"],
        "question": row["question"], "gold_sql": row["gold_sql"],
        "final_sql": final_sql, "finish_reason": finish_reason,
        "turns": turns, "tool_calls": tool_calls,
        "schema_lookups": schema_lookups, "sql_executes": sql_executes,
        "invalid_actions": invalid, "invalid_events": invalid_events,
        "parse_modes": parse_modes, "tool_events": tool_events,
        "prompt_tokens": len(tokenizer.encode(prompt)),
        "transcript_tail": transcript[-4000:],
        "api_usages": usages,
        **scored,
    }


def extract_oneshot_sql(text):
    sql = extract_final_sql(text)
    if sql:
        return sql, "answer_tag"
    fence = FENCE_RE.search(text or "")
    if fence:
        return fence.group(1).strip() or None, "fence"
    stripped = (text or "").strip()
    if stripped.lower().startswith(("select", "with")):
        return stripped, "raw"
    return None, "missing"


def run_oneshot_arm(row, schema_text, model, max_tokens, arm):
    prompt = ONESHOT_INSTRUCTION.format(
        schema=schema_text, question=row["question"],
        evidence=row.get("evidence") or "(not supplied)")
    out = complete(model, [{"role": "user", "content": prompt}], max_tokens)
    final_sql, extraction = extract_oneshot_sql(out["text"])
    scored = score_record(row, final_sql)
    return {
        "arm": arm, "id": row["id"], "db_id": row["db_id"],
        "question": row["question"], "gold_sql": row["gold_sql"],
        "final_sql": final_sql, "extraction": extraction,
        "finish_reason": "answer_submitted" if final_sql else "missing_answer",
        "raw_output": out["text"][:4000],
        "api_usages": [{k: out.get(k) for k in
                        ("input_tokens", "output_tokens", "stop_reason")}],
        **scored,
    }


def run_prompt_arm(row, prompt, model, max_tokens, arm):
    """One API call with an arbitrary prompt; no tools, no local harness."""
    out = complete(model, [{"role": "user", "content": prompt}], max_tokens)
    final_sql, extraction = extract_oneshot_sql(out["text"])
    scored = score_record(row, final_sql)
    return {
        "arm": arm, "id": row["id"], "db_id": row["db_id"],
        "question": row["question"], "gold_sql": row["gold_sql"],
        "final_sql": final_sql, "extraction": extraction,
        "finish_reason": "answer_submitted" if final_sql else "missing_answer",
        "prompt_chars": len(prompt),
        "raw_output": out["text"][:4000],
        "api_usages": [{k: out.get(k) for k in
                        ("input_tokens", "output_tokens", "stop_reason")}],
        **scored,
    }


def load_jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def mcnemar_exact(b, c):
    if b + c == 0:
        return 1.0
    n = b + c
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def paired(arm_a, arm_b, key):
    """arm_a/arm_b: dict id -> record.  Returns a-vs-b flips and McNemar p."""
    ids = sorted(set(arm_a) & set(arm_b))
    gain = sum((not arm_a[i].get(key)) and arm_b[i].get(key) for i in ids)
    loss = sum(arm_a[i].get(key) and (not arm_b[i].get(key)) for i in ids)
    return {"n": len(ids), "gain": gain, "loss": loss, "net": gain - loss,
            "mcnemar_exact_p": mcnemar_exact(gain, loss)}


def summarize(records, baseline):
    dedup = {}
    for record in records:
        dedup[(record["id"], record["arm"])] = record
    records = list(dedup.values())
    by_arm = {}
    for record in records:
        by_arm.setdefault(record["arm"], {})[record["id"]] = record

    summary = {"n_records": len(records), "arms": {}}
    for arm, rows in sorted(by_arm.items()):
        values = list(rows.values())
        n = len(values)
        summary["arms"][arm] = {
            "n": n,
            "correct_strict": sum(bool(r.get("correct")) for r in values),
            "ex_strict": round(sum(bool(r.get("correct")) for r in values) / n, 4) if n else 0,
            "correct_set": sum(bool(r.get("correct_set")) for r in values),
            "ex_set": round(sum(bool(r.get("correct_set")) for r in values) / n, 4) if n else 0,
            "valid": sum(bool(r.get("valid")) for r in values),
            "valid_rate": round(sum(bool(r.get("valid")) for r in values) / n, 4) if n else 0,
            "invalid": sum(not bool(r.get("valid")) for r in values),
            "finish_reasons": {k: sum(1 for r in values if r.get("finish_reason") == k)
                               for k in sorted({r.get("finish_reason") for r in values})},
            "avg_turns": round(sum(r.get("turns", 1) for r in values) / n, 3) if n else 0,
            "avg_tool_calls": round(sum(r.get("tool_calls", 0) for r in values) / n, 3) if n else 0,
            "avg_schema_lookups": round(sum(r.get("schema_lookups", 0) for r in values) / n, 3) if n else 0,
            "avg_sql_executes": round(sum(r.get("sql_executes", 0) for r in values) / n, 3) if n else 0,
            "avg_input_tokens": round(sum(u.get("input_tokens") or 0 for r in values
                                          for u in r.get("api_usages", [])) / max(n, 1), 1),
            "avg_output_tokens": round(sum(u.get("output_tokens") or 0 for r in values
                                           for u in r.get("api_usages", [])) / max(n, 1), 1),
        }
        per_db = {}
        for db in sorted({r["db_id"] for r in values}):
            subset = [r for r in values if r["db_id"] == db]
            per_db[db] = {"n": len(subset),
                          "correct_strict": sum(bool(r.get("correct")) for r in subset),
                          "correct_set": sum(bool(r.get("correct_set")) for r in subset)}
        summary["arms"][arm]["per_db"] = per_db

    # paired comparisons between teacher arms
    summary["paired"] = {}
    arms = sorted(by_arm)
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            summary["paired"][f"{a}_vs_{b}"] = {
                "strict": paired(by_arm[a], by_arm[b], "correct"),
                "set": paired(by_arm[a], by_arm[b], "correct_set"),
            }
    # paired comparisons against the 3B SFT step788
    if baseline:
        summary["baseline_3b"] = {
            "n": len(baseline),
            "correct_strict": sum(v["strict"] for v in baseline.values()),
            "ex_strict": round(sum(v["strict"] for v in baseline.values()) / len(baseline), 4),
            "correct_set": sum(v["set"] for v in baseline.values()),
            "ex_set": round(sum(v["set"] for v in baseline.values()) / len(baseline), 4),
        }
        summary["teacher_vs_3b"] = {}
        for arm, rows in by_arm.items():
            common = {i: v for i, v in rows.items() if i in baseline}
            base = {i: {"correct": baseline[i]["strict"], "correct_set": baseline[i]["set"]}
                    for i in common}
            summary["teacher_vs_3b"][arm] = {
                "strict": paired(base, common, "correct"),
                "set": paired(base, common, "correct_set"),
            }
    return summary


def baseline_map(path):
    """3B dev200 records, with set-only rescoring for valid-but-wrong."""
    if not path.exists():
        return {}
    records = load_jsonl(path)
    out = {}
    for row in records:
        strict = bool(row.get("correct"))
        set_correct = strict
        if row.get("valid") and not strict:
            result = run_sql(
                row["db_path"], row["final_sql"], operation="score_diagnostic", timeout=30,
                gold_sql=row["gold_sql"], score_mode="project_strict")
            set_correct = bool(result.get("bird_set_only_correct"))
        out[row["id"]] = {"strict": strict, "set": set_correct, "valid": bool(row.get("valid"))}
    return out


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")


def write_markdown(path, summary):
    lines = ["# Teacher dev200 rollout", "",
             "Scoring: `strict` is the historical project contract; `set` is the official BIRD set-only contract.", ""]
    lines += ["| arm | n | EX strict | EX set | valid | avg turns | avg tools |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for arm in sorted(summary["arms"]):
        value = summary["arms"][arm]
        lines.append(f"| {arm} | {value['n']} | {value['correct_strict']} ({value['ex_strict']:.1%}) | "
                     f"{value['correct_set']} ({value['ex_set']:.1%}) | "
                     f"{value['valid_rate']:.1%} | {value['avg_turns']} | {value['avg_tool_calls']} |")
    lines += ["", "## Teacher vs 3B SFT step788", ""]
    lines += ["| arm | strict gain/loss | p | set gain/loss | p |",
              "|---|---:|---:|---:|---:|"]
    for arm, value in sorted(summary.get("teacher_vs_3b", {}).items()):
        s, b = value["strict"], value["set"]
        lines.append(f"| {arm} | {s['gain']}/{s['loss']} | {s['mcnemar_exact_p']:.4g} | "
                     f"{b['gain']}/{b['loss']} | {b['mcnemar_exact_p']:.4g} |")
    lines += ["", "## Teacher arm comparisons", ""]
    for key, value in sorted(summary.get("paired", {}).items()):
        s, b = value["strict"], value["set"]
        lines.append(f"- {key}: strict {s['gain']}/{s['loss']} (p={s['mcnemar_exact_p']:.4g}); "
                     f"set {b['gain']}/{b['loss']} (p={b['mcnemar_exact_p']:.4g})")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="deepseek-reasoner")
    parser.add_argument("--data", default=str(DATA))
    parser.add_argument("--out", default="reports/m6/harness_audit_20261117/teacher_dev200")
    parser.add_argument("--arms", default="tool,fullschema,oracle")
    parser.add_argument("--limit", type=int, default=0, help="0 means all dev200")
    parser.add_argument("--ids", default="", help="comma-separated question ids")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens-tool", type=int, default=1024)
    parser.add_argument("--max-tokens-oneshot", type=int, default=1024)
    parser.add_argument("--lenient-parse", action="store_true")
    parser.add_argument("--legacy-relationships", action="store_true",
                        help="keep the historical null FK endpoints (debug only)")
    parser.add_argument("--compare", default=str(BASELINE_3B))
    parser.add_argument("--skip-baseline-rescore", action="store_true")
    args = parser.parse_args()

    rows = load_jsonl(args.data)
    if args.ids:
        wanted = {x.strip() for x in args.ids.split(",") if x.strip()}
        rows = [row for row in rows if row["id"] in wanted]
    if args.limit:
        rows = rows[:args.limit]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    stores = {}
    for row in rows:
        db = row["db_id"]
        if db not in stores:
            store = CatalogStore(CATALOG_DIR / db)
            if not args.legacy_relationships:
                resolve_null_relationship_endpoints(store)
            _ = store.adjacency  # initialize the lazy graph before threads
            stores[db] = store

    requirements_cache = {}
    for row in rows:
        if "oracle" not in arms:
            break
        req = sqlite_requirements(row["db_path"], row["gold_sql"])
        if not req["ok"]:
            raise ValueError(f"unresolved gold requirements for {row['id']}: {req}")
        requirements_cache[row["id"]] = req

    records_path = out / "records.jsonl"
    done = {}
    if records_path.exists():
        for record in load_jsonl(records_path):
            done[(record["id"], record["arm"])] = record

    jobs = [(row, arm) for row in rows for arm in arms
            if (row["id"], arm) not in done]
    print(f"questions={len(rows)} arms={arms} pending_jobs={len(jobs)}", flush=True)

    def run_job(job):
        row, arm = job
        started = time.monotonic()
        try:
            if arm == "tool":
                record = run_tool_arm(row, stores[row["db_id"]], args.model,
                                      EpisodeConfig(max_rounds=5, max_context=8192),
                                      args.max_tokens_tool, args.lenient_parse)
            elif arm == "fullschema":
                record = run_oneshot_arm(row, full_schema_context(stores[row["db_id"]]),
                                         args.model, args.max_tokens_oneshot, "fullschema")
            elif arm == "oracle":
                record = run_oneshot_arm(
                    row, oracle_schema_context(requirements_cache[row["id"]],
                                               stores[row["db_id"]]),
                    args.model, args.max_tokens_oneshot, "oracle")
            elif arm == "direct_original":
                record = run_prompt_arm(row, row["prompt"], args.model,
                                         args.max_tokens_oneshot, "direct_original")
            elif arm == "direct_catalog":
                record = run_prompt_arm(row, direct_catalog_prompt(row, stores[row["db_id"]]),
                                         args.model, args.max_tokens_oneshot, "direct_catalog")
            elif arm == "direct_question":
                record = run_prompt_arm(row, direct_question_prompt(row), args.model,
                                         args.max_tokens_oneshot, "direct_question")
            else:
                raise ValueError(f"unknown arm: {arm}")
        except Exception as exc:
            record = {
                "arm": arm, "id": row["id"], "db_id": row["db_id"],
                "question": row["question"], "gold_sql": row["gold_sql"],
                "final_sql": None, "correct": False, "correct_set": False,
                "valid": False, "error": f"{type(exc).__name__}: {exc}",
                "error_type": "teacher_error", "infrastructure_error": True,
                "finish_reason": "teacher_error",
            }
        record["wall_seconds"] = round(time.monotonic() - started, 3)
        return record

    with records_path.open("a", encoding="utf-8") as stream:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(run_job, job): job for job in jobs}
            for future in as_completed(futures):
                record = future.result()
                with _WRITE_LOCK:
                    stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                    stream.flush()
                print(json.dumps({
                    "id": record["id"], "arm": record["arm"],
                    "correct": record.get("correct"), "correct_set": record.get("correct_set"),
                    "valid": record.get("valid"), "finish": record.get("finish_reason"),
                    "wall_s": record.get("wall_seconds"),
                }), flush=True)

    records = load_jsonl(records_path)
    baseline = {} if args.skip_baseline_rescore else baseline_map(Path(args.compare))
    summary = summarize(records, baseline)
    summary["manifest"] = {
        "model": args.model,
        "endpoint_host": urlparse(os.environ.get("ANTHROPIC_BASE_URL", "")).netloc,
        "data": str(Path(args.data).resolve()),
        "data_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        "arms": arms,
        "config": {"max_rounds": 5, "max_context": 8192, "max_tools": 4,
                   "max_schema_lookups": 2, "max_sql_executes": 2,
                   "max_observation_tokens": 1024, "observation_mode": "compact",
                   "retrieval_version": "E1.1",
                   "max_tokens_tool": args.max_tokens_tool,
                   "max_tokens_oneshot": args.max_tokens_oneshot,
                   "lenient_parse": args.lenient_parse,
                   "legacy_relationships": args.legacy_relationships},
        "note": ("Teacher is the model behind the configured ANTHROPIC-compatible endpoint; "
                 "in this environment that is DeepSeek, not Anthropic Claude. "
                 "Outputs are diagnostic only and never enter SFT/RL data."),
    }
    write_json(out / "summary.json", summary)
    write_markdown(out / "summary.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=1, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
