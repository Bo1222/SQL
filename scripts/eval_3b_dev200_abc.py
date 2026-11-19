"""Prepare, execute and summarize the fixed-step788 dev200 input ablation.

One vLLM instance batches requests from independent synchronous episodes. The
real episode/parser/tools are reused; only A/B/C's initial information and C's
one-shot budget differ. No teacher calls, weight updates or training data writes.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from dataclasses import asdict
import hashlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import queue
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.sql_episode import episode_to_record, run_episode
from eval.harness_audit import AuditTokenizer
from eval.input_ablation import (ARMS, VERSION, episode_config, full_schema_context,
                                 input_prompt, resolve_implicit_foreign_keys)
from eval.paired_statistics import bootstrap_ci, mcnemar
from retrieval.catalog_store import CatalogStore
from tools.schema_lookup import SchemaLookupTool
from tools.sql_execute import SQLExecuteTool, run_sql

DEFAULT_MODEL = "outputs/m6_08m/student_sft_full_lowlr_5e6/epoch_2/hf_model"
DEFAULT_OUT = "reports/m6/harness_audit_20261117/3b_dev200_input_abc"
DATA = "artifacts/datasets/dev200.jsonl"
EXPECTED_DATA_SHA256 = "bda04835e9404f872aafa892bc179ff80a9db1a11a946fc3e9c198b44c57a2ee"
CODE_FILES = ["scripts/eval_3b_dev200_abc.py", "eval/input_ablation.py",
              "agent/protocol.py", "agent/sql_episode.py", "retrieval/catalog_store.py",
              "retrieval/e1_1.py", "retrieval/e1_2.py", "retrieval/retriever.py",
              "retrieval/observation_formatter.py", "retrieval/schema_extractor.py",
              "tools/schema_lookup.py", "tools/sql_execute.py", "tools/sql_worker.py",
              "eval/harness_audit.py", "eval/paired_statistics.py"]


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def read_jsonl(path):
    if not Path(path).exists():
        return []
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def stores_for(rows):
    stores, repairs = {}, {}
    for row in rows:
        db = row["db_id"]
        if db not in stores:
            store = CatalogStore(ROOT / "artifacts/catalog" / db)
            repairs[db] = resolve_implicit_foreign_keys(store, row["db_path"])
            if repairs[db]["unresolved"]:
                raise ValueError(f"unresolved FK endpoints: {db}")
            _ = store.adjacency
            stores[db] = store
    return stores, repairs


def prepare(out, model):
    if out.exists():
        raise SystemExit("output already exists; use run to resume or choose a fresh --out")
    rows = read_jsonl(ROOT / DATA)
    if (sha_file(ROOT / DATA) != EXPECTED_DATA_SHA256 or len(rows) != 200
            or len({r["id"] for r in rows}) != 200):
        raise ValueError("not the frozen dev200")
    tokenizer = AuditTokenizer(model)
    stores, repairs = stores_for(rows)
    prompts = []
    for row in rows:
        if not Path(row["db_path"]).is_file():
            raise ValueError(f"missing database: {row['db_id']}")
        for arm in ARMS:
            prompt = input_prompt(row, stores[row["db_id"]], arm)
            tokens = len(tokenizer.encode(prompt))
            if tokens + episode_config(arm).max_action_tokens > 8192:
                raise ValueError(f"prompt does not fit; no silent truncation: {row['id']}/{arm}")
            prompts.append({"id": row["id"], "db_id": row["db_id"], "arm": arm,
                            "prompt": prompt, "prompt_sha256": sha_text(prompt),
                            "prompt_tokens": tokens})
    paths = [ROOT / f for f in CODE_FILES]
    paths += [ROOT / DATA]
    for db in stores:
        paths += sorted((ROOT / "artifacts/catalog" / db).glob("*.json"))
    # Hash the actual weights once, before loading; their inventory is also
    # checked after inference. Evaluation never opens any model file for writing.
    model_files = sorted(p for p in model.iterdir() if p.is_file())
    paths += model_files
    inputs = {str(p.resolve()): sha_file(p) for p in paths}
    inventory = {str(p.resolve()): {"size": p.stat().st_size,
                                  "mtime_ns": p.stat().st_mtime_ns} for p in model_files}
    out.mkdir(parents=True)
    prompts_path = out / "prompts.jsonl"
    prompts_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in prompts))
    manifest = {
        "version": VERSION, "diagnostic_only": True, "training_eligible": False,
        "model": str(model.resolve()), "data": str((ROOT / DATA).resolve()),
        "data_sha256": EXPECTED_DATA_SHA256, "n_questions": 200, "n_episodes": 600,
        "question_ids": [r["id"] for r in rows], "arms": list(ARMS),
        "definitions": {"A": "table catalog + question/evidence + tools",
                        "B": "A plus all catalog columns/descriptions/relationships before acting",
                        "C": "same schema data as B, answer-only instruction, one generation, no tools"},
        "configs": {arm: asdict(episode_config(arm)) for arm in ARMS},
        "generation": {"temperature": 0.0, "top_p": 1.0, "seed": 0,
                       "max_action_tokens": 512, "chat_template": False,
                       "stop_after_action": False, "max_model_len": 8192,
                       "max_batch_size": 16, "concurrent_episodes": 16,
                       "gpu_memory_utilization": 0.55},
        "retrieval_version": "E1.2", "observation_mode": "compact_v2",
        "value_lookup": False, "score_timeout": 30,
        "primary_metric": "bird_set_only", "secondary_metric": "project_strict",
        "schema_rendering": {"all_tables_and_columns": True, "gold_selection": False,
                             "description_max_chars": 200, "value_description_max_chars": 160},
        "relationship_repairs": repairs, "input_sha256": inputs,
        "model_inventory": inventory, "prompts_sha256": sha_file(prompts_path),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "limits": ["dev200 is a repeatedly used selection set, not an independent test.",
                   "A/B share instruction and tools; only B has initial full schema.",
                   "C changes instruction, tool availability and generation rounds together.",
                   "New A uses E1.2/compact_v2/resolved FKs; historical step788 used E1.1/compact/legacy FKs.",
                   "Schema field/value description strings use the teacher's 200/160-char limits.",
                   "Latency includes concurrent scheduling and is not isolated single-request latency."]}
    write_json(out / "manifest.json", manifest)
    print(json.dumps({"prepared": str(out), "episodes": 600,
                      "prompt_tokens": {arm: {"min": min(r['prompt_tokens'] for r in prompts if r['arm'] == arm),
                                              "max": max(r['prompt_tokens'] for r in prompts if r['arm'] == arm)}
                                        for arm in ARMS},
                      "fk_repairs": sum(len(r['fixes']) for r in repairs.values())}), flush=True)


def validate_manifest(out, verify_weights=False):
    manifest = json.loads((out / "manifest.json").read_text())
    weight_paths = set(manifest["model_inventory"])
    for path, digest in manifest["input_sha256"].items():
        if path in weight_paths and not verify_weights:
            continue
        if sha_file(path) != digest:
            raise ValueError(f"frozen input changed: {path}")
    for path, old in manifest["model_inventory"].items():
        stat = Path(path).stat()
        if stat.st_size != old["size"] or stat.st_mtime_ns != old["mtime_ns"]:
            raise ValueError(f"model file changed: {path}")
    if sha_file(out / "prompts.jsonl") != manifest["prompts_sha256"]:
        raise ValueError("frozen prompts changed")
    return manifest


def concurrent_episodes(tasks, worker, batch_generate, workers=16, max_batch=16):
    """Only the calling thread touches vLLM. Episode workers wait on Futures."""
    requests = queue.Queue()
    abort = threading.Event()

    def generate(ids, remaining):
        if abort.is_set():
            raise RuntimeError("batch generation aborted")
        future = Future()
        requests.put((list(ids), remaining, future))
        while True:
            try:
                return future.result(timeout=0.1)
            except TimeoutError:
                if abort.is_set():
                    raise RuntimeError("batch generation aborted")

    pool = ThreadPoolExecutor(max_workers=workers)
    pending = {pool.submit(worker, task, generate) for task in tasks}
    try:
        while pending:
            completed = {f for f in pending if f.done()}
            pending -= completed
            for future in completed:
                yield future.result()
            if not pending:
                break
            try:
                batch = [requests.get(timeout=0.02)]
            except queue.Empty:
                continue
            until = time.monotonic() + 0.02
            while len(batch) < max_batch:
                try:
                    batch.append(requests.get(timeout=max(0.0001, until - time.monotonic())))
                except queue.Empty:
                    break
                if time.monotonic() >= until:
                    break
            try:
                outputs = batch_generate([(ids, remaining) for ids, remaining, _ in batch])
                if len(outputs) != len(batch):
                    raise RuntimeError("batch output count mismatch")
            except BaseException as exc:
                for _, _, future in batch:
                    future.set_exception(exc)
                raise
            for (_, _, future), ids in zip(batch, outputs):
                future.set_result(ids)
    finally:
        abort.set()
        # Unblock waiting workers even if inference, scoring or output writing fails.
        while not requests.empty():
            _, _, future = requests.get_nowait()
            future.set_exception(RuntimeError("evaluation stopped"))
        pool.shutdown(wait=True, cancel_futures=True)


def scored_record(row, state, arm, prompt_info):
    record = episode_to_record(state, row)
    sql = record["final_sql"]
    score = (run_sql(row["db_path"], sql, operation="score_diagnostic", timeout=30,
                     gold_sql=row["gold_sql"], score_mode="bird_set") if sql else
             {"ok": False, "error": "missing_answer", "error_type": "missing_answer"})
    record.update({"arm": arm, "diagnostic_only": True, "training_eligible": False,
                   "correct": bool(score.get("bird_set_only_correct")),
                   "correct_set": bool(score.get("bird_set_only_correct")),
                   "correct_strict": bool(score.get("project_strict_correct")),
                   "valid": bool(score.get("ok")), "missing": not bool(sql),
                   "infrastructure_error": bool(record["infrastructure_error"] or score.get("infrastructure_error")),
                   "score": score, "error_type": score.get("error_type"),
                   "prompt_sha256": prompt_info["prompt_sha256"],
                   "prompt_tokens": prompt_info["prompt_tokens"],
                   "response_tokens": sum(len(s["ids"]) for s in state["segments"] if s["kind"] == "action"),
                   "observation_tokens": sum(len(s["ids"]) for s in state["segments"] if s["kind"] == "observation"),
                   "config": asdict(episode_config(arm))})
    return record


def run(out):
    manifest = validate_manifest(out)
    rows = read_jsonl(manifest["data"])
    stores, repairs = stores_for(rows)
    if repairs != manifest["relationship_repairs"]:
        raise ValueError("relationship repair differs from preflight")
    prompts = {(p["id"], p["arm"]): p for p in read_jsonl(out / "prompts.jsonl")}
    records_path = out / "records.jsonl"
    prior = read_jsonl(records_path)
    done = {(r["id"], r["arm"]) for r in prior}
    if len(done) != len(prior) or not done <= set(prompts):
        raise ValueError("duplicate or unexpected completed episodes")
    if any(r["prompt_sha256"] != prompts[(r["id"], r["arm"])]["prompt_sha256"] for r in prior):
        raise ValueError("saved record/prompt mismatch")
    tasks = []
    for index, row in enumerate(rows):
        order = ARMS[index % 3:] + ARMS[:index % 3]
        tasks.extend((row, arm) for arm in order if (row["id"], arm) not in done)
    if not tasks:
        print("all 600 episodes already complete", flush=True)
        return
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tokenizer = AutoTokenizer.from_pretrained(manifest["model"], local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # The real tokenizer and the preflight fast tokenizer must agree exactly.
    for item in prompts.values():
        if len(tokenizer.encode(item["prompt"], add_special_tokens=False)) != item["prompt_tokens"]:
            raise ValueError("runtime/preflight tokenizer mismatch")
    started = time.monotonic()
    llm = LLM(model=manifest["model"], dtype="bfloat16", gpu_memory_utilization=0.55,
              enforce_eager=True, max_model_len=8192, max_num_seqs=16,
              max_num_batched_tokens=8192, trust_remote_code=False, seed=0)
    runtime = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "python": sys.version, "pid": os.getpid(), "n_pending": len(tasks),
               "devices": {key: os.environ.get(key) for key in (
                   "ASCEND_VISIBLE_DEVICES", "NPU_VISIBLE_DEVICES", "ASCEND_RT_VISIBLE_DEVICES")},
               "versions": {key: importlib.metadata.version(key) for key in (
                   "torch", "torch-npu", "transformers", "vllm", "vllm-ascend")}}
    write_json(out / f"runtime_{len(prior):03d}.json", runtime)
    schema_tools = {db: SchemaLookupTool(store, token_budget=1024,
                    count_tokens=lambda text: len(tokenizer.encode(text, add_special_tokens=False)),
                    enable_value_lookup=False, retrieval_version="E1.2") for db, store in stores.items()}

    def generate_batch(requests):
        outputs = llm.generate(prompts=[{"prompt_token_ids": ids} for ids, _ in requests],
            sampling_params=[SamplingParams(max_tokens=remaining, temperature=0.0, top_p=1.0)
                             for _, remaining in requests], use_tqdm=False)
        return [list(output.outputs[0].token_ids) for output in outputs]

    def worker(task, generate):
        row, arm = task
        prompt_info = prompts[(row["id"], arm)]
        state = run_episode(row, tokenizer, generate, schema_tools[row["db_id"]],
                            SQLExecuteTool(row["db_path"], timeout=10, preview_rows=3),
                            episode_config(arm), prompt=prompt_info["prompt"])
        record = scored_record(row, state, arm, prompt_info)
        if arm == "C" and record["tool_calls"]:
            raise AssertionError("one-shot arm executed a tool")
        return record

    counts = Counter(r["arm"] for r in prior)
    correct = Counter(r["arm"] for r in prior if r["correct_set"])
    with records_path.open("a") as stream:
        for record in concurrent_episodes(tasks, worker, generate_batch):
            stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            stream.flush()
            counts[record["arm"]] += 1
            correct[record["arm"]] += int(record["correct_set"])
            print(json.dumps({"completed": sum(counts.values()), "of": 600,
                  "arm": record["arm"], "id": record["id"], "set_correct": record["correct_set"],
                  "finish": record["finish_reason"], "counts": dict(counts),
                  "set_correct_counts": dict(correct),
                  "elapsed_s": round(time.monotonic() - started, 1)}), flush=True)
    validate_manifest(out)
    runtime.update({"completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "wall_seconds_including_engine_load": round(time.monotonic() - started, 3)})
    write_json(out / f"runtime_{len(prior):03d}.json", runtime)


def paired(left, right, field):
    if set(left) != set(right):
        raise ValueError("paired ID mismatch")
    a_wins = [qid for qid in left if left[qid][field] and not right[qid][field]]
    b_wins = [qid for qid in left if right[qid][field] and not left[qid][field]]
    by_db = defaultdict(list)
    for qid in left:
        if left[qid]["db_id"] != right[qid]["db_id"]:
            raise ValueError("paired DB mismatch")
        by_db[left[qid]["db_id"]].append(int(bool(right[qid][field])) - int(bool(left[qid][field])))
    return {"n": len(left), "A_wins": len(a_wins), "B_wins": len(b_wins),
            "net_B_minus_A": len(b_wins) - len(a_wins),
            "delta_B_minus_A": (len(b_wins) - len(a_wins)) / len(left),
            "mcnemar_exact_p": mcnemar(len(a_wins), len(b_wins)),
            "db_bootstrap_ci95_B_minus_A": bootstrap_ci(by_db),
            "A_win_ids": a_wins, "B_win_ids": b_wins}


def summary_data(records, ids):
    groups = {arm: {} for arm in ARMS}
    for record in records:
        arm, qid = record["arm"], record["id"]
        if arm not in groups or qid in groups[arm]:
            raise ValueError("duplicate or unknown arm record")
        groups[arm][qid] = record
    if any(set(group) != set(ids) for group in groups.values()):
        raise ValueError("incomplete ablation; all three arms need every frozen ID")
    metrics, by_db = {}, {}
    for arm, group in groups.items():
        rs = list(group.values()); n = len(rs)
        metrics[arm] = {"n": n, "strict": sum(r["correct_strict"] for r in rs),
                        "set_only": sum(r["correct_set"] for r in rs),
                        "valid": sum(r["valid"] for r in rs),
                        "missing": sum(not r.get("final_sql") for r in rs),
                        "infrastructure_errors": sum(r["infrastructure_error"] for r in rs),
                        "finish_reasons": dict(Counter(r["finish_reason"] for r in rs)),
                        "invalid_actions": sum(r["invalid_actions"] for r in rs)}
        for field in ("tool_calls", "schema_lookups", "sql_executes", "prompt_tokens",
                      "response_tokens", "observation_tokens", "latency_ms"):
            metrics[arm]["avg_" + field] = sum(r[field] for r in rs) / n
        metrics[arm]["set_ex"] = metrics[arm]["set_only"] / n
        metrics[arm]["strict_ex"] = metrics[arm]["strict"] / n
    for db in sorted({r["db_id"] for r in records}):
        by_db[db] = {arm: {"n": sum(r["db_id"] == db for r in group.values()),
                          "strict": sum(r["correct_strict"] for r in group.values() if r["db_id"] == db),
                          "set_only": sum(r["correct_set"] for r in group.values() if r["db_id"] == db)}
                     for arm, group in groups.items()}
    pairs = {f"{a}_vs_{b}": {metric: paired(groups[a], groups[b], field)
              for metric, field in (("strict", "correct_strict"), ("set_only", "correct_set"))}
             for a, b in itertools.combinations(ARMS, 2)}
    return {"complete": True, "n_questions": len(ids), "n_episodes": len(records),
            "arms": metrics, "paired": pairs, "per_db": by_db}


def summarize(out):
    manifest = validate_manifest(out)
    records = read_jsonl(out / "records.jsonl")
    summary = summary_data(records, manifest["question_ids"])
    summary.update({"manifest_sha256": sha_file(out / "manifest.json"),
                    "records_sha256": sha_file(out / "records.jsonl"),
                    "limits": manifest["limits"]})
    order = {qid: index for index, qid in enumerate(manifest["question_ids"])}
    for arm in ARMS:
        rs = sorted((r for r in records if r["arm"] == arm), key=lambda r: order[r["id"]])
        (out / f"{arm}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rs))
    paired_rows = []
    maps = {arm: {r["id"]: r for r in records if r["arm"] == arm} for arm in ARMS}
    for qid in manifest["question_ids"]:
        row = {"id": qid, "db_id": maps["A"][qid]["db_id"]}
        for arm in ARMS:
            row[arm] = {k: maps[arm][qid].get(k) for k in (
                "correct_strict", "correct_set", "valid", "missing", "final_sql",
                "finish_reason", "tool_calls", "invalid_actions", "infrastructure_error")}
        paired_rows.append(row)
    (out / "paired_cases.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in paired_rows))
    write_json(out / "summary.json", summary)
    lines = ["# 固定 3B step788：dev200 A/B/C 输入对照", "",
             "本报告记录实验配置、观测值、配对统计与限制；未训练、未调用教师。", "",
             "## 数值", "",
             "| arm | strict | 官方 set-only | valid | missing | infra | tools/题 | action tokens/题 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm, m in summary["arms"].items():
        lines.append(f"| {arm} | {m['strict']}/200 ({m['strict_ex']:.1%}) | "
                     f"{m['set_only']}/200 ({m['set_ex']:.1%}) | {m['valid']}/200 | "
                     f"{m['missing']} | {m['infrastructure_errors']} | {m['avg_tool_calls']:.3f} | "
                     f"{m['avg_response_tokens']:.1f} |")
    lines += ["", "## 实验定义", "", f"- 模型：`{manifest['model']}`，三臂共用一次加载、固定权重。",
              "- A：表名 catalog＋question/evidence＋三动作 instruction，可检索、执行、提交。",
              "- B：与 A 相同的 instruction、catalog 和工具，首次生成前额外提供全部表列、描述和关系。",
              "- C：与 B 相同的 schema 数据，answer-only instruction，单次生成，无工具。",
              "- A/B：5 rounds，4 tools，lookup≤2，execute≤2；C：1 round，0 tools。",
              "- 三臂：greedy，action≤512，context=8192，严格 parse_action，无 chat template，无 action stop。",
              "- 共用：E1.2、compact_v2、真实 SQLite FK 端点解析、工具超时10s、评分超时30s。",
              "- 字段/value 描述最多200/160字符；全部表列均保留，无 Gold 选列，无初始 schema 截断。",
              "- official set-only 为主指标，strict 为补充；同一 final SQL 在一次 worker 执行中计算两者。",
              "- 完整实际 prompt 见 prompts.jsonl；逐动作/观察/token 见 records.jsonl 和 A/B/C.jsonl。",
              "", "## 逐题配对", "",
              "下表差值为右臂减左臂，p 为双侧 McNemar exact；CI 为按 DB 聚类 bootstrap。", "",
              "| 左 | 右 | metric | 左 wins | 右 wins | net 右−左 | p | DB CI 95% (pp) |",
              "|---|---|---|---:|---:|---:|---:|---|"]
    for name, pair in summary["paired"].items():
        a, b = name.split("_vs_")
        for metric, p in pair.items():
            ci = [round(v * 100, 3) for v in p["db_bootstrap_ci95_B_minus_A"]]
            lines.append(f"| {a} | {b} | {metric} | {p['A_wins']} | {p['B_wins']} | "
                         f"{p['net_B_minus_A']:+} | {p['mcnemar_exact_p']:.6g} | {ci} |")
    lines += ["", "## 分 DB 正确数（strict / set-only）", "",
              "| DB | n | A | B | C |", "|---|---:|---:|---:|---:|"]
    for db, arms in summary["per_db"].items():
        values = [f"{arms[a]['strict']} / {arms[a]['set_only']}" for a in ARMS]
        lines.append(f"| {db} | {arms['A']['n']} | " + " | ".join(values) + " |")
    lines += ["", "## 格式与结束原因", ""]
    for arm, m in summary["arms"].items():
        lines.append(f"- {arm}：invalid actions={m['invalid_actions']}；finish={m['finish_reasons']}。")
    lines += ["", "## 限制", ""] + ["- " + item for item in manifest["limits"]]
    lines += ["", "## 文件", "", "- manifest.json：模型/数据/代码/catalog 哈希、预算、FK 修复。",
              "- prompts.jsonl：600条实际初始输入，不含 Gold SQL。",
              "- records.jsonl、A.jsonl、B.jsonl、C.jsonl：完整轨迹及两种评分。",
              "- paired_cases.jsonl：200题逐题 A/B/C SQL、正确性与结束原因。",
              "- summary.json：完整数值与配对统计。", ""]
    (out / "RESULTS.md").write_text("\n".join(lines))
    print(json.dumps({"complete": True, "arms": summary["arms"],
                      "report": str(out / "RESULTS.md")}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "run", "summarize"])
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    out = (ROOT / args.out).resolve()
    if args.phase == "prepare":
        prepare(out, (ROOT / args.model).resolve())
    elif args.phase == "run":
        run(out)
    else:
        summarize(out)


if __name__ == "__main__":
    main()
