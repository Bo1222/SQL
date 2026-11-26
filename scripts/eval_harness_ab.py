"""Fixed-step788 dev200 harness ablation: E1.1/E1.2 x compact/compact_v2.

All four arms use the same A-protocol prompt (table catalog + question/evidence +
three-action instruction), the same model weights, the same tools and budgets,
greedy decoding, and the same real SQLite scoring. They differ only in the
retrieval backend used by schema_lookup and in the model-visible observation
encoder. Both retrieval arms use the same in-memory repaired FK endpoints so the
comparison isolates the retrieval version rather than the FK fix.

This is a diagnostic-only experiment: no training, no teacher calls, no writes
to training data or benchmark labels.
"""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.sql_episode import episode_to_record, run_episode
from eval.harness_audit import AuditTokenizer
from eval.input_ablation import episode_config, input_prompt, resolve_implicit_foreign_keys
from eval.paired_statistics import bootstrap_ci, mcnemar
from retrieval.catalog_store import CatalogStore
from scripts.eval_3b_dev200_abc import (concurrent_episodes, read_jsonl, sha_file,
                                        sha_text, write_json)
from tools.schema_lookup import SchemaLookupTool
from tools.sql_execute import SQLExecuteTool, run_sql

DEFAULT_MODEL = "outputs/m6_08m/student_sft_full_lowlr_5e6/epoch_2/hf_model"
DEFAULT_OUT = "reports/m6/harness_audit_20261117/3b_dev200_harness_ab"
DATA = "artifacts/datasets/dev200.jsonl"
EXPECTED_DATA_SHA256 = "bda04835e9404f872aafa892bc179ff80a9db1a11a946fc3e9c198b44c57a2ee"
VERSION = "3b-harness-ab-v1"

ARMS = {
    "E11_compact": ("E1.1", "compact"),
    "E11_compact_v2": ("E1.1", "compact_v2"),
    "E12_compact": ("E1.2", "compact"),
    "E12_compact_v2": ("E1.2", "compact_v2"),
}
ARM_ORDER = list(ARMS)
CODE_FILES = ["scripts/eval_harness_ab.py", "scripts/eval_3b_dev200_abc.py",
              "eval/input_ablation.py", "eval/harness_audit.py",
              "eval/paired_statistics.py", "agent/protocol.py", "agent/sql_episode.py",
              "retrieval/catalog_store.py", "retrieval/e1_1.py", "retrieval/e1_2.py",
              "retrieval/retriever.py", "retrieval/observation_formatter.py",
              "retrieval/schema_extractor.py", "tools/schema_lookup.py",
              "tools/sql_execute.py", "tools/sql_worker.py"]


def harness_config(arm):
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    retrieval_version, observation_mode = ARMS[arm]
    return replace(episode_config("A"), observation_mode=observation_mode)


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
        # Every arm uses the identical A-protocol prompt; only the retrieval
        # backend and observation encoder differ at tool-observation time.
        prompt = input_prompt(row, stores[row["db_id"]], "A")
        tokens = len(tokenizer.encode(prompt))
        if tokens + episode_config("A").max_action_tokens > 8192:
            raise ValueError(f"prompt does not fit; no silent truncation: {row['id']}")
        for arm in ARM_ORDER:
            prompts.append({"id": row["id"], "db_id": row["db_id"], "arm": arm,
                            "retrieval_version": ARMS[arm][0],
                            "observation_mode": ARMS[arm][1],
                            "prompt": prompt, "prompt_sha256": sha_text(prompt),
                            "prompt_tokens": tokens})
    paths = [ROOT / f for f in CODE_FILES]
    paths += [ROOT / DATA]
    for db in stores:
        paths += sorted((ROOT / "artifacts/catalog" / db).glob("*.json"))
    model_files = sorted(p for p in model.iterdir() if p.is_file())
    paths += model_files
    inputs = {str(p.resolve()): sha_file(p) for p in paths}
    inventory = {str(p.resolve()): {"size": p.stat().st_size,
                                    "mtime_ns": p.stat().st_mtime_ns}
                 for p in model_files}
    out.mkdir(parents=True)
    prompts_path = out / "prompts.jsonl"
    prompts_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in prompts))
    manifest = {
        "version": VERSION, "diagnostic_only": True, "training_eligible": False,
        "model": str(model.resolve()), "data": str((ROOT / DATA).resolve()),
        "data_sha256": EXPECTED_DATA_SHA256, "n_questions": 200, "n_episodes": 800,
        "question_ids": [r["id"] for r in rows],
        "arms": {arm: {"retrieval_version": ARMS[arm][0],
                       "observation_mode": ARMS[arm][1],
                       "config": asdict(harness_config(arm))} for arm in ARM_ORDER},
        "definitions": {
            "common": "A protocol: table catalog + question/evidence + three-action instruction; "
                      "schema_lookup/sql_execute/answer; 5 rounds, 4 tools, lookup<=2, execute<=2; "
                      "action<=512; observation<=1024; context=8192; greedy; strict parse.",
            "difference": "Only schema_lookup retrieval backend (E1.1 vs E1.2) and the "
                          "model-visible observation encoder (compact vs compact_v2) differ."},
        "generation": {"temperature": 0.0, "top_p": 1.0, "seed": 0,
                       "max_action_tokens": 512, "chat_template": False,
                       "stop_after_action": False, "max_model_len": 8192,
                       "max_batch_size": 16, "concurrent_episodes": 16,
                       "gpu_memory_utilization": 0.55},
        "primary_metric": "bird_set_only", "secondary_metric": "project_strict",
        "score_timeout": 30, "tool_timeout": 10,
        "fk_policy": "Both arms use the same repaired in-memory FK endpoints; this is not "
                     "the historical E1.1/legacy-FK configuration.",
        "relationship_repairs": repairs, "input_sha256": inputs,
        "model_inventory": inventory, "prompts_sha256": sha_file(prompts_path),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "limits": [
            "dev200 is a repeatedly used selection set, not an independent test.",
            "This experiment measures model EX under four harness variants; it does not "
            "establish an oracle upper bound or a causal explanation of all errors.",
            "Both arms use repaired FK endpoints; the historical step788 E1.1 run used "
            "legacy FK endpoints, so the E1.1 arm here is a new E1.1/compact run, not an "
            "exact reproduction of the historical 58.0%/60.5% numbers.",
            "The A protocol prompt is identical across arms; only tool observations differ.",
            "200 questions / 7 DBs gives limited power; p-values are uncorrected and the "
            "DB-clustered bootstrap interval is descriptive.",
            "No training, no teacher calls, no benchmark label changes.",
        ],
    }
    write_json(out / "manifest.json", manifest)
    print(json.dumps({"prepared": str(out), "episodes": 800,
                      "prompt_tokens": {"min": min(r["prompt_tokens"] for r in prompts),
                                        "max": max(r["prompt_tokens"] for r in prompts)},
                      "fk_repairs": sum(len(r["fixes"]) for r in repairs.values()),
                      "arms": ARM_ORDER}), flush=True)


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


def scored_record(row, state, arm, prompt_info):
    record = episode_to_record(state, row)
    sql = record["final_sql"]
    score = (run_sql(row["db_path"], sql, operation="score_diagnostic", timeout=30,
                     gold_sql=row["gold_sql"], score_mode="bird_set") if sql else
             {"ok": False, "error": "missing_answer", "error_type": "missing_answer"})
    record.update({
        "arm": arm, "diagnostic_only": True, "training_eligible": False,
        "retrieval_version": ARMS[arm][0], "observation_mode": ARMS[arm][1],
        "correct": bool(score.get("bird_set_only_correct")),
        "correct_set": bool(score.get("bird_set_only_correct")),
        "correct_strict": bool(score.get("project_strict_correct")),
        "valid": bool(score.get("ok")), "missing": not bool(sql),
        "infrastructure_error": bool(record["infrastructure_error"]
                                     or score.get("infrastructure_error")),
        "score": score, "error_type": score.get("error_type"),
        "prompt_sha256": prompt_info["prompt_sha256"],
        "prompt_tokens": prompt_info["prompt_tokens"],
        "response_tokens": sum(len(s["ids"]) for s in state["segments"]
                               if s["kind"] == "action"),
        "observation_tokens": sum(len(s["ids"]) for s in state["segments"]
                                  if s["kind"] == "observation"),
        "config": asdict(harness_config(arm)),
    })
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
        offset = index % len(ARM_ORDER)
        order = ARM_ORDER[offset:] + ARM_ORDER[:offset]
        tasks.extend((row, arm) for arm in order if (row["id"], arm) not in done)
    if not tasks:
        print("all 800 episodes already complete", flush=True)
        return
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tokenizer = AutoTokenizer.from_pretrained(manifest["model"], local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
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

    count = len(tokenizer.encode("x", add_special_tokens=False))  # keep tokenizer loaded
    del count
    schema_tools = {
        (db, arm): SchemaLookupTool(store, token_budget=1024,
                                    count_tokens=lambda text: len(tokenizer.encode(
                                        text, add_special_tokens=False)),
                                    enable_value_lookup=False,
                                    retrieval_version=ARMS[arm][0])
        for db, store in stores.items() for arm in ARM_ORDER
    }

    def generate_batch(requests):
        outputs = llm.generate(
            prompts=[{"prompt_token_ids": ids} for ids, _ in requests],
            sampling_params=[SamplingParams(max_tokens=remaining, temperature=0.0, top_p=1.0)
                             for _, remaining in requests], use_tqdm=False)
        return [list(output.outputs[0].token_ids) for output in outputs]

    def worker(task, generate):
        row, arm = task
        prompt_info = prompts[(row["id"], arm)]
        state = run_episode(row, tokenizer, generate,
                            schema_tools[(row["db_id"], arm)],
                            SQLExecuteTool(row["db_path"], timeout=10, preview_rows=3),
                            harness_config(arm), prompt=prompt_info["prompt"])
        return scored_record(row, state, arm, prompt_info)

    counts = Counter(r["arm"] for r in prior)
    correct = Counter(r["arm"] for r in prior if r["correct_set"])
    with records_path.open("a") as stream:
        for record in concurrent_episodes(tasks, worker, generate_batch):
            stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            stream.flush()
            counts[record["arm"]] += 1
            correct[record["arm"]] += int(record["correct_set"])
            print(json.dumps({"completed": sum(counts.values()), "of": 800,
                              "arm": record["arm"], "id": record["id"],
                              "set_correct": record["correct_set"],
                              "finish": record["finish_reason"],
                              "counts": dict(counts),
                              "set_correct_counts": dict(correct),
                              "elapsed_s": round(time.monotonic() - started, 1)}),
                  flush=True)
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
        by_db[left[qid]["db_id"]].append(
            int(bool(right[qid][field])) - int(bool(left[qid][field])))
    return {"n": len(left), "A_wins": len(a_wins), "B_wins": len(b_wins),
            "net_B_minus_A": len(b_wins) - len(a_wins),
            "delta_B_minus_A": (len(b_wins) - len(a_wins)) / len(left),
            "mcnemar_exact_p": mcnemar(len(a_wins), len(b_wins)),
            "db_bootstrap_ci95_B_minus_A": bootstrap_ci(by_db),
            "A_win_ids": a_wins, "B_win_ids": b_wins}


def summary_data(records, ids):
    groups = {arm: {} for arm in ARM_ORDER}
    for record in records:
        arm, qid = record["arm"], record["id"]
        if arm not in groups or qid in groups[arm]:
            raise ValueError("duplicate or unknown arm record")
        groups[arm][qid] = record
    if any(set(group) != set(ids) for group in groups.values()):
        raise ValueError("incomplete ablation; all four arms need every frozen ID")
    metrics, by_db = {}, {}
    for arm, group in groups.items():
        rs = list(group.values())
        n = len(rs)
        metrics[arm] = {
            "retrieval_version": ARMS[arm][0], "observation_mode": ARMS[arm][1],
            "n": n, "strict": sum(r["correct_strict"] for r in rs),
            "set_only": sum(r["correct_set"] for r in rs),
            "valid": sum(r["valid"] for r in rs),
            "missing": sum(not r.get("final_sql") for r in rs),
            "infrastructure_errors": sum(r["infrastructure_error"] for r in rs),
            "finish_reasons": dict(Counter(r["finish_reason"] for r in rs)),
            "invalid_actions": sum(r["invalid_actions"] for r in rs),
            "error_types": dict(Counter(r.get("error_type") for r in rs)),
        }
        for field in ("tool_calls", "schema_lookups", "sql_executes", "prompt_tokens",
                      "response_tokens", "observation_tokens", "latency_ms"):
            metrics[arm]["avg_" + field] = sum(r[field] for r in rs) / n
        metrics[arm]["set_ex"] = metrics[arm]["set_only"] / n
        metrics[arm]["strict_ex"] = metrics[arm]["strict"] / n
    for db in sorted({r["db_id"] for r in records}):
        by_db[db] = {
            arm: {"n": sum(r["db_id"] == db for r in group.values()),
                  "strict": sum(r["correct_strict"] for r in group.values()
                                if r["db_id"] == db),
                  "set_only": sum(r["correct_set"] for r in group.values()
                                  if r["db_id"] == db)}
            for arm, group in groups.items()
        }
    pairs = {f"{a}_vs_{b}": {metric: paired(groups[a], groups[b], field)
                             for metric, field in (("strict", "correct_strict"),
                                                   ("set_only", "correct_set"))}
             for a, b in itertools.combinations(ARM_ORDER, 2)}
    # Focused 2x2 effects: right minus left on set-only accuracy.
    effects = {
        "retrieval_effect_at_compact": pairs["E11_compact_vs_E12_compact"]["set_only"],
        "retrieval_effect_at_compact_v2": pairs["E11_compact_v2_vs_E12_compact_v2"]["set_only"],
        "observation_effect_at_E11": pairs["E11_compact_vs_E11_compact_v2"]["set_only"],
        "observation_effect_at_E12": pairs["E12_compact_vs_E12_compact_v2"]["set_only"],
        "new_vs_old_deployed": pairs["E11_compact_vs_E12_compact_v2"]["set_only"],
    }
    return {"complete": True, "n_questions": len(ids), "n_episodes": len(records),
            "arms": metrics, "paired": pairs, "effects": effects, "per_db": by_db}


def summarize(out):
    manifest = validate_manifest(out)
    records = read_jsonl(out / "records.jsonl")
    summary = summary_data(records, manifest["question_ids"])
    summary.update({"manifest_sha256": sha_file(out / "manifest.json"),
                    "records_sha256": sha_file(out / "records.jsonl"),
                    "limits": manifest["limits"]})
    order = {qid: index for index, qid in enumerate(manifest["question_ids"])}
    maps = {arm: {} for arm in ARM_ORDER}
    for record in records:
        maps[record["arm"]][record["id"]] = record
    for arm in ARM_ORDER:
        rs = sorted(maps[arm].values(), key=lambda r: order[r["id"]])
        (out / f"{arm}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rs))
    paired_rows = []
    for qid in manifest["question_ids"]:
        row = {"id": qid, "db_id": maps[ARM_ORDER[0]][qid]["db_id"]}
        for arm in ARM_ORDER:
            row[arm] = {k: maps[arm][qid].get(k) for k in (
                "correct_strict", "correct_set", "valid", "missing", "final_sql",
                "finish_reason", "tool_calls", "schema_lookups", "sql_executes",
                "invalid_actions", "infrastructure_error", "error_type")}
        paired_rows.append(row)
    (out / "paired_cases.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in paired_rows))
    write_json(out / "summary.json", summary)

    lines = [
        "# 固定 3B step788：dev200 harness 2×2 消融（E1.1/E1.2 × compact/compact_v2）", "",
        "本报告只记录实验定义、逐题观测值、配对统计和限制；未训练、未调用教师、未修改数据或 Gold。", "",
        "## 数值摘要", "",
        "| arm | retrieval | observation | strict | 官方 set-only | valid | missing | infra | tools/题 | action tok/题 | obs tok/题 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, m in summary["arms"].items():
        lines.append(
            f"| {arm} | {m['retrieval_version']} | {m['observation_mode']} | "
            f"{m['strict']}/200 ({m['strict_ex']:.1%}) | "
            f"{m['set_only']}/200 ({m['set_ex']:.1%}) | {m['valid']}/200 | "
            f"{m['missing']} | {m['infrastructure_errors']} | "
            f"{m['avg_tool_calls']:.3f} | {m['avg_response_tokens']:.1f} | "
            f"{m['avg_observation_tokens']:.1f} |")
    lines += [
        "", "## 实验定义", "",
        f"- 模型：`{manifest['model']}`，四臂共用一次加载、固定权重。",
        "- 四臂共用同一 A 协议 prompt：表名 catalog + question/evidence + 三动作 instruction；"
        "schema_lookup / sql_execute / answer；5 rounds、4 tools、lookup≤2、execute≤2；"
        "action≤512、observation≤1024、context=8192、greedy、严格 parse。",
        "- 唯一差异：schema_lookup 的检索后端（E1.1 vs E1.2）与模型可见的 observation 编码器"
        "（compact vs compact_v2）。",
        "- 两臂均使用同一套内存中修复后的 FK 端点；这不是历史 E1.1/legacy-FK 配置，"
        "因此 E1.1 臂是新的 E1.1/compact 运行，不等同于历史 58.0%/60.5% 的逐位复现。",
        "- official set-only 为主指标，strict 为补充；同一 final SQL 在一次 worker 执行中计算两者。",
        "- 无训练、无教师调用、无 Gold 进入 prompt 或 observation。",
        "", "## 主对照（配对，set-only 为主）", "",
        "下表差值为右臂减左臂；p 为双侧 McNemar exact；CI 为按 DB 聚类 bootstrap。", "",
        "| 左 | 右 | 指标 | 左 wins | 右 wins | net 右−左 | p | DB CI 95% (pp) |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    focus = [
        ("E11_compact", "E12_compact_v2", "部署新 vs 旧"),
        ("E11_compact", "E12_compact", "仅换检索 E1.1→E1.2"),
        ("E11_compact_v2", "E12_compact_v2", "仅换检索 E1.1→E1.2（v2 观测）"),
        ("E11_compact", "E11_compact_v2", "仅换观测 compact→compact_v2"),
        ("E12_compact", "E12_compact_v2", "仅换观测 compact→compact_v2（E1.2）"),
    ]
    seen = set()
    for left, right, note in focus:
        seen.add((left, right))
        for metric in ("set_only", "strict"):
            p = summary["paired"][f"{left}_vs_{right}"][metric]
            ci = [round(v * 100, 3) for v in p["db_bootstrap_ci95_B_minus_A"]]
            lines.append(f"| {left} | {right} | {metric} | {p['A_wins']} | "
                         f"{p['B_wins']} | {p['net_B_minus_A']:+} | "
                         f"{p['mcnemar_exact_p']:.6g} | {ci} |")
    lines += ["", "说明：表中主对照已按上表顺序列出；其余两两组合见 `summary.json` 的 `paired`。", "",
              "## 2×2 主效应（set-only，右−左）", "",
              "| 效应 | net | Δ pp | p | DB CI 95% (pp) |", "|---|---:|---:|---:|---|"]
    for name, p in summary["effects"].items():
        ci = [round(v * 100, 3) for v in p["db_bootstrap_ci95_B_minus_A"]]
        lines.append(f"| {name} | {p['net_B_minus_A']:+} | "
                     f"{p['delta_B_minus_A'] * 100:+.2f} | "
                     f"{p['mcnemar_exact_p']:.6g} | {ci} |")
    lines += ["", "## 分 DB 正确数（strict / set-only）", "",
              "| DB | n | " + " | ".join(ARM_ORDER) + " |",
              "|---|---:|" + "---:|" * len(ARM_ORDER)]
    for db, arms in summary["per_db"].items():
        values = [f"{arms[a]['strict']} / {arms[a]['set_only']}" for a in ARM_ORDER]
        lines.append(f"| {db} | {arms[ARM_ORDER[0]]['n']} | " + " | ".join(values) + " |")
    lines += ["", "## 行为与格式", ""]
    for arm, m in summary["arms"].items():
        lines.append(f"- {arm}：tools/题={m['avg_tool_calls']:.3f}，"
                     f"lookup/题={m['avg_schema_lookups']:.3f}，"
                     f"execute/题={m['avg_sql_executes']:.3f}，"
                     f"invalid actions={m['invalid_actions']}，finish={m['finish_reasons']}。")
    lines += ["", "## 限制", ""] + ["- " + item for item in manifest["limits"]]
    lines += ["", "## 文件", "",
              "- manifest.json：模型/数据/代码/catalog 哈希、四臂配置、FK 修复。",
              "- prompts.jsonl：800 条实际初始输入（每臂同一 prompt）。",
              "- records.jsonl / E11_compact.jsonl 等：完整轨迹及两种评分。",
              "- paired_cases.jsonl：200 题逐题四臂 SQL、正确性、工具数与结束原因。",
              "- summary.json：完整数值、两两配对与 2×2 主效应。", ""]
    (out / "RESULTS.md").write_text("\n".join(lines))
    print(json.dumps({"complete": True, "arms": summary["arms"],
                      "effects": {k: {kk: v[kk] for kk in ("net_B_minus_A",
                                                           "delta_B_minus_A",
                                                           "mcnemar_exact_p",
                                                           "db_bootstrap_ci95_B_minus_A")}
                                  for k, v in summary["effects"].items()},
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
