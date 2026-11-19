"""Rebuild the original 3B SFT trajectories under the E1 query/compact contract.

Input: artifacts/datasets/sft_trajectories.jsonl (4,471 old table-prompt trajectories)
For each trajectory:
  - extract question/evidence from the old prompt suffix;
  - build the new E1 query prompt (catalog-three-action-query-v2);
  - replay the original action sequence against the real E1 schema/SQLite tools;
  - encode the new compact observation;
  - keep only final_ex=1 trajectories;
  - preserve the old `supervise` flags, but mask any action whose replayed tool
    result is an error;
  - write a new trajectory record with input_ids/loss_mask and segment texts.

This is a data rebuild, not training. It does not touch the old data.
"""
import argparse
import hashlib
import json
import re
import time
from pathlib import Path

from agent.protocol import parse_action
from agent.sql_episode import encode_observation
from e1_alignment.runtime import Runtime, RuntimeContract


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def extract_old_prompt(text):
    """Return (question, evidence, old_catalog) or None."""
    marker = "### Table catalog\n"
    if marker not in text:
        return None
    rest = text.split(marker, 1)[1]
    if "\n### Question\n" not in rest:
        return None
    catalog, rest = rest.split("\n### Question\n", 1)
    if "\n### Evidence\n" not in rest:
        return None
    question, rest = rest.split("\n### Evidence\n", 1)
    if "\n### Trajectory\n" in rest:
        evidence = rest.split("\n### Trajectory\n", 1)[0]
    else:
        evidence = rest
    return question.strip(), evidence.strip(), catalog.strip()


def canonical_action_text(name, arguments):
    if name == "schema_lookup":
        body = json.dumps(arguments, ensure_ascii=False)
    else:
        body = arguments
    return f"<{name}>{body}</{name}>"


def lookup_has_tables(result):
    """E1 SchemaLookupTool returns retrieval_metadata.matched_tables, not a `tables` list."""
    if not isinstance(result, dict):
        return False
    md = result.get("retrieval_metadata") or {}
    if isinstance(md, dict) and md.get("matched_tables"):
        return True
    if result.get("tables"):
        return True
    return "Tables:" in str(result.get("observation") or "")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="artifacts/datasets/sft_trajectories.jsonl")
    p.add_argument("--out", default="artifacts/m6/e1_contract_rebuild/sft_trajectories_e1.jsonl")
    p.add_argument("--manifest", default="artifacts/m6/e1_contract_rebuild/manifest.json")
    p.add_argument("--tokenizer", default="/home/ma-user/work/env_verify/models/Qwen2.5-3B-Instruct")
    p.add_argument("--runtime-fingerprint", default="reports/m6/e1_alignment/runtime_fingerprint.json")
    p.add_argument("--dev200", default="artifacts/datasets/dev200.jsonl")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--max-context", type=int, default=6144)
    p.add_argument("--observation-tokens", type=int, default=1024)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--sample-out", default="")
    args = p.parse_args()

    for path in (args.out, args.manifest):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.sample_out or args.out)
    if args.sample_out:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    fingerprint = None
    if Path(args.runtime_fingerprint).is_file():
        fingerprint = json.loads(Path(args.runtime_fingerprint).read_text(encoding="utf-8")).get("fingerprint_sha256")
    forbidden = set()
    if Path(args.dev200).is_file():
        for row in read_jsonl(args.dev200):
            if row.get("id"):
                forbidden.add(row["id"])

    contract = RuntimeContract(model_path=args.tokenizer, max_context=args.max_context,
                               max_observation_tokens=args.observation_tokens,
                               prompt_mode="query", observation_mode="compact")
    runtime = Runtime(contract)
    tok = runtime.tokenizer
    print(json.dumps({"tokenizer": args.tokenizer, "eos": tok.eos_token_id,
                      "prompt_mode": contract.prompt_mode,
                      "observation_mode": contract.observation_mode,
                      "max_context": contract.max_context,
                      "dev200_forbidden": len(forbidden)}), flush=True)

    done = set()
    if args.resume and Path(out_path).is_file():
        for row in read_jsonl(out_path):
            done.add(row.get("original_trajectory_id"))
        print(json.dumps({"resume_done": len(done)}), flush=True)

    rows = list(read_jsonl(args.data))
    if args.start:
        rows = rows[args.start:]
    if args.limit:
        rows = rows[:args.limit]

    counts = {"seen": 0, "kept": 0, "skipped": 0, "actions": 0, "tool_calls": 0,
              "supervised_actions": 0, "lookup_empty": 0, "lookup_error": 0,
              "sql_error": 0, "over_context": 0, "dev_overlap": 0, "bad_prompt": 0,
              "no_answer": 0, "final_ex_false": 0, "action_parse_fail": 0,
              "metadata_error": 0}
    skip_reasons = {}
    started = time.monotonic()
    mode = "a" if args.resume and out_path.is_file() else "w"
    with out_path.open(mode, encoding="utf-8") as fout:
        for old in rows:
            counts["seen"] += 1
            old_id = old.get("trajectory_id")
            if old_id in done:
                continue
            qid = old.get("question_id")
            if qid in forbidden:
                counts["dev_overlap"] += 1
                skip_reasons["dev200_overlap"] = skip_reasons.get("dev200_overlap", 0) + 1
                continue
            if old.get("final_ex") is not True:
                counts["final_ex_false"] += 1
                skip_reasons["final_ex_false"] = skip_reasons.get("final_ex_false", 0) + 1
                continue
            segs = (old.get("trajectory") or {}).get("segments") or []
            prompt_text = next((s.get("text") for s in segs if s.get("kind") == "prompt"), None)
            if not prompt_text:
                counts["bad_prompt"] += 1
                skip_reasons["no_prompt"] = skip_reasons.get("no_prompt", 0) + 1
                continue
            parsed = extract_old_prompt(prompt_text)
            if not parsed:
                counts["bad_prompt"] += 1
                skip_reasons["prompt_markers"] = skip_reasons.get("prompt_markers", 0) + 1
                continue
            question, evidence, _old_catalog = parsed
            row = {"id": qid, "question": question, "evidence": evidence,
                   "db_id": old.get("db_id"), "db_path": old.get("db_path")}
            try:
                new_prompt = runtime.prompt_for_row(row)
            except Exception as exc:
                counts["metadata_error"] += 1
                skip_reasons["metadata_error"] = skip_reasons.get("metadata_error", 0) + 1
                continue
            prompt_ids = tok.encode(new_prompt, add_special_tokens=False)

            new_segments = [{"kind": "prompt", "ids": prompt_ids, "text": new_prompt}]
            tool_events = []
            actions = []
            final_sql = None
            supervised = 0
            lookup_empty = 0
            lookup_error = 0
            sql_error = 0
            ok = True
            for seg in segs:
                if seg.get("kind") != "action":
                    continue
                text = (seg.get("text") or "").strip()
                try:
                    action = parse_action(text)
                except Exception:
                    counts["action_parse_fail"] += 1
                    skip_reasons["action_parse_fail"] = skip_reasons.get("action_parse_fail", 0) + 1
                    ok = False
                    break
                name, arguments = action.name, action.arguments
                action_text = canonical_action_text(name, arguments)
                action_ids = tok.encode(action_text, add_special_tokens=False)
                old_sup = seg.get("supervise") is True
                result = None
                if name == "schema_lookup":
                    try:
                        result = runtime.schema_tool(old["db_id"])(arguments)
                    except Exception:
                        counts["metadata_error"] += 1
                        skip_reasons["metadata_error"] = skip_reasons.get("metadata_error", 0) + 1
                        ok = False
                        break
                    status = str((result or {}).get("status"))
                    if status != "ok":
                        lookup_error += 1
                    elif not lookup_has_tables(result):
                        lookup_empty += 1
                elif name == "sql_execute":
                    try:
                        result = runtime.sql_tool(old["db_path"])(arguments)
                    except Exception:
                        result = {"status": "error", "error_type": "tool_exception"}
                    if str((result or {}).get("status")) != "ok":
                        sql_error += 1
                elif name == "answer":
                    final_sql = arguments
                else:
                    ok = False
                    break
                sup = old_sup and (result is None or str((result or {}).get("status")) == "ok")
                new_segments.append({"kind": "action", "ids": action_ids, "text": action_text,
                                     "action": name, "arguments": arguments,
                                     "supervise": sup, "tool_event_index": len(tool_events)})
                actions.append(name)
                counts["actions"] += 1
                if sup:
                    supervised += 1
                if result is not None:
                    obs_ids, obs_text = encode_observation(
                        tok, result, contract.max_observation_tokens, mode=contract.observation_mode)
                    new_segments.append({"kind": "observation", "ids": obs_ids, "text": obs_text,
                                         "tool_event_index": len(tool_events)})
                    tool_events.append({"action": name, "arguments": arguments, "result": result})
                    counts["tool_calls"] += 1
                if name == "answer":
                    break
            if not ok:
                counts["skipped"] += 1
                continue
            if not final_sql:
                counts["no_answer"] += 1
                skip_reasons["no_answer"] = skip_reasons.get("no_answer", 0) + 1
                continue
            input_ids = [i for s in new_segments for i in s["ids"]]
            loss_mask = []
            for s in new_segments:
                supervised_seg = s.get("kind") == "action" and s.get("supervise") is True
                loss_mask.extend([1 if supervised_seg else 0] * len(s["ids"]))
            if len(input_ids) > args.max_context:
                counts["over_context"] += 1
                skip_reasons["over_context"] = skip_reasons.get("over_context", 0) + 1
                continue
            new_id = "e1rebuild_" + str(old_id)
            record = {
                "id": new_id,
                "original_trajectory_id": old_id,
                "question_id": qid,
                "db_id": old.get("db_id"),
                "db_path": old.get("db_path"),
                "source_type": old.get("source_type"),
                "error_types": old.get("error_types"),
                "final_sql": final_sql,
                "final_ex": True,
                "supervision_policy": "e1-contract-rebuild-v1",
                "e1_prompt_version": "catalog-three-action-query-v2",
                "e1_observation_mode": "compact",
                "runtime_fingerprint_sha256": fingerprint,
                "tokenizer": args.tokenizer,
                "trajectory": {
                    "segments": new_segments,
                    "input_ids": input_ids,
                    "loss_mask": loss_mask,
                    "tool_calls": len(tool_events),
                    "actions": actions,
                    "tool_events": tool_events,
                    "stop_reason": "submitted",
                    "final_sql": final_sql,
                    "final_ex": True,
                },
                "token_count": len(input_ids),
                "masked_tokens": sum(loss_mask),
                "rebuild": {
                    "old_prompt_chars": len(prompt_text),
                    "new_prompt_chars": len(new_prompt),
                    "old_prompt_sha256": sha256_text(prompt_text),
                    "new_prompt_sha256": sha256_text(new_prompt),
                    "action_count": len(actions),
                    "supervised_actions": supervised,
                    "lookup_empty": lookup_empty,
                    "lookup_error": lookup_error,
                    "sql_error": sql_error,
                },
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()
            counts["kept"] += 1
            counts["supervised_actions"] += supervised
            counts["lookup_empty"] += lookup_empty
            counts["lookup_error"] += lookup_error
            counts["sql_error"] += sql_error
            if counts["seen"] % 50 == 0:
                print(json.dumps({"seen": counts["seen"], "kept": counts["kept"],
                                  "elapsed_s": round(time.monotonic() - started, 1)}), flush=True)

    summary = {"stage": "M6-08m-rebuild", "data": args.data, "out": str(out_path),
               "tokenizer": args.tokenizer, "max_context": args.max_context,
               "observation_tokens": args.observation_tokens,
               "runtime_fingerprint_sha256": fingerprint,
               "counts": counts, "skip_reasons": skip_reasons,
               "elapsed_s": round(time.monotonic() - started, 1)}
    Path(args.manifest).write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
