"""CPU-only diagnostics for SQL harness reachability and visible information.

Gold is used ONLY by this offline diagnostic. Neither a successful gold
injection nor a finite search over gold-guided lookup queries measures model
accuracy, and a failed lookup search does not prove global unreachability.
"""
import itertools
import json
from pathlib import Path

from agent.protocol import build_prompt
from agent.sql_episode import EpisodeConfig, encode_observation, run_episode
from e1_alignment.rebuild.visibility import visible_evidence
from eval.sql_dependencies import sqlite_requirements


class AuditTokenizer:
    """Use the actual local fast tokenizer without loading torch or a model."""
    def __init__(self, model_dir):
        from tokenizers import Tokenizer
        self.inner = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))

    def encode(self, text, add_special_tokens=False):
        return self.inner.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, ids, skip_special_tokens=True):
        return self.inner.decode(list(ids), skip_special_tokens=skip_special_tokens)


def coverage_of(requirements, segments):
    visible = visible_evidence(segments, len(segments), requirements["schema"])
    missing = {field: sorted(set(requirements[field]) - set(visible[field]))
               for field in ("tables", "columns")}
    return {"resolved": requirements["ok"],
            "all_required_schema_visible": requirements["ok"] and not any(missing.values()),
            "missing": missing,
            "visible_tables": sorted(visible["tables"]),
            "visible_columns": sorted(visible["columns"])}


def lookup_candidates(row, requirements):
    """A finite, explicitly gold-informed search; never called by a policy."""
    candidates = [{"query": (row["question"] + " " + row.get("evidence", "")).strip()}]
    by_table = {}
    for table in requirements["tables"]:
        cols = [c[len(table) + 1:] for c in requirements["columns"] if c.startswith(table + ".")]
        by_table[table] = cols
        candidates.append({"table": table})
        candidates.append({"query": " ".join([table, *cols])})
        for col in cols:
            candidates.extend([{"query": f"{table} {col}"}, {"table": table, "column": col}])
    if by_table:
        candidates.append({"query": " ".join(requirements["tables"] + requirements["columns"])})
        candidates.append({"query": row["gold_sql"]})
    unique = {json.dumps(c, sort_keys=True, ensure_ascii=False): c for c in candidates}
    return list(unique.values())


def best_lookup_plan(requirements, candidates, observations, max_lookups=2):
    """Exact set-cover search ONLY over the supplied finite candidate pool."""
    needed = {"t:" + t for t in requirements["tables"]} | {"c:" + c for c in requirements["columns"]}
    sets = []
    for text in observations:
        cov = coverage_of(requirements, [{"kind": "observation", "text": text}])
        sets.append(({"t:" + t for t in cov["visible_tables"]} |
                     {"c:" + c for c in cov["visible_columns"]}) & needed)
    best, best_hits = (), set()
    for count in range(1, min(max_lookups, len(candidates)) + 1):
        for choice in itertools.combinations(range(len(candidates)), count):
            hits = set().union(*(sets[i] for i in choice))
            if len(hits) > len(best_hits):
                best, best_hits = choice, hits
            if hits == needed:
                best, best_hits = choice, hits
                break
        if best_hits == needed:
            break
    return {"resolved": requirements["ok"], "candidate_count": len(candidates),
            "max_lookups": max_lookups, "selected_arguments": [candidates[i] for i in best],
            "all_required_schema_visible": requirements["ok"] and best_hits == needed,
            "required_items": len(needed), "covered_items": len(best_hits),
            "missing": sorted(needed - best_hits), "selected_indices": list(best)}


def gold_injection(row, tokenizer, config=None, lookup_actions=(), schema_tool=None):
    """Exercise the REAL episode/parser with gold as a scripted action.

    SQL execution/scoring is separate, and is not short-circuited by this test.
    Gold injection proves only that a known solution can be submitted.
    """
    config = config or EpisodeConfig(max_rounds=5, max_context=8192)
    prompt = build_prompt(row["question"], row.get("evidence", ""),
                          "\n".join(row.get("catalog_tables", [])), config.max_tools)
    actions = [tokenizer.encode("<schema_lookup>" + json.dumps(a, ensure_ascii=False) +
                               "</schema_lookup>", add_special_tokens=False) for a in lookup_actions]
    answer = tokenizer.encode("<answer>" + row["gold_sql"] + "</answer>", add_special_tokens=False)
    actions.append(answer)
    position = 0

    def generate(ids, remaining):
        nonlocal position
        action = actions[min(position, len(actions) - 1)]
        position += 1
        return action[:remaining]

    def unexpected_tool(_):
        raise AssertionError("direct answer must not invoke a tool")

    state = run_episode(row, tokenizer, generate, schema_tool or unexpected_tool,
                        unexpected_tool, config, prompt=prompt)
    return {"submitted_exact_gold": state["final_sql"] == row["gold_sql"].strip(),
            "gold_had_outer_whitespace": row["gold_sql"] != row["gold_sql"].strip(),
            "finish_reason": state["finish_reason"], "invalid_actions": state["invalid"],
            "prompt_tokens": len(tokenizer.encode(prompt, add_special_tokens=False)),
            "action_tokens": len(answer), "max_action_tokens": config.max_action_tokens,
            "lookup_actions": len(lookup_actions), "executed_lookups": state["schema_lookups"],
            "max_context": config.max_context}


def materialize_observation(tokenizer, result, config):
    ids, text = encode_observation(tokenizer, result, config.max_observation_tokens,
                                   mode=config.observation_mode)
    return {"kind": "observation", "text": text, "tokens": len(ids)}
