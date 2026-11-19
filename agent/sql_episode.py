"""Synchronous multi-turn SQL episode: the shared core used by offline eval and tests.

The async veRL AgentLoop reuses `encode_observation`, `check_budgets` and the
same finish-reason taxonomy, so offline and training rollouts stay comparable.
"""
import json
import time
from dataclasses import dataclass, field

from agent.protocol import build_prompt, parse_action

FINISH_REASONS = ("answer_submitted", "turn_limit", "tool_limit", "invalid_action",
                  "generation_error", "infrastructure_error", "budget_error")


@dataclass
class EpisodeConfig:
    max_rounds: int = 8
    max_tools: int = 4
    max_schema_lookups: int = 2
    max_sql_executes: int = 2
    max_invalid_actions: int = 2
    max_observation_tokens: int = 1024
    max_action_tokens: int = 512
    max_context: int = 8192
    tool_timeout: int = 10
    preview_rows: int = 3
    force_first_schema_lookup: bool = False
    # "compact" drops logging-only fields (retrieval_metadata / elapsed_ms) from the
    # model-visible Observation; "legacy" reproduces the pre-2026-11-08 encoder.
    observation_mode: str = "compact"
    # Optional veRL response tensor budget, including actions and observations.
    # None preserves the historical offline evaluator's context-only behavior.
    max_response_tokens: int | None = None


def _wrap_observation(payload):
    """Serialize one Observation; escape protocol delimiters inside data values."""
    body = json.dumps(payload, ensure_ascii=False, default=str)
    body = body.replace("<", "\\u003c").replace(">", "\\u003e")
    return "<observation>" + body + "</observation>\n"


def _encode_legacy(tokenizer, payload, max_tokens):
    """Old behavior: serialize the full payload, fall back to a 400-char observation."""
    text = "<observation>" + json.dumps(payload, ensure_ascii=False, default=str) + "</observation>\n"
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) > max_tokens:
        trimmed = {"status": payload.get("status"), "truncated": True,
                   "note": "observation truncated to fit the budget"}
        for key in ("error_type", "message", "observation"):
            if key in payload:
                trimmed[key] = str(payload[key])[:400]
        text = "<observation>" + json.dumps(trimmed, ensure_ascii=False, default=str) + "</observation>\n"
        ids = tokenizer.encode(text, add_special_tokens=False)[:max_tokens]
    return ids, tokenizer.decode(ids, skip_special_tokens=True)


def _encode_compact(tokenizer, payload, max_tokens):
    """Model-visible payload only; logging-only fields never consume the budget.

    The E1 retrieval payload carries `retrieval_metadata` (matched columns /
    relationships) for evaluation. Including it in the Observation made the
    full JSON exceed 1024 tokens on most dev200 lookups, which triggered a
    400-character fallback and removed the Relationships section entirely.
    Here the model sees the observation text itself; the metadata stays in
    `tool_events` for metrics.
    """
    visible = {k: v for k, v in payload.items() if k not in ("retrieval_metadata", "elapsed_ms")}
    text = _wrap_observation(visible)
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return ids, tokenizer.decode(ids, skip_special_tokens=True)

    # Drop/shorten the longest fields first, while keeping the JSON valid.
    trimmed = {}
    for key, value in visible.items():
        if isinstance(value, str) and len(value) > 400:
            trimmed[key] = value[:400] + " [truncated]"
        elif isinstance(value, list) and len(value) > 20:
            trimmed[key] = value[:20]
        elif isinstance(value, dict):
            continue
        else:
            trimmed[key] = value
    trimmed["truncated"] = True
    text = _wrap_observation(trimmed)
    if len(tokenizer.encode(text, add_special_tokens=False)) <= max_tokens:
        ids = tokenizer.encode(text, add_special_tokens=False)
        return ids, tokenizer.decode(ids, skip_special_tokens=True)

    # Last resort: keep status/error plus a token-bounded prefix of `observation`.
    base = {"status": visible.get("status"), "truncated": True,
            "note": "observation truncated to fit the budget"}
    for key in ("error_type", "message"):
        if key in visible:
            base[key] = visible[key]
    observation = str(visible.get("observation", ""))
    wrapper_tokens = len(tokenizer.encode(_wrap_observation({**base, "observation": ""}),
                                          add_special_tokens=False))
    available = max_tokens - wrapper_tokens - 16
    while available > 0:
        obs_ids = tokenizer.encode(observation, add_special_tokens=False)[:available]
        candidate = {**base, "observation": tokenizer.decode(obs_ids, skip_special_tokens=True) + " [truncated]"}
        text = _wrap_observation(candidate)
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) <= max_tokens:
            return ids, tokenizer.decode(ids, skip_special_tokens=True)
        available -= 32
    text = _wrap_observation({**base, "observation": "[truncated]"})
    ids = tokenizer.encode(text, add_special_tokens=False)[:max_tokens]
    return ids, tokenizer.decode(ids, skip_special_tokens=True)


def encode_observation(tokenizer, payload, max_tokens, mode="compact"):
    """Observation is data, not a policy target: wrapped and length-bounded."""
    if mode == "legacy":
        return _encode_legacy(tokenizer, payload, max_tokens)
    if mode == "compact_v2":
        return _encode_compact_v2(tokenizer, payload, max_tokens)
    if mode != "compact":
        raise ValueError(f"unknown observation mode: {mode}")
    return _encode_compact(tokenizer, payload, max_tokens)


def _encode_compact_v2(tokenizer, payload, max_tokens):
    """Fit the ACTUAL JSON wrapper without the legacy 400-character cliff.

    Opt-in to keep historical experiments reproducible. Preserve a valid JSON
    object and a closed observation tag; reject budgets too small to do so.
    """
    visible = {k: v for k, v in payload.items() if k not in ("retrieval_metadata", "elapsed_ms")}

    def encoded(value):
        text = _wrap_observation(value)
        return tokenizer.encode(text, add_special_tokens=False)

    ids = encoded(visible)
    if len(ids) <= max_tokens:
        return ids, tokenizer.decode(ids, skip_special_tokens=True)
    # Lookup text usually already fits its own budget. Reserve only the actual
    # wrapper overhead instead of discarding most of the retrieved schema.
    field = "observation" if isinstance(visible.get("observation"), str) else None
    if field is None:
        field = max((k for k, v in visible.items() if isinstance(v, str) and k not in
                     {"status", "error_type", "failure_origin"}),
                    key=lambda k: len(visible[k]), default=None)
    base = {**visible, "truncated": True}
    if field:
        value = visible[field]
        low, high = 0, len(value)
        best = None
        while low <= high:
            middle = (low + high) // 2
            candidate = {**base, field: value[:middle] + " [truncated]"}
            candidate_ids = encoded(candidate)
            if len(candidate_ids) <= max_tokens:
                best = candidate_ids
                low = middle + 1
            else:
                high = middle - 1
        if best is not None:
            return best, tokenizer.decode(best, skip_special_tokens=True)
    minimal = {"status": visible.get("status"), "truncated": True}
    for key in ("error_type", "failure_origin"):
        if key in visible:
            minimal[key] = visible[key]
    ids = encoded(minimal)
    if len(ids) > max_tokens:
        raise ValueError("observation budget cannot fit a valid status wrapper")
    return ids, tokenizer.decode(ids, skip_special_tokens=True)


def check_budgets(state, config):
    """Return an error string when an action cannot be executed."""
    if state["tool_calls"] >= config.max_tools:
        return "tool budget exhausted"
    if state["schema_lookups"] >= config.max_schema_lookups and state["pending"] == "schema_lookup":
        return "schema_lookup budget exhausted"
    if state["sql_executes"] >= config.max_sql_executes and state["pending"] == "sql_execute":
        return "sql_execute budget exhausted"
    return None


def new_state(segment_ids, prompt_text):
    return {"segments": [{"kind": "prompt", "ids": list(segment_ids), "text": prompt_text}],
            "tool_calls": 0, "schema_lookups": 0, "sql_executes": 0, "invalid": 0,
            "rounds": 0, "pending": None, "final_sql": None, "first_sql": None,
            "executions": [], "tool_events": [], "finish_reason": None,
            "infrastructure_error": False, "started": time.monotonic()}


def run_episode(row, tokenizer, generate, schema_tool, sql_tool, config=None,
                prompt=None, recorder=None):
    """Run one episode. `generate(ids, max_new_tokens)` returns token ids."""
    config = config or EpisodeConfig()
    catalog = "\n".join(row.get("catalog_tables") or [])
    prompt = prompt or build_prompt(row["question"], row.get("evidence", ""), catalog, config.max_tools)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    state = new_state(prompt_ids, prompt)
    if config.force_first_schema_lookup:
        # Validation-stage mode: every question sees one retrieval before the model acts.
        seeded = schema_tool({"query": row.get("question", "")})
        ids, text = encode_observation(tokenizer, seeded, config.max_observation_tokens, mode=config.observation_mode)
        state["segments"].append({"kind": "observation", "ids": list(ids), "text": text})
        state["tool_calls"] += 1
        state["schema_lookups"] += 1
        state["tool_events"].append({"action": "schema_lookup", "arguments": {"query": row.get("question", "")},
                                     "result": seeded, "seeded": True})

    def append(kind, ids, text):
        state["segments"].append({"kind": kind, "ids": list(ids), "text": text})

    def context_ids():
        return [i for s in state["segments"] for i in s["ids"]]

    def fail(reason):
        state["finish_reason"] = reason
        return state

    def observation_fits(ids):
        if config.max_response_tokens is None:
            return True  # preserve legacy branch behavior
        used = len(context_ids())
        return (used + len(ids) <= config.max_context
                and used - len(prompt_ids) + len(ids) <= config.max_response_tokens)

    for _ in range(config.max_rounds):
        state["rounds"] += 1
        used = len(context_ids())
        remaining = min(config.max_action_tokens, config.max_context - used)
        if config.max_response_tokens is not None:
            response_room = config.max_response_tokens - (used - len(prompt_ids))
            if response_room <= 0:
                return fail("turn_limit")
            remaining = min(remaining, response_room)
        if remaining <= 0:
            return fail("generation_error")
        try:
            action_ids = list(generate(context_ids(), remaining))
        except Exception as exc:  # generation failure is not a model error
            state["infrastructure_error"] = True
            state["generation_error"] = str(exc)[:200]
            return fail("generation_error")
        if not action_ids or len(action_ids) > remaining:
            return fail("generation_error")
        action_text = tokenizer.decode(action_ids, skip_special_tokens=True)
        append("action", action_ids, action_text)
        try:
            action = parse_action(action_text)
        except (ValueError, TypeError) as exc:
            state["invalid"] += 1
            if state["invalid"] >= config.max_invalid_actions:
                return fail("invalid_action")
            obs = {"status": "error", "error_type": "invalid_action", "message": str(exc)[:300]}
            ids, text = encode_observation(tokenizer, obs, config.max_observation_tokens, mode=config.observation_mode)
            if not observation_fits(ids):
                return fail("generation_error")
            append("observation", ids, text)
            continue
        if action.name == "answer":
            state["final_sql"] = action.arguments
            return fail("answer_submitted")
        state["pending"] = action.name
        budget_error = check_budgets(state, config)
        state["pending"] = None
        if budget_error:
            state["tool_calls"] = state["tool_calls"]
            obs = {"status": "error", "error_type": "budget", "message": budget_error}
            ids, text = encode_observation(tokenizer, obs, config.max_observation_tokens, mode=config.observation_mode)
            if not observation_fits(ids):
                return fail("generation_error")
            append("observation", ids, text)
            if state["tool_calls"] >= config.max_tools:
                return fail("tool_limit")
            continue
        state["tool_calls"] += 1
        if action.name == "schema_lookup":
            state["schema_lookups"] += 1
            payload = schema_tool(action.arguments)
        else:
            state["sql_executes"] += 1
            if state["first_sql"] is None:
                state["first_sql"] = action.arguments
            payload = sql_tool(action.arguments)
            state["executions"].append({"sql": action.arguments, "result": payload,
                                        "action_index": sum(1 for s in state["segments"] if s["kind"] == "action") - 1})
        state["tool_events"].append({"action": action.name, "arguments": action.arguments, "result": payload})
        if payload.get("failure_origin") == "infrastructure":
            state["infrastructure_error"] = True
            ids, text = encode_observation(tokenizer, payload, config.max_observation_tokens, mode=config.observation_mode)
            if observation_fits(ids):
                append("observation", ids, text)
            return fail("infrastructure_error")
        ids, text = encode_observation(tokenizer, payload, config.max_observation_tokens, mode=config.observation_mode)
        if (config.max_response_tokens is not None
                and len(context_ids()) - len(prompt_ids) + len(ids) > config.max_response_tokens):
            return fail("generation_error")
        if (len(context_ids()) + len(ids) >= config.max_context
                and not (config.max_response_tokens is not None
                         and len(context_ids()) + len(ids) == config.max_context)):
            return fail("generation_error")
        append("observation", ids, text)
    return fail("turn_limit")


def episode_to_record(state, row, config=None):
    """Convert the run state into the evaluation record shape."""
    return {
        "id": row.get("id"),
        "db_id": row.get("db_id"),
        "question": row.get("question"),
        "gold_sql": row.get("gold_sql"),
        "final_sql": state.get("final_sql"),
        "first_sql": state.get("first_sql"),
        "finish_reason": state.get("finish_reason"),
        "tool_calls": state.get("tool_calls"),
        "schema_lookups": state.get("schema_lookups"),
        "sql_executes": state.get("sql_executes"),
        "invalid_actions": state.get("invalid"),
        "rounds": state.get("rounds"),
        "executions": state.get("executions", []),
        "tool_events": state.get("tool_events", []),
        "infrastructure_error": state.get("infrastructure_error", False),
        "segments": state.get("segments", []),
        "latency_ms": round((time.monotonic() - state.get("started", time.monotonic())) * 1000, 1),
    }

