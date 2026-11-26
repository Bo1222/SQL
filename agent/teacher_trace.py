"""Teacher-trace collection: the same typed harness driven by a provider model,
with explanatory prose masked out of its raw responses instead of being
suppressed by the prompt.

Deliberate differences from `run_episode` (diagnostic-only, documented in the
artifact manifest):

  * A response with no complete action block does not consume a round or an
    invalid-action slot: it is recorded as `no_action` and the same context is
    re-sent, bounded by `max_no_action_retries`.
  * A response with several complete action blocks yields the first one; the
    rest are recorded as `extra_actions` and dropped, never executed.
  * The executed action is the MASKED (action-only) text, so the context the
    model sees next stays protocol-clean and its token cost is countable in the
    student tokenizer.

Everything else -- prompt, catalog, tools, observation encoder, budgets,
round/tool limits and scoring -- is identical to the student episode.
"""
import time

from agent.protocol import PATTERN, build_prompt, parse_action
from agent.sql_episode import (
    EpisodeConfig,
    check_budgets,
    encode_observation,
    new_state,
)

TEACHER_FINISH_REASONS = ("answer_submitted", "turn_limit", "tool_limit", "invalid_action",
                          "no_action_limit", "action_over_budget", "generation_error",
                          "infrastructure_error", "budget_error", "seed_observation_overflow")


def mask_action(tokenizer, text):
    """Keep only complete action blocks; return (masked_text, ids, info).

    The ground truth is the exact character slice of the first complete action
    block, never a set of whole tokens: a BPE token can straddle the closing
    `>` (it merges with the prose that follows), and dropping that token would
    hand the harness a block without its delimiter -- a valid action scored as
    `invalid_action`. So the offset-filtered ids are used only when they
    round-trip to the exact slice; otherwise the slice is re-encoded. Prose
    before, between and after actions is dropped and reported in
    `dropped_char_ranges`.
    """
    spans = [(m.start(), m.end()) for m in PATTERN.finditer(text)]
    info = {"raw_characters": len(text), "action_spans": spans, "extra_actions": len(spans) - 1 if spans else 0}
    if not spans:
        dropped_all = [(0, len(text))] if text else []
        return "", [], {**info, "kept_tokens": 0, "dropped_tokens": len(text) > 0,
                        "kept_char_ranges": [], "dropped_char_ranges": dropped_all,
                        "mask_granularity": "token" if getattr(tokenizer, "inner", None) is not None else "character"}
    start, end = spans[0]
    exact = text[start:end]
    inner = getattr(tokenizer, "inner", None)
    encoding = None
    if inner is not None and hasattr(inner, "encode"):
        try:
            encoding = inner.encode(text, add_special_tokens=False)
        except TypeError:
            encoding = inner.encode(text)
    if encoding is not None and getattr(encoding, "offsets", None):
        kept = [i for i, (s, e) in enumerate(encoding.offsets) if s >= start and e <= end]
        straddles = sum(1 for s, e in encoding.offsets if e > start and s < end and not (s >= start and e <= end))
        ids = [encoding.ids[i] for i in kept]
        masked = tokenizer.decode(ids, skip_special_tokens=True)
        if straddles or masked.strip() != exact.strip():
            ids = tokenizer.encode(exact, add_special_tokens=False)
            masked = tokenizer.decode(ids, skip_special_tokens=True)
            mask_granularity = "token" if not straddles else "token_reencoded"
        else:
            mask_granularity = "token"
        return masked, ids, {**info, "kept_tokens": len(ids), "dropped_tokens": len(encoding.ids) - len(kept),
                             "boundary_tokens_reencoded": straddles,
                             "kept_char_ranges": [[start, end]], "dropped_char_ranges": _complement_ranges(text, [(start, end)]),
                             "token_char_spans": [list(encoding.offsets[i]) for i in kept] if mask_granularity == "token" else None,
                             "mask_granularity": mask_granularity}
    ids = tokenizer.encode(exact, add_special_tokens=False)
    return exact, ids, {**info, "kept_tokens": len(ids), "dropped_tokens": len(
        tokenizer.encode(text, add_special_tokens=False)) - len(ids),
        "kept_char_ranges": [[start, end]], "dropped_char_ranges": _complement_ranges(text, [(start, end)]),
        "mask_granularity": "character"}


def _complement_ranges(text, kept):
    out, cursor = [], 0
    for start, end in kept:
        if start > cursor:
            out.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < len(text):
        out.append((cursor, len(text)))
    return out


def restore_stop_delimiter(text, finish_reason=None):
    """Re-add the closing tag the stop sequence consumed.

    A `stop` request returns content WITHOUT the stop string, so a response whose
    action was terminated by the delimiter arrives with an unclosed tag -- with or
    without a prose prefix. Only restore when the provider reported `stop`: a
    `length` cut is a truncated action and must never be repaired into an action.
    """
    if finish_reason not in (None, "stop"):
        return text, False
    opened = [(text.rfind("<" + tag + ">"), tag) for tag in ("schema_lookup", "sql_execute", "answer")]
    opened = [(index, tag) for index, tag in opened if index >= 0 and "</" + tag + ">" not in text[index:]]
    if not opened:
        return text, False
    _, tag = max(opened)
    return text.rstrip() + "</" + tag + ">", True


def run_teacher_episode(row, tokenizer, generate_raw, schema_tool, sql_tool, config=None,
                        prompt=None, max_no_action_retries=3):
    """Run one teacher episode; `generate_raw(ids, remaining)` returns a raw response dict."""
    config = config or EpisodeConfig()
    catalog = "\n".join(row.get("catalog_tables") or [])
    prompt = prompt or build_prompt(row["question"], row.get("evidence", ""), catalog, config.max_tools)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    state = new_state(prompt_ids, prompt)
    state.update(steps=[], no_action_responses=0, action_over_budget=0, extra_actions=0, masked_tokens_total=0)

    def append(kind, ids, text):
        state["segments"].append({"kind": kind, "ids": list(ids), "text": text})

    def context_ids():
        return [i for s in state["segments"] for i in s["ids"]]

    def fail(reason):
        state["finish_reason"] = reason
        return state

    def observation_fits(ids):
        if config.max_response_tokens is None:
            return True
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

        attempt = 0
        while True:
            attempt += 1
            try:
                raw = generate_raw(context_ids(), remaining)
            except Exception as exc:  # provider/transport failure is not a model error
                state["infrastructure_error"] = True
                state["generation_error"] = str(exc)[:200]
                return fail("generation_error")
            text, restored = restore_stop_delimiter(raw.get("content") or "", raw.get("finish_reason"))
            masked_text, action_ids, mask_info = mask_action(tokenizer, text)
            step = {"round": state["rounds"], "attempt": attempt, "raw_content": text,
                    "raw_content_pre_restore": raw.get("content") if restored else None,
                    "stop_delimiter_restored": restored,
                    "raw_finish_reason": raw.get("finish_reason"), "usage": raw.get("usage"),
                    "request_id": raw.get("request_id"), "mask": mask_info}
            state["steps"].append(step)
            if not action_ids:
                state["no_action_responses"] += 1
                step["outcome"] = "no_action"
                if attempt > max_no_action_retries:
                    return fail("no_action_limit")
                continue
            if len(action_ids) > remaining:
                state["action_over_budget"] += 1
                step["outcome"] = "action_over_budget"
                step["masked_tokens"] = len(action_ids)
                return fail("action_over_budget")
            break
        state["extra_actions"] += mask_info["extra_actions"]
        state["masked_tokens_total"] += len(action_ids)
        step.update(outcome="action", masked_action_text=masked_text, masked_tokens=len(action_ids))
        append("action", action_ids, masked_text)
        try:
            action = parse_action(masked_text)
        except (ValueError, TypeError) as exc:
            state["invalid"] += 1
            step["outcome"] = "invalid_action"
            step["invalid_reason"] = str(exc)[:300]
            if state["invalid"] >= config.max_invalid_actions:
                return fail("invalid_action")
            obs = {"status": "error", "error_type": "invalid_action", "message": str(exc)[:300]}
            ids, text = encode_observation(tokenizer, obs, config.max_observation_tokens,
                                           mode=config.observation_mode)
            if not observation_fits(ids):
                return fail("generation_error")
            append("observation", ids, text)
            step["observation_text"] = text
            continue
        if action.name == "answer":
            state["final_sql"] = action.arguments
            step["action"] = {"name": action.name, "arguments": action.arguments}
            return fail("answer_submitted")
        state["pending"] = action.name
        budget_error = check_budgets(state, config)
        state["pending"] = None
        if budget_error:
            obs = {"status": "error", "error_type": "budget", "message": budget_error}
            ids, text = encode_observation(tokenizer, obs, config.max_observation_tokens,
                                           mode=config.observation_mode)
            if not observation_fits(ids):
                return fail("generation_error")
            append("observation", ids, text)
            step["action"] = {"name": action.name, "arguments": action.arguments}
            step["observation_text"] = text
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
        step["action"] = {"name": action.name, "arguments": action.arguments}
        step["tool_result"] = payload
        if payload.get("failure_origin") == "infrastructure":
            state["infrastructure_error"] = True
            ids, text = encode_observation(tokenizer, payload, config.max_observation_tokens,
                                           mode=config.observation_mode)
            if observation_fits(ids):
                append("observation", ids, text)
            return fail("infrastructure_error")
        ids, text = encode_observation(tokenizer, payload, config.max_observation_tokens,
                                       mode=config.observation_mode)
        if (config.max_response_tokens is not None
                and len(context_ids()) - len(prompt_ids) + len(ids) > config.max_response_tokens):
            return fail("generation_error")
        if (len(context_ids()) + len(ids) >= config.max_context
                and not (config.max_response_tokens is not None
                         and len(context_ids()) + len(ids) == config.max_context)):
            return fail("generation_error")
        append("observation", ids, text)
        step["observation_text"] = text
    return fail("turn_limit")


def sft_pairs(state, row, contract_id, fingerprint):
    """(context, action) token pairs: the distillation view of one trace."""
    pairs, context = [], []
    step_index = 0
    for segment in state["segments"]:
        if segment["kind"] == "action":
            step_index += 1
            pairs.append({"id": row.get("id"), "db_id": row.get("db_id"), "step": step_index,
                          "context_ids": list(context), "target_ids": list(segment["ids"]),
                          "target_text": segment["text"], "runtime_contract": contract_id,
                          "runtime_fingerprint": fingerprint})
        context.extend(segment["ids"])
    return pairs


def trace_record(state, row, config=None):
    """Provenance-preserving trace: raw provider text, mask, action, observation."""
    return {
        "id": row.get("id"), "db_id": row.get("db_id"), "question": row.get("question"),
        "gold_sql": row.get("gold_sql"), "final_sql": state.get("final_sql"),
        "first_sql": state.get("first_sql"), "finish_reason": state.get("finish_reason"),
        "rounds": state.get("rounds"), "tool_calls": state.get("tool_calls"),
        "schema_lookups": state.get("schema_lookups"), "sql_executes": state.get("sql_executes"),
        "invalid_actions": state.get("invalid"), "no_action_responses": state.get("no_action_responses"),
        "extra_actions_dropped": state.get("extra_actions"), "masked_action_tokens": state.get("masked_tokens_total"),
        "infrastructure_error": state.get("infrastructure_error", False),
        "generation_error": state.get("generation_error"),
        "steps": state.get("steps", []), "executions": state.get("executions", []),
        "tool_events": state.get("tool_events", []), "segments": state.get("segments", []),
        "prompt": state["segments"][0]["text"], "prompt_tokens": len(state["segments"][0]["ids"]),
        "latency_ms": round((time.monotonic() - state.get("started", time.monotonic())) * 1000, 1),
    }
