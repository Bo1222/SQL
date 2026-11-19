"""Fixed E1 runtime contract and a lockstep batch episode runner.

The runner mirrors `agent.sql_episode.run_episode` (finish reasons, budgets,
observation encoding) but can advance several episodes in one vLLM call.
"""
import copy
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from agent.protocol import build_prompt, parse_action
from agent.sql_episode import EpisodeConfig, encode_observation
from retrieval.catalog_store import CatalogStore
from tools.schema_lookup import SchemaLookupTool
from tools.sql_execute import SQLExecuteTool

from e1_alignment.common import CATALOG_DIR, MODEL_PATH, RL_METADATA, load_metadata


@dataclass
class RuntimeContract:
    """E1-01 frozen values shared by collection, SFT and evaluation."""

    catalog_dir: str = str(CATALOG_DIR)
    model_path: str = str(MODEL_PATH)
    prompt_mode: str = "query"
    observation_mode: str = "compact"
    max_rounds: int = 5
    max_tools: int = 4
    max_schema_lookups: int = 2
    max_sql_executes: int = 2
    max_invalid_actions: int = 2
    max_observation_tokens: int = 1024
    max_action_tokens: int = 512
    max_context: int = 6144
    tool_timeout: int = 10
    preview_rows: int = 3
    score_timeout: int = 30
    enable_value_lookup: bool = False
    retrieval_version: str = "E1"

    def episode_config(self):
        return EpisodeConfig(
            max_rounds=self.max_rounds, max_tools=self.max_tools,
            max_schema_lookups=self.max_schema_lookups,
            max_sql_executes=self.max_sql_executes,
            max_invalid_actions=self.max_invalid_actions,
            max_observation_tokens=self.max_observation_tokens,
            max_action_tokens=self.max_action_tokens,
            max_context=self.max_context, tool_timeout=self.tool_timeout,
            preview_rows=self.preview_rows, observation_mode=self.observation_mode,
        )

    def as_dict(self):
        return dict(self.__dict__)


class Runtime:
    """Loaded E1 tools + tokenizer bound to one frozen contract."""

    def __init__(self, contract=None, tokenizer=None):
        from transformers import AutoTokenizer

        self.contract = contract or RuntimeContract()
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(self.contract.model_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self._stores = {}
        self._metadata = {}
        self._tools = {}

    # -- data -----------------------------------------------------------
    def metadata(self, db_id):
        if db_id not in self._metadata:
            self._metadata[db_id] = load_metadata({"db_id": db_id, "metadata_path": RL_METADATA / f"{db_id}.json"})
        return self._metadata[db_id]

    def catalog_text(self, db_id):
        return "\n".join(t["name"] for t in self.metadata(db_id)["tables"])

    def prompt_for_row(self, row):
        return build_prompt(row["question"], row.get("evidence", ""), self.catalog_text(row["db_id"]),
                            max_tools=self.contract.max_tools, prompt_mode=self.contract.prompt_mode)

    # -- tools ----------------------------------------------------------
    def store(self, db_id):
        if db_id not in self._stores:
            self._stores[db_id] = CatalogStore(Path(self.contract.catalog_dir) / db_id)
        return self._stores[db_id]

    def schema_tool(self, db_id):
        key = ("schema", db_id)
        if key not in self._tools:
            self._tools[key] = SchemaLookupTool(
                self.store(db_id), token_budget=self.contract.max_observation_tokens,
                count_tokens=self.count_tokens,
                enable_value_lookup=self.contract.enable_value_lookup,
                retrieval_version=self.contract.retrieval_version)
        return self._tools[key]

    def sql_tool(self, db_path):
        key = ("sql", str(db_path))
        if key not in self._tools:
            self._tools[key] = SQLExecuteTool(db_path, timeout=self.contract.tool_timeout,
                                              preview_rows=self.contract.preview_rows)
        return self._tools[key]

    def count_tokens(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    # -- state ----------------------------------------------------------
    def new_state(self, row, prompt=None, prompt_ids=None, prefix_segments=None, tool_events=None):
        prompt = prompt or self.prompt_for_row(row)
        prompt_ids = prompt_ids or self.tokenizer.encode(prompt, add_special_tokens=False)
        state = EpisodeState(row=row, prompt=prompt, prompt_ids=list(prompt_ids),
                             contract=self.contract, runtime=self)
        if tool_events:
            state.tool_events = list(tool_events)
        if prefix_segments:
            for segment in copy.deepcopy(prefix_segments):
                state.append_existing(segment)
        return state


@dataclass
class EpisodeState:
    row: dict
    prompt: str
    prompt_ids: list
    contract: RuntimeContract
    runtime: Runtime = field(repr=False)
    segments: list = field(default_factory=list)
    tool_events: list = field(default_factory=list)
    finish_reason: str | None = None
    final_sql: str | None = None
    first_sql: str | None = None
    rounds: int = 0
    tool_calls: int = 0
    schema_lookups: int = 0
    sql_executes: int = 0
    invalid_actions: int = 0
    infrastructure_error: bool = False
    generation_error: str | None = None
    started: float = field(default_factory=time.monotonic)
    index: int = 0

    def __post_init__(self):
        if not self.segments:
            self.segments.append({"kind": "prompt", "ids": list(self.prompt_ids), "text": self.prompt})

    @property
    def db_path(self):
        return self.row["db_path"]

    @property
    def done(self):
        return self.finish_reason is not None

    def context_ids(self):
        return [token for segment in self.segments for token in segment["ids"]]

    def context_length(self):
        return sum(len(segment["ids"]) for segment in self.segments)

    def append_existing(self, segment):
        """Add a reconstructed prefix segment and update budget counters."""
        self.segments.append(segment)
        kind = segment.get("kind")
        if kind == "action":
            self.rounds += 1
            action = segment.get("action")
            if action == "answer":
                self.final_sql = segment.get("arguments")
            elif action in ("schema_lookup", "sql_execute"):
                self.tool_calls += 1
                if action == "schema_lookup":
                    self.schema_lookups += 1
                else:
                    self.sql_executes += 1
                    if self.first_sql is None:
                        self.first_sql = segment.get("arguments")
            elif segment.get("invalid"):
                self.invalid_actions += 1

    def append_action(self, ids, text, action=None, arguments=None, invalid=False):
        segment = {"kind": "action", "ids": list(ids), "text": text, "action": action,
                   "arguments": arguments, "invalid": invalid}
        self.segments.append(segment)
        self.rounds += 1
        return segment

    def append_observation(self, payload, tool_event_index=None):
        ids, text = encode_observation(self.runtime.tokenizer, payload,
                                       self.contract.max_observation_tokens,
                                       mode=self.contract.observation_mode)
        if self.context_length() + len(ids) >= self.contract.max_context:
            return False
        segment = {"kind": "observation", "ids": list(ids), "text": text,
                   "tool_event_index": tool_event_index}
        if tool_event_index is None:
            segment["payload"] = payload
        self.segments.append(segment)
        return True

    def fail(self, reason):
        self.finish_reason = reason
        return self

    def record(self):
        return {
            "id": self.row["id"], "db_id": self.row["db_id"],
            "question": self.row["question"], "evidence": self.row.get("evidence", ""),
            "gold_sql": self.row["gold_sql"], "db_path": self.db_path,
            "prompt": self.prompt, "prompt_ids": list(self.prompt_ids),
            "segments": self.segments, "tool_events": self.tool_events,
            "final_sql": self.final_sql, "first_sql": self.first_sql,
            "finish_reason": self.finish_reason, "rounds": self.rounds,
            "tool_calls": self.tool_calls, "schema_lookups": self.schema_lookups,
            "sql_executes": self.sql_executes, "invalid_actions": self.invalid_actions,
            "infrastructure_error": self.infrastructure_error,
            "generation_error": self.generation_error,
            "latency_ms": round((time.monotonic() - self.started) * 1000, 1),
        }


def _budget_error(state):
    config = state.contract
    if state.tool_calls >= config.max_tools:
        return "tool budget exhausted"
    if state.schema_lookups >= config.max_schema_lookups and state._pending == "schema_lookup":
        return "schema_lookup budget exhausted"
    if state.sql_executes >= config.max_sql_executes and state._pending == "sql_execute":
        return "sql_execute budget exhausted"
    return None


def _execute_action(state, action):
    """Mirror run_episode's budget/tool/observation sequence for one tool action."""
    config = state.contract
    state._pending = action.name
    budget_error = _budget_error(state)
    state._pending = None
    if budget_error:
        payload = {"status": "error", "error_type": "budget", "message": budget_error}
        state.append_observation(payload)
        if state.tool_calls >= config.max_tools:
            return state.fail("tool_limit")
        return state

    state.tool_calls += 1
    if action.name == "schema_lookup":
        state.schema_lookups += 1
        payload = state.runtime.schema_tool(state.row["db_id"])(action.arguments)
    else:
        state.sql_executes += 1
        if state.first_sql is None:
            state.first_sql = action.arguments
        payload = state.runtime.sql_tool(state.db_path)(action.arguments)
    state.tool_events.append({"action": action.name, "arguments": action.arguments, "result": payload})
    event_index = len(state.tool_events) - 1

    state.segments[-1]["tool_event_index"] = event_index
    if payload.get("failure_origin") == "infrastructure":
        state.infrastructure_error = True
        state.append_observation(payload, tool_event_index=event_index)
        return state.fail("infrastructure_error")
    if not state.append_observation(payload, tool_event_index=event_index):
        return state.fail("generation_error")
    return state


def process_generation(state, new_ids, action_index):
    """Apply one generated action to a state (shared by batch and replay paths)."""
    contract = state.contract
    room = contract.max_context - state.context_length()
    budget = min(contract.max_action_tokens, room)
    new_ids = list(new_ids)[:budget]
    if not new_ids:
        return state.fail("generation_error")
    action_text = state.runtime.tokenizer.decode(new_ids, skip_special_tokens=True)
    try:
        action = parse_action(action_text)
    except (ValueError, TypeError) as exc:
        state.invalid_actions += 1
        state.append_action(new_ids, action_text, invalid=True)
        if state.invalid_actions >= contract.max_invalid_actions:
            return state.fail("invalid_action")
        payload = {"status": "error", "error_type": "invalid_action", "message": str(exc)[:300]}
        state.append_observation(payload)
        return state
    if action.name == "answer":
        state.final_sql = action.arguments
        state.append_action(new_ids, action_text, action="answer", arguments=action.arguments)
        return state.fail("answer_submitted")
    state.append_action(new_ids, action_text, action=action.name, arguments=action.arguments)
    return _execute_action(state, action)


def run_states(states, engine, temperature=1.0, top_p=1.0, seed_base=None, stop_on_infra=True):
    """Advance a list of states in lockstep, batching generations per round."""
    states = [state for state in states if not state.done]
    round_number = 0
    while states:
        active, contexts, budgets, seeds = [], [], [], []
        for state in states:
            if state.rounds >= state.contract.max_rounds:
                state.fail("turn_limit")
                continue
            room = state.contract.max_context - state.context_length()
            budget = min(state.contract.max_action_tokens, room)
            if budget <= 0:
                state.fail("generation_error")
                continue
            active.append(state)
            contexts.append(state.context_ids())
            budgets.append(budget)
            if seed_base is not None:
                seeds.append(seed_base + state.index * 1009 + round_number)
        if not active:
            break
        try:
            outputs = engine.generate(contexts, budgets, temperature=temperature,
                                      top_p=top_p, seeds=seeds or None)
        except Exception as exc:  # engine failure is infrastructure, not policy
            for state in active:
                state.infrastructure_error = True
                state.generation_error = str(exc)[:300]
                state.fail("generation_error")
            break
        for state, new_ids in zip(active, outputs):
            if state.rounds >= state.contract.max_rounds:
                state.fail("turn_limit")
                continue
            process_generation(state, new_ids, round_number)
            if state.done and state.finish_reason == "infrastructure_error" and stop_on_infra:
                continue
        states = [state for state in states if not state.done]
        round_number += 1
    for state in states:
        if not state.done:
            state.fail("turn_limit")
    return states


def run_group(row, runtime, engine, n=4, temperature=1.0, top_p=1.0, seed_base=0):
    states = [runtime.new_state(row) for _ in range(n)]
    for index, state in enumerate(states):
        state.index = index
    run_states(states, engine, temperature=temperature, top_p=top_p, seed_base=seed_base)
    return states


def run_continuations(row, runtime, engine, prefix_segments, n=4, temperature=1.0,
                      top_p=1.0, seed_base=0, prefix_tool_events=None):
    states = [runtime.new_state(row, prefix_segments=prefix_segments,
                                tool_events=prefix_tool_events) for _ in range(n)]
    for index, state in enumerate(states):
        state.index = index
    run_states(states, engine, temperature=temperature, top_p=top_p, seed_base=seed_base)
    return states
