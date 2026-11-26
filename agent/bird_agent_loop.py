"""veRL AgentLoop adapter for the three-action BIRD SQL agent.

Registered as `bird_sql_agent`; selected with
`actor_rollout_ref.rollout.agent.default_agent_loop=bird_sql_agent`.
All task-specific knobs live under the top-level `bird_agent` config node.
"""
import copy
import json
import os
import time
from pathlib import Path

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer

from agent.protocol import parse_action
from agent.sql_episode import EpisodeConfig, encode_observation
from tools.schema_lookup import SchemaLookupTool
from tools.sql_execute import SQLExecuteTool

_STORE_CACHE = {}


def _load_store(catalog_dir, db_id):
    from retrieval.catalog_store import CatalogStore
    key = (catalog_dir, db_id)
    if key not in _STORE_CACHE:
        _STORE_CACHE[key] = CatalogStore(Path(catalog_dir) / db_id)
    return _STORE_CACHE[key]


@register("bird_sql_agent")
class BirdSQLAgentLoop(AgentLoopBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = self.config.get("bird_agent", {}) or {}
        self.catalog_dir = cfg.get("catalog_dir", "artifacts/catalog")
        max_context = int(cfg.get(
            "max_context",
            getattr(self.rollout_config, "max_model_len", None) or 6144))
        if getattr(self.rollout_config, "max_model_len", None):
            max_context = min(max_context, int(self.rollout_config.max_model_len))
        self.ep_cfg = EpisodeConfig(
            max_rounds=int(cfg.get("max_rounds", 8)),
            max_tools=int(cfg.get("max_tools", 4)),
            max_schema_lookups=int(cfg.get("max_schema_lookups", 2)),
            max_sql_executes=int(cfg.get("max_sql_executes", 2)),
            max_invalid_actions=int(cfg.get("max_invalid_actions", 2)),
            max_observation_tokens=int(cfg.get("max_observation_tokens", 1024)),
            max_action_tokens=int(cfg.get("max_action_tokens", 512)),
            max_context=max_context,
            tool_timeout=int(cfg.get("tool_timeout", 10)),
            preview_rows=int(cfg.get("preview_rows", 3)),
            observation_mode=str(cfg.get("observation_mode", "compact")),
        )
        self.enable_value_lookup = bool(cfg.get("enable_value_lookup", False))
        # The frozen runtime contract is E1.1.  SchemaLookupTool defaults to the
        # legacy E1 backend, so this must be passed explicitly: otherwise the
        # RL rollout silently uses a different retriever than dev200/SFT data.
        self.retrieval_version = str(cfg.get("retrieval_version", "E1.1"))
        self.response_length = self.rollout_config.response_length
        self.runtime_contract = str(cfg.get("runtime_contract", ""))
        # SFT data and the offline evaluator use the raw prompt string (no chat template);
        # wrapping it in <|im_start|>... puts the policy off-distribution.
        self.use_chat_template = bool(cfg.get("use_chat_template", False))
        self.trace_dir = os.environ.get("SQLRL_TRACE_DIR", "")
        self.prompt_mode = None
        if self.runtime_contract:
            from agent.runtime_contract import contract_for
            contract = contract_for(self.runtime_contract)
            self.prompt_mode = contract["prompt_mode"]
            self.ep_cfg.max_response_tokens = self.response_length
            for key, value in contract.items():
                if key in EpisodeConfig.__dataclass_fields__ and getattr(self.ep_cfg, key) != value:
                    raise ValueError(f"runtime contract config mismatch: {key}")
            if self.retrieval_version != contract["retrieval_version"] or self.use_chat_template:
                raise ValueError("runtime contract retrieval/template mismatch")

    def _count_tokens(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _sampling_with_budget(self, sampling_params, budget):
        """Copy sampling params and cap one action at the frozen token budget.

        veRL otherwise defaults each turn to ``rollout.response_length`` tokens,
        which is much larger than the offline ``max_action_tokens=512``.  That
        makes the RL rollout distribution differ from the SFT/eval contract and
        lets one malformed action consume the whole response budget.
        """
        if isinstance(sampling_params, dict):
            params = dict(sampling_params)
            params["max_tokens"] = int(budget)
            params.pop("max_new_tokens", None)
            return params
        params = copy.copy(sampling_params)
        if hasattr(params, "max_tokens"):
            params.max_tokens = int(budget)
        return params

    def _record(self, payload):
        if not self.trace_dir:
            return
        Path(self.trace_dir).mkdir(parents=True, exist_ok=True)
        # one file per worker process: concurrent appends to a shared file can interleave
        with open(Path(self.trace_dir) / f"rollouts.worker-{os.getpid()}.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")

    async def run(self, sampling_params, **kwargs):
        messages = list(kwargs["raw_prompt"])
        extra = kwargs.get("extra_info") or {}
        data_retrieval = extra.get("retrieval_version")
        if data_retrieval and data_retrieval != self.retrieval_version:
            raise ValueError(
                f"retrieval version mismatch: data={data_retrieval!r} "
                f"agent={self.retrieval_version!r}")
        db_id = extra.get("db_id")
        typed_runtime = bool(getattr(self, "runtime_contract", ""))
        if typed_runtime:
            from agent.runtime_contract import load_store, validate_extra
            validate_extra(extra, contract_id=self.runtime_contract)
            key = (self.catalog_dir, db_id, self.runtime_contract)
            if key not in _STORE_CACHE:
                _STORE_CACHE[key] = load_store(self.catalog_dir, db_id, extra["db_path"])
            store = _STORE_CACHE[key]
        else:
            store = _load_store(self.catalog_dir, db_id)
        schema_tool = SchemaLookupTool(store, token_budget=self.ep_cfg.max_observation_tokens,
                                       count_tokens=self._count_tokens,
                                       enable_value_lookup=self.enable_value_lookup,
                                       retrieval_version=self.retrieval_version)
        sql_tool = SQLExecuteTool(extra.get("db_path"), timeout=self.ep_cfg.tool_timeout,
                                  preview_rows=self.ep_cfg.preview_rows,
                                  failure_policy="typed-v1" if typed_runtime else "legacy")

        request_id = f"{db_id}-{extra.get('question_id', 'na')}-{time.time_ns()}"
        if self.use_chat_template:
            prompt_ids = await self.apply_chat_template(messages)
        else:
            contents = [m.get("content", "") for m in messages]
            if typed_runtime:
                from agent.runtime_contract import prompt_for
                if "\n".join(contents) != prompt_for(extra, store, contract_id=self.runtime_contract):
                    raise ValueError("dataset prompt differs from shared A runtime prompt")
            prompt_ids = self.tokenizer.encode("\n".join(contents), add_special_tokens=False)
        response_ids, response_mask, logprobs = [], [], []
        logprobs_complete = True
        metrics, tool_events = {}, []
        final_sql, first_sql = None, None
        finish_reason = "turn_limit"
        turns = 0
        sql_executes = schema_lookups = invalid = 0

        def append_observation(payload):
            """Append observation tokens as mask 0, respecting both budgets."""
            nonlocal response_ids, response_mask, logprobs, logprobs_complete
            ids, _ = encode_observation(
                self.tokenizer, payload, self.ep_cfg.max_observation_tokens,
                mode=self.ep_cfg.observation_mode)
            room = min(
                self.response_length - len(response_mask),
                self.ep_cfg.max_context - (len(prompt_ids) + len(response_ids)),
            )
            # Match runtime.run_states: an observation that does not fully fit
            # is a generation error, not a silently truncated observation.
            if room <= 0 or len(ids) > room:
                return False
            response_ids += ids
            response_mask += [0] * len(ids)
            if logprobs_complete:
                # Tool observations are not policy actions.  They still occupy
                # a position in the response tensor, so old log-probs need a
                # zero placeholder to stay aligned with response_ids.
                logprobs += [0.0] * len(ids)
            return True

        for _ in range(self.ep_cfg.max_rounds):
            response_room = self.response_length - len(response_mask)
            context_room = self.ep_cfg.max_context - (len(prompt_ids) + len(response_ids))
            budget = min(self.ep_cfg.max_action_tokens, response_room, context_room)
            if budget <= 0:
                finish_reason = "turn_limit" if response_room <= 0 else "generation_error"
                break
            turns += 1
            params = self._sampling_with_budget(sampling_params, budget)
            with simple_timer("generate_sequences", metrics):
                output = await self.server_manager.generate(
                    request_id=request_id, prompt_ids=prompt_ids + response_ids,
                    sampling_params=params, image_data=None, video_data=None)
            new_ids = list(output.token_ids)[:budget]
            if not new_ids:
                finish_reason = "generation_error"
                break
            response_ids += new_ids
            response_mask += [1] * len(new_ids)
            turn_logprobs = getattr(output, "log_probs", None)
            if turn_logprobs is None or len(turn_logprobs) < len(new_ids):
                # Partial/missing log-probs cannot be aligned with response_ids.
                # Return None rather than a shorter list that veRL would pad at
                # the end, which would silently shift every later token.
                logprobs_complete = False
                logprobs.clear()
            elif logprobs_complete:
                logprobs += [float(value) for value in list(turn_logprobs)[:len(new_ids)]]

            action_text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
            try:
                action = parse_action(action_text)
            except (ValueError, TypeError) as exc:
                invalid += 1
                if invalid >= self.ep_cfg.max_invalid_actions:
                    finish_reason = "invalid_action"
                    break
                obs = {"status": "error", "error_type": "invalid_action", "message": str(exc)[:300]}
                if not append_observation(obs):
                    finish_reason = "generation_error"
                    break
                continue
            if action.name == "answer":
                final_sql = action.arguments
                finish_reason = "answer_submitted"
                break
            if action.name == "schema_lookup" and schema_lookups >= self.ep_cfg.max_schema_lookups:
                payload = {"status": "error", "error_type": "budget", "message": "schema_lookup budget exhausted"}
            elif action.name == "sql_execute" and sql_executes >= self.ep_cfg.max_sql_executes:
                payload = {"status": "error", "error_type": "budget", "message": "sql_execute budget exhausted"}
            elif len(tool_events) >= self.ep_cfg.max_tools:
                payload = {"status": "error", "error_type": "budget", "message": "tool budget exhausted"}
                finish_reason = "tool_limit"
            else:
                if action.name == "schema_lookup":
                    schema_lookups += 1
                    payload = schema_tool(action.arguments)
                else:
                    sql_executes += 1
                    if first_sql is None:
                        first_sql = action.arguments
                    payload = sql_tool(action.arguments)
                tool_events.append({"action": action.name, "arguments": action.arguments,
                                    "result": payload})
            appended = append_observation(payload)
            if payload.get("failure_origin") == "infrastructure":
                finish_reason = "infrastructure_error"
                break
            if not appended:
                finish_reason = "generation_error"
                break
            if finish_reason == "tool_limit":
                # Offline run_episode terminates immediately when the total
                # tool budget is exhausted; do not let the model keep acting
                # in an environment where every tool call is now an error.
                break
        else:
            finish_reason = "turn_limit"

        traj_info = kwargs.get("_trajectory_info") or {}
        self._record({"db_id": db_id, "question_id": extra.get("question_id"),
                      "step": traj_info.get("step"), "sample_index": traj_info.get("sample_index"),
                      "rollout_n": traj_info.get("rollout_n"),
                      "final_sql": final_sql, "first_sql": first_sql,
                      "finish_reason": finish_reason, "turns": turns,
                      "schema_lookups": schema_lookups, "sql_executes": sql_executes,
                      "invalid_actions": invalid, "tool_events": tool_events,
                      "retrieval_version": self.retrieval_version,
                      "prompt_mode": getattr(self, "prompt_mode", None),
                      "runtime_contract": getattr(self, "runtime_contract", ""),
                      "runtime_fingerprint": extra.get("runtime_fingerprint"),
                      "max_context": self.ep_cfg.max_context,
                      "max_response_tokens": self.response_length,
                      "max_action_tokens": self.ep_cfg.max_action_tokens,
                      "logprobs_complete": logprobs_complete,
                      "prompt_tokens": len(prompt_ids),
                      "action_tokens": sum(1 for m in response_mask if m == 1),
                      "observation_tokens": sum(1 for m in response_mask if m == 0),
                      "response_tokens": len(response_ids)})
        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=(logprobs if logprobs_complete and logprobs else None),
            num_turns=turns,
            metrics=metrics,
            # The typed trainer combines these fields with scorer availability
            # before group selection and advantage computation.
            extra_fields={
                "finish_reason": finish_reason,
                "infrastructure_error": finish_reason == "infrastructure_error",
                "turns": turns,
                "invalid_actions": invalid,
            },
        )



