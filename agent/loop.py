"""Budgeted agent interaction with model-generated actions and trusted observations."""
from dataclasses import dataclass

from agent.protocol import build_prompt, encode_observation, parse_action
from agent.trajectory import Segment, Trajectory


@dataclass
class AgentConfig:
    max_tools: int = 4
    max_context: int = 6144
    max_prompt: int = 1536
    max_generated: int = 2048
    max_action: int = 512
    max_observation: int = 384

    def __post_init__(self):
        if self.max_tools < 0 or min(self.max_context, self.max_prompt, self.max_generated,
                                     self.max_action, self.max_observation) < 1:
            raise ValueError("invalid agent budgets")
        if self.max_prompt >= self.max_context:
            raise ValueError("prompt budget must leave room for actions")


def run_episode(row, tokenizer, generator, tools, config=None, static_context=""):
    config = config or AgentConfig()
    prompt = build_prompt(row, tools.catalog(), config.max_tools)
    if static_context:
        prompt += "\n### Fixed retrieval\n" + static_context + "\n"
    if config.max_tools == 0:
        prompt += "\nNo tool calls are available. Return only <answer>final SQL</answer>.\n"
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(prompt_ids) > config.max_prompt:
        raise ValueError(f"prompt exceeds budget: {len(prompt_ids)} > {config.max_prompt}; not silently truncated")
    trajectory = Trajectory(segments=[Segment("prompt", prompt_ids, prompt)])
    consecutive_invalid = 0

    def observe(result):
        ids, text = encode_observation(tokenizer, result, config.max_observation)
        if len(trajectory.input_ids) + len(ids) >= config.max_context:
            return False
        trajectory.segments.append(Segment("observation", ids, text))
        return True

    # Invalid attempts consume action slots too; no unlimited free retries.
    for turn in range(config.max_tools + 1):
        remaining = min(config.max_action, config.max_generated - trajectory.generated_tokens,
                        config.max_context - len(trajectory.input_ids))
        if remaining <= 0:
            trajectory.stop_reason = "token_budget"
            break
        ids = list(generator(trajectory.input_ids, remaining))
        if not ids or len(ids) > remaining:
            trajectory.stop_reason = "generation_budget"
            break
        text = tokenizer.decode(ids, skip_special_tokens=True)
        trajectory.segments.append(Segment("action", ids, text))
        try:
            action = parse_action(text)
        except (ValueError, TypeError) as exc:
            trajectory.invalid_actions += 1
            consecutive_invalid += 1
            if consecutive_invalid >= 2:
                trajectory.stop_reason = "invalid_actions"
                break
            if not observe({"ok": False, "error": str(exc), "instruction": "Return one complete action"}):
                trajectory.stop_reason = "context_budget"
                break
            continue
        consecutive_invalid = 0
        trajectory.actions.append(action.name)
        if action.name in {"execute_sql", "submit"} and trajectory.first_sql is None:
            trajectory.first_sql = action.arguments
        if action.name == "submit":
            trajectory.final_sql = action.arguments
            trajectory.stop_reason = "submitted"
            break
        if turn == config.max_tools:
            trajectory.stop_reason = "tool_budget"
            break
        trajectory.tool_calls += 1
        result = tools.dispatch(action)
        if result.get("infrastructure_error"):
            trajectory.stop_reason = "infrastructure_error"
            break
        if turn == config.max_tools - 1:
            result = {**result, "instruction": "Tool budget exhausted; submit <answer>SQL</answer> next"}
        if not observe(result):
            trajectory.stop_reason = "context_budget"
            break
    if not trajectory.stop_reason:
        trajectory.stop_reason = "action_budget"
    return trajectory
