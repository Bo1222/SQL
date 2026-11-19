"""Verifiable process signals for multi-turn SQL agent rollouts.

This module is intentionally independent of models and trainers.  It is used by
the offline reward audit first; only signals that survive that audit should be
wired into the training reward.

Design rules:
* terminal EX remains the primary reward;
* process signals may only distinguish *failed* trajectories (rank-preserving);
* "SQL executed successfully", "lookup returned something" and "answer was
  submitted" are not rewards by themselves -- previous C1 validity shaping
  showed that these proxies can increase valid-but-wrong SQL.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class ProcessEvents:
    """Compact, auditable events extracted from one trajectory."""

    tool_calls: int
    schema_lookups: int
    sql_executes: int
    invalid_actions: int
    turns: int
    finish_reason: str | None
    answer_submitted: bool
    first_sql: bool
    final_sql: bool
    schema_error_count: int
    syntax_error_count: int
    other_error_count: int
    schema_error_then_lookup: bool
    schema_error_then_answer: bool
    schema_error_repaired: bool
    lookup_nonempty: bool
    valid_sql_count: int
    repeat_sql: int
    repeat_lookup: int
    actions: tuple[str, ...] = field(default_factory=tuple)
    error_types: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).lower()


def _tool_result(event: Mapping[str, Any]) -> Mapping[str, Any]:
    result = event.get("result")
    return result if isinstance(result, Mapping) else {}


def _is_ok(event: Mapping[str, Any]) -> bool:
    return _tool_result(event).get("status") == "ok"


def _error_type(event: Mapping[str, Any]) -> str | None:
    result = _tool_result(event)
    if result.get("status") != "error":
        return None
    value = result.get("error_type")
    return str(value) if value else "other"


def _lookup_nonempty(event: Mapping[str, Any]) -> bool:
    result = _tool_result(event)
    if result.get("status") != "ok":
        return False
    observation = result.get("observation")
    return bool(str(observation or "").strip())


def extract_process_events(
    *,
    tool_events: Sequence[Mapping[str, Any]] | None,
    finish_reason: str | None,
    final_sql: Any = None,
    first_sql: Any = None,
    invalid_actions: Any = 0,
    turns: Any = 0,
    schema_lookups: Any = None,
    sql_executes: Any = None,
    tool_calls: Any = None,
) -> ProcessEvents:
    """Extract deterministic process signals from one trajectory.

    ``tool_events`` is the ordered list recorded by the agent loop.  Each item
    has ``action``, ``arguments`` and ``result``.  Observation text is not
    parsed; only structured status/error fields and action arguments are used.
    """
    events = [event for event in (tool_events or []) if isinstance(event, Mapping)]
    actions = [str(event.get("action") or "") for event in events]
    if finish_reason == "answer_submitted" and final_sql:
        actions.append("answer")

    schema_error_positions: list[int] = []
    syntax_errors = 0
    other_errors = 0
    for index, event in enumerate(events):
        kind = _error_type(event)
        if kind == "schema_error":
            schema_error_positions.append(index)
        elif kind == "syntax_error":
            syntax_errors += 1
        elif kind is not None:
            other_errors += 1

    schema_error_then_lookup = False
    for index in schema_error_positions:
        next_index = index + 1
        if next_index < len(events) and events[next_index].get("action") == "schema_lookup":
            schema_error_then_lookup = True
            break

    # An answer ends the trajectory.  If the last tool event is a schema error
    # and the trajectory ended with an answer, no tool call happened in between.
    schema_error_then_answer = bool(
        schema_error_positions
        and schema_error_positions[-1] == len(events) - 1
        and finish_reason == "answer_submitted"
        and final_sql
    )

    schema_error_repaired = False
    for position in schema_error_positions:
        for later in events[position + 1:]:
            if later.get("action") == "sql_execute" and _is_ok(later):
                schema_error_repaired = True
                break
        if schema_error_repaired:
            break

    sql_keys = [
        _normalize_text(event.get("arguments"))
        for event in events
        if event.get("action") == "sql_execute"
    ]
    lookup_keys = [
        _normalize_text(json.dumps(event.get("arguments"), sort_keys=True, ensure_ascii=False))
        for event in events
        if event.get("action") == "schema_lookup"
    ]
    repeat_sql = max(len(sql_keys) - len(set(sql_keys)), 0)
    repeat_lookup = max(len(lookup_keys) - len(set(lookup_keys)), 0)

    return ProcessEvents(
        tool_calls=int(tool_calls if tool_calls is not None else len(events)),
        schema_lookups=int(
            schema_lookups if schema_lookups is not None
            else sum(1 for action in actions if action == "schema_lookup")
        ),
        sql_executes=int(
            sql_executes if sql_executes is not None
            else sum(1 for action in actions if action == "sql_execute")
        ),
        invalid_actions=int(invalid_actions or 0),
        turns=int(turns or 0),
        finish_reason=finish_reason,
        answer_submitted=finish_reason == "answer_submitted",
        first_sql=bool(first_sql),
        final_sql=bool(final_sql),
        schema_error_count=len(schema_error_positions),
        syntax_error_count=syntax_errors,
        other_error_count=other_errors,
        schema_error_then_lookup=schema_error_then_lookup,
        schema_error_then_answer=schema_error_then_answer,
        schema_error_repaired=schema_error_repaired,
        lookup_nonempty=any(
            event.get("action") == "schema_lookup" and _lookup_nonempty(event)
            for event in events
        ),
        valid_sql_count=sum(
            1 for event in events
            if event.get("action") == "sql_execute" and _is_ok(event)
        ),
        repeat_sql=repeat_sql,
        repeat_lookup=repeat_lookup,
        actions=tuple(actions),
        error_types=tuple(
            kind for kind in (_error_type(event) for event in events) if kind
        ),
    )


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def progress_score(events: ProcessEvents, mode: str = "recovery_penalties") -> float:
    """Return a verifiable progress score in [0, 1] for a failed trajectory.

    ``recovery_only`` is the simplest candidate: 1 if a schema error was
    followed by a successful SQL execute, else 0.

    ``recovery_penalties`` subtracts small penalties for invalid actions,
    exact repeated SQL and exact repeated lookups.

    ``penalties_only`` is a conservative candidate that never rewards
    execution success; it only penalizes protocol/waste events.

    ``avoid_abandon_only`` penalizes one specific bad branch: answering
    immediately after a schema error.  The Stage 0 audit found this branch has
    EX 6.4% vs 40.8% otherwise, while its valid-but-wrong rate is much lower
    than the recovery branch.

    ``safe_penalties`` combines the abandon penalty with invalid/repeat
    penalties; it never rewards recovery or execution success.
    """
    recovery = 1.0 if events.schema_error_repaired else 0.0
    abandon = 1.0 if events.schema_error_then_answer else 0.0
    invalid_penalty = 0.10 * min(events.invalid_actions, 3)
    repeat_sql_penalty = 0.10 * min(events.repeat_sql, 3)
    repeat_lookup_penalty = 0.05 * min(events.repeat_lookup, 3)
    if mode == "recovery_only":
        return recovery
    if mode == "recovery_penalties":
        return _clip01(
            recovery - invalid_penalty - repeat_sql_penalty - repeat_lookup_penalty
        )
    if mode == "penalties_only":
        return _clip01(
            1.0 - 0.20 * min(events.invalid_actions, 3)
            - 0.10 * min(events.repeat_sql, 3)
            - 0.05 * min(events.repeat_lookup, 3)
        )
    if mode == "avoid_abandon_only":
        return 1.0 - abandon
    if mode == "safe_penalties":
        return _clip01(
            1.0 - 0.50 * abandon
            - 0.10 * min(events.invalid_actions, 3)
            - 0.05 * min(events.repeat_sql, 3)
            - 0.05 * min(events.repeat_lookup, 3)
        )
    raise ValueError(f"unknown process reward mode: {mode}")


def compose_reward(
    terminal_ex: bool,
    events: ProcessEvents,
    *,
    beta: float = 0.05,
    mode: str = "recovery_penalties",
) -> float:
    """Rank-preserving terminal + process reward.

    A correct trajectory always receives 1.0.  A failed trajectory receives
    ``-1.0 + beta * progress`` in [-1, -1+beta].  Therefore any success ranks
    strictly above any failure, independent of the process signal.
    """
    if beta < 0:
        raise ValueError("beta must be non-negative")
    if terminal_ex:
        return 1.0
    return -1.0 + float(beta) * progress_score(events, mode=mode)


def iter_jsonl(path: str) -> Iterable[dict[str, Any]]:
    """Stream a JSONL file without loading it into memory."""
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)
