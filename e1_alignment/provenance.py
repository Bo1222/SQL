"""Provenance contract for M6-08m student-only SFT data.

The contract separates *how a trajectory was produced* from *whether it was
successful*.  Only two sources are trainable:

``raw_success``
    The 3B policy generated every action from scratch and the final SQL
    matched Gold.  No gold SQL was injected and no failed prefix was reused.

``self_repair_success``
    The policy failed a first attempt, then continued from a failed prefix
    (instead of a gold correction) and eventually produced a correct final
    SQL.  ``parent_trajectory_id`` and ``repair_round`` are mandatory.

Everything else is audit-only and must never enter a training manifest:
``raw_failure``, ``self_repair_failure``, ``gold_constructed``,
``legacy_model_replay`` and ``unmarked``.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Iterator

RAW_SUCCESS = "raw_success"
SELF_REPAIR_SUCCESS = "self_repair_success"
RAW_FAILURE = "raw_failure"
SELF_REPAIR_FAILURE = "self_repair_failure"
GOLD_CONSTRUCTED = "gold_constructed"
LEGACY_MODEL_REPLAY = "legacy_model_replay"
UNMARKED = "unmarked"

TRAINABLE_SOURCES = frozenset({RAW_SUCCESS, SELF_REPAIR_SUCCESS})
SUCCESS_SOURCES = frozenset({RAW_SUCCESS, SELF_REPAIR_SUCCESS})
AUDIT_ONLY_SOURCES = frozenset(
    {RAW_FAILURE, SELF_REPAIR_FAILURE, GOLD_CONSTRUCTED, LEGACY_MODEL_REPLAY, UNMARKED}
)
ALL_SOURCES = TRAINABLE_SOURCES | AUDIT_ONLY_SOURCES

GENERATION_MODES = frozenset(
    {"n1_greedy", "n1_sample", "n4_sample", "n8_sample", "prefix_repair", "legacy_replay"}
)
FINAL_EX_SOURCES = frozenset({"gold_scoring", "inherited_legacy", "unscored", "score_error"})


class ProvenanceError(ValueError):
    """Raised when a record violates the provenance contract."""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path) -> Iterator[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ProvenanceError(f"{path}:{line_number}: invalid JSON: {exc}") from exc


def classify_legacy_origin(origin: str | None) -> str:
    """Map the old ``full_tool_unique.jsonl`` origin to the new vocabulary."""
    if origin == "constructed_recovery_gold":
        return GOLD_CONSTRUCTED
    if origin == "model_rollout_replay":
        return LEGACY_MODEL_REPLAY
    return UNMARKED


def classify_legacy_source_type(source_type: str | None) -> str:
    """Map the old ``sft_trajectories.jsonl`` source_type conservatively.

    The old file only distinguishes ``autonomous_success`` and
    ``autonomous_recovery``; it cannot tell whether a recovery was built with a
    gold correction.  Both are therefore audit-only until joined to a finer
    provenance file.
    """
    if source_type in ("autonomous_success", "autonomous_recovery"):
        return LEGACY_MODEL_REPLAY
    return UNMARKED


def build_provenance(
    *,
    source: str,
    generation_mode: str,
    model_path: str,
    model_fingerprint: str,
    runtime_fingerprint: str,
    retrieval_version: str,
    prompt_mode: str,
    observation_mode: str,
    split: str,
    final_ex: bool,
    final_ex_source: str,
    sample_index: int = 0,
    parent_trajectory_id: str | None = None,
    repair_round: int = 0,
    gold_used_for_construction: bool = False,
    tool_replay: bool = False,
    created_at: str | None = None,
) -> dict:
    """Build and validate one provenance object."""
    import datetime as _dt

    provenance = {
        "source": source,
        "trainable": source in TRAINABLE_SOURCES,
        "generation_mode": generation_mode,
        "model_path": str(model_path),
        "model_fingerprint": str(model_fingerprint),
        "runtime_fingerprint": str(runtime_fingerprint),
        "retrieval_version": str(retrieval_version),
        "prompt_mode": str(prompt_mode),
        "observation_mode": str(observation_mode),
        "split": str(split),
        "final_ex": bool(final_ex),
        "final_ex_source": str(final_ex_source),
        "sample_index": int(sample_index),
        "parent_trajectory_id": parent_trajectory_id,
        "repair_round": int(repair_round),
        "gold_used_for_construction": bool(gold_used_for_construction),
        "tool_replay": bool(tool_replay),
        "created_at": created_at or _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    validate_provenance(provenance)
    return provenance


def build_legacy_provenance(
    *,
    origin: str | None = None,
    source_type: str | None = None,
    trajectory_id: str | None = None,
    question_id: str | None = None,
    matched: bool = False,
    tool_replay: bool = False,
) -> dict:
    """Build an audit-only provenance object for legacy data."""
    if origin is not None:
        source = classify_legacy_origin(origin)
        reason = f"legacy origin={origin}"
    else:
        source = classify_legacy_source_type(source_type)
        reason = f"legacy source_type={source_type}"
    return {
        "source": source,
        "trainable": False,
        "generation_mode": "legacy_replay",
        "model_path": None,
        "model_fingerprint": None,
        "runtime_fingerprint": None,
        "retrieval_version": "legacy",
        "prompt_mode": "legacy",
        "observation_mode": "legacy",
        "split": None,
        "final_ex": None,
        "final_ex_source": "inherited_legacy",
        "sample_index": None,
        "parent_trajectory_id": None,
        "repair_round": None,
        "gold_used_for_construction": source == GOLD_CONSTRUCTED,
        "tool_replay": bool(tool_replay),
        "legacy": {
            "origin": origin,
            "source_type": source_type,
            "trajectory_id": trajectory_id,
            "question_id": question_id,
            "matched": bool(matched),
            "reason": reason,
        },
    }


def validate_provenance(provenance: dict, *, require_trainable: bool = False) -> None:
    """Validate mandatory fields and cross-field invariants."""
    if not isinstance(provenance, dict):
        raise ProvenanceError("provenance must be an object")
    source = provenance.get("source")
    if source not in ALL_SOURCES:
        raise ProvenanceError(f"unknown provenance source: {source!r}")
    mode = provenance.get("generation_mode")
    if mode not in GENERATION_MODES:
        raise ProvenanceError(f"unknown generation_mode: {mode!r}")
    final_ex_source = provenance.get("final_ex_source")
    if final_ex_source not in FINAL_EX_SOURCES:
        raise ProvenanceError(f"unknown final_ex_source: {final_ex_source!r}")
    if provenance.get("trainable") != (source in TRAINABLE_SOURCES):
        raise ProvenanceError("trainable flag contradicts source")
    final_ex = provenance.get("final_ex")
    legacy_audit = (
        provenance.get("generation_mode") == "legacy_replay"
        and provenance.get("final_ex_source") == "inherited_legacy"
    )
    if final_ex is None and legacy_audit:
        pass
    elif not isinstance(final_ex, bool):
        raise ProvenanceError("final_ex must be a bool")
    elif source in SUCCESS_SOURCES and not final_ex:
        raise ProvenanceError(f"{source} requires final_ex=True")
    elif source in (RAW_FAILURE, SELF_REPAIR_FAILURE) and final_ex:
        raise ProvenanceError(f"{source} requires final_ex=False")
    if source == SELF_REPAIR_SUCCESS:
        if not provenance.get("parent_trajectory_id"):
            raise ProvenanceError("self_repair_success requires parent_trajectory_id")
        if int(provenance.get("repair_round") or 0) < 1:
            raise ProvenanceError("self_repair_success requires repair_round >= 1")
    if source == RAW_SUCCESS:
        if provenance.get("parent_trajectory_id"):
            raise ProvenanceError("raw_success must not have parent_trajectory_id")
        if int(provenance.get("repair_round") or 0) != 0:
            raise ProvenanceError("raw_success must have repair_round=0")
    if source in TRAINABLE_SOURCES:
        if provenance.get("gold_used_for_construction"):
            raise ProvenanceError(f"{source} must not use gold for construction")
        if provenance.get("tool_replay"):
            raise ProvenanceError(f"{source} must not be a tool replay")
        if provenance.get("final_ex_source") != "gold_scoring":
            raise ProvenanceError(f"{source} requires final_ex_source=gold_scoring")
        if provenance.get("split") != "sft_train":
            raise ProvenanceError(f"{source} requires split=sft_train")
    if require_trainable and source not in TRAINABLE_SOURCES:
        raise ProvenanceError(f"source {source!r} is not trainable")


def ensure_trainable(record: dict) -> None:
    """Validate that a full record is admissible into a training manifest."""
    provenance = record.get("provenance")
    if not isinstance(provenance, dict):
        raise ProvenanceError("missing provenance")
    validate_provenance(provenance, require_trainable=True)
    if record.get("final_ex") is not True and provenance.get("final_ex") is not True:
        raise ProvenanceError("trainable record must have final_ex=True")
    if record.get("final_sql") in (None, ""):
        raise ProvenanceError("trainable record must have final_sql")


# ---------------------------------------------------------------------------
# rollout helpers
# ---------------------------------------------------------------------------
def provenance_for_state(
    *,
    state,
    source: str,
    generation_mode: str,
    split: str,
    model_path: str,
    model_fingerprint: str,
    runtime_fingerprint: str,
    retrieval_version: str,
    prompt_mode: str,
    observation_mode: str,
    sample_index: int = 0,
    parent_trajectory_id: str | None = None,
    repair_round: int = 0,
) -> dict:
    """Build provenance for one EpisodeState produced by a rollout.

    The outcome is not known yet, so success sources are stored as their
    failure counterpart until ``finalize_provenance`` is called after scoring.
    """
    initial_source = source
    if source == RAW_SUCCESS:
        initial_source = RAW_FAILURE
    elif source == SELF_REPAIR_SUCCESS:
        initial_source = SELF_REPAIR_FAILURE
    return build_provenance(
        source=initial_source,
        generation_mode=generation_mode,
        model_path=model_path,
        model_fingerprint=model_fingerprint,
        runtime_fingerprint=runtime_fingerprint,
        retrieval_version=retrieval_version,
        prompt_mode=prompt_mode,
        observation_mode=observation_mode,
        split=split,
        final_ex=False,  # filled after Gold scoring
        final_ex_source="unscored",
        sample_index=sample_index,
        parent_trajectory_id=parent_trajectory_id,
        repair_round=repair_round,
    )


def finalize_provenance(provenance: dict, *, final_ex: bool, final_sql: str | None,
                        final_ex_source: str | None = None) -> dict:
    """Fill the outcome after Gold scoring, normalise source, and re-validate."""
    if final_ex and not final_sql:
        raise ProvenanceError("final_ex=True requires a non-empty final_sql")
    if final_ex_source is not None and final_ex_source not in FINAL_EX_SOURCES:
        raise ProvenanceError(f"unknown final_ex_source: {final_ex_source!r}")
    source = provenance.get("source")
    if source in (RAW_SUCCESS, RAW_FAILURE):
        provenance["source"] = RAW_SUCCESS if final_ex else RAW_FAILURE
    elif source in (SELF_REPAIR_SUCCESS, SELF_REPAIR_FAILURE):
        provenance["source"] = SELF_REPAIR_SUCCESS if final_ex else SELF_REPAIR_FAILURE
    elif source not in (GOLD_CONSTRUCTED, LEGACY_MODEL_REPLAY, UNMARKED):
        raise ProvenanceError(f"cannot finalize source {source!r}")
    provenance["final_ex"] = bool(final_ex)
    provenance["final_ex_source"] = (
        final_ex_source or ("gold_scoring" if final_sql else "unscored"))
    provenance["trainable"] = (
        provenance["final_ex"]
        and provenance["source"] in TRAINABLE_SOURCES
        and not provenance.get("gold_used_for_construction")
        and not provenance.get("tool_replay")
    )
    validate_provenance(provenance)
    return provenance


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------
def audit_records(records: Iterable[dict], *, provenance_source: dict | None = None,
                  tool_replay: bool = False) -> dict:
    """Count provenance sources and validate every record.

    ``provenance_source`` optionally maps ``question_id``/``id`` to a record
    from the old ``full_tool_unique.jsonl`` so legacy data can be classified
    more precisely than ``source_type`` alone.
    """
    from collections import Counter

    provenance_source = provenance_source or {}
    counts = Counter()
    trainable = Counter()
    final_ex = Counter()
    invalid = []
    unmarked = []
    matched_legacy = 0
    total = 0
    for index, record in enumerate(records):
        total += 1
        prov = record.get("provenance")
        if prov is None:
            key = record.get("question_id") or record.get("id")
            old = provenance_source.get(key) if key else None
            if old is not None:
                matched_legacy += 1
                prov = build_legacy_provenance(
                    origin=(old.get("provenance") or {}).get("origin"),
                    source_type=record.get("source_type"),
                    trajectory_id=record.get("trajectory_id"),
                    question_id=record.get("question_id"),
                    matched=True,
                    tool_replay=tool_replay,
                )
            else:
                prov = build_legacy_provenance(
                    source_type=record.get("source_type"),
                    trajectory_id=record.get("trajectory_id"),
                    question_id=record.get("question_id"),
                    matched=False,
                    tool_replay=tool_replay,
                )
        try:
            validate_provenance(prov)
        except ProvenanceError as exc:
            invalid.append({"index": index, "id": record.get("id") or record.get("trajectory_id"),
                            "reason": str(exc)})
            counts[UNMARKED] += 1
            continue
        source = prov["source"]
        counts[source] += 1
        if source in TRAINABLE_SOURCES:
            trainable[source] += 1
        else:
            unmarked.append({"index": index, "id": record.get("id") or record.get("trajectory_id"),
                             "source": source})
        if record.get("final_ex") is True:
            final_ex[source] += 1
    return {
        "total": total,
        "matched_legacy_provenance": matched_legacy,
        "source_counts": dict(counts),
        "trainable_counts": dict(trainable),
        "trainable_total": sum(trainable.values()),
        "final_ex_counts": dict(final_ex),
        "invalid_count": len(invalid),
        "invalid": invalid[:50],
        "audit_only_count": len(unmarked),
        "audit_only": unmarked[:50],
    }


def write_json(path, payload) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n",
                          encoding="utf-8")
