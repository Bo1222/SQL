import copy
import json
from pathlib import Path

import pytest

from e1_alignment.provenance import (
    GOLD_CONSTRUCTED,
    LEGACY_MODEL_REPLAY,
    RAW_FAILURE,
    RAW_SUCCESS,
    SELF_REPAIR_FAILURE,
    SELF_REPAIR_SUCCESS,
    ProvenanceError,
    audit_records,
    build_legacy_provenance,
    build_provenance,
    ensure_trainable,
    finalize_provenance,
)


def make_provenance(source=RAW_SUCCESS, **overrides):
    payload = dict(
        source=source,
        generation_mode="n1_greedy",
        model_path="artifacts/checkpoints/step_600_merged",
        model_fingerprint="m" * 64,
        runtime_fingerprint="r" * 64,
        retrieval_version="E1",
        prompt_mode="query",
        observation_mode="compact",
        split="sft_train",
        final_ex=True,
        final_ex_source="gold_scoring",
        sample_index=0,
        parent_trajectory_id=None,
        repair_round=0,
        gold_used_for_construction=False,
        tool_replay=False,
    )
    payload.update(overrides)
    return build_provenance(**payload)


def make_record(provenance, final_ex=True, final_sql="SELECT 1"):
    return {"id": "q1", "final_ex": final_ex, "final_sql": final_sql,
            "provenance": provenance}


def test_raw_success_is_trainable():
    prov = make_provenance()
    assert prov["trainable"] is True
    ensure_trainable(make_record(prov))


def test_raw_success_rejects_false_final_ex():
    with pytest.raises(ProvenanceError):
        make_provenance(final_ex=False)


def test_self_repair_requires_parent_and_round():
    with pytest.raises(ProvenanceError):
        make_provenance(source=SELF_REPAIR_SUCCESS)
    prov = make_provenance(source=SELF_REPAIR_SUCCESS,
                           generation_mode="prefix_repair",
                           parent_trajectory_id="parent-1", repair_round=1)
    assert prov["trainable"] is True


def test_gold_constructed_is_not_trainable():
    prov = make_provenance(source=GOLD_CONSTRUCTED, generation_mode="legacy_replay",
                           final_ex_source="inherited_legacy", split=None,
                           model_path=None, model_fingerprint=None,
                           runtime_fingerprint=None, prompt_mode="legacy",
                           observation_mode="legacy", retrieval_version="legacy",
                           final_ex=None)
    assert prov["trainable"] is False
    with pytest.raises(ProvenanceError):
        ensure_trainable(make_record(prov))


def test_tool_replay_is_rejected_for_success_source():
    with pytest.raises(ProvenanceError):
        make_provenance(tool_replay=True)
    replay = build_legacy_provenance(origin="model_rollout_replay")
    assert replay["trainable"] is False
    with pytest.raises(ProvenanceError):
        ensure_trainable(make_record(replay))


def test_finalize_normalises_failure_to_success():
    prov = make_provenance(source=RAW_FAILURE, final_ex=False,
                           final_ex_source="unscored")
    finalize_provenance(prov, final_ex=True, final_sql="SELECT 1")
    assert prov["source"] == RAW_SUCCESS
    assert prov["trainable"] is True
    assert prov["final_ex_source"] == "gold_scoring"


def test_finalize_keeps_failure_audit_only():
    prov = make_provenance(source=SELF_REPAIR_FAILURE, final_ex=False,
                           final_ex_source="unscored",
                           generation_mode="prefix_repair",
                           parent_trajectory_id="parent-1", repair_round=1)
    finalize_provenance(prov, final_ex=False, final_sql="SELECT bad")
    assert prov["source"] == SELF_REPAIR_FAILURE
    assert prov["trainable"] is False


def test_finalize_rejects_success_without_sql():
    prov = make_provenance(source=RAW_FAILURE, final_ex=False,
                           final_ex_source="unscored")
    with pytest.raises(ProvenanceError):
        finalize_provenance(prov, final_ex=True, final_sql=None)


def test_score_error_is_audit_only():
    prov = make_provenance(source=RAW_FAILURE, final_ex=False,
                           final_ex_source="unscored")
    finalize_provenance(prov, final_ex=False, final_sql="SELECT 1",
                        final_ex_source="score_error")
    assert prov["final_ex_source"] == "score_error"
    assert prov["trainable"] is False


def test_legacy_classification():
    gold = build_legacy_provenance(origin="constructed_recovery_gold")
    replay = build_legacy_provenance(origin="model_rollout_replay")
    assert gold["source"] == GOLD_CONSTRUCTED
    assert replay["source"] == LEGACY_MODEL_REPLAY
    assert gold["trainable"] is False
    assert replay["trainable"] is False


def test_audit_records_counts_and_join(tmp_path):
    legacy = [
        {"trajectory_id": "t1", "question_id": "q1", "source_type": "autonomous_success",
         "final_ex": True},
        {"trajectory_id": "t2", "question_id": "q2", "source_type": "autonomous_recovery",
         "final_ex": True},
    ]
    source = {"q1": {"id": "q1", "provenance": {"origin": "model_rollout_replay"}},
              "q2": {"id": "q2", "provenance": {"origin": "constructed_recovery_gold"}}}
    result = audit_records(legacy, provenance_source=source)
    assert result["total"] == 2
    assert result["matched_legacy_provenance"] == 2
    assert result["source_counts"][LEGACY_MODEL_REPLAY] == 1
    assert result["source_counts"][GOLD_CONSTRUCTED] == 1
    assert result["trainable_total"] == 0
    assert result["invalid_count"] == 0


def test_audit_records_flags_unmarked():
    result = audit_records([{"final_ex": True}])
    assert result["source_counts"]["unmarked"] == 1
    assert result["trainable_total"] == 0
