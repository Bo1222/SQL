"""Durable quota reservations, including failed and interrupted requests."""
import fcntl
import json
import os
from pathlib import Path
from e1_alignment.common import canonical_hash


def reserve(path, kind, identity, policy):
    if kind not in ("gold", "model"):
        raise ValueError("unknown_acquisition_kind")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0)
        ledger = [json.loads(line) for line in stream if line.strip()]
        if any(r["kind"] == kind and r["identity"] == identity for r in ledger):
            raise ValueError("already_reserved_no_retry")
        used = [r for r in ledger if r["kind"] == kind]
        cap = policy["gold_attempt_cap" if kind == "gold" else "model_state_cap"]
        if len(used) >= cap:
            raise ValueError("acquisition_quota_exhausted")
        row = {"kind": kind, "identity": identity, "ordinal": len(used),
               "reserved_candidates": 1 if kind == "gold" else policy["model_candidates_per_state"],
               "policy_sha256": canonical_hash(policy)}
        if any(r["policy_sha256"] != row["policy_sha256"] for r in ledger):
            raise ValueError("quota_policy_changed")
        stream.seek(0, 2)
        stream.write(json.dumps(row, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
        return row


def original_identity(record):
    # Renaming the parent version or candidate ID cannot reset the revision budget.
    origin = record.get("origin", record.get("history_origin"))
    return origin["sha256"]
