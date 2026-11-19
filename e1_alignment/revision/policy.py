import json
from pathlib import Path
from e1_alignment.common import canonical_hash
from e1_alignment.revision import RULE_VERSION

DEFAULT = Path("configs/data/m6_08k_v1.json")


def load_policy(path=DEFAULT):
    policy = json.loads(Path(path).read_text())
    if policy.get("format") != "sqlrl-m6-08k-policy-v1" or policy.get("rule_version") != RULE_VERSION:
        raise ValueError("unsupported_data_policy")
    return policy


def policy_hash(policy):
    return canonical_hash(policy)
