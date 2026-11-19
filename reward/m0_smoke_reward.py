"""M0-only placeholder reward.

Purpose: validate the veRL + FSDP2 + vLLM-Ascend training loop end to end.
It deliberately does NOT implement BIRD EX (that is M4). The value depends on
response length parity so that groups contain mixed rewards and the actor
receives a non-zero gradient.
"""


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    text = solution_str or ""
    return 1.0 if len(text) % 2 == 0 else 0.0
