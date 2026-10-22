"""Store original token boundaries; observations are context, never policy targets."""
from dataclasses import dataclass, field


@dataclass
class Segment:
    kind: str
    ids: list
    text: str


@dataclass
class Trajectory:
    segments: list = field(default_factory=list)
    final_sql: str | None = None
    first_sql: str | None = None
    stop_reason: str = ""
    tool_calls: int = 0
    invalid_actions: int = 0
    actions: list = field(default_factory=list)
    source: str = "on_policy"
    diagnostics: dict = field(default_factory=dict)

    @property
    def input_ids(self):
        return [i for segment in self.segments for i in segment.ids]

    @property
    def loss_mask(self):
        return [int(segment.kind == "action") for segment in self.segments for _ in segment.ids]

    @property
    def generated_tokens(self):
        return sum(len(s.ids) for s in self.segments if s.kind == "action")

    def to_dict(self):
        from dataclasses import asdict
        return {"trajectory_version": 1, **asdict(self)}

    @classmethod
    def from_dict(cls, record):
        record = {k: v for k, v in record.items() if k != "trajectory_version"}
        record["segments"] = [Segment(**s) for s in record["segments"]]
        if not record["segments"] or record["segments"][0].kind != "prompt":
            raise ValueError("trajectory must begin with a prompt")
        if any(s.kind not in {"prompt", "action", "observation"} for s in record["segments"]):
            raise ValueError("unknown token provenance")
        return cls(**record)


def masked_logprob(model, ids, mask, adapter=None, requires_grad=False):
    import torch
    ids = torch.as_tensor(ids, dtype=torch.long, device=next(model.parameters()).device)
    mask = torch.as_tensor(mask, dtype=torch.bool, device=ids.device)
    if ids.ndim != 1 or mask.shape != ids.shape or mask[0] or not mask.any():
        raise ValueError("aligned 1D ids/mask required, with an unscored prefix and nonempty actions")
    if adapter is not None:
        model.set_adapter(adapter)
    positions = mask.nonzero(as_tuple=True)[0]
    with torch.set_grad_enabled(requires_grad):
        # Compute vocabulary logits only where an action token is predicted.
        output = model(input_ids=ids[None], attention_mask=torch.ones_like(ids)[None],
                       use_cache=False, logits_to_keep=positions - 1)
        logits = output.logits[0]
        return logits.float().log_softmax(-1).gather(1, ids[positions, None]).squeeze(1)
