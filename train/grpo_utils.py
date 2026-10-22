"""Shared causal scoring and unpadded sampling for single-device and DDP GRPO."""
import torch
from transformers import EosTokenCriteria, StopStringCriteria, StoppingCriteria


class CompletionLengths(StoppingCriteria):
    """Remember each sequence's first stop, including EOS or the stop string.

    Recording the actual generation step also works when pad == eos, or when a
    stop string ends a sequence before any EOS. Inspecting trailing token IDs
    alone cannot distinguish those cases.
    """

    def __init__(self, tokenizer, eos_token_id):
        self.stop_string = StopStringCriteria(tokenizer, ["</answer>"])
        self.eos = EosTokenCriteria(eos_token_id) if eos_token_id is not None else None
        self.lengths = None

    def __call__(self, input_ids, scores, **kwargs):
        stopped = self.stop_string(input_ids, scores)
        if self.eos is not None:
            stopped = stopped | self.eos(input_ids, scores)
        if self.lengths is None:
            self.lengths = torch.zeros_like(stopped, dtype=torch.long)
        first_stop = stopped & self.lengths.eq(0)
        self.lengths = torch.where(
            first_stop, torch.full_like(self.lengths, input_ids.shape[1]), self.lengths
        )
        return stopped


def sequence_logprob(model, ids, start, adapter, requires_grad):
    """Score response token t using logits at t-1, with no padding in ids."""
    if ids.ndim != 1 or not 0 < start < ids.numel():
        raise ValueError("expected a nonempty prompt and completion in a 1D token sequence")
    model.set_adapter(adapter)
    keep = int(ids.numel() - start)
    with torch.set_grad_enabled(requires_grad):
        output = model(
            input_ids=ids.unsqueeze(0),
            attention_mask=torch.ones_like(ids).unsqueeze(0),
            logits_to_keep=keep + 1,
        )
        # The extra position is the last prompt token, which predicts the first
        # completion token. The final input position predicts beyond the target.
        logits = output.logits[0, :-1]
        targets = ids[start:]
        return torch.log_softmax(logits.float(), dim=-1).gather(1, targets.unsqueeze(1)).squeeze(1)


def generate_group(model, tokenizer, prompt, n, max_new_tokens, temperature,
                   max_prompt_tokens, device="npu:0"):
    enc = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=max_prompt_tokens
    ).to(device)
    prompt_len = enc["input_ids"].shape[1]
    eos_token_id = model.generation_config.eos_token_id
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id
    completion_lengths = CompletionLengths(tokenizer, eos_token_id)
    with torch.no_grad():
        output = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            num_return_sequences=n,
            stopping_criteria=[completion_lengths],
            pad_token_id=tokenizer.pad_token_id,
        )
    ends = completion_lengths.lengths.cpu().tolist()
    samples = []
    for row, end in zip(output, ends):
        # A zero end means generation hit its token budget without an early stop.
        ids = row[:end] if end else row
        text = tokenizer.decode(ids[prompt_len:], skip_special_tokens=True)
        samples.append({"ids": ids, "start": prompt_len, "text": text})
    return samples
