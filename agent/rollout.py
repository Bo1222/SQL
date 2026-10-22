"""Single-active-trajectory HF generation, preserving exact generated token IDs."""
from agent.protocol import STOP_STRINGS


class HFGenerator:
    def __init__(self, model, tokenizer, temperature=1.0):
        self.model, self.tokenizer, self.temperature = model, tokenizer, temperature
        self.input_tokens = self.output_tokens = 0

    def __call__(self, ids, max_new_tokens):
        import torch
        from transformers import EosTokenCriteria, StopStringCriteria, StoppingCriteriaList
        device = next(self.model.parameters()).device
        inputs = torch.tensor([ids], dtype=torch.long, device=device)
        stops = [StopStringCriteria(self.tokenizer, STOP_STRINGS)]
        eos = self.model.generation_config.eos_token_id or self.tokenizer.eos_token_id
        if eos is not None:
            stops.append(EosTokenCriteria(eos))
        kwargs = {"do_sample": self.temperature > 0, "top_p": 1.0, "top_k": 0,
                  "repetition_penalty": 1.0, "no_repeat_ngram_size": 0, "min_new_tokens": 0}
        if self.temperature > 0:
            kwargs["temperature"] = self.temperature
        self.model.eval()
        self.input_tokens += len(ids)
        with torch.no_grad():
            result = self.model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                         max_new_tokens=max_new_tokens, use_cache=True,
                                         stopping_criteria=StoppingCriteriaList(stops),
                                         pad_token_id=self.tokenizer.pad_token_id, **kwargs)
        output = result[0, len(ids):].tolist()
        self.output_tokens += len(output)
        return output
