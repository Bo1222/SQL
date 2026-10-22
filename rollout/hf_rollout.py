"""Minimal HuggingFace rollout on Ascend NPU."""
import time

import torch
import torch_npu  # noqa: F401
from transformers import AutoModelForCausalLM, AutoTokenizer


class HFRollout:
    def __init__(self, model_path, device="npu:0", max_new_tokens=200, max_prompt_tokens=4096, adapter=""):
        torch.npu.set_device(0)
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.max_prompt_tokens = max_prompt_tokens
        self.tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, trust_remote_code=True
        )
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model = self.model.to(device)
        self.model.eval()
        self.model.generation_config.pad_token_id = self.tok.pad_token_id

    def generate(self, prompts, batch_size=4, temperature=0.0, n=1, progress=None):
        outputs = []
        started = time.time()
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start:start + batch_size]
            enc = self.tok(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_prompt_tokens,
            ).to(self.device)
            with torch.no_grad():
                generated = self.model.generate(
                    **enc,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=temperature > 0,
                    temperature=max(temperature, 1e-5),
                    top_p=0.95,
                    num_return_sequences=n,
                    stop_strings=["</answer>"],
                    tokenizer=self.tok,
                )
            new_tokens = generated[:, enc["input_ids"].shape[1]:]
            texts = self.tok.batch_decode(new_tokens, skip_special_tokens=True)
            if n == 1:
                outputs.extend(texts)
            else:
                for idx in range(len(batch)):
                    outputs.append(texts[idx * n:(idx + 1) * n])
            if progress is not None:
                progress(min(start + len(batch), len(prompts)), len(prompts), time.time() - started)
        return outputs
