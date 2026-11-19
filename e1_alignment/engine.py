"""vLLM batch engine for E1 alignment collection, continuation and replay."""
import json
from pathlib import Path


class VLLMBatchEngine:
    """Thin wrapper around vLLM offline generation.

    ``generate`` receives a list of token-id contexts plus one max-new-token
    budget per context.  The budgets can differ (an episode near its context
    limit gets fewer tokens), so a SamplingParams object is built per request.
    """

    def __init__(self, model_path, max_model_len=6144, gpu_memory_utilization=0.5,
                 enforce_eager=True, dtype="bfloat16"):
        from vllm import LLM, SamplingParams

        self.SamplingParams = SamplingParams
        self.model_path = str(model_path)
        self.max_model_len = max_model_len
        self.llm = LLM(model=self.model_path, dtype=dtype,
                       gpu_memory_utilization=gpu_memory_utilization,
                       enforce_eager=enforce_eager, max_model_len=max_model_len,
                       trust_remote_code=False)

    def generate(self, contexts, max_new_tokens, temperature=0.0, top_p=1.0, seeds=None):
        """Return one token-id list per context."""
        if not contexts:
            return []
        budgets = [int(m) for m in max_new_tokens]
        if any(b <= 0 for b in budgets):
            raise ValueError("max_new_tokens must be positive")
        params = []
        for index, budget in enumerate(budgets):
            kwargs = {"max_tokens": budget, "temperature": float(temperature), "top_p": float(top_p)}
            if seeds is not None:
                kwargs["seed"] = int(seeds[index])
            params.append(self.SamplingParams(**kwargs))
        prompts = [{"prompt_token_ids": list(ids)} for ids in contexts]
        try:
            outputs = self.llm.generate(prompts=prompts, sampling_params=params, use_tqdm=False)
        except (TypeError, ValueError):
            # Older vLLM builds reject a list of SamplingParams; same values are
            # still applied when budgets happen to be equal.
            if len(set(budgets)) != 1:
                raise
            outputs = self.llm.generate(prompts=prompts, sampling_params=params[0], use_tqdm=False)
        results = []
        for output in outputs:
            if not output.outputs:
                results.append([])
            else:
                results.append(list(output.outputs[0].token_ids))
        return results


def engine_from_args(args):
    return VLLMBatchEngine(args.model, max_model_len=args.max_model_len,
                           gpu_memory_utilization=args.gpu_memory_utilization)
