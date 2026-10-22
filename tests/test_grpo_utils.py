"""Independent checks against causal LM loss and real HF generation semantics."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from peft import LoraConfig, get_peft_model
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import LogitsProcessor, PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from train.grpo_utils import generate_group, sequence_logprob


def tiny_model(vocab_size=32):
    return Qwen2ForCausalLM(Qwen2Config(
        vocab_size=vocab_size, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        attention_dropout=0.0, attn_implementation="eager",
    )).eval()


def byte_tokenizer(pad_equals_eos=False):
    vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2, "<end2>": 3}
    vocab.update({s: i + 4 for i, s in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))})
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", eos_token="<eos>",
        pad_token="<eos>" if pad_equals_eos else "<pad>",
        additional_special_tokens=["<end2>"],
        model_input_names=["input_ids", "attention_mask"],
    )


class ScriptedTokens(LogitsProcessor):
    def __init__(self, schedules, prompt_len):
        self.schedules = schedules
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores):
        step = input_ids.shape[1] - self.prompt_len
        scores.fill_(-float("inf"))
        for row, schedule in enumerate(self.schedules):
            token = schedule[step] if step < len(schedule) else 0
            scores[row, token] = 0
        return scores


class GrpoScoringTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.model = get_peft_model(tiny_model(), LoraConfig(
            r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"
        ), adapter_name="policy").eval()

    def test_logprob_matches_builtin_causal_loss_and_gradients(self):
        ids = torch.tensor([3, 5, 7, 11, 13, 17])
        for start in [1, 3, 5]:
            with self.subTest(start=start):
                labels = ids[None].clone()
                labels[:, :start] = -100
                logp = sequence_logprob(self.model, ids, start, "policy", True)
                loss = self.model(input_ids=ids[None], labels=labels).loss
                torch.testing.assert_close(-logp.mean(), loss)
                params = [p for p in self.model.parameters() if p.requires_grad]
                actual_grad = torch.autograd.grad(-logp.mean(), params, allow_unused=True)
                expected_grad = torch.autograd.grad(loss, params, allow_unused=True)
                for actual, expected in zip(actual_grad, expected_grad):
                    if expected is not None:
                        torch.testing.assert_close(actual, expected)

    def test_invalid_prompt_or_completion_rejected(self):
        ids = torch.tensor([3, 5])
        for start in [0, 2]:
            with self.assertRaises(ValueError):
                sequence_logprob(self.model, ids, start, "policy", False)


class GrpoGenerationTest(unittest.TestCase):
    def test_true_stops_preserve_eos_and_exclude_padding(self):
        for shared_pad_eos in [False, True]:
            with self.subTest(pad_equals_eos=shared_pad_eos):
                tok = byte_tokenizer(shared_pad_eos)
                model = tiny_model(len(tok))
                model.generation_config.eos_token_id = [tok.eos_token_id, tok.convert_tokens_to_ids("<end2>")]
                model.generation_config.pad_token_id = tok.pad_token_id
                schedules = [tok.encode("ok</answer>"),
                             tok.encode("yes") + [tok.eos_token_id],
                             tok.encode("no") + [tok.convert_tokens_to_ids("<end2>")],
                             tok.encode("abcdefghijklmn")]
                real_generate = model.generate
                forced = ScriptedTokens(schedules, len(tok.encode("Q")))
                def controlled_generate(**kwargs):
                    return real_generate(**kwargs, logits_processor=[forced])
                with patch.object(model, "generate", side_effect=controlled_generate):
                    samples = generate_group(model, tok, "Q", 4, 14, 1.0, 32, "cpu")
                self.assertEqual(len(samples), 4)
                for sample, expected in zip(samples, schedules):
                    self.assertEqual(sample["ids"][sample["start"]:].tolist(), expected)
                self.assertEqual(samples[0]["text"], "ok</answer>")
                self.assertEqual(samples[1]["text"], "yes")
                self.assertEqual(samples[2]["text"], "no")

    def test_nonleader_tracking_does_not_import_or_init_wandb(self):
        from train.grpo_tracking import training_logs
        with training_logs(SimpleNamespace(), 1, 4) as result:
            self.assertEqual(result, (None, None))


if __name__ == "__main__":
    unittest.main()
