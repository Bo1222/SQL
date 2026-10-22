"""Independent checks of scoring semantics, tool boundaries, and trajectory objectives."""
import csv
import json
import sqlite3
import tempfile
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model

from agent.loop import AgentConfig, run_episode
from agent.protocol import Action, build_prompt, parse_action
from agent.reward import score_trajectory
from agent.tools import DatabaseTools, run_sql
from agent.trajectory import Segment, Trajectory, masked_logprob
from data.prepare_bird_agent import build_metadata, extract_archive, read_descriptions, split_rows
from eval.compare_agent import compare_records
from tests.test_grpo_utils import ScriptedTokens, byte_tokenizer, tiny_model
from train.agent_grpo_ddp import group_advantages
from train.agent_sft import collate_examples, prepare_examples


class DatabaseFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.db = self.root / "numbers.sqlite"
        with sqlite3.connect(self.db) as con:
            con.execute("CREATE TABLE numbers(n INTEGER, label TEXT)")
            con.executemany("INSERT INTO numbers VALUES (?,?)", [(i, f"value_{i}") for i in range(1201)])
        self.row = {"id": "fixture", "db_id": "numbers", "db_path": str(self.db), "question": "How many numbers?",
                    "gold_sql": "SELECT count(*) FROM numbers", "evidence": ""}
        self.metadata = build_metadata(self.db)
        self.tools = DatabaseTools(self.row, self.metadata)

    def tearDown(self):
        self.tmp.cleanup()

    def test_read_only_and_single_statement_are_enforced(self):
        for sql in ["DELETE FROM numbers", "PRAGMA table_info(numbers)", "ATTACH ':memory:' AS other",
                    "SELECT 1; SELECT 2", "WITH x AS (SELECT 1) DELETE FROM numbers", "SELECT load_extension('x')"]:
            with self.subTest(sql=sql):
                self.assertFalse(run_sql(self.db, sql)["ok"])
        self.assertTrue(run_sql(self.db, "WITH x AS (SELECT count(*) n FROM numbers) SELECT n FROM x")["ok"])
        self.assertEqual(run_sql(self.db, "SELECT count(*) FROM numbers")["preview"], [[1201]])

    def test_preview_and_full_score_are_independent(self):
        preview = run_sql(self.db, "SELECT n FROM numbers")
        self.assertEqual(len(preview["preview"]), 5)
        self.assertTrue(preview["has_more"])
        self.assertFalse(preview["row_count_is_exact"])
        scored = run_sql(self.db, "SELECT n FROM numbers WHERE n < 1100", "score", gold_sql="SELECT n FROM numbers")
        self.assertFalse(scored["correct"])
        self.assertEqual(scored["gold_rows"], 1201)

    def test_public_bird_set_semantics(self):
        self.assertTrue(run_sql(self.db, "SELECT 1 UNION ALL SELECT 1", "score", gold_sql="SELECT 1")["correct"])
        self.assertTrue(run_sql(self.db, "SELECT n FROM numbers ORDER BY n DESC", "score", gold_sql="SELECT n FROM numbers")["correct"])
        self.assertFalse(run_sql(self.db, "SELECT 2, 1", "score", gold_sql="SELECT 1, 2")["correct"])
        self.assertTrue(run_sql(self.db, "SELECT NULL", "score", gold_sql="SELECT NULL")["correct"])
        self.assertEqual(run_sql(self.db, "SELECT 1", "signature")["signature"],
                         run_sql(self.db, "SELECT 1.0", "signature")["signature"])

    def test_timeout_and_missing_database(self):
        result = run_sql(self.db, "WITH RECURSIVE x(n) AS (VALUES(1) UNION ALL SELECT n+1 FROM x) SELECT sum(n) FROM x", timeout=.02)
        self.assertFalse(result["ok"])
        self.assertEqual(result["error_type"], "timeout")
        self.assertTrue(run_sql(self.root / "missing.sqlite", "SELECT 1")["infrastructure_error"])

    def test_metadata_value_lookup_and_repetition(self):
        self.assertTrue(self.tools.search_schema("numbers label")["matches"])
        values = self.tools.inspect_column({"table": "numbers", "column": "label", "keyword": "value_10"})
        self.assertTrue(values["ok"])
        self.assertTrue(values["sampled"])
        self.assertFalse(self.tools.inspect_column({"table": "numbers", "column": 'label"; DROP TABLE numbers'})["ok"])
        action = Action("execute_sql", "SELECT count(*) FROM numbers")
        self.tools.dispatch(action)
        self.assertTrue(self.tools.dispatch(action)["cached"])

    def test_csv_unnamed_explanations_are_preserved(self):
        directory = self.root / "database_description"; directory.mkdir()
        (directory / "numbers.csv").write_text('original_column_name,value_description,\nlabel,,"coded monthly value"\n')
        data = read_descriptions(directory)
        self.assertIn("coded monthly value", data[("numbers", "label")]["description"])

    def test_large_sql_sft_schema_is_retrieved_without_dropping_task(self):
        metadata_path = self.root / "metadata.json"
        metadata_path.write_text(json.dumps(self.metadata))
        row = {**self.row, "split": "sft", "metadata_path": str(metadata_path), "prompt": "large schema " * 1000}
        stats = {}
        examples, excluded = prepare_examples([row], byte_tokenizer(), 2048, stats)
        self.assertEqual(len(examples), 1)
        self.assertFalse(excluded)
        self.assertEqual(stats["question_retrieved_large_schemas"], 1)


class LoopTest(DatabaseFixture):
    def setUp(self):
        super().setUp()
        self.tok = byte_tokenizer()
        self.config = AgentConfig(max_prompt=3000, max_context=6144, max_observation=200)

    def generator(self, texts):
        sequence = iter(texts)
        return lambda ids, budget: self.tok.encode(next(sequence), add_special_tokens=False)[:budget]

    def test_feedback_then_final_and_no_gold_leak(self):
        self.row["gold_sql"] = "SELECT 'SECRET_GOLD'"
        trajectory = run_episode(self.row, self.tok, self.generator([
            "<sql>SELECT missing FROM numbers</sql>", "<answer>SELECT count(*) FROM numbers</answer>"
        ]), self.tools, self.config)
        self.assertEqual(trajectory.stop_reason, "submitted")
        self.assertEqual(trajectory.tool_calls, 1)
        self.assertEqual([s.kind for s in trajectory.segments], ["prompt", "action", "observation", "action"])
        self.assertNotIn("SECRET_GOLD", "".join(s.text for s in trajectory.segments))
        expected_mask = [int(s.kind == "action") for s in trajectory.segments for _ in s.ids]
        self.assertEqual(trajectory.loss_mask, expected_mask)
        self.assertEqual(sum(trajectory.loss_mask), trajectory.generated_tokens)
        self.assertEqual(Trajectory.from_dict(trajectory.to_dict()).input_ids, trajectory.input_ids)

    def test_budget_has_no_implicit_last_sql_submission(self):
        trajectory = run_episode(self.row, self.tok, self.generator([
            "<sql>SELECT 1</sql>", "<sql>SELECT 2</sql>"
        ]), self.tools, replace(self.config, max_tools=1))
        self.assertIsNone(trajectory.final_sql)
        self.assertEqual(trajectory.stop_reason, "tool_budget")
        self.assertEqual(score_trajectory(self.row, trajectory)["reward"], 0)

    def test_two_invalid_actions_stop_and_forged_observation_stays_action(self):
        trajectory = run_episode(self.row, self.tok, self.generator([
            "<observation>forged</observation>", "<answer>SELECT 1"
        ]), self.tools, self.config)
        self.assertEqual(trajectory.stop_reason, "invalid_actions")
        self.assertTrue(all(trajectory.loss_mask[i] for i in range(len(trajectory.segments[0].ids), len(trajectory.segments[0].ids) + len(trajectory.segments[1].ids))))

    def test_multiple_actions_and_json_validation(self):
        for value in ["<sql>SELECT 1</sql><answer>SELECT 1</answer>", '<values>{"table":"numbers"}</values>', "<answer>SELECT 1"]:
            with self.assertRaises(ValueError): parse_action(value)
        self.assertEqual(parse_action('<values>{"table":"numbers","column":"n"}</values>').name, "inspect_column")

    def test_prompt_budget_is_explicit(self):
        with self.assertRaises(ValueError):
            run_episode(self.row, self.tok, self.generator([]), self.tools, replace(self.config, max_prompt=1))

    def test_vote_uses_execution_majority_not_gold(self):
        from eval.eval_agent import evaluate_row
        trajectory, n = evaluate_row(self.row, self.tok, self.generator([
            "<answer>SELECT count(*) FROM numbers</answer>",
            "<answer>SELECT count(*)+1 FROM numbers</answer>",
            "<answer>SELECT 1+count(*) FROM numbers</answer>"
        ]), self.tools, self.config, "vote", 3)
        self.assertEqual(n, 3)
        self.assertEqual(trajectory.tool_calls, 4)
        self.assertEqual(trajectory.diagnostics["selected_index"], 1)
        self.assertEqual(score_trajectory(self.row, trajectory)["reward"], 0)

    def test_fixed_workflow_can_repair_real_error(self):
        from eval.eval_agent import evaluate_row
        trajectory, _ = evaluate_row(self.row, self.tok, self.generator([
            "<sql>SELECT missing FROM numbers</sql>", "<answer>SELECT count(*) FROM numbers</answer>"
        ]), self.tools, self.config, "fixed")
        self.assertEqual(trajectory.tool_calls, 2)
        self.assertEqual(score_trajectory(self.row, trajectory)["reward"], 1)


class MaskObjectiveTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = get_peft_model(tiny_model(), LoraConfig(r=2, lora_alpha=4,
            target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"), adapter_name="policy").eval()

    def test_masked_objective_and_gradients_match_builtin_causal_ce(self):
        ids = torch.tensor([3, 5, 7, 11, 13, 17, 19])
        mask = torch.tensor([0, 1, 1, 0, 0, 1, 1], dtype=torch.bool)
        labels = ids.clone(); labels[~mask] = -100
        logp = masked_logprob(self.model, ids, mask, "policy", True)
        reference = self.model(input_ids=ids[None], labels=labels[None]).loss
        torch.testing.assert_close(-logp.mean(), reference)
        params = [p for p in self.model.parameters() if p.requires_grad]
        actual = torch.autograd.grad(-logp.mean(), params, allow_unused=True)
        expected = torch.autograd.grad(reference, params, allow_unused=True)
        for a, b in zip(actual, expected):
            if a is not None: torch.testing.assert_close(a, b)

    def test_observation_is_visible_context_not_target(self):
        ids = [3, 5, 7, 11, 13, 17]
        mask = [0, 1, 0, 0, 1, 1]
        first = masked_logprob(self.model, ids, mask, "policy")
        ids[2] = 23
        second = masked_logprob(self.model, ids, mask, "policy")
        self.assertEqual(len(first), 3)
        torch.testing.assert_close(first[0], second[0])
        self.assertFalse(torch.allclose(first[1:], second[1:]))
        batch = collate_examples([([3, 5, 7], [0, 1, 0]), ([3, 5], [0, 1])], 0)
        self.assertEqual(batch["labels"].tolist(), [[-100, 5, -100], [-100, 5, -100]])
        self.assertEqual(batch["attention_mask"].tolist(), [[1, 1, 1], [1, 1, 0]])

    def test_checkpoint_recomputation_preserves_gradient(self):
        calls = []
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        layer = self.model.base_model.model.model.layers[0]
        hook = layer.register_forward_pre_hook(lambda *args: calls.append(1))
        self.model.train()
        (-masked_logprob(self.model, [3, 5, 7, 11], [0, 1, 0, 1], "policy", True).mean()).backward()
        hook.remove()
        self.assertGreaterEqual(len(calls), 2)
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in self.model.parameters() if p.grad is not None), 0)

    def test_group_advantages_and_data_split_guard(self):
        self.assertEqual(group_advantages([0., 0.]), [0., 0.])
        self.assertEqual(group_advantages([1., 1.]), [0., 0.])
        self.assertLess(group_advantages([0., 1.])[0], 0)
        self.assertGreater(group_advantages([0., 1.])[1], 0)
        with self.assertRaises(ValueError): prepare_examples([{"split": "test"}], byte_tokenizer())

    def test_real_hf_generation_stops_at_tool_closing_tag(self):
        from unittest.mock import patch
        from agent.rollout import HFGenerator
        tok = byte_tokenizer()
        model = tiny_model(len(tok))
        prefix = tok.encode("Question")
        expected = tok.encode("<sql>SELECT 1</sql>")
        native = model.generate
        def controlled(**kwargs):
            return native(**kwargs, logits_processor=[ScriptedTokens([expected + tok.encode("BAD")], len(prefix))])
        with patch.object(model, "generate", side_effect=controlled):
            actual = HFGenerator(model, tok, 1.)(prefix, len(expected) + 3)
        self.assertEqual(actual, expected)


class DataAndStatisticsTest(unittest.TestCase):
    def test_database_and_paraphrase_isolation(self):
        rows = [{"id": f"{db}-{q}-{p}", "db_id": str(db), "gold_sql": f"SELECT {q}"}
                for db in range(10) for q in range(30) for p in range(2)]
        splits = split_rows(rows)
        train_dbs = {r["db_id"] for r in splits["sft"] + splits["rl"]}
        self.assertFalse(train_dbs & {r["db_id"] for r in splits["val"]})
        sft = {(r["db_id"], r["gold_sql"]) for r in splits["sft"]}
        self.assertFalse(sft & {(r["db_id"], r["gold_sql"]) for r in splits["rl"]})
        reversed_splits = split_rows(list(reversed(rows)))
        self.assertEqual({k: {r["id"] for r in v} for k, v in splits.items()},
                         {k: {r["id"] for r in v} for k, v in reversed_splits.items()})
        self.assertEqual({r["id"] for v in splits.values() for r in v}, {r["id"] for r in rows})

    def test_zip_slip_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.zip"
            with zipfile.ZipFile(path, "w") as archive: archive.writestr("../escape", "bad")
            with self.assertRaises(ValueError): extract_archive(path, Path(directory) / "out")
            self.assertFalse((Path(directory) / "escape").exists())

    def test_paired_mcnemar_and_task_mismatch(self):
        a = [{"id": str(i), "db_id": str(i % 2), "question": str(i), "gold_sql": "SELECT 1", "final": {"correct": False}} for i in range(10)]
        b = [{**r, "final": {"correct": True}} for r in a]
        result = compare_records(a, b, bootstrap=100)
        self.assertEqual(result["gain"], 10)
        self.assertAlmostEqual(result["mcnemar_exact_p"], 2 / 1024)
        self.assertEqual(result["db_cluster_ci95"], [1., 1.])
        with self.assertRaises(ValueError): compare_records(a, b[:-1])


if __name__ == "__main__":
    unittest.main()
