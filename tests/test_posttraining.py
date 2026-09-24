"""Offline safety/correctness checks for the separate inference harness."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from posttraining.runtime import (
    Runtime, longest_token_run, read_config, system_prompt, validate_panel,
)


class PosttrainingTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.panel = json.loads((self.root / "posttraining/eval_scenarios_v1.json").read_text(encoding="utf-8"))

    def test_panel_is_dev_and_languages_are_paired(self):
        validate_panel(self.panel)
        scenarios = self.panel["scenarios"]
        self.assertEqual(len(scenarios), 20)
        self.assertEqual(sum(len(s["turns"]) for s in scenarios) * 3, 108)
        families = {}
        for scenario in scenarios:
            families.setdefault(scenario["family"], set()).add(scenario["language"])
        self.assertEqual(len(families), 10)
        self.assertTrue(all(languages == {"en", "fr"} for languages in families.values()))

    def test_refuses_test_split_and_duplicate_ids(self):
        changed = copy.deepcopy(self.panel)
        changed["split"] = "test"
        with self.assertRaises(ValueError):
            validate_panel(changed)
        changed = copy.deepcopy(self.panel)
        changed["scenarios"].append(changed["scenarios"][0])
        with self.assertRaises(ValueError):
            validate_panel(changed)

    def test_expected_answer_is_not_in_prompt(self):
        for scenario in self.panel["scenarios"]:
            prompt = system_prompt(scenario["persona"], scenario["language"])
            self.assertIn(scenario["persona"], prompt)
            for turn in scenario["turns"]:
                self.assertNotIn(turn["expect"], prompt)

    def test_refuses_unpinned_model_revision(self):
        config = json.loads((self.root / "configs/ministral_p0.json").read_text())
        config["revision"] = "main"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                read_config(path)

    def test_token_loop_metric_handles_subword_runs(self):
        self.assertEqual(longest_token_run([]), 0)
        self.assertEqual(longest_token_run([1, 1, 1, 2, 2, 1]), 3)

    def test_context_overflow_is_rejected_not_truncated(self):
        import torch

        class Tokenizer:
            def apply_chat_template(self, *args, **kwargs):
                return {"input_ids": torch.tensor([[1, 2, 3, 4]])}

        runtime = Runtime.__new__(Runtime)
        runtime.tokenizer = Tokenizer()
        runtime.config = {"max_context_tokens": 3}
        with self.assertRaisesRegex(ValueError, "silent truncation"):
            runtime.encode([{"role": "user", "content": "hello"}])


if __name__ == "__main__":
    unittest.main()
