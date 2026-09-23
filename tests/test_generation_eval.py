"""Held-out evaluation must preserve token inputs and report degenerate replies."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from datasets import Dataset
from test_model import tiny_model, tokenizer_fixture

import evaluate_generation
from conversation import write_dataset_metadata
from workflow import ModelConfig


class GenerationEvaluationTests(unittest.TestCase):
    def test_repetition_and_short_response_denominators(self):
        empty = evaluate_generation.repetition_metrics("")
        short = evaluate_generation.repetition_metrics("Hello friend")
        loop = evaluate_generation.repetition_metrics("door door door door door")
        self.assertTrue(empty["empty"])
        self.assertIsNone(short["repeated_word_4gram_fraction"])
        self.assertEqual(loop["repeated_word_4gram_fraction"], 0.5)
        rows = [
            dict(r, generated_tokens=0, stopped_on_eos=True)
            for r in (empty, short, loop)
        ]
        summary = evaluate_generation.summarize(rows)
        self.assertEqual(summary["responses_eligible_for_4gram_metric"], 1)
        self.assertEqual(summary["mean_repeated_word_4gram_fraction"], 0.5)
        self.assertEqual(summary["empty_response_fraction"], 1 / 3)

    def test_token_generation_preserves_prefix_and_reports_eos(self):
        model = tiny_model().train()
        original = [2, 3, 1]
        seen = []

        def forward(ids):
            seen.append(ids.tolist()[0])
            logits = torch.zeros(1, ids.size(1), model.vocab)
            logits[:, -1, 1] = 10
            return {"main": logits}

        with patch.object(model, "forward", side_effect=forward):
            tokens = model.generate_token_ids(
                original, stop_token=1, temperature=0, max_new_tokens=5
            )
        self.assertEqual(seen, [original])
        self.assertEqual(tokens, [1])
        self.assertTrue(model.training)

    def test_evaluation_saves_exact_ids_and_reproducible_panel(self):
        torch.set_num_threads(1)
        tokenizer = tokenizer_fixture()
        config = ModelConfig(
            max_len=32,
            d_model=32,
            n_layers=2,
            n_heads=4,
            d_ff=64,
            dropout=0,
            gradient_checkpointing=False,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint_dir = root / "model"
            checkpoint_dir.mkdir()
            config.save(checkpoint_dir / "config.json")
            tokenizer.save_pretrained(checkpoint_dir / "tokenizer")
            torch.save(
                config.build(len(tokenizer)).state_dict(), checkpoint_dir / "best.pt"
            )
            data = root / "validation"
            contexts = [[2, 3, 1], [2, 4, 1], [8, 2, 3, 1]]
            Dataset.from_dict(
                {"input_ids": contexts, "labels": [[5, 1], [6, 1], [5, 7, 1]]}
            ).save_to_disk(data)
            write_dataset_metadata(data, tokenizer)
            panels = []
            for name in ("first", "second"):
                command = [
                    "evaluate_generation.py",
                    "--checkpoint",
                    str(checkpoint_dir / "best.pt"),
                    "--data-dir",
                    str(data),
                    "--output-dir",
                    str(root / name),
                    "--samples",
                    "2",
                    "--seeds",
                    "42",
                    "43",
                    "--max-new-tokens",
                    "3",
                    "--device",
                    "cpu",
                ]
                with (
                    patch.object(sys, "argv", command),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    evaluate_generation.main()
                rows = [
                    json.loads(line)
                    for line in (root / name / "generations.jsonl")
                    .read_text()
                    .splitlines()
                ]
                self.assertEqual(len(rows), 4)
                for row in rows:
                    self.assertEqual(row["context_ids"], contexts[row["dataset_index"]])
                    self.assertEqual(
                        row["generated_tokens"],
                        len(row["generated_ids"]) - row["stopped_on_eos"],
                    )
                summary = json.loads((root / name / "summary.json").read_text())
                self.assertEqual(summary["generations"], 4)
                self.assertTrue((root / name / "human_review.csv").is_file())
                panels.append(rows)
            self.assertEqual(*panels)


if __name__ == "__main__":
    unittest.main()
