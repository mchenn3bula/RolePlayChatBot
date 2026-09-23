"""Offline checks for the notebook-to-script workflow."""

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from datasets import load_from_disk
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

import prepare_data
import train
from mini_deepseek import RolePlayTransformer
from workflow import ModelConfig, load_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def tiny_tokenizer():
    vocabulary = {
        word: index
        for index, word in enumerate(
            ("[UNK]", "[EOS]", "one", "two", "three", "four", "five")
        )
    }
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        eos_token="[EOS]",
        pad_token="[EOS]",
    )


def sample_rows():
    rows = []
    for title in ("alpha", "beta", "gamma", "delta"):
        for month, text in (
            ("Jan", "one two"),
            ("Feb", "two three"),
            ("Mar", "three four"),
            ("Apr", "four five"),
            ("May", "five one"),
        ):
            rows.append(
                {
                    "thread_title": title,
                    "message_timestamp": f"{month} 1, 2020 at 1:00 AM",
                    "message": text,
                }
            )
    return list(reversed(rows))


class DataTests(unittest.TestCase):
    def test_chronological_pairs_and_disjoint_reproducible_split(self):
        threads = prepare_data.group_threads(sample_rows())
        train_titles, validation_titles, test_titles = prepare_data.split_titles(
            threads
        )
        self.assertFalse(set(train_titles) & set(test_titles))
        self.assertFalse(set(train_titles) & set(validation_titles))
        self.assertFalse(set(test_titles) & set(validation_titles))
        self.assertEqual(
            set(train_titles + validation_titles + test_titles), set(threads)
        )
        self.assertEqual(
            (train_titles, validation_titles, test_titles),
            prepare_data.split_titles(threads),
        )
        pairs = list(prepare_data.make_pairs(threads, ["alpha"]))
        self.assertEqual(
            pairs[0],
            {
                "input_text": "one two\n\ntwo three\n\nthree four",
                "target_text": "four five",
            },
        )
        self.assertEqual(pairs[1]["target_text"], "five one")

    def test_tokenization_and_collation_preserve_targets_and_eos(self):
        tokenizer = tiny_tokenizer()
        batch = prepare_data.tokenize_batch(
            {"input_text": ["one two", "one"], "target_text": ["three four", "five"]},
            tokenizer,
        )
        self.assertEqual([len(ids) for ids in batch["input_ids"]], [3, 2])
        examples = [
            {key: values[index] for key, values in batch.items()} for index in range(2)
        ]
        inputs, mask, labels = RolePlayTransformer.make_collate_fn(tokenizer)(examples)
        self.assertEqual(inputs.shape, labels.shape)
        self.assertEqual(labels[0].tolist(), [-100, -100, -100, 4, 5, 1])
        self.assertEqual(labels[1].tolist(), [-100, -100, 6, 1, -100, -100])
        self.assertEqual(mask[1].tolist(), [1, 1, 1, 1, 0, 0])
        self.assertEqual(
            labels[0, 1:][labels[0, 1:] != -100].tolist(), batch["labels"][0]
        )


class WorkflowTests(unittest.TestCase):
    def test_prepare_train_reload_evaluate_chat_and_baseline(self):
        torch.set_num_threads(1)
        tokenizer = tiny_tokenizer()
        config = ModelConfig(
            max_len=64,
            d_model=64,
            n_layers=2,
            n_heads=1,
            d_ff=128,
            moe_start=1,
            dropout=0.0,
        )
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            data_dir = directory / "data"
            output_dir = directory / "weights"
            with (
                patch("datasets.load_dataset", return_value=sample_rows()),
                patch("workflow.load_tokenizer", return_value=tokenizer),
                patch.object(
                    sys, "argv", ["prepare_data.py", "--output-dir", str(data_dir)]
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                prepare_data.main()

            dataset = load_from_disk(str(data_dir / "bluemoon_train_tok_ds"))
            self.assertEqual(len(dataset), 4)
            torch.manual_seed(42)
            initial = config.build(len(tokenizer)).embed.weight.detach().clone()
            # Four batches and accumulation=8 exercises an incomplete final group.
            with (
                patch("workflow.ModelConfig", return_value=config),
                patch("workflow.load_tokenizer", return_value=tokenizer),
                patch.object(
                    sys,
                    "argv",
                    [
                        "train.py",
                        "--data-dir",
                        str(data_dir / "bluemoon_train_tok_ds"),
                        "--output-dir",
                        str(output_dir),
                        "--epochs",
                        "1",
                        "--micro-batch",
                        "1",
                        "--grad-accum",
                        "8",
                        "--warmup-updates",
                        "0",
                        "--device",
                        "cpu",
                        "--validation-dir",
                        str(data_dir / "bluemoon_validation_tok_ds"),
                    ],
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                train.main()

            checkpoint = output_dir / "ckpt_ep1.pt"
            self.assertTrue(checkpoint.is_file())
            self.assertTrue((output_dir / "best.pt").is_file())
            self.assertTrue((output_dir / "metrics.json").is_file())
            self.assertEqual(
                json.loads((output_dir / "config.json").read_text())["d_model"], 64
            )
            model, saved_tokenizer, device = load_checkpoint(str(checkpoint), "cpu")
            self.assertFalse(torch.equal(initial, model.embed.weight))
            self.assertEqual(len(saved_tokenizer), len(tokenizer))
            self.assertEqual(device, "cpu")
            logits = model(torch.tensor([[2, 3, 4]]))["main"]
            self.assertTrue(torch.isfinite(logits).all())
            second, _, _ = load_checkpoint(str(checkpoint), "cpu")
            torch.testing.assert_close(
                logits, second(torch.tensor([[2, 3, 4]]))["main"]
            )

            commands = [
                [
                    "evaluate.py",
                    "--checkpoint",
                    str(checkpoint),
                    "--data-dir",
                    str(data_dir / "bluemoon_test_tok_ds"),
                    "--device",
                    "cpu",
                ],
                [
                    "chat.py",
                    "--checkpoint",
                    str(checkpoint),
                    "--prompt",
                    "one two",
                    "--max-new-tokens",
                    "2",
                    "--temperature",
                    "0",
                    "--full-text",
                    "--device",
                    "cpu",
                ],
                [
                    "baseline.py",
                    "--train-dir",
                    str(data_dir / "bluemoon_train_ds"),
                    "--test-dir",
                    str(data_dir / "bluemoon_test_ds"),
                    "--prompt",
                    "one two",
                ],
            ]
            expected = ("Test perplexity:", "one two", "Markov baseline perplexity:")
            environment = {
                **os.environ,
                "HF_HUB_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
            }
            for command, marker in zip(commands, expected):
                result = subprocess.run(
                    [sys.executable, *command],
                    cwd=ROOT,
                    env=environment,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=60,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn(marker, result.stdout)


if __name__ == "__main__":
    unittest.main()
