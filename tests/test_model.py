"""Regression tests for causal learning, padding, generation, and saved formats."""

import contextlib
import copy
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from conversation import (
    encode_context,
    validate_dataset_metadata,
    write_dataset_metadata,
)
from mini_deepseek import MoEBlock, RolePlayTransformer, next_token_loss
from prepare_data import tokenize_batch
from workflow import load_checkpoint, read_config


def tokenizer_fixture():
    words = [
        "[UNK]",
        "[EOS]",
        "Say",
        "hello",
        "goodbye",
        "Hello",
        "Goodbye",
        "friend",
        "please",
    ]
    backend = Tokenizer(
        WordLevel({word: index for index, word in enumerate(words)}, unk_token="[UNK]")
    )
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        eos_token="[EOS]",
        pad_token="[EOS]",
    )


def tiny_model(**options):
    defaults = dict(
        vocab=9,
        max_len=32,
        d_model=32,
        n_layers=2,
        n_heads=4,
        d_ff=64,
        dropout=0.0,
        gradient_checkpointing=False,
    )
    return RolePlayTransformer(**(defaults | options))


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(13)

    def test_future_changes_cannot_change_prefix_logits_dense_or_moe(self):
        for moe_start in (None, 0):
            model = tiny_model(moe_start=moe_start).eval()
            original = torch.tensor([[2, 3, 4, 5, 6, 7]])
            altered = torch.tensor([[2, 3, 4, 8, 8, 8]])
            torch.testing.assert_close(
                model(original)["main"][:, :3],
                model(altered)["main"][:, :3],
                atol=1e-7,
                rtol=0,
            )
            # Full-sequence teacher forcing must agree with prefix-only inference.
            for index in range(1, original.size(1) + 1):
                torch.testing.assert_close(
                    model(original)["main"][:, index - 1],
                    model(original[:, :index])["main"][:, -1],
                    atol=2e-7,
                    rtol=1e-5,
                )

    def test_future_positions_have_zero_gradient_from_earlier_prediction(self):
        model = tiny_model()
        model(torch.tensor([[2, 3, 4, 5, 6]]))["main"][0, 1, 8].backward()
        self.assertGreater(model.pos.grad[0, :2].abs().sum().item(), 0)
        self.assertEqual(model.pos.grad[0, 2:].abs().sum().item(), 0)

    def test_padding_does_not_change_real_logits_or_moe_balance(self):
        model = tiny_model(moe_start=0).eval()
        ids = torch.tensor([[2, 3, 1]])
        reference = model(ids)
        for padded, mask, start in (
            ([[2, 3, 1, 8, 8]], [[1, 1, 1, 0, 0]], 0),
            ([[8, 8, 2, 3, 1]], [[0, 0, 1, 1, 1]], 2),
        ):
            output = model(torch.tensor(padded), torch.tensor(mask))
            torch.testing.assert_close(
                reference["main"],
                output["main"][:, start : start + 3],
                atol=2e-7,
                rtol=1e-5,
            )
            torch.testing.assert_close(reference["lb"], output["lb"])
            self.assertTrue(torch.isfinite(output["main"]).all())

    def test_balance_loss_trains_router_without_padding_contributions(self):
        block = MoEBlock(4, 8, n_experts=4, top_k=2, p=0.0)
        with torch.no_grad():
            block.router.weight.copy_(
                torch.tensor([[2.0] * 4, [1.0] * 4, [-1.0] * 4, [-2.0] * 4])
            )
        inputs = torch.ones(1, 3, 4)
        _, loss = block(inputs)
        loss.backward()
        self.assertGreater(block.router.weight.grad.abs().sum().item(), 0)
        padded = torch.cat([inputs, torch.full((1, 2, 4), -100.0)], dim=1)
        _, padded_loss = block(padded, torch.tensor([[1, 1, 1, 0, 0]]))
        torch.testing.assert_close(loss, padded_loss)
        _, empty_loss = block(inputs, torch.zeros(1, 3))
        self.assertEqual(empty_loss.item(), 0)

    def test_single_head_and_small_tied_initialization(self):
        model = tiny_model()
        self.assertFalse(hasattr(model, "mtp_head"))
        self.assertIs(model.lm_head.weight, model.embed.weight)
        self.assertLess(model.embed.weight.std().item(), 0.03)
        self.assertEqual(set(model(torch.tensor([[2, 3]]))), {"main", "lb"})

    def test_loss_counts_only_shifted_targets_including_final_eos(self):
        logits = torch.zeros(2, 5, 9)
        labels = torch.tensor([[-100, -100, 5, 7, 1], [-100, 6, 1, -100, -100]])
        loss, count = next_token_loss(logits, labels)
        self.assertEqual(count.item(), 5)
        self.assertAlmostEqual(loss.item(), 5 * math.log(9), places=5)
        with self.assertRaises(ValueError):
            next_token_loss(logits, torch.full_like(labels, -100))

    def test_top_p_filters_each_row_without_cross_row_indices(self):
        logits = torch.tensor([[9.0, 1.0, 0.0, -1.0], [-1.0, 0.0, 1.0, 9.0]])
        filtered = RolePlayTransformer.top_k_top_p_filtering(logits, top_p=0.7)
        self.assertEqual(
            torch.isfinite(filtered).tolist(),
            [[True, False, False, False], [False, False, False, True]],
        )
        self.assertTrue(torch.isfinite(logits).all())
        top_k = RolePlayTransformer.top_k_top_p_filtering(logits, top_k=2)
        self.assertEqual(torch.isfinite(top_k).sum(-1).tolist(), [2, 2])

    def test_repetition_penalty_handles_both_signs_per_row(self):
        scores = torch.tensor([[4.0, -4.0, 2.0], [-2.0, 8.0, 1.0]])
        result = RolePlayTransformer.apply_repetition_penalty(
            scores, torch.tensor([[0, 1], [1, 1]]), 2.0
        )
        torch.testing.assert_close(
            result, torch.tensor([[2.0, -8.0, 2.0], [-2.0, 4.0, 1.0]])
        )

    def test_stop_id_zero_and_mode_restoration_even_on_failure(self):
        model = tiny_model().train()
        tokenizer = tokenizer_fixture()
        logits = torch.zeros(1, 3, 9)
        logits[..., 0] = 10
        with patch.object(model, "forward", return_value={"main": logits}):
            self.assertEqual(
                model.generate(
                    "Say hello",
                    tokenizer=tokenizer,
                    max_new_tokens=2,
                    temperature=0,
                    stop_token=0,
                ),
                "",
            )
        self.assertTrue(model.training)
        with (
            patch.object(model, "forward", side_effect=RuntimeError("test")),
            self.assertRaises(RuntimeError),
        ):
            model.generate("Say hello", tokenizer=tokenizer, max_new_tokens=2)
        self.assertTrue(model.training)
        for options in (
            {"temperature": float("nan")},
            {"repetition_penalty": -1},
            {"top_p": 0},
            {"max_new_tokens": 50},
        ):
            with self.assertRaises(ValueError):
                model.generate("Say hello", tokenizer=tokenizer, **options)

    def test_training_and_generation_use_same_boundary_and_latest_context(self):
        tokenizer = tokenizer_fixture()
        batch = tokenize_batch(
            {"input_text": ["Say hello"], "target_text": ["Hello friend"]}, tokenizer
        )
        self.assertEqual(batch["input_ids"][0], encode_context("Say hello", tokenizer))
        self.assertEqual(batch["input_ids"][0][-1], tokenizer.eos_token_id)
        self.assertEqual(
            encode_context("Say goodbye please", tokenizer, max_length=3), [4, 8, 1]
        )

    def test_tiny_model_learns_and_generates_two_exact_replies(self):
        tokenizer = tokenizer_fixture()
        encoded = tokenize_batch(
            {
                "input_text": ["Say hello", "Say goodbye"],
                "target_text": ["Hello friend", "Goodbye friend"],
            },
            tokenizer,
        )
        examples = [
            {key: value[index] for key, value in encoded.items()} for index in range(2)
        ]
        ids, mask, labels = RolePlayTransformer.make_collate_fn(tokenizer)(examples)
        model = tiny_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        for _ in range(120):
            optimizer.zero_grad()
            total, count = next_token_loss(model(ids, mask)["main"], labels)
            loss = total / count
            loss.backward()
            optimizer.step()
        self.assertLess(loss.item(), 0.03)
        for prompt, answer in (
            ("Say hello", "Hello friend"),
            ("Say goodbye", "Goodbye friend"),
        ):
            self.assertEqual(
                model.generate_chat(
                    prompt, tokenizer=tokenizer, temperature=0, max_new_tokens=5
                ),
                answer,
            )

    def test_accumulation_matches_full_batch_for_unequal_target_lengths(self):
        tokenizer = tokenizer_fixture()
        examples = [
            {"input_ids": [2, 3, 1], "labels": [5, 7, 1]},
            {"input_ids": [2, 4, 1], "labels": [6, 1]},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            Dataset.from_list(examples).save_to_disk(str(path / "dataset"))
            write_dataset_metadata(path / "dataset", tokenizer)
            tokenizer.save_pretrained(str(path / "tokenizer"))
            full = tiny_model()
            accumulated = copy.deepcopy(full)
            for model, micro_batch, grad_accum in ((full, 2, 1), (accumulated, 1, 3)):
                torch.manual_seed(3)
                with contextlib.redirect_stdout(io.StringIO()):
                    model.train_model(
                        str(path / "dataset"),
                        str(path / "tokenizer"),
                        micro_batch=micro_batch,
                        grad_accum=grad_accum,
                        warmup_updates=0,
                    )
            for left, right in zip(full.parameters(), accumulated.parameters()):
                torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-4)

    def test_legacy_checkpoint_and_wrong_tokenizer_data_are_rejected(self):
        tokenizer = tokenizer_fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            torch.save({}, path / "legacy.pt")
            with self.assertRaisesRegex(ValueError, "Legacy"):
                load_checkpoint(str(path / "legacy.pt"))
            (path / "config.json").write_text(json.dumps({"use_mtp": True}))
            with self.assertRaisesRegex(ValueError, "Legacy"):
                read_config(path / "config.json")
            with self.assertRaisesRegex(ValueError, "Rebuild"):
                validate_dataset_metadata(path, tokenizer)
            write_dataset_metadata(path, tokenizer)
            tokenizer.add_tokens(["new"])
            with self.assertRaisesRegex(ValueError, "differs"):
                validate_dataset_metadata(path, tokenizer)

    def test_evaluation_matches_prefix_only_scoring_and_is_batch_invariant(self):
        tokenizer = tokenizer_fixture()
        examples = [
            {"input_ids": [2, 3, 1], "labels": [5, 7, 1]},
            {"input_ids": [2, 4, 8, 1], "labels": [6, 1]},
        ]
        model = tiny_model().eval()
        losses = []
        with torch.no_grad():
            for example in examples:
                prefix = list(example["input_ids"])
                for target in example["labels"]:
                    logits = model(torch.tensor([prefix]))["main"][0, -1]
                    losses.append(-logits.log_softmax(-1)[target].item())
                    prefix.append(target)
        reference = math.exp(sum(losses) / len(losses))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            Dataset.from_list(examples).save_to_disk(str(path))
            write_dataset_metadata(path, tokenizer)
            model.train()
            for batch_size in (1, 2):
                with contextlib.redirect_stdout(io.StringIO()):
                    perplexity = model.evaluate_perplexity(
                        str(path), tokenizer, batch_size=batch_size
                    )
                self.assertAlmostEqual(perplexity, reference, places=5)
                self.assertTrue(model.training)

    def test_gradient_checkpointing_matches_ordinary_backward(self):
        regular = tiny_model(moe_start=1)
        checkpointed = copy.deepcopy(regular)
        checkpointed.gc = True
        ids = torch.tensor([[2, 3, 1, 5, 7, 1]])
        labels = torch.tensor([[-100, -100, -100, 5, 7, 1]])
        for model in (regular, checkpointed):
            output = model(ids)
            total, count = next_token_loss(output["main"], labels)
            (total / count + 0.01 * output["lb"]).backward()
        for left, right in zip(regular.parameters(), checkpointed.parameters()):
            if left.grad is None:
                self.assertIsNone(right.grad)
            else:
                torch.testing.assert_close(left.grad, right.grad)


if __name__ == "__main__":
    unittest.main()
