"""Precision selection and exact same-device continuation across interruptions."""

import contextlib
import copy
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from datasets import Dataset
from test_model import tiny_model, tokenizer_fixture

from conversation import write_dataset_metadata
from training import autocast_context, choose_precision


class TrainingTests(unittest.TestCase):
    def test_time_budget_saves_at_update_boundary_and_resumes_exactly(self):
        torch.set_num_threads(1)
        tokenizer = tokenizer_fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            Dataset.from_list(
                [
                    {"input_ids": [2, 3, 1], "labels": [5, 7, 1]},
                    {"input_ids": [2, 4, 1], "labels": [6, 1]},
                ]
                * 4
            ).save_to_disk(str(path / "data"))
            write_dataset_metadata(path / "data", tokenizer)
            tokenizer.save_pretrained(str(path / "tokenizer"))
            torch.manual_seed(77)
            initial = tiny_model(dropout=0.2, gradient_checkpointing=True)
            complete, partial = copy.deepcopy(initial), copy.deepcopy(initial)
            arguments = dict(
                dataset_path=str(path / "data"),
                tokenizer_name=str(path / "tokenizer"),
                epochs=1,
                micro_batch=1,
                grad_accum=2,
                warmup_updates=1,
                precision="fp32",
                seed=42,
                save_every=100,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                torch.manual_seed(100)
                full_history = complete.train_model(
                    **arguments, save_path=path / "full"
                )
                torch.manual_seed(100)
                # Start, pre-loop, first group, second group, stop report timestamp.
                with patch("training.monotonic", side_effect=[0, 0, 0, 3600, 3600]):
                    partial.train_model(
                        **arguments, save_path=path / "partial", max_hours=1
                    )
                saved = torch.load(path / "partial/latest.pt", weights_only=True)
                self.assertEqual(saved["global_step"], 1)
                self.assertEqual(saved["next_example"], 2)
                self.assertNotIn("max_hours", saved["settings"])
                self.assertTrue((path / "partial/time_limit.json").exists())
                resumed = tiny_model(dropout=0.2, gradient_checkpointing=True)
                history = resumed.train_model(
                    **arguments,
                    save_path=path / "partial",
                    resume=path / "partial/latest.pt",
                    max_hours=2,
                )
            self.assertEqual(full_history, history)
            for expected, actual in zip(complete.parameters(), resumed.parameters()):
                torch.testing.assert_close(expected, actual, rtol=0, atol=0)

    def test_time_budget_rejects_invalid_limits_and_missing_checkpoints(self):
        model = tiny_model()
        for value in (0, -1, float("nan"), float("inf")):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "max_hours"),
            ):
                model.train_model(
                    dataset_path="unused", save_path="unused", max_hours=value
                )
        with self.assertRaisesRegex(ValueError, "save_path"):
            model.train_model(dataset_path="unused", max_hours=1)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA/ROCm GPU required")
    def test_fp16_gpu_attention_is_causal_and_padding_isolated(self):
        torch.manual_seed(42)
        model = tiny_model(dropout=0.0).cuda().eval()
        ids = torch.tensor([[2, 3, 4, 5, 1]], device="cuda")
        changed = torch.tensor([[2, 3, 6, 7, 1]], device="cuda")
        padded = torch.tensor([[2, 3, 4, 5, 1, 8, 8]], device="cuda")
        mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0]], device="cuda")
        with torch.no_grad(), autocast_context("cuda", "fp16"):
            expected = model(ids)["main"]
            future_changed = model(changed)["main"]
            padding_added = model(padded, mask)["main"]
        self.assertTrue(torch.isfinite(expected).all())
        torch.testing.assert_close(
            expected[:, :2], future_changed[:, :2], atol=2e-4, rtol=2e-3
        )
        torch.testing.assert_close(expected, padding_added[:, :5], atol=2e-4, rtol=2e-3)

    @patch("torch.version.hip", None)
    def test_auto_precision_uses_fp16_for_t4_not_emulated_bf16(self):
        with patch("torch.cuda.get_device_capability", return_value=(7, 5)):
            self.assertEqual(choose_precision("cuda"), "fp16")
            with self.assertRaisesRegex(ValueError, "native BF16"):
                choose_precision("cuda", "bf16")
        with patch("torch.cuda.get_device_capability", return_value=(8, 9)):
            self.assertEqual(choose_precision("cuda"), "bf16")
            self.assertEqual(choose_precision("cuda", "fp16"), "fp16")
        self.assertEqual(choose_precision("cpu"), "fp32")

    @patch("torch.version.hip", "7.2.1")
    def test_rocm_precision_does_not_use_nvidia_capability(self):
        with patch(
            "torch.cuda.get_device_capability",
            side_effect=AssertionError("NVIDIA capability queried on AMD"),
        ):
            self.assertEqual(choose_precision("cuda"), "fp16")
            self.assertEqual(choose_precision("cuda", "fp16"), "fp16")
            self.assertEqual(choose_precision("cuda", "fp32"), "fp32")
            with (
                patch(
                    "torch.cuda.device", return_value=contextlib.nullcontext()
                ) as scope,
                patch("torch.cuda.is_bf16_supported", return_value=True),
            ):
                self.assertEqual(choose_precision("cuda:1", "bf16"), "bf16")
                scope.assert_called_once_with(torch.device("cuda:1"))
            with (
                patch("torch.cuda.device", return_value=contextlib.nullcontext()),
                patch("torch.cuda.is_bf16_supported", return_value=False),
            ):
                with self.assertRaisesRegex(ValueError, "no BF16 support"):
                    choose_precision("cuda", "bf16")

    def test_resume_mid_epoch_matches_uninterrupted_training_with_dropout(self):
        torch.set_num_threads(1)
        tokenizer = tokenizer_fixture()
        examples = [
            {"input_ids": [2, 3, 1], "labels": [5, 7, 1]},
            {"input_ids": [2, 4, 1], "labels": [6, 1]},
        ] * 5
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            Dataset.from_list(examples).save_to_disk(str(path / "data"))
            write_dataset_metadata(path / "data", tokenizer)
            tokenizer.save_pretrained(str(path / "tokenizer"))
            torch.manual_seed(77)
            initial = tiny_model(dropout=0.2, gradient_checkpointing=True)
            complete = copy.deepcopy(initial)
            interrupted = copy.deepcopy(initial)
            arguments = dict(
                dataset_path=str(path / "data"),
                tokenizer_name=str(path / "tokenizer"),
                epochs=2,
                micro_batch=2,
                grad_accum=2,
                warmup_updates=1,
                save_every=1,
                seed=123,
                precision="fp32",
            )
            with contextlib.redirect_stdout(io.StringIO()):
                torch.manual_seed(100)
                full_history = complete.train_model(
                    **arguments, save_path=path / "full"
                )
                torch.manual_seed(100)
                interrupted.train_model(
                    **arguments, save_path=path / "partial", max_updates=2
                )
                halfway = torch.load(path / "partial/latest.pt", weights_only=True)
                self.assertEqual(halfway["epoch"], 0)
                self.assertEqual(halfway["next_example"], 8)
                resumed = tiny_model(dropout=0.2, gradient_checkpointing=True)
                resumed_history = resumed.train_model(
                    **arguments,
                    save_path=path / "partial",
                    resume=path / "partial/latest.pt",
                )
            self.assertEqual(full_history, resumed_history)
            for expected, actual in zip(complete.parameters(), resumed.parameters()):
                torch.testing.assert_close(expected, actual, atol=0, rtol=0)
            full_state = torch.load(path / "full/latest.pt", weights_only=True)
            resumed_state = torch.load(path / "partial/latest.pt", weights_only=True)
            self.assertEqual(full_state["scheduler"], resumed_state["scheduler"])
            self.assertEqual(resumed_state["epoch"], 2)
            self.assertTrue((path / "partial/previous.pt").is_file())
            with self.assertRaisesRegex(ValueError, "settings or data changed"):
                resumed.train_model(
                    **(arguments | {"lr": 0.03}),
                    save_path=path / "partial",
                    resume=path / "partial/latest.pt",
                )

    @unittest.skipUnless(
        torch.cuda.is_available(), "CUDA GPU required for FP16 gradient-scaler test"
    )
    def test_fp16_cuda_checkpoint_restores_scaler_and_optimizer(self):
        tokenizer = tokenizer_fixture()
        examples = [{"input_ids": [2, 3, 1], "labels": [5, 7, 1]}] * 8
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            Dataset.from_list(examples).save_to_disk(str(path / "data"))
            write_dataset_metadata(path / "data", tokenizer)
            tokenizer.save_pretrained(str(path / "tokenizer"))
            arguments = dict(
                dataset_path=str(path / "data"),
                tokenizer_name=str(path / "tokenizer"),
                epochs=2,
                micro_batch=2,
                grad_accum=2,
                warmup_updates=0,
                save_every=1,
                device="cuda",
                precision="fp16",
                save_path=path / "run",
            )
            model = tiny_model(gradient_checkpointing=True)
            with contextlib.redirect_stdout(io.StringIO()):
                model.train_model(**arguments, max_updates=1)
                before = torch.load(
                    path / "run/latest.pt", weights_only=True, map_location="cpu"
                )
                self.assertTrue(before["scaler"])
                restarted = tiny_model(gradient_checkpointing=True)
                restarted.train_model(**arguments, resume=path / "run/latest.pt")
            after = torch.load(
                path / "run/latest.pt", weights_only=True, map_location="cpu"
            )
            self.assertEqual(after["epoch"], 2)
            self.assertGreater(after["global_step"], before["global_step"])
            self.assertTrue(
                all(torch.isfinite(weight).all() for weight in after["model"].values())
            )


if __name__ == "__main__":
    unittest.main()
