"""RoPE/SwiGLU correctness, sparse projection equivalence, and version isolation."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from test_model import tiny_model, tokenizer_fixture
import test_training
import test_generation_eval
import test_model

from mini_deepseek import RotaryEmbedding, next_token_loss
from training import autocast_context
from workflow import ModelConfig, load_checkpoint, read_config


def modern_model(**options):
    return tiny_model(
        **(
            dict(
                position_encoding="rope", ffn_type="swiglu", target_only_projection=True
            )
            | options
        )
    )


class ModernTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(42)

    def test_rotary_preserves_norm_and_relative_attention_scores(self):
        rotary = RotaryEmbedding(8, 32)
        q, k = torch.randn(2, 3, 5, 8), torch.randn(2, 3, 5, 8)
        positions = torch.arange(5).expand(2, -1)
        rotated_q, rotated_k = rotary(q, positions), rotary(k, positions)
        torch.testing.assert_close(rotated_q.norm(dim=-1), q.norm(dim=-1))
        torch.testing.assert_close(rotated_q[:, :, 0], q[:, :, 0])
        torch.testing.assert_close(
            rotated_q @ rotated_k.transpose(-1, -2),
            rotary(q, positions + 11) @ rotary(k, positions + 11).transpose(-1, -2),
            atol=3e-6,
            rtol=2e-5,
        )

    def check_causality_and_padding(self, device, precision):
        model = modern_model().to(device).eval()
        ids = torch.tensor([[2, 3, 4, 5, 1]], device=device)
        altered = torch.tensor([[2, 3, 8, 8, 1]], device=device)
        tol = 3e-4 if precision == "fp16" else 3e-7
        with torch.no_grad(), autocast_context(device, precision):
            reference = model(ids)["main"]
            torch.testing.assert_close(
                reference[:, :2], model(altered)["main"][:, :2], atol=tol, rtol=3e-3
            )
            for length in range(1, 6):
                torch.testing.assert_close(
                    reference[:, length - 1],
                    model(ids[:, :length])["main"][:, -1],
                    atol=tol,
                    rtol=3e-3,
                )
            for padded, mask, offset in (
                ([[8, 8, 2, 3, 4, 5, 1]], [[0, 0, 1, 1, 1, 1, 1]], 2),
                ([[2, 3, 4, 5, 1, 8, 8]], [[1, 1, 1, 1, 1, 0, 0]], 0),
            ):
                output = model(
                    torch.tensor(padded, device=device),
                    torch.tensor(mask, device=device),
                )["main"]
                self.assertTrue(torch.isfinite(output).all())
                torch.testing.assert_close(
                    reference, output[:, offset : offset + 5], atol=tol, rtol=3e-3
                )

    def test_causal_prefix_and_left_right_padding(self):
        self.check_causality_and_padding("cpu", "fp32")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA/ROCm GPU required")
    def test_fp16_causal_prefix_and_left_right_padding(self):
        self.check_causality_and_padding("cuda", "fp16")

    def test_future_embeddings_cannot_affect_earlier_state(self):
        model = modern_model()
        embedded = []

        def capture(module, args, output):
            output.retain_grad()
            embedded.append(output)

        with model.embed.register_forward_hook(capture):
            model(torch.tensor([[2, 3, 4, 5, 6]]))["main"][0, 1, 8].backward()
        self.assertGreater(embedded[0].grad[:, :2].abs().sum().item(), 0)
        self.assertEqual(embedded[0].grad[:, 2:].abs().sum().item(), 0)

    def check_projection(self, device, precision):
        # Unequal reply lengths, EOS targets, context and padding exclusions.
        ids = torch.tensor([[2, 3, 5, 7, 1], [2, 4, 6, 1, 1]], device=device)
        labels = torch.tensor(
            [[-100, -100, 5, 7, 1], [-100, -100, 6, 1, -100]], device=device
        )
        mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]], device=device)
        for checkpointing in (False, True):
            full = (
                modern_model(dropout=0.2, gradient_checkpointing=checkpointing)
                .to(device)
                .train()
            )
            selected = copy.deepcopy(full)
            torch.manual_seed(123)
            with autocast_context(device, precision):
                expected, count = next_token_loss(full(ids, mask)["main"], labels)
            expected.backward()
            projected_shapes = []
            torch.manual_seed(123)
            with selected.lm_head.register_forward_pre_hook(
                lambda module, args: projected_shapes.append(args[0].shape)
            ):
                with autocast_context(device, precision):
                    actual = selected.forward_loss(ids, mask, labels)
                actual["loss_sum"].backward()
            self.assertEqual(count.item(), 5)
            self.assertEqual(actual["target_count"].item(), 5)
            self.assertEqual(projected_shapes[0], (5, 32))
            tol = 3e-3 if precision == "fp16" else 3e-6
            torch.testing.assert_close(actual["loss_sum"], expected, atol=tol, rtol=tol)
            for a, b in zip(full.parameters(), selected.parameters()):
                torch.testing.assert_close(a.grad, b.grad, atol=tol, rtol=tol)

    def test_selected_projection_matches_full_loss_and_all_gradients(self):
        self.check_projection("cpu", "fp32")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA/ROCm GPU required")
    def test_fp16_selected_projection_matches_full_loss_and_gradients(self):
        self.check_projection("cuda", "fp16")

    def test_empty_supervision_and_invalid_rotary_config_rejected(self):
        ids = torch.tensor([[2, 3, 1]])
        with self.assertRaisesRegex(ValueError, "no target tokens"):
            modern_model().forward_loss(ids, None, torch.full_like(ids, -100))
        for dim, theta in ((7, 10000), (8, 0), (8, float("nan"))):
            with self.assertRaises(ValueError):
                RotaryEmbedding(dim, 32, theta)
        with self.assertRaises(ValueError):
            ModelConfig(architecture_version=2, position_encoding="rope").build(9)
        with self.assertRaises(ValueError):
            ModelConfig(architecture_version=3).build(9)

    def test_checkpoint_roundtrip_and_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            config = ModelConfig(
                architecture_version=3,
                max_len=32,
                d_model=32,
                n_layers=2,
                n_heads=4,
                d_ff=64,
                dropout=0,
                position_encoding="rope",
                ffn_type="swiglu",
                target_only_projection=True,
            )
            config.save(path / "config.json")
            self.assertEqual(config, read_config(path / "config.json"))
            tokenizer_fixture().save_pretrained(str(path / "tokenizer"))
            model = config.build(9).eval()
            torch.save(model.state_dict(), path / "best.pt")
            restored, _, _ = load_checkpoint(str(path / "best.pt"), "cpu")
            ids = torch.tensor([[2, 3, 1]])
            torch.testing.assert_close(
                model(ids)["main"], restored(ids)["main"], atol=0, rtol=0
            )
            self.assertEqual(
                model.generate_token_ids([2, 3, 1], max_new_tokens=4, temperature=0),
                restored.generate_token_ids([2, 3, 1], max_new_tokens=4, temperature=0),
            )
            self.assertIsNone(restored.pos)
            self.assertIs(restored.embed.weight, restored.lm_head.weight)

    def test_modern_exact_resume_with_dropout(self):
        with patch("test_training.tiny_model", side_effect=modern_model):
            test_training.TrainingTests().test_resume_mid_epoch_matches_uninterrupted_training_with_dropout()

    def test_modern_time_stop_resumes_exactly(self):
        with patch("test_training.tiny_model", side_effect=modern_model):
            test_training.TrainingTests().test_time_budget_saves_at_update_boundary_and_resumes_exactly()

    def test_modern_perplexity_matches_prefix_only_scoring(self):
        with patch("test_model.tiny_model", side_effect=modern_model):
            test_model.ModelTests().test_evaluation_matches_prefix_only_scoring_and_is_batch_invariant()

    def test_modern_fixed_generation_panel(self):
        def config(**kwargs):
            return ModelConfig(
                **(
                    kwargs
                    | dict(
                        architecture_version=3,
                        position_encoding="rope",
                        ffn_type="swiglu",
                        target_only_projection=True,
                    )
                )
            )

        with patch("test_generation_eval.ModelConfig", side_effect=config):
            test_generation_eval.GenerationEvaluationTests().test_evaluation_saves_exact_ids_and_reproducible_panel()

    def test_rope_theta_change_rejected_on_resume(self):
        # Reuse the integration fixture but construct a changed resume model.
        count = 0

        def build(**kwargs):
            nonlocal count
            count += 1
            return modern_model(**kwargs, rope_theta=10000 if count == 1 else 20000)

        with patch("test_training.tiny_model", side_effect=build):
            with self.assertRaisesRegex(ValueError, "model_variant"):
                test_training.TrainingTests().test_resume_mid_epoch_matches_uninterrupted_training_with_dropout()


if __name__ == "__main__":
    unittest.main()
