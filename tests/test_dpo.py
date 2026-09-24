"""Synthetic correctness checks for standard DPO and its data partitions."""

import copy
import math
import unittest

from posttraining.dpo_core import completion_labels, dpo_loss, summed_log_probs, train_group
from posttraining.dpo_data import authored_pairs


class DpoTests(unittest.TestCase):
    def test_initial_loss_and_analytic_gradient(self):
        import torch
        chosen = torch.tensor(-5.0, requires_grad=True)
        rejected = torch.tensor(-7.0, requires_grad=True)
        loss, margin = dpo_loss(chosen, rejected, -5.0, -7.0, 0.1)
        self.assertAlmostEqual(loss.item(), math.log(2), places=6)
        self.assertEqual(margin.item(), 0)
        loss.backward()
        self.assertAlmostEqual(chosen.grad.item(), -0.05, places=6)
        self.assertAlmostEqual(rejected.grad.item(), 0.05, places=6)

    def test_loss_uses_reference_and_sequence_sum(self):
        import torch
        first, _ = dpo_loss(torch.tensor(-5.0), torch.tensor(-7.0), -5.0, -7.0, 0.1)
        changed_reference, _ = dpo_loss(torch.tensor(-5.0), torch.tensor(-7.0), -2.0, -7.0, 0.1)
        improved_policy, _ = dpo_loss(torch.tensor(-4.0), torch.tensor(-7.0), -5.0, -7.0, 0.1)
        self.assertGreater(changed_reference, first)
        self.assertLess(improved_policy, first)
        with self.assertRaises(ValueError):
            dpo_loss(torch.tensor(0.), torch.tensor(0.), 0., 0., 0)

    def test_shift_mask_eos_padding_and_prefix_isolation(self):
        import torch
        logits = torch.zeros(1, 6, 8, requires_grad=True)
        labels = torch.tensor([[-100, -100, 4, 2, -100, -100]])
        score = summed_log_probs(logits, labels)
        self.assertAlmostEqual(score.item(), -2*math.log(8), places=6)
        score.sum().backward()
        self.assertEqual(logits.grad[:, 0].abs().sum().item(), 0)
        self.assertGreater(logits.grad[:, 1:3].abs().sum().item(), 0)
        self.assertEqual(logits.grad[:, 3:].abs().sum().item(), 0)
        changed = logits.detach().clone()
        changed[:, 4:] = 100
        self.assertEqual(summed_log_probs(changed, labels).item(), score.item())
        self.assertEqual(completion_labels([1, 5], [1, 5, 7, 2], 2), [-100, -100, 7, 2])
        with self.assertRaises(ValueError):
            completion_labels([1, 5], [1, 4, 7, 2], 2)

    def test_aliases_translations_and_templates_stay_in_one_partition(self):
        rows = list(authored_pairs())
        self.assertEqual(len(rows), 200)
        partitions = {}
        for row in rows:
            partitions.setdefault(row["template_family"], set()).add(row["split"])
            self.assertNotEqual(row["chosen_text"], row["rejected_text"])
            self.assertNotIn(row["rationale"], str(row["messages"]))
        self.assertTrue(all(len(v) == 1 for v in partitions.values()))
        self.assertEqual(sum(r["split"] == "train" for r in rows), 160)
        self.assertEqual(sum(r["split"] == "validation" for r in rows), 40)
        scenes = {}
        for row in rows:
            scenes.setdefault(row["scene_id"], set()).add(row["language"])
        self.assertTrue(all(v == {"en", "fr"} for v in scenes.values()))

    def test_accumulation_is_pair_mean_not_token_mean(self):
        import torch
        import torch.nn.functional as functional
        from types import SimpleNamespace
        from posttraining.lora_train import batch

        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = torch.nn.Embedding(16, 8)
                self.head = torch.nn.Linear(8, 16)

            def forward(self, input_ids, labels, **kwargs):
                logits = self.head(self.emb(input_ids))
                return SimpleNamespace(loss=functional.cross_entropy(
                    logits[:, :-1].reshape(-1, 16), labels[:, 1:].reshape(-1), ignore_index=-100))

        def row(ids, prefix):
            return {"input_ids": ids, "labels": [-100]*prefix+ids[prefix:], "target_tokens": len(ids)-prefix}

        pairs = [{"chosen": row([1, 3, 5, 2], 2), "rejected": row([1, 3, 7, 8, 2], 2),
                  "ref_chosen": -4., "ref_rejected": -6.},
                 {"chosen": row([1, 9, 2], 1), "rejected": row([1, 8, 7, 6, 5, 2], 1),
                  "ref_chosen": -3., "ref_rejected": -8.}]
        torch.manual_seed(9)
        model = Tiny()
        reference = copy.deepcopy(model)
        expected_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
        expected_loss = 0
        for pair in pairs:
            chosen = -reference(**batch(pair["chosen"], "cpu")).loss * pair["chosen"]["target_tokens"]
            rejected = -reference(**batch(pair["rejected"], "cpu")).loss * pair["rejected"]["target_tokens"]
            expected_loss += dpo_loss(chosen, rejected, pair["ref_chosen"], pair["ref_rejected"], .1)[0] / 2
        expected_loss.backward()
        expected_optimizer.step()
        runtime = SimpleNamespace(model=model, device="cpu", guard_memory=lambda: None)
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        train_group(runtime, pairs, optimizer, torch.amp.GradScaler("cuda", enabled=False),
                    {"beta": .1, "max_grad_norm": 1000})
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
