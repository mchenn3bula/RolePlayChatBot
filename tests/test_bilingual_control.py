"""Synthetic data isolation and objective-routing checks for the continued-SFT control."""

from collections import Counter
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from posttraining.control_train import update
from posttraining.preferences_v2 import rows


class BilingualControlTests(unittest.TestCase):
    def test_family_language_partition_and_no_alias_inflation(self):
        records = list(rows())
        self.assertEqual(Counter(r["split"] for r in records), {"train": 80, "validation": 24, "evaluation": 16})
        families = {}
        for r in records:
            families.setdefault(r["template_family"], []).append(r)
        self.assertEqual(len(families), 60)
        for group in families.values():
            self.assertEqual(len({r["split"] for r in group}), 1)
            self.assertEqual(Counter(r["language"] for r in group), {"en": 1, "fr": 1})
            if group[0]["split"] == "evaluation":
                self.assertFalse(any(r["mine"] for r in group))

    def test_current_state_follows_old_authored_history(self):
        records = {r["id"]: r for r in rows()}
        row = records["r2-film_session-en"]
        self.assertEqual(len(row["messages"]), 4)
        self.assertIn("Friday in hall 1", row["messages"][2]["content"])
        self.assertIn("Saturday in hall 4", row["messages"][0]["content"])
        self.assertIn('"through_user_turn":2', row["messages"][0]["content"])

    def test_sft_never_passes_rejected_labels_to_optimizer(self):
        chosen = {"labels": [-100, 11, 2]}
        rejected = {"labels": [-100, 99, 2]}
        with patch("posttraining.control_train.lora_train.train_group", return_value=(2., 3., 0)) as train:
            result = update("runtime", [{"chosen": chosen, "rejected": rejected}], "opt", "scaler", {"objective": "sft_chosen"})
        self.assertEqual(train.call_args.args[1], [chosen])
        self.assertEqual(result, (2., None, 3., 0))
        with self.assertRaises(ValueError):
            update(None, [], None, None, {"objective": "unknown"})

    def test_control_configs_match_except_objective_and_name(self):
        configs = [json.loads(Path(f"configs/ministral_v2_{arm}.json").read_text()) for arm in ("sft", "dpo")]
        for config in configs:
            config.pop("objective")
            config.pop("experiment")
            self.assertEqual(config["lora_dropout"], 0)
            self.assertEqual(config["sft_reference"], "checkpoints/ministral-s1-whole-lora-v1-r2/epoch-02")
        self.assertEqual(*configs)


if __name__ == "__main__":
    unittest.main()
