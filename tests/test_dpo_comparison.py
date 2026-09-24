"""Comparison integrity checks using authored toy replies only."""

import json
from pathlib import Path
import tempfile
import unittest

from posttraining.compare_dpo import pair_outputs
from posttraining.summarize_dpo import summarize


class ComparisonTests(unittest.TestCase):
    def test_pair_refuses_unequal_context_or_seed(self):
        for field in ("input_ids", "messages", "sampling_seed", "context"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                row = {"id": "toy", "input_ids": [1], "messages": [{"role": "user", "content": "Hello"}],
                       "sampling_seed": 42, "context": {"revision": 1}}
                for arm in ("sft", "dpo"):
                    (root / arm).mkdir()
                    (root / arm / "manifest.json").write_text(json.dumps({"config": {}, "requests_sha256": "same"}))
                    record = {**row, **({field: None} if arm == "dpo" else {})}
                    (root / arm / "generations.jsonl").write_text(json.dumps(record)+"\n")
                with self.assertRaisesRegex(ValueError, "Paired field mismatch"):
                    pair_outputs(root)
                self.assertFalse((root / "blind_key.json").exists())

    def test_incomplete_review_fails_before_unblinding(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "blind_review.jsonl").write_text('{"id":"toy"}\n')
            (root / "assistant_review.jsonl").write_text("")
            with self.assertRaisesRegex(ValueError, "cover each request"):
                summarize(root)
            self.assertFalse((root / "review_frozen.json").exists())

    def test_changed_frozen_review_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "blind_review.jsonl").write_text('{"id":"toy","fact_slots":[]}\n')
            review = {"id": "toy", "preferred": "tie", "A": [0, 0, 0, 1, 5, 5, 5], "B": [0, 0, 0, 1, 5, 5, 5]}
            (root / "assistant_review.jsonl").write_text(json.dumps(review)+"\n")
            (root / "review_frozen.json").write_text('{"review_sha256":"different"}')
            with self.assertRaisesRegex(ValueError, "changed after unblinding"):
                summarize(root)


if __name__ == "__main__":
    unittest.main()
