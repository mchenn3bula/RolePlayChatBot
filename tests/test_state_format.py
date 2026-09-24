"""Protect the single-variable boundary of the state-format ablation."""

import copy
import json
from pathlib import Path
import unittest

from posttraining.preferences_v2 import rows
from posttraining.state_format import replace_value


class StateFormatTests(unittest.TestCase):
    def test_only_scene_value_changes_and_input_remains_unmodified(self):
        fixtures = json.loads(Path("posttraining/state_format_fixtures.json").read_text(encoding="utf-8"))
        original_rows = [r for r in rows() if r["split"] == "evaluation"]
        self.assertEqual(len(original_rows), 16)
        self.assertEqual({r["template_family"] for r in original_rows}, set(fixtures))
        for row in original_rows:
            original = copy.deepcopy(row["messages"])
            changed, old_value = replace_value(row["messages"], fixtures[row["template_family"]][row["language"]])
            self.assertEqual(original, row["messages"])
            self.assertEqual(original[1:], changed[1:])
            before_prefix, before_raw = original[0]["content"].rsplit("\n", 1)
            after_prefix, after_raw = changed[0]["content"].rsplit("\n", 1)
            self.assertEqual(before_prefix, after_prefix)
            after = json.loads(after_raw)
            target = next(f for f in after["facts"] if f["predicate"] == "established")
            self.assertNotEqual(target["value"], old_value)
            target["value"] = old_value
            self.assertEqual(json.loads(before_raw), after)

    def test_rejects_non_assignment_and_reapplication(self):
        row = next(r for r in rows() if r["split"] == "evaluation")
        with self.assertRaises(ValueError):
            replace_value(row["messages"], ["Arbitrary new prose"])
        changed, _ = replace_value(row["messages"], ["object.holder = character"])
        with self.assertRaises(ValueError):
            replace_value(changed, ["object.holder = user"])


if __name__ == "__main__":
    unittest.main()
