"""Keep the natural-reference experiment a reversible single-value intervention."""

import copy
import json
from pathlib import Path
import re
import unittest

from posttraining.natural_reference import apply_edits, replace_value
from posttraining.preferences_v2 import rows


class NaturalReferenceTests(unittest.TestCase):
    def test_all_edits_preserve_history_state_and_sentence_boundaries(self):
        edits = json.loads(Path("posttraining/natural_reference_edits.json").read_text(encoding="utf-8"))
        counts = {"changed": 0, "unchanged": 0}
        for row in (r for r in rows() if r["split"] == "evaluation"):
            original = copy.deepcopy(row["messages"])
            spans = edits[row["template_family"]][row["language"]]
            changed, old_value = replace_value(original, spans)
            self.assertEqual(original, row["messages"])
            self.assertEqual(original[1:], changed[1:])
            prefix, raw = changed[0]["content"].rsplit("\n", 1)
            before_prefix, before_raw = original[0]["content"].rsplit("\n", 1)
            self.assertEqual(prefix, before_prefix)
            state = json.loads(raw)
            target = next(f for f in state["facts"] if f["predicate"] == "established")
            self.assertIsInstance(target["value"], str)
            self.assertEqual(re.findall(r"[.;!?]", target["value"]), re.findall(r"[.;!?]", old_value))
            target["value"] = old_value
            self.assertEqual(state, json.loads(before_raw))
            if spans:
                self.assertNotEqual(original, changed)
                counts["changed"] += 1
            else:
                self.assertEqual(original, changed)
                self.assertIn(row["template_family"], ("r2-film_session", "r2-song_sharing"))
                counts["unchanged"] += 1
        self.assertEqual(counts, {"changed": 12, "unchanged": 4})

    def test_missing_duplicate_and_nonreversible_spans_fail(self):
        for value, edits in (("You wait.", [["Missing", "The host"]]),
                             ("You wait. You leave.", [["You", "The host"]]),
                             ("You wait. The host watches.", [["You", "The host"]])):
            with self.assertRaises(ValueError):
                apply_edits(value, edits)


if __name__ == "__main__":
    unittest.main()
