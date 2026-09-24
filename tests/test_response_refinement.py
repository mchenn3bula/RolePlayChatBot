import copy
import json
import unittest

from posttraining.response_refinement import fresh_scenes, refine, RULES


class ResponseRefinementTests(unittest.TestCase):
    def test_exact_single_rule_preserves_facts_history_and_user_input(self):
        scenes = list(fresh_scenes())
        self.assertEqual(len(scenes), 36)
        self.assertEqual(len({r["family"] for r in scenes}), 6)
        self.assertEqual(len({r["id"] for r in scenes}), 36)
        for r in scenes:
            original = copy.deepcopy(r["messages"])
            changed = refine(r["messages"], r["language"])
            self.assertEqual(original, r["messages"])
            self.assertEqual(original[1:], changed[1:])
            self.assertEqual(changed[0]["content"].replace("\n"+RULES[r["language"]], "", 1), original[0]["content"])
            self.assertEqual(changed[0]["content"].rsplit("\n", 1)[1], original[0]["content"].rsplit("\n", 1)[1])
            with self.assertRaises(ValueError):
                refine(changed, r["language"])

    def test_fresh_state_updates_follow_complete_old_exchanges(self):
        for r in fresh_scenes():
            state = json.loads(r["messages"][0]["content"].rsplit("\n", 1)[1])
            if len(r["messages"]) == 4:
                self.assertEqual(state["through_user_turn"], 2)
                current = next(f for f in state["facts"] if f["predicate"] == "established")
                self.assertEqual(current["source"]["turn"], 2)
                self.assertEqual(r["messages"][1]["role"], "user")
                self.assertEqual(r["messages"][2]["role"], "assistant")


if __name__ == "__main__":
    unittest.main()
