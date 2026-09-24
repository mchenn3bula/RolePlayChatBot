"""Synthetic authored fixtures only; never load real model replies."""

import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path

from posttraining.artifacts import ReplyWriter
from posttraining.build_state_fixtures import build, fact
from posttraining.chat import reload_state
from posttraining.evaluate import run_panel, validate_states
from posttraining.state import build_context, snapshot, validate_state


class StateTests(unittest.TestCase):
    def test_future_events_are_absent_and_snapshots_are_detached(self):
        document = build(10, "en")
        initial = snapshot(document, 1)
        self.assertNotIn("Tomorrow morning", json.dumps(initial))
        self.assertNotIn("events", initial)
        self.assertEqual(initial["applied_event_turns"], [])
        later = snapshot(document, 3)
        self.assertEqual(later["applied_event_turns"], [2, 3])
        self.assertIn("Tomorrow morning", json.dumps(later))
        later["persona"]["voice"] = "changed"
        self.assertNotEqual(document["persona"]["voice"], "changed")

    def test_immutable_persona_and_stable_fact_ids(self):
        for removal in (False, True):
            document = build(1, "en")
            document["events"] = [{"turn": 1, "set": [] if removal else [
                fact("character", "role", "Changed", scope="persona", turn=1)],
                "remove": ["character.role"] if removal else []}]
            with self.assertRaises(ValueError):
                validate_state(document)
        document = build(1, "en")
        change = fact("user", "location", "Somewhere", turn=1)
        change["id"] = "character.location"
        document["events"] = [{"turn": 1, "set": [change], "remove": []}]
        with self.assertRaisesRegex(ValueError, "identity"):
            validate_state(document)

    def test_unknowns_claims_and_user_provenance(self):
        document = build(1, "en")
        claim = fact("user", "reported_weather", {"by": "user", "text": "Sunny"}, status="claim", turn=1)
        document["events"] = [{"turn": 1, "set": [claim], "remove": []}]
        current = {f["id"]: f for f in snapshot(document, 1)["facts"]}
        self.assertEqual(current["user.reported_weather"]["status"], "claim")
        self.assertIsNone(current["user.name"]["value"])
        claim["source"]["kind"] = "assistant"
        with self.assertRaisesRegex(ValueError, "authored/user"):
            validate_state(document)

    def test_invalid_sources_ids_references_and_slots(self):
        mutations = [
            lambda d: d["facts"].append(copy.deepcopy(d["facts"][0])),
            lambda d: d["facts"][0]["source"].update(turn=3),
            lambda d: d["facts"][0].update(value={"entity_id": "missing"}),
            lambda d: d["facts"][0].update(subject=[]),
            lambda d: d["facts"].append({**d["facts"][0], "id": "different"}),
        ]
        for mutate in mutations:
            document = build(1, "en")
            mutate(document)
            with self.assertRaises(ValueError):
                validate_state(document)

    def test_turn_budget_preserves_state_and_latest_user(self):
        history = [{"role": role, "content": f"Synthetic message {i}"}
                   for i, role in enumerate(["user", "assistant", "user", "assistant", "user"])]
        document = build(1, "en")
        messages, audit = build_context("Synthetic persona", "en", history, document, 3,
                                        lambda messages: len(messages) * 100, 400)
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "user"])
        self.assertEqual(messages[-1], history[-1])
        self.assertIn("character.role", messages[0]["content"])
        self.assertEqual(audit["dropped_history_turns"], [1])
        self.assertEqual(len(history), 5)
        with self.assertRaisesRegex(ValueError, "nothing was truncated"):
            build_context("Synthetic persona", "en", history, document, 3, lambda m: 900, 400)
        with self.assertRaisesRegex(ValueError, "languages differ"):
            build_context("Synthetic persona", "fr", history, document, 3, len, 400)

    def test_state_replay_tracks_transfer_and_deadline(self):
        for language in ("en", "fr"):
            def values(number, turn):
                return {f["id"]: f["value"] for f in snapshot(build(number, language), turn)["facts"]}
            self.assertEqual(values(9, 1)["parcel.holder"], {"entity_id": "user"})
            self.assertEqual(values(9, 2)["parcel.holder"], {"entity_id": "character"})
            self.assertEqual(values(9, 5)["parcel.holder"], {"entity_id": "character"})
            self.assertEqual(values(9, 4)["umbrella.holder"], {"entity_id": "user"})
            self.assertEqual(values(10, 3)["blue_atlas.reserved_until"], values(10, 5)["blue_atlas.reserved_until"])
            self.assertNotEqual(values(10, 1)["blue_atlas.reserved_until"], values(10, 3)["blue_atlas.reserved_until"])

    def test_reload_requires_revision_and_same_scene(self):
        original = build(1, "en")
        candidate = copy.deepcopy(original)
        candidate["persona"]["voice"] = "Synthetic change"
        with self.assertRaisesRegex(ValueError, "revision"):
            reload_state(original, candidate)
        candidate["revision"] += 1
        self.assertEqual(reload_state(original, candidate), candidate)
        candidate["scene_id"] = "another"
        with self.assertRaisesRegex(ValueError, "scene identity"):
            reload_state(original, candidate)

    def test_fixture_coverage(self):
        root = Path(__file__).resolve().parents[1]
        panel = json.loads((root / "posttraining/eval_scenarios_v1.json").read_text(encoding="utf-8"))
        states = json.loads((root / "posttraining/eval_states_v1.json").read_text(encoding="utf-8"))
        validate_states(panel, states)
        self.assertEqual(len(states), 20)

    def test_file_output_escapes_markup_and_does_not_print_replies(self):
        marker = "<script>SYNTHETIC_OUTPUT_ONLY</script>"

        class FakeRuntime:
            count_tokens = staticmethod(lambda messages: len(messages) * 100)

            def generate(self, messages, seed):
                return {"text": marker, "output_ids": [1, 2], "seconds": 0.5,
                        "tokens_per_second": 4, "memory": {"reserved_gib": 0}}

        panel = {"scenarios": [{"id": "synthetic", "family": "fixture", "language": "en",
                                "persona": "Synthetic character", "turns": [{"user": "Synthetic input"}]}]}
        stdout = io.StringIO()
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(stdout):
            with ReplyWriter(directory) as writer:
                metadata = run_panel(FakeRuntime(), {"seeds": [42], "max_context_tokens": 2048},
                                     panel, {"synthetic": build(1, "en")}, writer)
                # Read only our fake-runtime fixture, never an actual generation artifact.
                raw = json.loads((Path(directory) / "generations.jsonl").read_text(encoding="utf-8"))
                self.assertEqual(raw["text"], marker)
            html = (Path(directory) / "replies.html").read_text(encoding="utf-8")
            self.assertNotIn(marker, html)
            self.assertIn("&lt;script&gt;", html)
            self.assertNotIn("SYNTHETIC_OUTPUT_ONLY", stdout.getvalue())
            self.assertNotIn("text", metadata[0])
            self.assertNotIn("repetition", json.dumps(metadata))


if __name__ == "__main__":
    unittest.main()
