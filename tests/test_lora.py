"""LoRA preparation/training checks using only authored synthetic content."""

import copy
import importlib.util
from pathlib import Path
import unittest

from posttraining.lora_data import encode_conversation, whole_messages
from posttraining.lora_train import lr_at, train_group


class Encoder:
    eos_id = 2

    def __call__(self, messages):
        result = [1]
        for message in messages:
            role = message["role"]
            result.extend({"system": [4], "user": [5], "assistant": []}[role])
            result.extend([20 + ord(c) % 40 for c in message["content"]])
            result.append(2 if role == "assistant" else 6)
        return result


class LoraDataTests(unittest.TestCase):
    def messages(self):
        return [{"role": "system", "content": "Synthetic."},
                {"role": "user", "content": "Hello."},
                {"role": "assistant", "content": "Welcome."},
                {"role": "user", "content": "Goodbye."},
                {"role": "assistant", "content": "See you."}]

    def test_every_reply_including_eos_is_supervised_and_context_is_masked(self):
        messages, encoder = self.messages(), Encoder()
        row = encode_conversation(messages, encoder, 2048)
        supervised = set()
        for index in (2, 4):
            start, end = len(encoder(messages[:index])), len(encoder(messages[:index+1]))
            supervised.update(range(start, end))
            self.assertEqual(row["labels"][end-1], encoder.eos_id)
        self.assertEqual({i for i, label in enumerate(row["labels"]) if label != -100}, supervised)
        self.assertEqual(row["target_tokens"], len(supervised))
        self.assertEqual(row["input_ids"], encoder(messages))

    def test_rejects_entire_overlength_conversation_without_trimming(self):
        messages = self.messages()
        self.assertIsNone(encode_conversation(messages, Encoder(), 10))
        self.assertEqual(messages, self.messages())

    def test_template_prefix_mismatch_is_fatal(self):
        class BadEncoder(Encoder):
            def __call__(self, messages):
                return [len(messages)] + super().__call__(messages)
        with self.assertRaisesRegex(ValueError, "prefix"):
            encode_conversation(self.messages(), BadEncoder(), 2048)

    def test_complete_thread_structural_curation_and_order(self):
        rows = [{"thread_href": "/synthetic", "message": f"Authored fixture {i}",
                 "message_username": "A" if i % 2 == 0 else "B",
                 "message_timestamp": f"Jan 01, 2020 at 01:0{i} PM"} for i in range(4)]
        messages, reason = whole_messages(list(reversed(rows)))
        self.assertIsNone(reason)
        self.assertEqual(len(messages), 5)
        self.assertEqual([m["content"] for m in messages[1:]], [r["message"] for r in rows])
        changed = copy.deepcopy(rows)
        changed[2]["message_username"] = "C"
        self.assertEqual(whole_messages(changed)[1], "not_two_participants")
        changed = copy.deepcopy(rows)
        changed[2]["message_timestamp"] = changed[1]["message_timestamp"]
        self.assertEqual(whole_messages(changed)[1], "ambiguous_chronology")
        self.assertEqual(whole_messages(rows[:-1])[1], "incomplete_pair_count")

    def test_learning_rate_warmup_and_cosine_floor(self):
        config = {"learning_rate": 1e-4, "warmup_ratio": 0.1, "min_lr_ratio": 0.1}
        self.assertLess(lr_at(0, 30, config), lr_at(2, 30, config))
        self.assertEqual(lr_at(2, 30, config), 1e-4)
        self.assertAlmostEqual(lr_at(29, 30, config), 1e-5)

    def test_accumulation_is_weighted_by_supervised_tokens(self):
        import torch
        import torch.nn.functional as functional
        from types import SimpleNamespace

        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(32, 8)
                self.output = torch.nn.Linear(8, 32)

            def forward(self, input_ids, labels, **kwargs):
                logits = self.output(self.embedding(input_ids))[:, :-1]
                return SimpleNamespace(loss=functional.cross_entropy(
                    logits.reshape(-1, 32), labels[:, 1:].reshape(-1), ignore_index=-100))

        torch.manual_seed(7)
        model = Tiny()
        expected = copy.deepcopy(model)
        rows = [{"input_ids": [1, 4, 5, 2], "labels": [-100, -100, 5, 2], "target_tokens": 2},
                {"input_ids": [1, 8, 9, 10, 2], "labels": [-100, 8, 9, 10, 2], "target_tokens": 4}]
        runtime = SimpleNamespace(model=model, device="cpu", guard_memory=lambda: None)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        reference_optimizer = torch.optim.SGD(expected.parameters(), lr=0.01)
        reference_loss = sum(expected(input_ids=torch.tensor([r["input_ids"]]),
                                      labels=torch.tensor([r["labels"]])).loss * r["target_tokens"] / 6 for r in rows)
        reference_loss.backward()
        reference_optimizer.step()
        train_group(runtime, rows, optimizer, torch.amp.GradScaler("cuda", enabled=False), {"max_grad_norm": 1000})
        for actual, reference in zip(model.parameters(), expected.parameters()):
            torch.testing.assert_close(actual, reference)

    @unittest.skipUnless(importlib.util.find_spec("mistral_common"), "Native tokenizer tested in LoRA environment")
    def test_native_training_prefix_matches_serving_and_keeps_eos(self):
        from posttraining.lora_data import NativeEncoder, MODEL, MODEL_REVISION
        from huggingface_hub import snapshot_download
        from transformers import MistralCommonBackend
        path = Path(snapshot_download(MODEL, revision=MODEL_REVISION, local_files_only=True,
                                      allow_patterns=["*.json", "*.jinja", "*.txt", "README.md", "model-*.safetensors"]))
        encoder = NativeEncoder(path)
        serving = MistralCommonBackend.from_pretrained(path, local_files_only=True)
        messages = self.messages()
        row = encode_conversation(messages, encoder, 2048)
        for index, (start, end) in zip((2, 4), row["reply_ranges"]):
            prompt = serving.apply_chat_template(messages[:index], add_generation_prompt=True, return_dict=False)
            self.assertEqual(row["input_ids"][:start], prompt)
            self.assertEqual(row["labels"][end-1], serving.eos_token_id)
        changed = self.messages()
        changed[-1]["content"] = "Different future fixture."
        later = encode_conversation(changed, encoder, 2048)
        first_end = row["reply_ranges"][0][1]
        self.assertEqual(later["input_ids"][:first_end], row["input_ids"][:first_end])


if __name__ == "__main__":
    unittest.main()
