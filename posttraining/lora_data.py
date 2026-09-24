"""Whole-thread random selection and native reply-only labels; never print dialogue."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random

from posttraining.runtime import file_hash
from prepare_data import TIMESTAMP_FORMAT, split_titles

SOURCE = "rickRossie/bluemoon_roleplay_chat_data_300k_messages"
REVISION = "f8cf6b0cbd69294b084d502e2806dc60b9f9c4a0"
PARQUET = "data/train-00000-of-00001-9276d1ce89875933.parquet"
MODEL = "mistralai/Ministral-3-3B-Instruct-2512-BF16"
MODEL_REVISION = "b6d637bef2393152b3da2b2fde72eecdee30557e"
SYSTEM = ("Continue this fictional roleplay as the second participant. Respond in the "
          "conversation's language. Respect established characters and scene facts, "
          "and leave the other participant's choices to them.")


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


class NativeEncoder:
    def __init__(self, snapshot):
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
        from mistral_common.protocol.instruct.validator import ValidationMode
        # Both user-ending prompts and assistant-ending completed training sequences.
        self.tokenizer = MistralTokenizer.from_file(Path(snapshot) / "tekken.json", mode=ValidationMode.agnostic)
        self.eos_id = self.tokenizer.instruct_tokenizer.tokenizer.eos_id

    def __call__(self, messages):
        from mistral_common.protocol.instruct.request import ChatCompletionRequest
        return self.tokenizer.encode_chat_completion(ChatCompletionRequest.from_openai(messages=messages)).tokens


def encode_conversation(messages, encoder, max_length):
    """Mask system/user tokens; supervise every complete assistant reply, including EOS."""
    if len(messages) < 5 or messages[0]["role"] != "system":
        raise ValueError("A complete training conversation needs at least two exchanges.")
    for i, message in enumerate(messages[1:]):
        if message["role"] != ("user" if i % 2 == 0 else "assistant"):
            raise ValueError("Conversation must alternate two participants.")
    if messages[-1]["role"] != "assistant":
        raise ValueError("Whole conversation must end on the supervised participant.")
    ids = encoder(messages)
    if len(ids) > max_length:
        return None  # Reject the entire conversation; never trim or split it.
    labels = [-100] * len(ids)
    ranges = []
    for index in range(2, len(messages), 2):
        prompt = encoder(messages[:index])
        completed = encoder(messages[:index+1])
        if ids[:len(prompt)] != prompt or ids[:len(completed)] != completed:
            raise ValueError("Native template changed a prior prefix; cannot safely mask labels.")
        if completed[-1] != encoder.eos_id or len(completed) <= len(prompt) + 1:
            raise ValueError("Missing native EOS or empty target.")
        labels[len(prompt):len(completed)] = ids[len(prompt):len(completed)]
        ranges.append([len(prompt), len(completed)])
    return {"input_ids": ids, "labels": labels, "reply_ranges": ranges,
            "target_tokens": sum(x != -100 for x in labels)}


def whole_messages(rows):
    """Strict structural curation only. Authors map consistently to user/assistant."""
    if len(rows) < 4 or len(rows) % 2:
        return None, "incomplete_pair_count"
    if len({r["thread_href"] for r in rows}) != 1:
        return None, "ambiguous_thread_title"
    if any(not isinstance(r[k], str) or not r[k].strip()
           for r in rows for k in ("message", "message_username", "message_timestamp", "thread_href")):
        return None, "missing_fields"
    try:
        ordered = sorted(rows, key=lambda r: datetime.strptime(r["message_timestamp"], TIMESTAMP_FORMAT))
    except ValueError:
        return None, "invalid_timestamp"
    if len({r["message_timestamp"] for r in rows}) != len(rows):
        return None, "ambiguous_chronology"
    authors = list(dict.fromkeys(r["message_username"] for r in ordered))
    if len(authors) != 2:
        return None, "not_two_participants"
    if any(r["message_username"] != authors[i % 2] for i, r in enumerate(ordered)):
        return None, "nonalternating_authors"
    messages = [{"role": "system", "content": SYSTEM}]
    messages.extend({"role": "user" if i % 2 == 0 else "assistant", "content": r["message"].strip()}
                    for i, r in enumerate(ordered))
    return messages, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/ministral-s1-whole-v1"))
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--train-conversations", type=int, default=256)
    parser.add_argument("--validation-conversations", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260924)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError("Data output already exists; use a new version.")
    from huggingface_hub import hf_hub_download, snapshot_download
    import pyarrow.parquet as pq
    raw_path = hf_hub_download(SOURCE, PARQUET, revision=REVISION, repo_type="dataset", local_files_only=True)
    snapshot = snapshot_download(MODEL, revision=MODEL_REVISION, local_files_only=True,
                                 allow_patterns=["*.json", "*.jinja", "*.txt", "README.md", "model-*.safetensors"])
    encoder = NativeEncoder(snapshot)
    rows = pq.read_table(raw_path).to_pylist()
    groups = defaultdict(list)
    for row in rows:
        groups[row["thread_title"]].append(row)
    # Match the historical title-based split before curation or sampling.
    eligible_titles = {title: None for title, group in groups.items()
                       if sum(bool(r["message"] and r["message"].strip()) for r in group) >= 4}
    train, validation, test = split_titles(eligible_titles, seed=42)
    # Exact duplicate threads are removed globally, including cross-split duplicates.
    # This is contamination prevention, not test-set scoring or tokenization.
    content_hash = {title: digest([r["message"].strip() if r["message"] else "" for r in sorted(
        group, key=lambda r: (r["message_timestamp"] or "", r["Unnamed: 0"]))]) for title, group in groups.items()}
    hash_counts = Counter(content_hash.values())
    outputs, report = {}, {}
    for split, titles, limit in (("train", train, args.train_conversations),
                                 ("validation", validation, args.validation_conversations)):
        shuffled = sorted(titles)
        random.Random(args.seed + (0 if split == "train" else 1)).shuffle(shuffled)
        chosen, counts = [], Counter()
        for title in shuffled:
            if hash_counts[content_hash[title]] > 1:
                counts["duplicate_thread"] += 1
                continue
            messages, reason = whole_messages(groups[title])
            if reason:
                counts[reason] += 1
                continue
            # A cheap upper bound on work, not a length truncation.
            if sum(len(m["content"]) for m in messages) > 40000:
                counts["very_long_thread"] += 1
                continue
            try:
                encoded = encode_conversation(messages, encoder, args.max_length)
            except Exception:
                counts["native_encoding_rejected"] += 1
                continue  # Never print potentially content-bearing exception strings.
            if encoded is None:
                counts["over_token_budget"] += 1
                continue
            chosen.append({"thread_id": digest([title, groups[title][0]["thread_href"]]),
                           "content_sha256": content_hash[title], "messages": messages, **encoded})
            if len(chosen) == limit:
                break
        outputs[split] = chosen
        report[split] = {"conversations": len(chosen), "rejection_counts_before_sample_filled": dict(counts),
                         "messages": sum(len(r["messages"])-1 for r in chosen),
                         "assistant_replies": sum(len(r["reply_ranges"]) for r in chosen),
                         "input_tokens": sum(len(r["input_ids"]) for r in chosen),
                         "target_tokens": sum(r["target_tokens"] for r in chosen),
                         "max_tokens": max((len(r["input_ids"]) for r in chosen), default=0)}
    if len(outputs["train"]) < 32 or len(outputs["validation"]) < 8:
        print(json.dumps(report, indent=2))
        raise RuntimeError("Insufficient complete conversations for the pilot; no truncation fallback.")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for split, chosen in outputs.items():
        for name, keys in ((f"{split}.jsonl", ("thread_id", "input_ids", "labels", "target_tokens", "reply_ranges")),
                           (f"conversations_{split}.jsonl", ("thread_id", "messages", "content_sha256"))):
            path = args.output_dir / name
            with path.open("x", encoding="utf-8") as handle:
                for row in chosen:
                    handle.write(json.dumps({k: row[k] for k in keys}, ensure_ascii=False) + "\n")
            hashes[name] = file_hash(path)
    manifest = {"format": "native-whole-conversation-reply-loss-v1", "source": SOURCE,
                "source_revision": REVISION, "source_sha256": file_hash(raw_path),
                "source_license": "not declared in source metadata; personal local experiment only",
                "model": MODEL, "model_revision": MODEL_REVISION,
                "tokenizer_sha256": file_hash(Path(snapshot) / "tekken.json"),
                "seed": args.seed, "historical_split_seed": 42,
                "original_split_threads": {"train": len(train), "validation": len(validation), "test": len(test)},
                "curation": "structural only; no semantic/language/content review",
                "completeness": "all available source-thread messages; narrative completion is unknown",
                "role_mapping": "first chronological author=user; second author=assistant; strict alternation",
                "state_labels": "unavailable; no inferred persona or scene labels",
                "truncated_conversations": 0, "test_tokenization_or_training": False,
                "max_length": args.max_length, "eos_id": encoder.eos_id,
                "splits": report, "files": hashes, "code_sha256": file_hash(__file__)}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "splits": report}, indent=2))


if __name__ == "__main__":
    main()
