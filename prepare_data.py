"""Group, split, and tokenize Bluemoon conversations without a notebook runtime."""

import argparse
import random
from datetime import datetime
from pathlib import Path

from conversation import encode_context, format_messages, write_dataset_metadata

DATASET_NAME = "rickRossie/bluemoon_roleplay_chat_data_300k_messages"
TIMESTAMP_FORMAT = "%b %d, %Y at %I:%M %p"


def group_threads(rows):
    threads = {}
    for row in rows:
        if row["message"] and row["message"].strip():
            timestamp = datetime.strptime(row["message_timestamp"], TIMESTAMP_FORMAT)
            threads.setdefault(row["thread_title"], []).append(
                (timestamp, row["message"].strip())
            )
    return {title: turns for title, turns in threads.items() if len(turns) >= 4}


def split_titles(threads, test_fraction=0.2, seed=42, validation_fraction=0.1):
    if not 0 < test_fraction < 1 or not 0 < validation_fraction < 1 - test_fraction:
        raise ValueError(
            "Validation and test fractions must be positive and sum to less than 1."
        )
    titles = sorted(threads)
    random.Random(seed).shuffle(titles)
    test_count = max(1, int(len(titles) * test_fraction))
    validation_count = max(1, int(len(titles) * validation_fraction))
    cut = len(titles) - test_count - validation_count
    if cut < 1:
        raise ValueError(
            "Need enough threads for nonempty train, validation, and test splits."
        )
    return (
        titles[:cut],
        titles[cut : cut + validation_count],
        titles[cut + validation_count :],
    )


def make_pairs(threads, titles):
    for title in titles:
        messages = [
            message for _, message in sorted(threads[title], key=lambda turn: turn[0])
        ]
        for index in range(len(messages) - 3):
            yield {
                "input_text": format_messages(messages[index : index + 3]),
                "target_text": messages[index + 3],
            }


def tokenize_batch(batch, tokenizer, context_length=2048, target_length=512):
    # Dynamic padding belongs in the collator, not in stored dataset examples.
    if context_length < 2 or target_length < 2:
        raise ValueError("Context and target limits must allow text plus EOS.")
    contexts = [
        encode_context(text, tokenizer, context_length) for text in batch["input_text"]
    ]
    targets = tokenizer(
        batch["target_text"],
        truncation=True,
        max_length=target_length - 1,
        padding=False,
        add_special_tokens=False,
    )
    return {
        "input_ids": contexts,
        "labels": [ids + [tokenizer.eos_token_id] for ids in targets["input_ids"]],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--target-length", type=int, default=512)
    args = parser.parse_args()
    if args.context_length < 2 or args.target_length < 2:
        parser.error("context-length and target-length must each be at least 2.")
    if (
        not 0 < args.test_fraction < 1
        or not 0 < args.validation_fraction < 1 - args.test_fraction
    ):
        parser.error(
            "Validation and test fractions must be positive and sum to less than 1."
        )
    paths = {
        name: args.output_dir / f"bluemoon_{name}_ds"
        for name in (
            "train",
            "validation",
            "test",
            "train_tok",
            "validation_tok",
            "test_tok",
        )
    }
    for path in paths.values():
        if path.exists():
            parser.error(f"Output already exists: {path}. Choose a fresh --output-dir.")

    from datasets import Dataset, load_dataset
    from tqdm.auto import tqdm

    from workflow import load_tokenizer

    rows = load_dataset(args.dataset, split="train", streaming=True)
    threads = group_threads(tqdm(rows, desc="Grouping messages"))
    train_titles, validation_titles, test_titles = split_titles(
        threads, args.test_fraction, args.seed, args.validation_fraction
    )
    tokenizer = load_tokenizer()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Threads: {len(train_titles)} train / {len(validation_titles)} validation / {len(test_titles)} test"
    )
    for split, titles in (
        ("train", train_titles),
        ("validation", validation_titles),
        ("test", test_titles),
    ):
        dataset = Dataset.from_generator(
            make_pairs, gen_kwargs={"threads": threads, "titles": titles}
        )
        dataset.save_to_disk(str(paths[split]))
        tokenized = dataset.map(
            tokenize_batch,
            batched=True,
            fn_kwargs={
                "tokenizer": tokenizer,
                "context_length": args.context_length,
                "target_length": args.target_length,
            },
            remove_columns=dataset.column_names,
            desc=f"Tokenizing {split}",
        )
        tokenized.save_to_disk(str(paths[f"{split}_tok"]))
        write_dataset_metadata(paths[f"{split}_tok"], tokenizer)
        print(f"Saved {len(dataset)} {split} examples")


if __name__ == "__main__":
    main()
