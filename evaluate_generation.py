"""Save a reproducible held-out generation panel, metrics, and human review CSV."""

import argparse
import csv
import hashlib
import json
import math
import random
import re
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repetition_metrics(text):
    words = re.findall(r"\b\w+(?:'\w+)?\b", text.lower())
    grams = list(zip(words, words[1:], words[2:], words[3:]))
    return {
        "words": len(words),
        "word_4gram_count": len(grams),
        "repeated_word_4gram_fraction": (
            1 - len(set(grams)) / len(grams) if grams else None
        ),
        "empty": not text.strip(),
    }


def summarize(rows):
    if not rows:
        raise ValueError("No generations to summarize.")
    repetitions = [
        row["repeated_word_4gram_fraction"]
        for row in rows
        if row["repeated_word_4gram_fraction"] is not None
    ]
    return {
        "generations": len(rows),
        "mean_generated_tokens_excluding_eos": sum(r["generated_tokens"] for r in rows)
        / len(rows),
        "empty_response_fraction": sum(r["empty"] for r in rows) / len(rows),
        "fewer_than_four_words_fraction": sum(r["words"] < 4 for r in rows) / len(rows),
        "eos_termination_fraction": sum(r["stopped_on_eos"] for r in rows) / len(rows),
        "length_limit_fraction": sum(not r["stopped_on_eos"] for r in rows) / len(rows),
        "mean_repeated_word_4gram_fraction": sum(repetitions) / len(repetitions)
        if repetitions
        else None,
        "responses_eligible_for_4gram_metric": len(repetitions),
        "note": "Repetition excludes responses under four words; inspect lengths and empty responses. These metrics do not measure fluency or relevance. Seeds are repeated samples of the same prompts, not independent prompts.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/bluemoon_validation_tok_ds")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if (
        args.samples < 1
        or args.max_new_tokens < 1
        or args.top_k < 0
        or not math.isfinite(args.temperature)
        or args.temperature < 0
        or not 0 < args.top_p <= 1
        or not math.isfinite(args.repetition_penalty)
        or args.repetition_penalty <= 0
        or len(set(args.seeds)) != len(args.seeds)
        or any(not 0 <= seed < 2**63 for seed in args.seeds)
    ):
        parser.error(
            "Use positive lengths/penalties, valid sampling settings, and unique nonnegative seeds below 2**63."
        )
    if args.output_dir.exists() and (
        not args.output_dir.is_dir() or any(args.output_dir.iterdir())
    ):
        parser.error("Choose a new or empty evaluation output directory.")

    import torch
    from datasets import load_from_disk

    from conversation import validate_dataset_metadata
    from workflow import load_checkpoint

    torch.set_num_threads(2)
    model, tokenizer, device = load_checkpoint(str(args.checkpoint), args.device)
    validate_dataset_metadata(args.data_dir, tokenizer)
    dataset = load_from_disk(str(args.data_dir))
    if args.samples > len(dataset):
        parser.error(
            f"Requested {args.samples} prompts but dataset has only {len(dataset)}."
        )
    indices = random.Random(args.selection_seed).sample(
        range(len(dataset)), args.samples
    )
    for index in indices:
        context = dataset[index]["input_ids"]
        if not context or context[-1] != tokenizer.eos_token_id:
            parser.error(f"Example {index} is missing its context/reply EOS boundary.")
        if len(context) + args.max_new_tokens > model.max_len:
            parser.error(
                "Context plus requested generation exceeds model length; reduce --max-new-tokens."
            )

    settings = {
        key: getattr(args, key)
        for key in (
            "max_new_tokens",
            "temperature",
            "top_k",
            "top_p",
            "repetition_penalty",
        )
    }
    manifest = {
        "evaluation_version": 1,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "model_config": json.loads(
            (args.checkpoint.parent / "config.json").read_text()
        ),
        "dataset": str(args.data_dir.resolve()),
        "dataset_files_sha256": {
            str(path.relative_to(args.data_dir)): sha256(path)
            for path in sorted(args.data_dir.rglob("*"))
            if path.is_file()
        },
        "selection_seed": args.selection_seed,
        "dataset_indices": indices,
        "generation_seeds": args.seeds,
        "generation_settings": settings,
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name() if device == "cuda" else device,
        "precision": "fp32 (matches chat.py inference; training uses mixed precision)",
        "source_sha256": {
            name: sha256(Path(__file__).parent / name)
            for name in (
                "evaluate_generation.py",
                "mini_deepseek.py",
                "workflow.py",
                "conversation.py",
            )
        },
        "reproducibility": "Exact context IDs and generated IDs are saved. Same seeds need not produce identical outputs across GPU types or PyTorch versions.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    rows = []
    with (
        (args.output_dir / "generations.jsonl").open("w", encoding="utf-8") as output,
        (args.output_dir / "human_review.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as review,
    ):
        fields = [
            "dataset_index",
            "seed",
            "context",
            "reference",
            "generation",
            "grammar_1_to_5",
            "relevance_1_to_5",
            "coherence_1_to_5",
            "notes",
        ]
        writer = csv.DictWriter(review, fieldnames=fields)
        writer.writeheader()
        for index in indices:
            example = dataset[index]
            for seed in args.seeds:
                torch.manual_seed(seed)
                tokens = model.generate_token_ids(
                    example["input_ids"],
                    stop_token=tokenizer.eos_token_id,
                    device=device,
                    **settings,
                )
                stopped = bool(tokens and tokens[-1] == tokenizer.eos_token_id)
                text = tokenizer.decode(tokens, skip_special_tokens=True)
                row = {
                    "dataset_index": index,
                    "seed": seed,
                    "context_ids": example["input_ids"],
                    "reference_ids": example["labels"],
                    "generated_ids": tokens,
                    "context": tokenizer.decode(
                        example["input_ids"], skip_special_tokens=True
                    ),
                    "reference": tokenizer.decode(
                        example["labels"], skip_special_tokens=True
                    ),
                    "generation": text,
                    "generated_tokens": len(tokens) - int(stopped),
                    "stopped_on_eos": stopped,
                    **repetition_metrics(text),
                }
                rows.append(row)
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()
                writer.writerow({key: row.get(key, "") for key in fields})
                review.flush()
            print(
                f"Generated {len(rows)}/{len(indices) * len(args.seeds)} replies",
                flush=True,
            )
    summary = summarize(rows)
    summary["by_seed"] = {
        str(seed): summarize([r for r in rows if r["seed"] == seed])
        for seed in args.seeds
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved evaluation: {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
