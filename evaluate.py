"""Evaluate a saved transformer checkpoint on the tokenized test split."""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="bluemoon_test_tok_ds")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive.")

    from workflow import load_checkpoint

    model, tokenizer, device = load_checkpoint(args.checkpoint, args.device)
    model.evaluate_perplexity(
        dataset_path=args.data_dir,
        tokenizer=tokenizer,
        device=device,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
