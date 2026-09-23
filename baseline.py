"""Train/evaluate the Markov baseline using raw train/test conversation pairs."""

import argparse
import random


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", default="bluemoon_train_ds")
    parser.add_argument("--test-dir", default="bluemoon_test_ds")
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    args = parser.parse_args()
    if args.order < 1 or args.max_new_tokens < 1:
        parser.error("order and max-new-tokens must be positive.")
    if args.prompt is not None and len(args.prompt.split()) < args.order:
        parser.error("The prompt must contain at least --order words.")

    from markov_baseline import MarkovChain

    random.seed(args.seed)
    model = MarkovChain(n=args.order)
    model.train_dataset(args.train_dir, text_field="input_text")
    perplexity = model.perplexity_dataset(args.test_dir, text_field="input_text")
    print(f"Markov baseline perplexity: {perplexity:.2f}")
    if args.prompt:
        print(model.respond_to(args.prompt, max_tokens=args.max_new_tokens))


if __name__ == "__main__":
    main()
