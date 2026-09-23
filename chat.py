"""Generate a continuation from a checkpoint and optional prior messages."""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument(
        "--context",
        action="append",
        default=[],
        help="Prior message; repeat in conversation order.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument(
        "--full-text",
        action="store_true",
        help="Include the prompt before the generated reply.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if (
        args.max_new_tokens < 1
        or args.temperature < 0
        or args.top_k < 0
        or not 0 < args.top_p <= 1
        or args.repetition_penalty <= 0
    ):
        parser.error(
            "Invalid generation settings: use positive lengths/penalties, nonnegative temperature/top-k, and 0 < top-p <= 1."
        )

    from workflow import load_checkpoint

    model, tokenizer, device = load_checkpoint(args.checkpoint, args.device)
    print(
        model.generate_chat(
            args.prompt,
            context=args.context,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            device=device,
            tokenizer=tokenizer,
            return_full_text=args.full_text,
        )
    )


if __name__ == "__main__":
    main()
