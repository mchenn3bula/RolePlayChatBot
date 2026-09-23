"""Train the roleplay transformer and save weights, configuration, and tokenizer."""

import argparse
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="bluemoon_train_tok_ds")
    parser.add_argument(
        "--tokenizer",
        default="gpt2",
        help="Tokenizer name or local tokenizer directory.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("mini_ds_ckpts"))
    parser.add_argument(
        "--config",
        type=Path,
        help="Version 2 or 3 model configuration JSON; omit for the historical dense default.",
    )
    parser.add_argument(
        "--validation-dir",
        help="Separate validation split; saves best.pt based on validation perplexity.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--warmup-updates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument(
        "--precision", choices=("auto", "fp32", "fp16", "bf16"), default="auto"
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=250,
        help="Save resumable state every N optimizer updates.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume latest.pt or previous.pt in the original output directory.",
    )
    parser.add_argument(
        "--max-updates",
        type=int,
        help="Stop safely at this total optimizer-update count; omit on continuation.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--max-hours",
        type=float,
        help="Save and stop after this many hours per invocation, at a safe boundary. Does not change the planned LR schedule.",
    )
    args = parser.parse_args()
    if (
        min(
            args.epochs,
            args.micro_batch,
            args.grad_accum,
            args.log_every,
            args.cpu_threads,
            args.save_every,
        )
        < 1
        or args.lr <= 0
        or args.warmup_updates < 0
        or (args.max_updates is not None and args.max_updates < 1)
        or (
            args.max_hours is not None
            and (not math.isfinite(args.max_hours) or args.max_hours <= 0)
        )
    ):
        parser.error(
            "Epochs, batch sizes, learning rate, and optional max-hours must be positive; max-hours must be finite; warmup must be nonnegative."
        )
    if not Path(args.data_dir).is_dir():
        parser.error(f"Dataset not found: {args.data_dir}. Run prepare_data.py first.")
    if (
        not args.resume
        and args.output_dir.exists()
        and (not args.output_dir.is_dir() or any(args.output_dir.iterdir()))
    ):
        parser.error(
            "Output directory must be empty; choose a new --output-dir for each run."
        )
    if args.resume and (
        not args.resume.is_file()
        or args.resume.parent.resolve() != args.output_dir.resolve()
    ):
        parser.error("--resume must name latest.pt or previous.pt inside --output-dir.")

    import torch

    from conversation import validate_dataset_metadata
    from workflow import ModelConfig, load_tokenizer, read_config, resolve_device

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    tokenizer = load_tokenizer(
        str(args.output_dir / "tokenizer") if args.resume else args.tokenizer
    )
    validate_dataset_metadata(args.data_dir, tokenizer)
    if args.validation_dir:
        validate_dataset_metadata(args.validation_dir, tokenizer)
    config = (
        read_config(args.output_dir / "config.json")
        if args.resume
        else (read_config(args.config) if args.config else ModelConfig())
    )
    if args.resume and args.config and read_config(args.config) != config:
        parser.error("The requested config differs from the saved model config.")
    if args.no_gradient_checkpointing:
        config.gradient_checkpointing = False
    model = config.build(len(tokenizer))
    print(
        f"Architecture v{config.architecture_version}: {sum(p.numel() for p in model.parameters()):,} parameters | {config.position_encoding} | {config.ffn_type} | target-only projection: {config.target_only_projection}",
        flush=True,
    )
    print(
        f"Output: {args.output_dir.resolve()} | planned epochs: {args.epochs} | time cap: {args.max_hours} hours",
        flush=True,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        config.save(args.output_dir / "config.json")
        tokenizer.save_pretrained(str(args.output_dir / "tokenizer"))
    history = model.train_model(
        dataset_path=args.data_dir,
        tokenizer_name=str(args.output_dir / "tokenizer"),
        epochs=args.epochs,
        micro_batch=args.micro_batch,
        grad_accum=args.grad_accum,
        lr=args.lr,
        warmup_updates=args.warmup_updates,
        device=resolve_device(args.device),
        save_path=str(args.output_dir),
        validation_path=args.validation_dir,
        log_every=args.log_every,
        precision=args.precision,
        save_every=args.save_every,
        resume=str(args.resume) if args.resume else None,
        seed=args.seed,
        max_updates=args.max_updates,
        max_hours=args.max_hours,
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Checkpoints saved in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
