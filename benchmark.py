"""Measure CUDA/ROCm training speed and peak memory on prepared real examples."""

import argparse
import json
import math
import time
from itertools import islice
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/bluemoon_train_tok_ds")
    parser.add_argument("--config", default="configs/small.json")
    parser.add_argument("--tokenizer", default="tokenizer")
    parser.add_argument(
        "--precision", choices=("auto", "fp16", "bf16", "fp32"), default="auto"
    )
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=32)
    parser.add_argument(
        "--steps",
        type=int,
        default=256,
        help="Measured micro-batches, not optimizer updates.",
    )
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument(
        "--output", type=Path, default=Path("reports/local-benchmark.json")
    )
    args = parser.parse_args()
    if min(args.micro_batch, args.grad_accum, args.steps) < 1 or args.warmup_steps < 0:
        parser.error(
            "Batch sizes and steps must be positive; warmup cannot be negative."
        )

    import torch
    from datasets import load_from_disk
    from torch.utils.data import DataLoader

    from conversation import validate_dataset_metadata
    from training import autocast_context, choose_precision
    from workflow import load_tokenizer, read_config

    if not torch.cuda.is_available():
        parser.error(
            "GPU unavailable. See AMD_TRAINING.md for ROCm or LOCAL_TRAINING.md for NVIDIA."
        )
    torch.manual_seed(42)
    torch.set_num_threads(4)
    tokenizer = load_tokenizer(args.tokenizer)
    validate_dataset_metadata(args.data_dir, tokenizer)
    dataset = load_from_disk(args.data_dir)
    if not len(dataset):
        parser.error("Dataset is empty.")
    config = read_config(args.config)
    model = config.build(len(tokenizer)).cuda().train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1.5e-4, betas=(0.9, 0.95), weight_decay=0.01
    )
    loader = DataLoader(
        dataset.shuffle(seed=42),
        batch_size=args.micro_batch,
        collate_fn=model.make_collate_fn(tokenizer),
    )
    if len(loader) < args.steps + args.warmup_steps:
        parser.error("Not enough examples for the requested benchmark length.")
    iterator = iter(loader)
    precision = choose_precision("cuda", args.precision)
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    device_free, device_total = torch.cuda.mem_get_info()

    def backward_batch(batch, target_count, valid_count):
        ids, mask, labels = [value.cuda() for value in batch]
        with autocast_context("cuda", precision):
            output = model.forward_loss(ids, mask, labels)
            loss_sum, count = output["loss_sum"], output["target_count"]
            loss = loss_sum / target_count + model.lambda_lb * output["lb"] * (
                mask.sum() / valid_count
            )
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite benchmark loss.")
        scaler.scale(loss).backward()
        return ids.size(0), mask.sum().item(), count.item(), loss_sum.detach().item()

    def update():
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0, error_if_nonfinite=not scaler.is_enabled()
        )
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        return not scaler.is_enabled() or scaler.get_scale() >= old_scale

    print(f"GPU: {torch.cuda.get_device_name()} | precision: {precision}", flush=True)
    for _ in range(args.warmup_steps):
        optimizer.zero_grad(set_to_none=True)
        batch = next(iterator)
        backward_batch(batch, int((batch[2][:, 1:] != -100).sum()), int(batch[1].sum()))
        update()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    examples = tokens = targets = 0
    longest_sequence = 0
    total_loss = 0.0
    succeeded = skipped = measured = 0
    for group_start in range(0, args.steps, args.grad_accum):
        group = list(islice(iterator, min(args.grad_accum, args.steps - group_start)))
        group_targets = sum(int((batch[2][:, 1:] != -100).sum()) for batch in group)
        group_valid = sum(int(batch[1].sum()) for batch in group)
        for batch in group:
            longest_sequence = max(longest_sequence, batch[0].size(1))
            count, token_count, target_count, loss_sum = backward_batch(
                batch, group_targets, group_valid
            )
            examples += count
            tokens += token_count
            targets += target_count
            total_loss += loss_sum
            measured += 1
        if update():
            succeeded += 1
        else:
            skipped += 1
        print(f"Measured {measured}/{args.steps} micro-batches", flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    result = {
        "gpu": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "hip_runtime": torch.version.hip,
        "precision": precision,
        "successful_measured_updates": succeeded,
        "skipped_measured_updates": skipped,
        "model_config": args.config,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "micro_batch": args.micro_batch,
        "grad_accum": args.grad_accum,
        "context_limit": config.max_len,
        "longest_measured_sequence": longest_sequence,
        "measured_micro_batches": args.steps,
        "seconds": elapsed,
        "examples_per_second": examples / elapsed,
        "input_tokens_per_second": tokens / elapsed,
        "target_tokens_per_second": targets / elapsed,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        "gpu_total_gib": device_total / 1024**3,
        "gpu_free_gib_after_model_load": device_free / 1024**3,
        "training_examples": len(dataset),
        "estimated_training_hours_per_epoch": len(dataset)
        / (examples / elapsed)
        / 3600,
        "sample_target_nll": total_loss / targets,
        "note": "Short real-data benchmark; epoch estimate excludes validation, saving, and thermal/other-app variation. Benchmark weights are discarded.",
    }
    if not math.isfinite(result["sample_target_nll"]):
        raise RuntimeError("Benchmark produced a nonfinite loss.")
    if not succeeded:
        raise RuntimeError(
            "No measured optimizer updates succeeded; increase warmup or investigate FP16 overflow."
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
