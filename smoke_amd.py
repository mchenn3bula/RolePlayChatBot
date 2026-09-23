"""Validate the fixed baseline on ROCm with real data and a resumed short run."""

import argparse
import json
import math
import platform
import time
from pathlib import Path

import torch
from datasets import load_from_disk

from conversation import validate_dataset_metadata, write_dataset_metadata
from workflow import load_checkpoint, load_tokenizer, read_config


def finite_tensors(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tensors(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tensors(item) for item in value)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", default="configs/colab_t4.json")
    parser.add_argument("--tokenizer", default="tokenizer")
    parser.add_argument("--data-dir", default="data/bluemoon_train_tok_ds")
    parser.add_argument("--validation-dir", default="data/bluemoon_validation_tok_ds")
    parser.add_argument("--micro-batch", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--examples", type=int, default=256)
    parser.add_argument("--precision", choices=("fp16", "bf16", "fp32"), default="fp16")
    args = parser.parse_args()
    if min(args.micro_batch, args.grad_accum) < 1:
        parser.error("Batch sizes must be positive.")
    if args.examples < 4 * args.micro_batch * args.grad_accum:
        parser.error("Use at least four effective batches to exercise resume.")
    if args.output_dir.exists():
        parser.error(
            "Choose a new output directory; smoke results are never overwritten."
        )
    if not torch.version.hip or not torch.cuda.is_available():
        parser.error("A working ROCm PyTorch GPU is required. See AMD_TRAINING.md.")
    devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    matching = [i for i, name in enumerate(devices) if "7900 XTX" in name]
    if len(matching) != 1:
        parser.error(f"Expected one RX 7900 XTX; detected {devices}.")
    torch.cuda.set_device(matching[0])
    device = f"cuda:{matching[0]}"
    torch.set_num_threads(4)
    torch.manual_seed(42)
    tokenizer = load_tokenizer(args.tokenizer)
    config = read_config(args.config)
    for source in (args.data_dir, args.validation_dir):
        validate_dataset_metadata(source, tokenizer)
    training = load_from_disk(args.data_dir)
    validation = load_from_disk(args.validation_dir)
    if len(training) < args.examples:
        parser.error("Training dataset is smaller than the requested sample.")
    # Copies under the smoke directory; original prepared splits stay intact.
    args.output_dir.mkdir(parents=True)
    for name, dataset, count in (
        ("train", training, args.examples),
        ("validation", validation, 16),
    ):
        destination = args.output_dir / name
        dataset.shuffle(seed=42).select(range(min(count, len(dataset)))).save_to_disk(
            str(destination)
        )
        write_dataset_metadata(destination, tokenizer)
    run = args.output_dir / "run"
    run.mkdir()
    config.save(run / "config.json")
    tokenizer.save_pretrained(str(run / "tokenizer"))
    model = config.build(len(tokenizer))
    initial_embedding = model.embed.weight.detach().clone()
    arguments = dict(
        dataset_path=str(args.output_dir / "train"),
        validation_path=str(args.output_dir / "validation"),
        tokenizer_name=str(run / "tokenizer"),
        save_path=run,
        epochs=1,
        micro_batch=args.micro_batch,
        grad_accum=args.grad_accum,
        lr=1.5e-4,
        warmup_updates=0,
        precision=args.precision,
        device=device,
        seed=42,
        log_every=1,
        save_every=2,
    )
    print(f"GPU: {devices[matching[0]]} | torch: {torch.__version__}", flush=True)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    model.train_model(**arguments, max_updates=2)
    first = torch.load(run / "latest.pt", map_location="cpu", weights_only=True)
    if first["global_step"] != 2 or first["epoch"] != 0:
        raise RuntimeError("The smoke run did not stop at the requested resume point.")
    if args.precision == "fp16" and not first["scaler"]:
        raise RuntimeError("FP16 checkpoint has no gradient scaler state.")
    first_step = first["global_step"]
    del first, model
    torch.cuda.empty_cache()
    resumed = config.build(len(tokenizer))
    history = resumed.train_model(**arguments, resume=run / "latest.pt")
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak_allocated = torch.cuda.max_memory_allocated() / 1024**3
    peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
    final = torch.load(run / "latest.pt", map_location="cpu", weights_only=True)
    if final["epoch"] != 1 or final["global_step"] <= first_step:
        raise RuntimeError("Resume did not complete the smoke epoch.")
    if not finite_tensors((final["model"], final["optimizer"])):
        raise FloatingPointError(
            "Checkpoint contains nonfinite weights/optimizer tensors."
        )
    if not all(p.grad is None or finite_tensors(p.grad) for p in resumed.parameters()):
        raise FloatingPointError("Last gradient update contains nonfinite values.")
    if torch.equal(initial_embedding, final["model"]["embed.weight"]):
        raise RuntimeError("Training did not change the model weights.")
    if not history or not all(math.isfinite(v) for v in history[-1].values()):
        raise FloatingPointError(
            "Training/validation metrics are missing or nonfinite."
        )
    del final, initial_embedding
    resumed.eval()
    probe = torch.tensor([[10, 20, tokenizer.eos_token_id]], device=device)
    with torch.no_grad():
        expected = resumed(probe)["main"].cpu()
    del resumed
    torch.cuda.empty_cache()
    reloaded, _, _ = load_checkpoint(str(run / "latest.pt"), device)
    with torch.no_grad():
        actual = reloaded(probe)["main"].cpu()
    torch.testing.assert_close(expected, actual, rtol=1e-5, atol=1e-6)
    progress = json.loads((run / "progress.json").read_text())
    result = {
        "status": "passed",
        "gpu": devices[matching[0]],
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device_total_gib": torch.cuda.get_device_properties(device).total_memory
        / 1024**3,
        "precision": args.precision,
        "micro_batch": args.micro_batch,
        "grad_accum": args.grad_accum,
        "examples": args.examples,
        "parameters": sum(p.numel() for p in reloaded.parameters()),
        "seconds_including_validation_and_checkpoint_io": elapsed,
        "examples_per_second_including_validation_and_checkpoint_io": args.examples
        / elapsed,
        "peak_training_allocated_gib": peak_allocated,
        "peak_training_reserved_gib": peak_reserved,
        "progress": progress,
        "history": history,
        "resume_and_reload_verified": True,
        "finite_final_gradients_weights_and_optimizer": True,
        "note": "One short training-split sample with 16 validation examples. This is runtime validation, not a chatbot-quality evaluation or full-training speed estimate.",
    }
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
