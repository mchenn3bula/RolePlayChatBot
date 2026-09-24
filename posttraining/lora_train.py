"""FP16/FP32 LoRA SFT, whole conversations, immutable checkpoints and content-free logs."""

import argparse
import copy
import json
import math
from pathlib import Path
import random
import shutil
import sys
import time

from posttraining.runtime import Runtime, file_hash, read_config


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def load_data(config):
    root = Path(config["data_dir"])
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["model_revision"] != config["revision"] or manifest["max_length"] != config["max_context_tokens"]:
        raise ValueError("Data/model/token budget mismatch.")
    splits = {}
    for split in ("train", "validation"):
        path = root / f"{split}.jsonl"
        if file_hash(path) != manifest["files"][path.name]:
            raise ValueError("Prepared data fingerprint changed.")
        with path.open(encoding="utf-8") as handle:
            splits[split] = [json.loads(line) for line in handle]
        for row in splits[split]:
            ids, labels = row["input_ids"], row["labels"]
            if (not 1 < len(ids) <= config["max_context_tokens"] or len(ids) != len(labels)
                    or labels[0] != -100 or ids[-1] != manifest["eos_id"]
                    or labels[-1] != manifest["eos_id"]
                    or row["target_tokens"] != sum(x != -100 for x in labels)
                    or row["target_tokens"] < 1
                    or any(label != -100 and label != token for token, label in zip(ids, labels))):
                raise ValueError("Invalid reply-only training record.")
    if {r["thread_id"] for r in splits["train"]} & {r["thread_id"] for r in splits["validation"]}:
        raise ValueError("Train/validation thread overlap.")
    return splits["train"], splits["validation"], manifest


def lr_at(update, total, config):
    warmup = max(1, math.ceil(total * config["warmup_ratio"]))
    if update < warmup:
        return config["learning_rate"] * (update + 1) / warmup
    phase = (update - warmup) / max(1, total - warmup - 1)
    return config["learning_rate"] * (config["min_lr_ratio"] +
        (1-config["min_lr_ratio"]) * (1+math.cos(math.pi * min(phase, 1))) / 2)


def make_model(config):
    import torch
    from peft import LoraConfig, get_peft_model
    runtime = Runtime(config)
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    runtime.model = get_peft_model(runtime.model, LoraConfig(
        r=config["rank"], lora_alpha=config["lora_alpha"], lora_dropout=config["lora_dropout"],
        target_modules=config["target_modules"], bias="none", task_type="CAUSAL_LM",
    ))
    runtime.model.config.use_cache = False
    runtime.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    runtime.model.enable_input_require_grads()
    trainable = [(n, p) for n, p in runtime.model.named_parameters() if p.requires_grad]
    if not trainable or any("lora_" not in n or "language_model" not in n for n, p in trainable):
        raise RuntimeError("Unexpected trainable parameters.")
    for _, param in trainable:
        param.data = param.data.float()
    runtime.trainable_parameters = sum(p.numel() for _, p in trainable)
    print(f"LoRA trainable parameters: {runtime.trainable_parameters:,} | FP32 adapters | "
          f"FP16 frozen backbone | checkpointing enabled", flush=True)
    return runtime


def batch(row, device):
    import torch
    ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids),
            "labels": torch.tensor([row["labels"]], dtype=torch.long, device=device)}


def train_group(runtime, group, optimizer, scaler, config):
    import torch
    runtime.model.train()
    total_targets = sum(row["target_tokens"] for row in group)
    for attempt in range(8):
        optimizer.zero_grad(set_to_none=True)
        weighted_loss = 0.0
        for row in group:
            # Variable whole-thread lengths otherwise accumulate large, differently
            # sized vocabulary-projection buffers in the allocator's cache.
            torch.cuda.empty_cache()
            runtime.guard_memory()
            with torch.autocast("cuda", dtype=torch.float16):
                loss = runtime.model(**batch(row, runtime.device), use_cache=False).loss
                weighted = loss * (row["target_tokens"] / total_targets)
            if not torch.isfinite(loss).item():
                raise RuntimeError("Nonfinite loss; refusing an invalid update.")
            weighted_loss += loss.detach().float().item() * row["target_tokens"] / total_targets
            scaler.scale(weighted).backward()
            del loss, weighted
            runtime.guard_memory()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(
            [p for p in runtime.model.parameters() if p.requires_grad], config["max_grad_norm"])
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= old_scale:
            if not torch.isfinite(norm).item():
                raise RuntimeError("Nonfinite gradients passed the scaler.")
            optimizer.zero_grad(set_to_none=True)
            runtime.guard_memory()
            return weighted_loss, float(norm), attempt
        print(f"FP16 overflow: retrying same update ({attempt+1}/8), scale {scaler.get_scale():.0f}", flush=True)
    raise RuntimeError("Repeated FP16 overflow; no progress was advanced.")


def validate(runtime, rows):
    import torch
    runtime.model.eval()
    numerator = 0.0
    denominator = 0
    with torch.inference_mode():
        for row in rows:
            torch.cuda.empty_cache()
            runtime.guard_memory()
            with torch.autocast("cuda", dtype=torch.float16):
                loss = runtime.model(**batch(row, runtime.device), use_cache=False).loss
            if not torch.isfinite(loss).item():
                raise RuntimeError("Nonfinite validation loss.")
            numerator += loss.float().item() * row["target_tokens"]
            denominator += row["target_tokens"]
            del loss
            runtime.guard_memory()
    return numerator / denominator


def compatibility(config, manifest):
    return {"config": config, "data_manifest": manifest,
            "trainer_sha256": file_hash(__file__),
            "runtime_sha256": file_hash(Path(__file__).with_name("runtime.py"))}


def save_checkpoint(root, name, runtime, optimizer, scaler, progress, identity):
    import torch
    final = root / name
    temporary = root / (".saving-" + name)
    if final.exists() or temporary.exists():
        raise ValueError("Checkpoint exists; refusing overwrite.")
    temporary.mkdir()
    runtime.model.save_pretrained(temporary, safe_serialization=True)
    torch.save({"optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                "python_rng": random.getstate(), "progress": copy.deepcopy(progress)}, temporary / "training_state.pt")
    write_json(temporary / "manifest.json", {"identity": identity, "progress": progress,
        "files": {n: file_hash(temporary / n) for n in
                  ("adapter_model.safetensors", "adapter_config.json", "training_state.pt")}})
    temporary.rename(final)
    write_json(root / "latest.json", {"checkpoint": name})
    print(f"Checkpoint saved: {final}", flush=True)
    return final


def restore(path, runtime, optimizer, scaler, identity):
    import torch
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["identity"] != identity:
        raise ValueError("Resume compatibility mismatch; use the original code/config/data.")
    if any(file_hash(path / name) != expected for name, expected in manifest["files"].items()):
        raise ValueError("Checkpoint fingerprint changed.")
    set_peft_model_state_dict(runtime.model, load_file(path / "adapter_model.safetensors"))
    saved = torch.load(path / "training_state.pt", map_location="cpu", weights_only=False)
    optimizer.load_state_dict(saved["optimizer"])
    scaler.load_state_dict(saved["scaler"])
    torch.set_rng_state(saved["torch_rng"])
    torch.cuda.set_rng_state_all(saved["cuda_rng"])
    random.setstate(saved["python_rng"])
    return saved["progress"]


def smoke(runtime, config, root):
    """Synthetic worst-length backward, optimizer, checkpoint and exact resume checks."""
    import torch
    from peft import get_peft_model_state_dict
    optimizer = torch.optim.AdamW([p for p in runtime.model.parameters() if p.requires_grad], lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", init_scale=128)
    probe = {"input_ids": [1, 101, 102, 103, 2], "labels": [-100, -100, 102, 103, 2], "target_tokens": 3}
    before = {k: v.detach().clone() for k, v in get_peft_model_state_dict(runtime.model).items()}
    results = []
    for length in (1024, config["max_context_tokens"]):
        row = {"input_ids": [1] + [101, 102] * ((length-2)//2) + [2]}
        row["labels"] = [-100] + row["input_ids"][1:]
        row["target_tokens"] = len(row["input_ids"]) - 1
        start = time.perf_counter()
        loss, norm, retries = train_group(runtime, [row], optimizer, scaler, config)
        results.append({"length": len(row["input_ids"]), "loss": loss, "grad_norm": norm,
                        "seconds": time.perf_counter()-start, "overflow_retries": retries, "memory": runtime.memory()})
        print(f"Synthetic backward passed at {length} tokens | reserved {runtime.memory()['reserved_gib']:.2f} GiB", flush=True)
    if not any(not torch.equal(before[k], v) for k, v in get_peft_model_state_dict(runtime.model).items()):
        raise RuntimeError("Adapters did not change.")
    if any(p.grad is not None for p in runtime.model.parameters() if not p.requires_grad):
        raise RuntimeError("Frozen backbone accumulated gradients.")
    identity = compatibility(config, {"synthetic": True})
    checkpoint = save_checkpoint(root, "smoke-checkpoint", runtime, optimizer, scaler, {"update": 2}, identity)
    expected = train_group(runtime, [probe], optimizer, scaler, config)
    expected_weights = {k: v.detach().clone() for k, v in get_peft_model_state_dict(runtime.model).items()}
    restore(checkpoint, runtime, optimizer, scaler, identity)
    actual = train_group(runtime, [probe], optimizer, scaler, config)
    difference = max((v - expected_weights[k]).abs().max().item()
                     for k, v in get_peft_model_state_dict(runtime.model).items())
    if difference != 0 or expected != actual:
        raise RuntimeError("Synthetic resumed update differs from uninterrupted update.")
    write_json(root / "smoke.json", {"status": "passed", "checks": results,
        "adapter_changed": True, "only_lora_trainable": True, "resume_max_abs_diff": difference,
        "trainable_parameters": runtime.trainable_parameters, "memory": runtime.memory()})
    print("Synthetic checkpoint/resume update is exact. No dialogue was generated.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/ministral_s1_lora.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = read_config(args.config)
    if config["micro_batch"] != 1:
        raise ValueError("This whole-conversation pilot uses micro-batch 1 without packing.")
    if args.resume:
        if not args.output_dir.is_dir() or args.resume.resolve().parent != args.output_dir.resolve():
            raise ValueError("Resume must belong to the specified run directory.")
        if (args.output_dir / "complete.json").exists():
            raise ValueError("Completed runs are immutable; start a named new experiment.")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        sources = args.output_dir / "source_snapshot"
        sources.mkdir()
        for source in (args.config, Path(__file__), Path(__file__).with_name("runtime.py"),
                       Path(__file__).with_name("lora_data.py")):
            shutil.copyfile(source, sources / source.name)
    import torch
    start = time.perf_counter()
    try:
        runtime = make_model(config)
        if args.smoke:
            smoke(runtime, config, args.output_dir)
            return 0
        train, validation, manifest = load_data(config)
        if file_hash(runtime.snapshot / "tekken.json") != manifest["tokenizer_sha256"]:
            raise ValueError("Tokenizer fingerprint changed.")
        identity = compatibility(config, manifest)
        optimizer = torch.optim.AdamW([p for p in runtime.model.parameters() if p.requires_grad],
                                     lr=config["learning_rate"], betas=(0.9, 0.95), weight_decay=config["weight_decay"])
        scaler = torch.amp.GradScaler("cuda", init_scale=128)
        total = math.ceil(len(train) / config["accumulation"]) * config["epochs"]
        progress = {"epoch": 0, "offset": 0, "update": 0, "overflow_retries": 0,
                    "best_validation_nll": math.inf, "best_checkpoint": None, "validation_history": []}
        if args.resume:
            progress = restore(args.resume, runtime, optimizer, scaler, identity)
        else:
            initial = validate(runtime, validation)
            progress["initial_validation_nll"] = initial
            print(f"Initial held-out reply NLL: {initial:.4f}", flush=True)
            save_checkpoint(args.output_dir, "initial", runtime, optimizer, scaler, progress, identity)
        write_json(args.output_dir / "run.json", {"identity": identity, "planned_updates": total,
                   "trainable_parameters": runtime.trainable_parameters, "review_policy": "no dialogue inspection"})
        while progress["epoch"] < config["epochs"]:
            order = list(range(len(train)))
            random.Random(config["seed"] + progress["epoch"]).shuffle(order)
            while progress["offset"] < len(order):
                indices = order[progress["offset"]:progress["offset"]+config["accumulation"]]
                lr = lr_at(progress["update"], total, config)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                loss, norm, retries = train_group(runtime, [train[i] for i in indices], optimizer, scaler, config)
                progress["offset"] += len(indices)
                progress["update"] += 1
                progress["overflow_retries"] += retries
                print(f"Update {progress['update']}/{total} | epoch {progress['epoch']+1}/{config['epochs']} | "
                      f"conversations {progress['offset']}/{len(train)} | target NLL {loss:.4f} | "
                      f"LR {lr:.2e} | grad {norm:.3f} | reserved {runtime.memory()['reserved_gib']:.2f} GiB | "
                      f"elapsed {(time.perf_counter()-start)/60:.1f}m", flush=True)
                if progress["update"] % config["checkpoint_every"] == 0:
                    save_checkpoint(args.output_dir, f"step-{progress['update']:05d}", runtime, optimizer, scaler, progress, identity)
                if time.perf_counter()-start >= config["max_hours"]*3600:
                    save_checkpoint(args.output_dir, f"time-stop-{progress['update']:05d}", runtime, optimizer, scaler, progress, identity)
                    print("Six-hour soft cap reached; checkpoint is resumable.", flush=True)
                    return 0
            score = validate(runtime, validation)
            progress["epoch"] += 1
            progress["offset"] = 0
            progress["validation_history"].append({"epoch": progress["epoch"], "reply_nll": score})
            name = f"epoch-{progress['epoch']:02d}"
            if score < progress["best_validation_nll"]:
                progress["best_validation_nll"] = score
                progress["best_checkpoint"] = name
            save_checkpoint(args.output_dir, name, runtime, optimizer, scaler, progress, identity)
            print(f"Validation epoch {progress['epoch']}: target NLL {score:.4f}", flush=True)
        write_json(args.output_dir / "complete.json", {"status": "complete", "progress": progress,
                   "elapsed_seconds": time.perf_counter()-start, "memory": runtime.memory(),
                   "review_policy": "technical loss only; no generated-text review"})
        print(f"Training complete. Selected adapter: {args.output_dir / progress['best_checkpoint']}", flush=True)
        return 0
    except Exception as error:
        # Keep text-bearing third-party exception messages out of terminal/tool output.
        import traceback
        frames = [{"file": Path(f.filename).name, "line": f.lineno, "function": f.name}
                  for f in traceback.extract_tb(error.__traceback__)]
        failure = {"error_type": type(error).__name__, "frames": frames}
        if "runtime" in locals():
            failure["memory"] = runtime.memory()
        write_json(args.output_dir / "failure.json", failure)
        print(f"Training stopped ({type(error).__name__}); technical failure location saved. No checkpoint overwritten.", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
