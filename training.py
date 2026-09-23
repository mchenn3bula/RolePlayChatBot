"""Mixed-precision training and resumable optimizer-boundary checkpoints."""

import json
import math
import random
from contextlib import nullcontext
from itertools import islice
from pathlib import Path
from time import monotonic, perf_counter

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from conversation import tokenizer_fingerprint, validate_dataset_metadata

CHECKPOINT_VERSION = 1


def choose_precision(device, requested="auto"):
    if requested not in {"auto", "fp32", "fp16", "bf16"}:
        raise ValueError("precision must be auto, fp32, fp16, or bf16.")
    device = torch.device(device)
    if device.type != "cuda":
        if requested not in {"auto", "fp32"}:
            raise ValueError("FP16/BF16 training requires a CUDA or ROCm GPU.")
        return "fp32"
    if requested in {"fp32", "fp16"}:
        return requested
    if torch.version.hip:
        # ROCm uses torch.cuda too, but NVIDIA capability numbers do not apply.
        # FP16 is the documented Radeon/WSL training path. BF16 is opt-in and
        # still needs a workload smoke test; the runtime query is not that test.
        if requested == "auto":
            return "fp16"
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():
                raise ValueError("This ROCm device reports no BF16 support; use fp16.")
        return "bf16"
    # Do not count software BF16 emulation on pre-Ampere cards such as the T4.
    native_bf16 = torch.cuda.get_device_capability(device)[0] >= 8
    if requested == "bf16" and not native_bf16:
        raise ValueError("This GPU has no native BF16 support; use fp16.")
    return ("bf16" if native_bf16 else "fp16") if requested == "auto" else requested


def autocast_context(device, precision):
    if precision == "fp32":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(torch.device(device).type, dtype=dtype)


def capture_rng():
    return {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_training_state(directory, state):
    """Keep the last two complete snapshots; ignore any interrupted .tmp write."""
    directory = Path(directory)
    temporary = directory / "latest.pt.tmp"
    latest = directory / "latest.pt"
    previous = directory / "previous.pt"
    torch.save(state, temporary)
    if latest.exists():
        latest.replace(previous)
    temporary.replace(latest)


def train_model(
    model,
    dataset_path,
    tokenizer_name="gpt2",
    epochs=1,
    micro_batch=8,
    grad_accum=4,
    lr=1e-4,
    warmup_updates=1000,
    device="cpu",
    save_path=None,
    validation_path=None,
    log_every=100,
    precision="auto",
    save_every=250,
    resume=None,
    seed=42,
    max_updates=None,
    max_hours=None,
):
    started = monotonic()
    log_started = perf_counter()
    logged_examples = 0
    if (
        min(epochs, micro_batch, grad_accum, log_every, save_every) < 1
        or not math.isfinite(lr)
        or lr <= 0
        or warmup_updates < 0
    ):
        raise ValueError("Training settings must be positive; warmup may be zero.")
    if max_updates is not None and max_updates < 1:
        raise ValueError("max_updates must be positive.")
    if max_hours is not None and (not math.isfinite(max_hours) or max_hours <= 0):
        raise ValueError("max_hours must be finite and positive.")
    if max_hours is not None and not save_path:
        raise ValueError("A time-limited run requires save_path for recovery.")
    if resume and not save_path:
        raise ValueError("Resuming requires the original output directory.")
    precision = choose_precision(device, precision)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    validate_dataset_metadata(dataset_path, tokenizer)
    if validation_path:
        validate_dataset_metadata(validation_path, tokenizer)
        if Path(validation_path).resolve() == Path(dataset_path).resolve():
            raise ValueError("Training and validation must use different datasets.")
    dataset = load_from_disk(dataset_path)
    if not len(dataset):
        raise ValueError("Training dataset is empty.")
    validation_fingerprint = (
        load_from_disk(validation_path)._fingerprint if validation_path else None
    )
    micro_batches = math.ceil(len(dataset) / micro_batch)
    total_updates = math.ceil(micro_batches / grad_accum) * epochs
    warmup_updates = min(warmup_updates, max(0, total_updates - 1))
    settings = {
        "epochs": epochs,
        "micro_batch": micro_batch,
        "grad_accum": grad_accum,
        "lr": lr,
        "warmup_updates": warmup_updates,
        "seed": seed,
        "precision": precision,
        "dataset_fingerprint": dataset._fingerprint,
        "dataset_size": len(dataset),
        "validation_fingerprint": validation_fingerprint,
        "tokenizer_fingerprint": tokenizer_fingerprint(tokenizer),
        "lambda_lb": model.lambda_lb,
    }
    if (
        model.variant["position_encoding"] != "learned"
        or model.variant["ffn_type"] != "gelu"
        or model.target_only_projection
    ):
        # Preserve old v2 checkpoint settings while guarding non-weight semantics
        # (e.g. RoPE theta) for new experiments, even through the Python API.
        settings["model_variant"] = model.variant
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_updates, total_updates
    )
    # Scale only FP16 gradients; BF16 has a much wider exponent range.
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    epoch = cursor = global_step = skipped_updates = 0
    total_loss, total_targets = 0.0, 0
    history = []
    best_perplexity = math.inf
    if save_path:
        Path(save_path).mkdir(parents=True, exist_ok=True)
    if resume:
        state = torch.load(resume, map_location="cpu", weights_only=True)
        if state.get("checkpoint_version") != CHECKPOINT_VERSION:
            raise ValueError(
                "Resume requires latest.pt/previous.pt, not a weights-only checkpoint."
            )
        if state["settings"] != settings:
            changed = [
                key for key in settings if state["settings"].get(key) != settings[key]
            ]
            raise ValueError(
                f"Resume settings or data changed: {', '.join(changed)}. Restore the original settings or start a new run."
            )
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        epoch, cursor, global_step = (
            state["epoch"],
            state["next_example"],
            state["global_step"],
        )
        skipped_updates = state["skipped_updates"]
        total_loss, total_targets = state["epoch_loss_sum"], state["epoch_target_count"]
        best_perplexity, history = state["best_perplexity"], state["history"]
        restore_rng(state["rng"])
        print(
            f"Resumed epoch {epoch + 1}, example {cursor}, optimizer update {global_step}",
            flush=True,
        )
        del state
    print(
        f"Training precision: {precision} | effective batch: {micro_batch * grad_accum}",
        flush=True,
    )

    def save():
        if save_path:
            print(f"Saving checkpoint at update {global_step}...", flush=True)
            save_training_state(
                save_path,
                {
                    "checkpoint_version": CHECKPOINT_VERSION,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "settings": settings,
                    "epoch": epoch,
                    "next_example": cursor,
                    "global_step": global_step,
                    "skipped_updates": skipped_updates,
                    "epoch_loss_sum": total_loss,
                    "epoch_target_count": total_targets,
                    "best_perplexity": best_perplexity,
                    "history": history,
                    "rng": capture_rng(),
                },
            )
            write_json(
                Path(save_path) / "progress.json",
                {
                    "completed_epochs": epoch,
                    "next_example": cursor,
                    "global_step": global_step,
                    "planned_epochs": epochs,
                    "skipped_fp16_updates": skipped_updates,
                    "precision": precision,
                    "history": history,
                },
            )
            print(f"Checkpoint saved: {Path(save_path) / 'latest.pt'}", flush=True)

    if not resume:
        save()  # Recovery is possible even before the first periodic checkpoint.

    def stop_for_time():
        # A per-invocation limit, like max_updates: do not change the planned
        # cosine schedule or resume compatibility when continuing another night.
        if max_hours is None or monotonic() - started < max_hours * 3600:
            return False
        save()
        write_json(
            Path(save_path) / "time_limit.json",
            {
                "max_hours": max_hours,
                "global_step": global_step,
                "epoch": epoch,
                "next_example": cursor,
                "elapsed_seconds": monotonic() - started,
            },
        )
        print(
            f"Time budget reached at update {global_step}; resume from latest.pt.",
            flush=True,
        )
        return True

    if stop_for_time():
        return history
    if max_updates is not None and global_step >= max_updates:
        return history
    while epoch < epochs:
        model.train()
        # Explicit ordering and cursor avoid re-reading/skipping thousands of
        # already trained batches, and do not consume the dropout RNG on resume.
        generator = torch.Generator().manual_seed(seed + epoch)
        order = torch.randperm(len(dataset), generator=generator).tolist()[cursor:]
        loader = DataLoader(
            dataset,
            batch_size=micro_batch,
            sampler=order,
            generator=torch.Generator().manual_seed(seed + epoch),
            collate_fn=model.make_collate_fn(tokenizer),
        )
        iterator = iter(loader)
        while group := list(islice(iterator, grad_accum)):
            if stop_for_time():
                return history
            target_count = sum(
                int((labels[:, 1:] != -100).sum()) for _, _, labels in group
            )
            valid_count = sum(int(mask.sum()) for _, mask, _ in group)
            optimizer.zero_grad(set_to_none=True)
            group_nll = 0.0
            for ids, mask, labels in group:
                ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
                with autocast_context(device, precision):
                    output = model.forward_loss(ids, mask, labels)
                    summed_loss = output["loss_sum"]
                    loss = summed_loss / target_count + model.lambda_lb * output[
                        "lb"
                    ] * (mask.sum() / valid_count)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        "Nonfinite loss; training stopped. Resume the last checkpoint after investigating."
                    )
                scaler.scale(loss).backward()
                group_nll += summed_loss.detach().item()
            scaler.unscale_(optimizer)
            # FP16 overflow is handled by GradScaler, which skips the unsafe step.
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=not scaler.is_enabled()
            )
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            succeeded = not scaler.is_enabled() or scaler.get_scale() >= old_scale
            if succeeded:
                scheduler.step()
                global_step += 1
            else:
                skipped_updates += 1
                print(
                    f"FP16 overflow: skipped update | total skipped {skipped_updates} | scale {scaler.get_scale():.0f}",
                    flush=True,
                )
            group_examples = sum(ids.size(0) for ids, _, _ in group)
            cursor += group_examples
            logged_examples += group_examples
            total_loss += group_nll
            total_targets += target_count
            if succeeded and (global_step == 1 or global_step % log_every == 0):
                elapsed = perf_counter() - log_started
                rate = logged_examples / max(elapsed, 1e-9)
                eta_minutes = (len(dataset) - cursor) / max(rate, 1e-9) / 60
                memory = ""
                if torch.device(device).type == "cuda":
                    memory = f" | VRAM reserved {torch.cuda.memory_reserved(device) / 1024**3:.2f} GiB"
                print(
                    f"epoch {epoch + 1}/{epochs} | update {global_step}/{total_updates} | examples {cursor}/{len(dataset)} | target NLL {group_nll / target_count:.4f} | LR {scheduler.get_last_lr()[0]:.2e} | {rate:.1f} examples/s | elapsed {elapsed / 3600:.2f}h | epoch ETA ~{eta_minutes:.1f}m | skipped {skipped_updates}{memory}",
                    flush=True,
                )
            if succeeded and global_step % save_every == 0:
                save()
            if max_updates is not None and global_step >= max_updates:
                save()
                print(
                    f"Stopped at requested update {global_step}; resume from latest.pt.",
                    flush=True,
                )
                return history
        if stop_for_time():
            return history
        metrics = {"epoch": epoch + 1, "train_nll": total_loss / total_targets}
        print(
            f"epoch {epoch + 1} | train target NLL {metrics['train_nll']:.4f}",
            flush=True,
        )
        if validation_path:
            save()  # A disconnect in validation must not repeat an epoch of training.
            perplexity = model.evaluate_perplexity(
                validation_path, tokenizer, device, micro_batch, label="Validation"
            )
            metrics["validation_perplexity"] = perplexity
            if perplexity < best_perplexity:
                best_perplexity = perplexity
                if save_path:
                    model._save_weights(Path(save_path) / "best.pt")
        if save_path:
            model._save_weights(Path(save_path) / f"ckpt_ep{epoch + 1}.pt")
        history.append(metrics)
        epoch += 1
        cursor, total_loss, total_targets = 0, 0.0, 0
        if save_path:
            write_json(Path(save_path) / "metrics.json", history)
        save()
        if stop_for_time():
            return history
    return history
