"""Matched chosen-exposure continued-SFT and standard-DPO controls from immutable S1."""

import argparse
import math
from pathlib import Path
import random
import shutil
import time

from posttraining import dpo_core, dpo_train, lora_train
from posttraining.runtime import file_hash, read_config


def chosen_rows(pairs):
    """The SFT objective must never receive rejected labels or prior assistant targets."""
    return [p["chosen"] for p in pairs]


def update(runtime, pairs, optimizer, scaler, config):
    if config["objective"] == "sft_chosen":
        loss, norm, retries = lora_train.train_group(runtime, chosen_rows(pairs), optimizer, scaler, config)
        return loss, None, norm, retries
    if config["objective"] == "sigmoid_dpo":
        return dpo_core.train_group(runtime, pairs, optimizer, scaler, config)
    raise ValueError("Unknown control objective.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = read_config(args.config)
    if config["objective"] not in ("sft_chosen", "sigmoid_dpo") or config["lora_dropout"] != 0 or config["micro_batch"] != 1:
        raise ValueError("Fixed two-arm control settings required.")
    if config["reference_free"] or config["length_normalization"] or config["label_smoothing"]:
        raise ValueError("Control is standard DPO / reply-only SFT only.")
    if args.resume:
        if args.resume.resolve().parent != args.output_dir.resolve() or (args.output_dir / "complete.json").exists():
            raise ValueError("Resume requires a checkpoint inside an unfinished run.")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        source = args.output_dir / "source_snapshot"
        source.mkdir()
        for path in (args.config, Path(__file__), Path("posttraining/dpo_train.py"), Path("posttraining/dpo_core.py"),
                     Path("posttraining/lora_train.py"), Path("posttraining/runtime.py")):
            shutil.copyfile(path, source / path.name)
    started = time.perf_counter()
    try:
        import torch
        runtime = dpo_train.make_runtime(config)
        if args.smoke:
            if config["objective"] == "sft_chosen":
                lora_train.smoke(runtime, config, args.output_dir)
            else:
                dpo_train.smoke(runtime, config, args.output_dir)
            return
        splits, manifest = dpo_train.load_pairs(config)
        if file_hash(runtime.snapshot / "tekken.json") != manifest["tokenizer_sha256"]:
            raise ValueError("Tokenizer mismatch.")
        ident = dpo_train.identity(config, manifest)
        ident["control_trainer_sha256"] = file_hash(__file__)
        ident["protocol_sha256"] = file_hash("BILINGUAL_CONTROL_PROTOCOL.md")
        cache_started = time.perf_counter()
        cache_hash = dpo_train.cache_reference(runtime, splits, args.output_dir / "reference_logps.json", ident)
        cache_seconds = time.perf_counter()-cache_started
        ident["reference_cache_sha256"] = cache_hash
        optimizer = torch.optim.AdamW([p for p in runtime.model.parameters() if p.requires_grad], lr=config["learning_rate"],
                                     betas=(0.9, 0.95), weight_decay=config["weight_decay"])
        scaler = torch.amp.GradScaler("cuda", init_scale=128)
        total = math.ceil(len(splits["train"])/config["accumulation"])*config["epochs"]
        progress = {"epoch": 0, "offset": 0, "update": 0, "overflow_retries": 0, "chosen_presentations": 0,
                    "optimization_seconds": 0.0, "validation_history": []}
        if args.resume:
            progress = lora_train.restore(args.resume, runtime, optimizer, scaler, ident)
        else:
            progress["initial_chosen_nll"] = lora_train.validate(runtime, chosen_rows(splits["validation"]))
            progress["initial_preference"] = dpo_core.preference_metrics(runtime, splits["validation"], config["beta"])
            if abs(progress["initial_preference"]["loss"]-math.log(2)) > 1e-5:
                raise ValueError("Initial policy/reference mismatch.")
            lora_train.save_checkpoint(args.output_dir, "initial", runtime, optimizer, scaler, progress, ident)
        lora_train.write_json(args.output_dir / "run.json", {"identity": ident, "planned_updates": total,
            "matched_axis": "chosen presentations and optimizer updates; not FLOPs/wall time",
            "per_epoch": {"chosen_target_tokens": sum(p["chosen"]["target_tokens"] for p in splits["train"]),
                          "rejected_target_tokens": sum(p["rejected"]["target_tokens"] for p in splits["train"]),
                          "chosen_sequence_tokens": sum(len(p["chosen"]["input_ids"]) for p in splits["train"]),
                          "rejected_sequence_tokens": sum(len(p["rejected"]["input_ids"]) for p in splits["train"])},
            "reference_cache_seconds_this_invocation": cache_seconds})
        while progress["epoch"] < config["epochs"]:
            order = list(range(len(splits["train"])))
            random.Random(config["seed"]+progress["epoch"]).shuffle(order)
            while progress["offset"] < len(order):
                indices = order[progress["offset"]:progress["offset"]+config["accumulation"]]
                group = [splits["train"][i] for i in indices]
                lr = lora_train.lr_at(progress["update"], total, config)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr
                before = time.perf_counter()
                loss, margin, norm, retries = update(runtime, group, optimizer, scaler, config)
                progress["optimization_seconds"] += time.perf_counter()-before
                progress["offset"] += len(indices)
                progress["chosen_presentations"] += len(indices)
                progress["update"] += 1
                progress["overflow_retries"] += retries
                print(f"{config['objective']} {progress['update']}/{total} | epoch {progress['epoch']+1} | "
                      f"loss {loss:.4f} | LR {lr:.2e} | grad {norm:.3f} | chosen {progress['chosen_presentations']} | "
                      f"reserved {runtime.memory()['reserved_gib']:.2f} GiB | elapsed {(time.perf_counter()-started)/60:.1f}m", flush=True)
                if time.perf_counter()-started >= config["max_hours"]*3600:
                    lora_train.save_checkpoint(args.output_dir, f"time-stop-{progress['update']:05d}", runtime,
                                               optimizer, scaler, progress, ident)
                    print("Time cap reached at optimizer boundary. Resume available.", flush=True)
                    return
            metrics = dpo_core.preference_metrics(runtime, splits["validation"], config["beta"])
            nll = lora_train.validate(runtime, chosen_rows(splits["validation"]))
            progress["epoch"] += 1
            progress["offset"] = 0
            progress["validation_history"].append({"epoch": progress["epoch"], "chosen_nll": nll, "preference": metrics})
            lora_train.save_checkpoint(args.output_dir, f"epoch-{progress['epoch']:02d}", runtime, optimizer, scaler, progress, ident)
            print(f"Validation chosen NLL {nll:.4f} | preference loss {metrics['loss']:.4f}", flush=True)
        for name, fingerprint in ident["reference_files"].items():
            if file_hash(Path(config["sft_reference"])/name) != fingerprint:
                raise ValueError("Immutable S1 reference changed.")
        if file_hash(args.output_dir / "reference_logps.json") != cache_hash:
            raise ValueError("Reference cache changed.")
        lora_train.write_json(args.output_dir / "complete.json", {"status": "complete", "progress": progress,
            "selected_checkpoint": f"epoch-{config['epochs']:02d}", "selection": "prespecified final epoch, no output tuning",
            "reference_unchanged": True, "elapsed_seconds": time.perf_counter()-started, "memory": runtime.memory()})
        print("Control training complete; preserved final adapter.", flush=True)
    except Exception as error:
        import traceback
        lora_train.write_json(args.output_dir / "failure.json", {"type": type(error).__name__, "frames": [
            {"file": Path(f.filename).name, "line": f.lineno, "function": f.name} for f in traceback.extract_tb(error.__traceback__)]})
        raise RuntimeError(f"Control stopped: {type(error).__name__}; technical report saved") from None


if __name__ == "__main__":
    main()
