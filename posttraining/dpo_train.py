"""Standard offline sigmoid DPO from S1, with cached immutable S1 reference scores."""

import argparse
import json
import math
from pathlib import Path
import random
import shutil
import sys
import time

from posttraining.dpo_core import disable_dropout, preference_metrics, sequence_logp, train_group
from posttraining.lora_train import lr_at, restore, save_checkpoint, write_json
from posttraining.runtime import Runtime, file_hash, read_config


def make_runtime(config):
    import torch
    runtime = Runtime(config, adapter_path=config["sft_reference"])
    for name, param in runtime.model.named_parameters():
        param.requires_grad_("lora_" in name)
        if param.requires_grad:
            if "language_model" not in name:
                raise ValueError("Unexpected adapter module outside language decoder.")
            param.data = param.data.float()
    for peft_config in runtime.model.peft_config.values():
        if (peft_config.r, peft_config.lora_alpha) != (config["rank"], config["lora_alpha"]):
            raise ValueError("DPO adapter differs from SFT rank/alpha.")
        peft_config.inference_mode = False
    runtime.model.config.use_cache = False
    runtime.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    runtime.model.enable_input_require_grads()
    disable_dropout(runtime.model)
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    runtime.trainable_parameters = sum(p.numel() for p in runtime.model.parameters() if p.requires_grad)
    if runtime.trainable_parameters != 9371648:
        raise ValueError("Trainable parameter count differs from the fixed SFT adapter.")
    return runtime


def identity(config, data):
    reference = Path(config["sft_reference"])
    return {"config": config, "data_manifest": data,
            "reference_files": {n: file_hash(reference / n) for n in ("adapter_model.safetensors", "adapter_config.json")},
            "code": {str(p): file_hash(p) for p in (Path(__file__), Path("posttraining/dpo_core.py"),
                       Path("posttraining/lora_train.py"), Path("posttraining/runtime.py"))},
            "environment_lock": file_hash("requirements-lora-rocm.lock.txt")}


def load_pairs(config):
    root = Path(config["data_dir"])
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["model_revision"] != config["revision"] or manifest["max_length"] != config["max_context_tokens"]:
        raise ValueError("Preference data/model/budget mismatch.")
    result = {}
    for split in ("train", "validation"):
        path = root / f"{split}.jsonl"
        if file_hash(path) != manifest["files"][path.name]:
            raise ValueError("Preference data fingerprint mismatch.")
        result[split] = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if not result[split]:
            raise ValueError("Empty preference split.")
        for pair in result[split]:
            if pair["chosen"]["input_ids"] == pair["rejected"]["input_ids"]:
                raise ValueError("Identical preference candidates.")
            prefixes = []
            for side in ("chosen", "rejected"):
                row = pair[side]
                labels, ids = row["labels"], row["input_ids"]
                start = next(i for i, label in enumerate(labels) if label != -100)
                if (len(ids) != len(labels) or len(ids) > config["max_context_tokens"] or start < 1
                        or any(x != -100 for x in labels[:start]) or labels[start:] != ids[start:]
                        or ids[-1] != 2 or row["target_tokens"] != len(ids)-start):
                    raise ValueError("Invalid preference completion mask/EOS.")
                prefixes.append(ids[:start])
            if prefixes[0] != prefixes[1]:
                raise ValueError("Preference candidates do not share an exact prompt.")
    train_families = {r["template_family"] for r in result["train"]}
    if train_families & {r["template_family"] for r in result["validation"]}:
        raise ValueError("Preference template-family leakage.")
    return result, manifest


def cache_reference(runtime, splits, path, experiment_identity):
    import torch
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["identity"] != experiment_identity:
            raise ValueError("Reference cache identity mismatch.")
        scores = saved["scores"]
    else:
        runtime.model.eval()
        scores = {}
        pairs = [p for rows in splits.values() for p in rows]
        with torch.inference_mode():
            for index, pair in enumerate(pairs):
                torch.cuda.empty_cache()
                runtime.guard_memory()
                scores[pair["id"]] = {}
                for side in ("chosen", "rejected"):
                    score = sequence_logp(runtime, pair[side])
                    if not torch.isfinite(score):
                        raise RuntimeError("Nonfinite SFT reference log probability.")
                    scores[pair["id"]][side] = score.item()
                    del score
                runtime.guard_memory()
                if (index+1) % 20 == 0:
                    print(f"SFT reference cache: {index+1}/{len(pairs)} pairs", flush=True)
        write_json(path, {"identity": experiment_identity, "scores": scores})
    for pairs in splits.values():
        for pair in pairs:
            pair["ref_chosen"] = scores[pair["id"]]["chosen"]
            pair["ref_rejected"] = scores[pair["id"]]["rejected"]
    return file_hash(path)


def smoke(runtime, config, root):
    import torch
    from peft import get_peft_model_state_dict
    length = config["max_context_tokens"]
    prefix = [1] + [101] * (length//2-1)
    pair = {"id": "synthetic"}
    for side, token in (("chosen", 102), ("rejected", 103)):
        ids = prefix + [token] * (length-len(prefix)-1) + [2]
        pair[side] = {"input_ids": ids, "labels": [-100]*len(prefix)+ids[len(prefix):],
                      "target_tokens": len(ids)-len(prefix)}
    ident = identity(config, {"synthetic": True})
    cache_reference(runtime, {"train": [pair]}, root / "reference.json", ident)
    initial = preference_metrics(runtime, [pair], config["beta"])
    if abs(initial["loss"] - math.log(2)) > 1e-5:
        raise RuntimeError("Initial policy is not the cached SFT reference.")
    optimizer = torch.optim.AdamW([p for p in runtime.model.parameters() if p.requires_grad], lr=config["learning_rate"])
    scaler = torch.amp.GradScaler("cuda", init_scale=128)
    before = {k: v.detach().clone() for k, v in get_peft_model_state_dict(runtime.model).items()}
    train_group(runtime, [pair], optimizer, scaler, config)
    if not any(not torch.equal(before[k], v) for k, v in get_peft_model_state_dict(runtime.model).items()):
        raise RuntimeError("DPO update did not change policy parameters.")
    checkpoint = save_checkpoint(root, "smoke-checkpoint", runtime, optimizer, scaler, {"update": 1}, ident)
    expected = train_group(runtime, [pair], optimizer, scaler, config)
    weights = {k: v.detach().clone() for k, v in get_peft_model_state_dict(runtime.model).items()}
    restore(checkpoint, runtime, optimizer, scaler, ident)
    actual = train_group(runtime, [pair], optimizer, scaler, config)
    difference = max((v - weights[k]).abs().max().item() for k, v in get_peft_model_state_dict(runtime.model).items())
    if difference or expected != actual:
        raise RuntimeError("DPO resume differs from uninterrupted update.")
    if identity(config, {"synthetic": True}) != ident:
        raise RuntimeError("Frozen SFT reference or source changed.")
    write_json(root / "smoke.json", {"status": "passed", "initial": initial, "max_length": length,
        "resume_max_abs_diff": difference, "reference_unchanged": True, "memory": runtime.memory()})
    print(f"DPO smoke passed: initial loss log(2), {length}-token pairs, exact resume, immutable SFT reference.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/ministral_d1_dpo.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    config = read_config(args.config)
    if (config["objective"] != "sigmoid_dpo" or config["reference_free"] or config["length_normalization"]
            or config["label_smoothing"] or config["lora_dropout"]):
        raise ValueError("This runner implements only fixed-reference standard sigmoid DPO with dropout disabled.")
    if args.resume:
        if args.resume.resolve().parent != args.output_dir.resolve() or (args.output_dir / "complete.json").exists():
            raise ValueError("Resume must be inside an unfinished run.")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        source_dir = args.output_dir / "source_snapshot"
        source_dir.mkdir()
        for source in (args.config, Path(__file__), Path("posttraining/dpo_core.py"), Path("posttraining/dpo_data.py"),
                       Path("posttraining/lora_train.py"), Path("posttraining/runtime.py")):
            shutil.copyfile(source, source_dir / source.name)
    start = time.perf_counter()
    try:
        import torch
        runtime = make_runtime(config)
        if args.smoke:
            smoke(runtime, config, args.output_dir)
            return 0
        splits, manifest = load_pairs(config)
        if manifest["tokenizer_sha256"] != file_hash(runtime.snapshot / "tekken.json"):
            raise ValueError("Native tokenizer fingerprint mismatch.")
        ident = identity(config, manifest)
        cache_hash = cache_reference(runtime, splits, args.output_dir / "reference_logps.json", ident)
        ident["reference_cache_sha256"] = cache_hash
        optimizer = torch.optim.AdamW([p for p in runtime.model.parameters() if p.requires_grad],
                                     lr=config["learning_rate"], betas=(0.9, 0.95), weight_decay=config["weight_decay"])
        scaler = torch.amp.GradScaler("cuda", init_scale=128)
        total = math.ceil(len(splits["train"]) / config["accumulation"]) * config["epochs"]
        progress = {"epoch": 0, "offset": 0, "update": 0, "overflow_retries": 0,
                    "best_validation_loss": None, "best_checkpoint": None, "validation_history": []}
        if args.resume:
            progress = restore(args.resume, runtime, optimizer, scaler, ident)
        else:
            initial = preference_metrics(runtime, splits["validation"], config["beta"])
            if abs(initial["loss"]-math.log(2)) > 1e-5:
                raise RuntimeError("Initial DPO loss differs from log(2); reference mismatch.")
            progress["initial_validation"] = initial
            save_checkpoint(args.output_dir, "initial", runtime, optimizer, scaler, progress, ident)
        write_json(args.output_dir / "run.json", {"identity": ident, "planned_updates": total,
            "trainable_parameters": runtime.trainable_parameters, "reference": "cached exact SFT adapter, never disabled"})
        while progress["epoch"] < config["epochs"]:
            order = list(range(len(splits["train"])))
            random.Random(config["seed"] + progress["epoch"]).shuffle(order)
            while progress["offset"] < len(order):
                indices = order[progress["offset"]:progress["offset"]+config["accumulation"]]
                lr = lr_at(progress["update"], total, config)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                loss, margin, norm, retries = train_group(runtime, [splits["train"][i] for i in indices], optimizer, scaler, config)
                progress["offset"] += len(indices)
                progress["update"] += 1
                progress["overflow_retries"] += retries
                print(f"DPO {progress['update']}/{total} | epoch {progress['epoch']+1} | loss {loss:.4f} | "
                      f"relative margin {margin:.3f} | LR {lr:.2e} | grad {norm:.3f} | "
                      f"reserved {runtime.memory()['reserved_gib']:.2f} GiB | elapsed {(time.perf_counter()-start)/60:.1f}m", flush=True)
                if progress["update"] % config["checkpoint_every"] == 0:
                    save_checkpoint(args.output_dir, f"step-{progress['update']:05d}", runtime, optimizer, scaler, progress, ident)
                if time.perf_counter()-start >= config["max_hours"]*3600:
                    save_checkpoint(args.output_dir, f"time-stop-{progress['update']:05d}", runtime, optimizer, scaler, progress, ident)
                    print("Time cap reached at optimizer boundary; resume is available.", flush=True)
                    return 0
            metrics = preference_metrics(runtime, splits["validation"], config["beta"])
            progress["epoch"] += 1
            progress["offset"] = 0
            progress["validation_history"].append({"epoch": progress["epoch"], **metrics})
            name = f"epoch-{progress['epoch']:02d}"
            if progress["best_validation_loss"] is None or metrics["loss"] < progress["best_validation_loss"]:
                progress["best_validation_loss"] = metrics["loss"]
                progress["best_checkpoint"] = name
            save_checkpoint(args.output_dir, name, runtime, optimizer, scaler, progress, ident)
            print(f"Held-out synthetic preference loss {metrics['loss']:.4f}", flush=True)
        if file_hash(args.output_dir / "reference_logps.json") != cache_hash:
            raise RuntimeError("Reference cache changed during training.")
        for name, expected in ident["reference_files"].items():
            if file_hash(Path(config["sft_reference"]) / name) != expected:
                raise RuntimeError("SFT checkpoint changed during training.")
        write_json(args.output_dir / "complete.json", {"status": "complete", "progress": progress,
            "elapsed_seconds": time.perf_counter()-start, "memory": runtime.memory(), "reference_unchanged": True})
        print(f"DPO complete. Selected adapter: {args.output_dir / progress['best_checkpoint']}", flush=True)
        return 0
    except Exception as error:
        import traceback
        report = {"error_type": type(error).__name__, "frames": [
            {"file": Path(f.filename).name, "line": f.lineno, "function": f.name}
            for f in traceback.extract_tb(error.__traceback__)]}
        if "runtime" in locals():
            report["memory"] = runtime.memory()
        write_json(args.output_dir / "failure.json", report)
        print(f"DPO stopped ({type(error).__name__}); technical failure report saved.", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
