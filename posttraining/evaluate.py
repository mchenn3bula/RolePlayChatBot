"""Save bilingual replies for user review; log technical metadata only."""

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from posttraining.artifacts import ReplyWriter
from posttraining.runtime import Runtime, file_hash, read_config, system_prompt, validate_panel
from posttraining.state import build_context, validate_state


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate_states(panel, states):
    if set(states) != {s["id"] for s in panel["scenarios"]}:
        raise ValueError("State panel must cover every scenario exactly once.")
    for scenario in panel["scenarios"]:
        state = states[scenario["id"]]
        validate_state(state)
        if state["language"] != scenario["language"]:
            raise ValueError("State panel language mismatch.")
        if any(e["turn"] > len(scenario["turns"]) for e in state["events"]):
            raise ValueError("State event exceeds the scenario timeline.")


def run_panel(runtime, config, panel, states, writer):
    """Return metadata only. Reply text goes directly to local user artifacts."""
    metadata = []
    total = sum(len(s["turns"]) for s in panel["scenarios"]) * len(config["seeds"])
    for scenario in panel["scenarios"]:
        for seed in config["seeds"]:
            history = []
            for index, turn in enumerate(scenario["turns"]):
                history.append({"role": "user", "content": turn["user"]})
                if states is not None:
                    messages, audit = build_context(
                        scenario["persona"], scenario["language"], history,
                        states[scenario["id"]], index + 1, runtime.count_tokens,
                        config["max_context_tokens"],
                    )
                else:
                    messages = [{"role": "system", "content": system_prompt(
                        scenario["persona"], scenario["language"])}] + list(history)
                    audit = {"state_source": None, "dropped_history_turns": []}
                sampling_seed = seed + 1000003 * index
                result = runtime.generate(messages, sampling_seed)
                row = {"id": f"{scenario['id']}-s{seed}-t{index+1}",
                       "scenario_id": scenario["id"], "family": scenario["family"],
                       "language": scenario["language"], "seed": seed,
                       "sampling_seed": sampling_seed, "turn": index + 1,
                       "messages": messages, "context": audit, **result}
                writer.append(row)
                history.append({"role": "assistant", "content": result["text"]})
                metadata.append({"language": scenario["language"],
                                 "generated_tokens_including_eos": len(result["output_ids"]),
                                 "seconds": result["seconds"],
                                 "dropped_history_turns": len(audit["dropped_history_turns"])})
                print(f"[{len(metadata)}/{total}] saved | language {scenario['language']} | "
                      f"{result['tokens_per_second']:.1f} tok/s | "
                      f"reserved {result['memory']['reserved_gib']:.2f} GiB", flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/ministral_p1.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    args = parser.parse_args()
    config = read_config(args.config)
    panel = json.loads(Path(config["panel"]).read_text(encoding="utf-8"))
    validate_panel(panel)
    mode = config.get("context_mode", "p0")
    if mode not in ("p0", "p1"):
        raise ValueError("Unknown context mode.")
    states = None
    if mode == "p1":
        states = json.loads(Path(config["state_panel"]).read_text(encoding="utf-8"))
        validate_states(panel, states)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sources = [args.config, Path(config["panel"]), Path(__file__),
               Path("posttraining/runtime.py"), Path("posttraining/state.py"),
               Path("posttraining/artifacts.py")]
    if states is not None:
        sources.append(Path(config["state_panel"]))
    source_dir = args.output_dir / "source_snapshot"
    source_dir.mkdir()
    for source in sources:
        (source_dir / source.name).write_bytes(source.read_bytes())
    manifest = {
        "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "config": config, "source_hashes": {str(p): file_hash(p) for p in sources},
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in (
            "torch", "transformers", "accelerate", "mistral-common", "huggingface-hub")},
        "review_policy": "user-only; no content scoring or assistant review",
        "protocol": (f"{mode.upper()} frozen inference. P1 adds authored point-in-time state; "
                     "no automatic extraction or training. Per-turn sampling seed = base seed + "
                     "1000003 * zero-based turn. State/persona/current input stay intact; "
                     "P1 drops oldest complete exchanges only when required by token budget."),
    }
    write_json(args.output_dir / "manifest.json", manifest)
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                            capture_output=True, text=True, check=True)
    (args.output_dir / "pip-freeze.txt").write_text(freeze.stdout, encoding="utf-8")
    start = time.perf_counter()
    try:
        runtime = Runtime(config, adapter_path=args.adapter)
        manifest.update({"snapshot": str(runtime.snapshot), "parameters": runtime.parameters,
                         "before_load_memory": runtime.before_load,
                         "load_seconds": runtime.load_seconds, "adapter": runtime.adapter,
                         "model_file_hashes": {p.name: file_hash(p) for p in sorted(
                             runtime.snapshot.iterdir()) if p.is_file()},
                         "preflight": runtime.preflight()})
        write_json(args.output_dir / "manifest.json", manifest)
        print("Finite-logit preflight passed. Replies will be saved without content review.", flush=True)
        with ReplyWriter(args.output_dir) as writer:
            metadata = run_panel(runtime, config, panel, states, writer)
        seconds = sum(r["seconds"] for r in metadata)
        tokens = sum(r["generated_tokens_including_eos"] for r in metadata)
        summary = {"saved_generations": len(metadata),
                   "by_language": {lang: sum(r["language"] == lang for r in metadata)
                                   for lang in ("en", "fr")},
                   "generation_seconds": seconds, "generated_tokens_including_eos": tokens,
                   "aggregate_tokens_per_second": tokens / seconds if seconds else None,
                   "contexts_with_history_dropped": sum(bool(r["dropped_history_turns"]) for r in metadata),
                   "final_memory": runtime.memory(), "review_policy": "user-only"}
        write_json(args.output_dir / "summary.json", summary)
        manifest.update(status="complete", saved_generations=len(metadata))
    except Exception as error:
        # Third-party exception strings can contain input/output text.
        manifest.update(status="failed", error_type=type(error).__name__)
        print(f"Run failed ({type(error).__name__}); see technical manifest. Partial replies remain saved.", flush=True)
        return 1
    finally:
        manifest["elapsed_seconds"] = time.perf_counter() - start
        write_json(args.output_dir / "manifest.json", manifest)
    print(f"Complete. User reply gallery: {args.output_dir / 'replies.html'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
