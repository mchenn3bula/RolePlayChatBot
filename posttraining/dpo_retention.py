"""Matched token-only validation NLL; never decode or display private dialogue."""

import argparse
import gc
import json
from pathlib import Path

from posttraining.lora_train import load_data, validate, write_json
from posttraining.runtime import Runtime, file_hash


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sft", type=Path, required=True)
    parser.add_argument("--dpo", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite validation results.")
    config = json.loads(Path("configs/ministral_s1_lora.json").read_text())
    _, rows, manifest = load_data(config)
    result = {"scope": "Secondary technical retention check, added before generated-output review; not checkpoint selection",
              "rows": len(rows), "target_tokens": sum(r["target_tokens"] for r in rows),
              "validation_sha256": manifest["files"]["validation.jsonl"], "config": config,
              "code_sha256": file_hash(__file__), "arms": {}}
    for arm in ("sft", "dpo"):
        runtime = Runtime(config, adapter_path=getattr(args, arm))
        result["arms"][arm] = {"nll": validate(runtime, rows), "adapter": runtime.adapter, "memory": runtime.memory()}
        print(f"{arm.upper()} retention NLL: {result['arms'][arm]['nll']:.6f}", flush=True)
        del runtime
        gc.collect()
        torch.cuda.empty_cache()
    write_json(args.output, result)


if __name__ == "__main__":
    main()
