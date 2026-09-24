"""Compare actual panel logits across ROCm attention backends and precision."""

import argparse
import json
from pathlib import Path

from posttraining.runtime import Runtime, read_config


def main():
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = [json.loads(line) for line in (args.run_dir / "generations.jsonl").read_text(encoding="utf-8").splitlines()]
    selected = [r for r in rows if r["id"] in ("en03-s42-t1", "fr03-s42-t1", "fr04-s42-t1")]
    runtime = Runtime(read_config("configs/ministral_p0.json"))
    result = {"model_revision": runtime.config["revision"], "comparisons": {}}
    reference = {}

    def inspect(label, device, math_backend):
        for row in selected:
            inputs = runtime.encode(row["messages"]).to(device)
            context = sdpa_kernel(SDPBackend.MATH) if math_backend else __import__("contextlib").nullcontext()
            with context, torch.inference_mode():
                logits = runtime.model(**inputs, use_cache=False).logits[0, -1].float().cpu()
            if not torch.isfinite(logits).all():
                raise RuntimeError(f"Non-finite logits: {label}")
            if label == "fp16_default":
                reference[row["id"]] = logits
            top = logits.topk(5)
            probabilities = logits.softmax(-1)
            item = {
                "max_abs_logit_diff_vs_fp16": (logits-reference[row["id"]]).abs().max().item(),
                "probability_l1_diff_vs_fp16": (probabilities-reference[row["id"]].softmax(-1)).abs().sum().item(),
                "top5_ids": top.indices.tolist(),
                "top5_text": [runtime.tokenizer.decode([i]) for i in top.indices.tolist()],
                "finite": True,
            }
            result["comparisons"].setdefault(row["id"], {})[label] = item
            print(row["id"], label, item, flush=True)

    inspect("fp16_default", runtime.device, False)
    inspect("fp16_math", runtime.device, True)
    runtime.model.to(dtype=torch.bfloat16)
    inspect("bf16_default", runtime.device, False)
    # Source weights are BF16. Cast them to FP32 on CPU for an independent forward.
    runtime.model.to(device="cpu", dtype=torch.float32)
    torch.cuda.empty_cache()
    torch.set_num_threads(12)
    inspect("fp32_cpu_math", "cpu", True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
