"""Exact-input SFT/DPO generation and randomized A/B review artifacts."""

import argparse
import html
import json
from pathlib import Path
import random
import time

from posttraining.artifacts import ReplyWriter
from posttraining.lora_data import digest
from posttraining.lora_train import write_json
from posttraining.runtime import Runtime, file_hash


# Required factual slots; omissions and contradictions are scored separately.
SLOTS = {
    "01-1": ["Respond as innkeeper from inside, not as the traveler outside"],
    "02-1": ["Brass key in blue drawer", "Red drawer contains screws"],
    "03-1": ["Never served royal navy", "Ivo is brother, not sister", "Ivo is baker, not ferry pilot"],
    "04-1": ["Contents of sealed letter unknown; do not fabricate a quotation"],
    "05-1": [], "06-1": ["Station/life support normal", "Kettle/heating element is broken"],
    "07-1": ["Sam is adult colleague, not son", "Mint unsuitable because of allergy"],
    "08-1": [],
    "09-1": ["User holds green parcel"], "09-2": ["Now at bakery"],
    "09-3": ["Runa holds green parcel", "Runa holds red umbrella"],
    "09-4": ["Runa still holds parcel"],
    "09-5": ["At lighthouse doorway", "Runa holds parcel, not yet delivered", "Parcel intended for keeper"],
    "10-1": ["Green atlas available", "Blue held for Jun"],
    "10-2": ["Reservation still active at noon, until sunset"],
    "10-3": ["Updated deadline tomorrow morning"],
    "10-4": ["Jun adult customer, not daughter", "Blue remains reserved", "Updated deadline tomorrow morning"],
    "10-5": ["Green available again", "Blue reserved for Jun", "Until tomorrow morning"],
}


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def prepare(source, output):
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("SFT source run must be complete.")
    rows = read_rows(source / "generations.jsonl")
    if len(rows) != 108 or len({r["id"] for r in rows}) != 108:
        raise ValueError("Expected fixed complete 108-request SFT panel.")
    panel = json.loads(Path(manifest["config"]["panel"]).read_text(encoding="utf-8"))
    scenarios = {s["id"]: s for s in panel["scenarios"]}
    requests = []
    for row in rows:
        record = {k: row[k] for k in ("id", "scenario_id", "family", "language", "seed", "sampling_seed", "turn",
                                      "messages", "context", "input_ids")}
        record["prior_sft_output_ids"] = row["output_ids"]
        record["expected"] = scenarios[row["scenario_id"]]["turns"][row["turn"]-1]["expect"]
        record["fact_slots"] = SLOTS[f"{row['scenario_id'][2:]}-{row['turn']}"]
        record["prompt_sha256"] = digest(record["messages"])
        requests.append(record)
    output.mkdir(parents=True, exist_ok=False)
    path = output / "requests.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in requests), encoding="utf-8")
    write_json(output / "protocol.json", {"config": manifest["config"], "requests_sha256": file_hash(path),
        "source_run": str(source), "source_generations_sha256": file_hash(source / "generations.jsonl"),
        "source_adapter": manifest["adapter"], "requests": 108, "first_turns": 60, "followups": 48,
        "fact_opportunities": sum(len(r["fact_slots"]) for r in requests),
        "history": "Exact shared historical SFT-generated history; not an independent rollout comparison",
        "protocol_document_sha256": file_hash("DPO_COMPARISON_PROTOCOL.md")})
    print(f"Frozen {len(requests)} exact requests with {sum(len(r['fact_slots']) for r in requests)} factual slots.")


def generate(root, arm, adapter):
    protocol = json.loads((root / "protocol.json").read_text())
    if file_hash(root / "requests.jsonl") != protocol["requests_sha256"]:
        raise ValueError("Comparison inputs changed.")
    rows = read_rows(root / "requests.jsonl")
    output = root / arm
    output.mkdir(exist_ok=False)
    runtime = Runtime(protocol["config"], adapter_path=adapter)
    preflight = runtime.preflight()
    started = time.perf_counter()
    timings, tokens, reproduced = [], [], 0
    with ReplyWriter(output) as writer:
        for i, row in enumerate(rows):
            result = runtime.generate(row["messages"], row["sampling_seed"])
            if result["input_ids"] != row["input_ids"]:
                raise ValueError("Input tokens differ from the frozen reference request.")
            writer.append({**row, **result})
            reproduced += result["output_ids"] == row["prior_sft_output_ids"]
            timings.append(result["seconds"])
            tokens.append(len(result["output_ids"]))
            print(f"{arm.upper()} [{i+1}/{len(rows)}] | {result['tokens_per_second']:.1f} tok/s | "
                  f"reserved {runtime.memory()['reserved_gib']:.2f} GiB", flush=True)
    write_json(output / "manifest.json", {"status": "complete", "arm": arm, "adapter": runtime.adapter,
        "config": protocol["config"], "requests_sha256": protocol["requests_sha256"], "preflight": preflight,
        "saved_generations": len(rows), "input_ids_identical": True,
        "prior_sft_output_ids_reproduced": reproduced if arm == "sft" else None,
        "generation_seconds": sum(timings), "tokens_including_eos": sum(tokens),
        "tokens_per_second": sum(tokens)/sum(timings), "elapsed_seconds": time.perf_counter()-started,
        "memory": runtime.memory(), "code_sha256": file_hash(__file__)})


def pair_outputs(root):
    rows = {arm: read_rows(root / arm / "generations.jsonl") for arm in ("sft", "dpo")}
    manifests = {arm: json.loads((root / arm / "manifest.json").read_text()) for arm in rows}
    if manifests["sft"]["config"] != manifests["dpo"]["config"]:
        raise ValueError("Evaluation config mismatch.")
    if manifests["sft"]["requests_sha256"] != manifests["dpo"]["requests_sha256"]:
        raise ValueError("Evaluation request mismatch.")
    if len(rows["sft"]) != len(rows["dpo"]):
        raise ValueError("Evaluation coverage mismatch.")
    rng = random.Random(20260925)
    blind, key, gallery = [], {}, []
    for first, second in zip(rows["sft"], rows["dpo"]):
        for field in ("id", "messages", "input_ids", "sampling_seed", "context"):
            if first[field] != second[field]:
                raise ValueError(f"Paired field mismatch: {field}")
        names = ["sft", "dpo"]
        rng.shuffle(names)
        by_arm = {"sft": first, "dpo": second}
        record = {k: first[k] for k in ("id", "family", "language", "turn", "expected", "fact_slots")}
        record.update(user=first["messages"][-1]["content"], A=by_arm[names[0]]["text"], B=by_arm[names[1]]["text"])
        blind.append(record)
        key[first["id"]] = {"A": names[0], "B": names[1]}
        gallery.append("<article><h2>" + html.escape(first["id"]) + "</h2><p>" + html.escape(record["user"]) +
            "</p><div class='pair'><section><h3>SFT</h3><pre>" + html.escape(first["text"]) +
            "</pre></section><section><h3>DPO</h3><pre>" + html.escape(second["text"]) +
            "</pre></section></div><details><summary>Expected response / factual slots</summary><pre>" +
            html.escape(json.dumps({"expected": record["expected"], "fact_slots": record["fact_slots"]},
                                   ensure_ascii=False, indent=2)) + "</pre></details></article>")
    (root / "blind_review.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in blind), encoding="utf-8")
    write_json(root / "blind_key.json", key)
    (root / "comparison.html").write_text("""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'">
<title>SFT versus standard DPO</title><style>body{font:16px/1.6 system-ui;max-width:1200px;margin:40px auto;padding:0 20px}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:24px}pre{white-space:pre-wrap;overflow-wrap:anywhere}
article{border-top:1px solid #aaa;padding:20px 0}section{background:#eee;padding:15px;color:#111}
@media(max-width:650px){.pair{grid-template-columns:1fr}}</style><h1>SFT versus standard DPO</h1>
<p>Exact same message arrays, state, native input tokens, sampling seeds and decoding settings.
Follow-up inputs use shared historical SFT conversation context. These are development scenarios.</p>
""" + "\n".join(gallery) + "</html>", encoding="utf-8")
    print(f"Paired {len(blind)} identical-input requests. Gallery: {root / 'comparison.html'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "generate", "pair"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--arm", choices=("sft", "dpo"))
    parser.add_argument("--adapter", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.source_run, args.output_dir)
    elif args.mode == "generate":
        if not args.arm or not args.adapter:
            parser.error("Generate requires arm and adapter.")
        generate(args.output_dir, args.arm, args.adapter)
    else:
        pair_outputs(args.output_dir)


if __name__ == "__main__":
    main()
