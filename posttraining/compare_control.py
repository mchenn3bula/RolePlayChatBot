"""Frozen three-arm evaluation of S1, chosen-response SFT and ordinary DPO."""

import argparse
import html
import json
from pathlib import Path
import random

from posttraining.compare_dpo import read_rows
from posttraining.lora_train import write_json
from posttraining.preferences_v2 import ROOT as DATA
from posttraining.runtime import Runtime, file_hash


ROOT = Path("reports/ministral-bilingual-control-v2")
ARMS = {"baseline": "ministral-s1-whole-lora-v1-r2", "csft": "ministral-v2-csft-v1", "dpo": "ministral-v2-dpo-v1"}


def prepare():
    metadata = json.loads((DATA / "manifest.json").read_text())
    if file_hash(DATA / "evaluation.jsonl") != metadata["files"]["evaluation.jsonl"]:
        raise ValueError("Evaluation fingerprint mismatch.")
    requests = []
    for scene in read_rows(DATA / "evaluation.jsonl"):
        for seed in (42, 43, 44):
            requests.append({"id": f"{scene['id']}-s{seed}", "family": scene["template_family"],
                "language": scene["language"], "messages": scene["messages"], "input_ids": scene["input_ids"],
                "expected": scene["chosen_text"], "seed": seed})
    ROOT.mkdir(parents=True, exist_ok=False)
    path = ROOT / "requests.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in requests), encoding="utf-8")
    write_json(ROOT / "protocol.json", {"config": json.loads(Path("configs/ministral_p1.json").read_text()),
        "requests_sha256": file_hash(path), "data_manifest_sha256": file_hash(DATA / "manifest.json"),
        "protocol_sha256": file_hash("BILINGUAL_CONTROL_PROTOCOL.md"), "requests": len(requests),
        "history": "Identical authored history; no model-generated follow-ups"})
    print(f"Frozen {len(requests)} three-arm evaluation requests.")


def verify_training():
    runs = {}
    for arm in ("csft", "dpo"):
        root = Path("checkpoints") / ARMS[arm]
        complete = json.loads((root / "complete.json").read_text())
        run = json.loads((root / "run.json").read_text())
        if complete["status"] != "complete" or complete["selected_checkpoint"] != "epoch-02":
            raise ValueError("Requires both prespecified completed controls.")
        runs[arm] = {"complete": complete, "run": run}
    for field in ("update", "chosen_presentations", "epoch"):
        if runs["csft"]["complete"]["progress"][field] != runs["dpo"]["complete"]["progress"][field]:
            raise ValueError("Unequal chosen exposure or updates.")
    if runs["csft"]["run"]["per_epoch"] != runs["dpo"]["run"]["per_epoch"]:
        raise ValueError("Unequal paired training data.")
    identities = [runs[a]["run"]["identity"] for a in ("csft", "dpo")]
    for field in ("reference_files", "data_manifest", "protocol_sha256"):
        if identities[0][field] != identities[1][field]:
            raise ValueError("Unequal reference, data or protocol.")
    common = [{k: v for k, v in identity["config"].items() if k not in ("objective", "experiment")}
              for identity in identities]
    if common[0] != common[1]:
        raise ValueError("Control training settings differ.")
    return runs


def generate(arm):
    training = verify_training()
    protocol = json.loads((ROOT / "protocol.json").read_text())
    if file_hash(ROOT / "requests.jsonl") != protocol["requests_sha256"]:
        raise ValueError("Evaluation requests changed.")
    output = ROOT / arm
    output.mkdir(exist_ok=False)
    runtime = Runtime(protocol["config"], adapter_path=Path("checkpoints") / ARMS[arm] / "epoch-02")
    preflight = runtime.preflight()
    requests = read_rows(ROOT / "requests.jsonl")
    with (output / "generations.jsonl").open("x", encoding="utf-8") as handle:
        for index, request in enumerate(requests):
            result = runtime.generate(request["messages"], request["seed"])
            if result["input_ids"] != request["input_ids"]:
                raise ValueError("Native input mismatch.")
            handle.write(json.dumps({**request, **result}, ensure_ascii=False)+"\n")
            handle.flush()
            print(f"{arm} {index+1}/{len(requests)} | {result['tokens_per_second']:.1f} tok/s | "
                  f"reserved {runtime.memory()['reserved_gib']:.2f} GiB", flush=True)
    write_json(output / "manifest.json", {"status": "complete", "config": protocol["config"], "adapter": runtime.adapter,
        "requests_sha256": protocol["requests_sha256"], "preflight": preflight, "memory": runtime.memory(),
        "training": training, "generations_sha256": file_hash(output / "generations.jsonl"), "code_sha256": file_hash(__file__)})


def pair():
    records = {a: read_rows(ROOT / a / "generations.jsonl") for a in ARMS}
    protocol = json.loads((ROOT / "protocol.json").read_text())
    for arm in ARMS:
        manifest = json.loads((ROOT / arm / "manifest.json").read_text())
        if (manifest["status"] != "complete" or manifest["config"] != protocol["config"]
                or manifest["requests_sha256"] != protocol["requests_sha256"]
                or file_hash(ROOT / arm / "generations.jsonl") != manifest["generations_sha256"]
                or len(records[arm]) != protocol["requests"]):
            raise ValueError("Unequal or incomplete evaluation arm.")
    rng = random.Random(20260929)
    blind, key, articles = [], {}, []
    for index, reference in enumerate(records["baseline"]):
        for arm in ARMS:
            for field in ("id", "input_ids", "messages", "seed"):
                if records[arm][index][field] != reference[field]:
                    raise ValueError("Unequal paired inputs.")
        names = list(ARMS)
        rng.shuffle(names)
        mapping = dict(zip("ABC", names))
        key[reference["id"]] = mapping
        row = {k: reference[k] for k in ("id", "family", "language", "expected", "messages")}
        row.update({letter: records[arm][index]["text"] for letter, arm in mapping.items()})
        blind.append(row)
        columns = "".join("<section><h3>"+arm+"</h3><pre>"+html.escape(records[arm][index]["text"])+"</pre></section>" for arm in ARMS)
        articles.append("<article><h2>"+html.escape(row["id"])+"</h2><p>"+html.escape(row["messages"][-1]["content"])+
            "</p><div class='columns'>"+columns+"</div><details><summary>Expected answer and identical authored context</summary><pre>"+
            html.escape(row["expected"])+"</pre><pre>"+html.escape(json.dumps(row["messages"],ensure_ascii=False,indent=2))+"</pre></details></article>")
    with (ROOT / "blind_review.jsonl").open("x", encoding="utf-8") as f:
        for row in blind:
            f.write(json.dumps(row, ensure_ascii=False)+"\n")
    write_json(ROOT / "blind_key.json", key)
    (ROOT / "comparison.html").write_text("""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'"><title>Bilingual controls</title>
<style>body{font:16px/1.6 system-ui;max-width:1450px;margin:32px auto;padding:16px}.columns{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}section{background:#eee;color:#111;padding:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere}article{border-top:1px solid #888;padding:20px 0}@media(max-width:850px){.columns{grid-template-columns:1fr}}</style>
<h1>Bilingual preference controls</h1><p>baseline = unchanged S1; csft = continued SFT on chosen replies; dpo = standard DPO on the same pairs. All receive identical native inputs, authored state/history and sampling seeds. This is a development panel.</p>
"""+"\n".join(articles)+"</html>", encoding="utf-8")
    print("All three arms paired. Randomized review and labeled gallery saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "generate", "pair"))
    parser.add_argument("--arm", choices=tuple(ARMS))
    args = parser.parse_args()
    if args.mode == "generate":
        if not args.arm:
            parser.error("generate requires --arm")
        generate(args.arm)
    else:
        {"prepare": prepare, "pair": pair}[args.mode]()
