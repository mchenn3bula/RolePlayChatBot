"""Bilingual preference curation with model-error provenance and untouched evaluation scenes."""

import argparse
from collections import Counter
import json
from pathlib import Path

from posttraining.build_state_fixtures import fact
from posttraining.dpo_core import completion_labels
from posttraining.dpo_data import authored_pairs
from posttraining.lora_data import NativeEncoder
from posttraining.lora_train import write_json
from posttraining.runtime import Runtime, file_hash
from posttraining.state import build_context


SCENES = Path("posttraining/preferences_v2_scenes.json")
ROOT = Path("data/ministral-preferences-v2")
CURATION = Path("reports/ministral-preferences-v2-curation")
SFT = Path("checkpoints/ministral-s1-whole-lora-v1-r2/epoch-02")
HISTORY = {
    "radio": (("Please hold the radio and stopwatch for now.", "I have both the radio and the stopwatch."),
              ("Garde la radio et le chronomètre pour le moment.", "J'ai la radio et le chronomètre.")),
    "rehearsal": (("Where was rehearsal originally scheduled?", "Tuesday in room A."),
                  ("Où la répétition était-elle prévue au départ ?", "Mardi en salle A.")),
    "wet_trail": (("Which route did you recommend earlier?", "The eastern path was open then."),
                  ("Quel itinéraire conseillais-tu auparavant ?", "Le sentier est était ouvert à ce moment-là.")),
    "visitor_pass": (("What did my pass originally say?", "It originally expired on Friday."),
                     ("Qu'indiquait mon laissez-passer à l'origine ?", "Il expirait à l'origine vendredi.")),
    "ferry_stop": (("What was the original departure?", "Dock 2 at nine."),
                   ("Quel était le départ initial ?", "Quai 2 à neuf heures.")),
    "tent_location": (("Where was the tent earlier?", "By the entrance."),
                      ("Où était la tente auparavant ?", "À l'entrée.")),
    "film_session": (("What were the original screening details?", "Friday in hall 1."),
                     ("Quels étaient les détails de la séance initiale ?", "Vendredi en salle 1.")),
    "equipment_booking": (("What was Vic's original deadline?", "Monday."),
                          ("Quelle était l'échéance initiale de Vic ?", "Lundi.")),
}


def rows():
    # One original alias only; renaming a character is not new independent data.
    for row in authored_pairs():
        if row["id"].split("-")[2] == "0":
            yield {**row, "id": "d1-"+row["id"], "template_family": "d1-"+row["template_family"],
                   "origin": "deduplicated D1 authored contrast", "mine": False}
    for scene in json.loads(SCENES.read_text(encoding="utf-8")):
        for lang in ("en", "fr"):
            role, setting, user, chosen, rejected = scene[lang]
            history = []
            if scene["id"] in HISTORY:
                previous = HISTORY[scene["id"]][lang == "fr"]
                history = [{"role": "user", "content": previous[0]}, {"role": "assistant", "content": previous[1]}]
            history.append({"role": "user", "content": user})
            turn = (len(history)+1)//2
            state = {"schema_version": 1, "scene_id": "r2-"+scene["id"], "revision": turn,
                     "language": lang, "persona": {"character_id": "character", "voice": "Natural",
                     "traits": [], "goals": []}, "entities": {"character": role, "scene": "Scene", "user": "User"},
                     "facts": [fact("character", "role", role, scope="persona")],
                     "events": [{"turn": turn, "set": [fact("scene", "established", setting, turn=turn)], "remove": []}]}
            messages, context = build_context(role, lang, history, state, turn, lambda m: 0, 2048)
            yield {"id": f"r2-{scene['id']}-{lang}", "template_family": "r2-"+scene["id"],
                   "split": scene["split"], "language": lang, "category": scene["category"],
                   "scene_id": state["scene_id"], "messages": messages, "context": context,
                   "chosen_text": chosen, "rejected_text": rejected, "mine": scene["split"] != "evaluation",
                   "rationale": "Preserve explicit current state, model role and user choice.",
                   "origin": "new authored scene", "label_source": "assistant-authored synthetic contrast"}


def encoder():
    from huggingface_hub import snapshot_download
    config = json.loads(Path("configs/ministral_p1.json").read_text())
    snapshot = Path(snapshot_download(config["model_id"], revision=config["revision"], local_files_only=True,
                                      allow_patterns=["*.json"]))
    return NativeEncoder(snapshot), snapshot, config


def prepare():
    records = list(rows())
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("Duplicate row IDs.")
    families = {}
    for r in records:
        if families.setdefault(r["template_family"], r["split"]) != r["split"]:
            raise ValueError("Scenario family crosses partitions.")
    enc, snapshot, config = encoder()
    for r in records:
        r["input_ids"] = enc(r["messages"])
        if len(r["input_ids"]) > 2048:
            raise ValueError("No prompt truncation permitted.")
    ROOT.mkdir(parents=True, exist_ok=False)
    path = ROOT / "draft.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in records), encoding="utf-8")
    write_json(ROOT / "draft_manifest.json", {"draft_sha256": file_hash(path), "scenes_sha256": file_hash(SCENES),
        "builder_sha256": file_hash(__file__), "protocol_sha256": file_hash("BILINGUAL_CONTROL_PROTOCOL.md"),
        "tokenizer_sha256": file_hash(snapshot / "tekken.json"), "model_revision": config["revision"],
        "rows": dict(Counter(r["split"] for r in records)), "scenario_families": dict(Counter(families.values()))})
    print("Draft and held-out evaluation requests frozen.")


def read_draft():
    manifest = json.loads((ROOT / "draft_manifest.json").read_text())
    if file_hash(ROOT / "draft.jsonl") != manifest["draft_sha256"]:
        raise ValueError("Frozen draft changed.")
    return [json.loads(line) for line in (ROOT / "draft.jsonl").read_text(encoding="utf-8").splitlines()], manifest


def mine():
    records, draft = read_draft()
    config = json.loads(Path("configs/ministral_p1.json").read_text())
    runtime = Runtime(config, adapter_path=SFT)
    CURATION.mkdir(parents=True, exist_ok=False)
    selected = [r for r in records if r["mine"]]
    with (CURATION / "candidates.jsonl").open("x", encoding="utf-8") as f:
        for i, r in enumerate(selected):
            result = runtime.generate(r["messages"], 20260927+i)
            if result["input_ids"] != r["input_ids"]:
                raise ValueError("Native input mismatch.")
            f.write(json.dumps({"id": r["id"], "candidate": result["text"], "output_ids": result["output_ids"],
                                "seed": 20260927+i}, ensure_ascii=False)+"\n")
            f.flush()
            print(f"Preference mining {i+1}/{len(selected)} | reserved {runtime.memory()['reserved_gib']:.2f} GiB", flush=True)
    write_json(CURATION / "manifest.json", {"adapter": runtime.adapter, "draft": draft,
        "candidates_sha256": file_hash(CURATION / "candidates.jsonl"), "memory": runtime.memory()})


def finalize():
    records, draft = read_draft()
    metadata = json.loads((CURATION / "manifest.json").read_text())
    if file_hash(CURATION / "candidates.jsonl") != metadata["candidates_sha256"]:
        raise ValueError("Candidate fingerprint changed.")
    candidates = {r["id"]: r for r in map(json.loads, (CURATION / "candidates.jsonl").read_text(encoding="utf-8").splitlines())}
    reviews = json.loads((CURATION / "decisions.json").read_text(encoding="utf-8"))
    expected = {r["id"] for r in records if r["mine"]}
    if set(candidates) != expected or set(reviews) != expected:
        raise ValueError("All mined replies require explicit curation decisions.")
    enc, snapshot, config = encoder()
    for r in records:
        if r["mine"]:
            review = reviews[r["id"]]
            if type(review["use_as_rejected"]) is not bool or not review["reason"].strip():
                raise ValueError("Missing curation rationale.")
            if review["use_as_rejected"]:
                r["rejected_text"] = candidates[r["id"]]["candidate"]
                r["label_source"] = "assistant-reviewed SFT error versus authored correction; not human preference"
            r["curation"] = review
        if r["chosen_text"].strip() == r["rejected_text"].strip():
            raise ValueError("Identical candidate pair.")
        for side in ("chosen", "rejected"):
            ids = enc(r["messages"]+[{"role": "assistant", "content": r[side+"_text"]}])
            if len(ids) > 2048:
                raise ValueError("No completion truncation permitted.")
            labels = completion_labels(r["input_ids"], ids, enc.eos_id)
            r[side] = {"input_ids": ids, "labels": labels, "target_tokens": sum(x != -100 for x in labels)}
    files, stats = {}, {}
    for split in ("train", "validation", "evaluation"):
        group = [r for r in records if r["split"] == split]
        path = ROOT / f"{split}.jsonl"
        with path.open("x", encoding="utf-8") as f:
            for r in group:
                f.write(json.dumps(r, ensure_ascii=False)+"\n")
        files[path.name] = file_hash(path)
        stats[split] = {"pairs": len(group), "scenario_families": len({r["template_family"] for r in group}),
            "languages": dict(Counter(r["language"] for r in group)),
            "mined_rejections": sum(r.get("curation", {}).get("use_as_rejected", False) for r in group),
            "chosen_tokens": sum(r["chosen"]["target_tokens"] for r in group),
            "rejected_tokens": sum(r["rejected"]["target_tokens"] for r in group),
            "history_examples": sum(len(r["messages"]) > 2 for r in group),
            "max_length": max(len(r[side]["input_ids"]) for r in group for side in ("chosen", "rejected"))}
    write_json(ROOT / "manifest.json", {"format": "bilingual-preferences-v2", "model_revision": config["revision"],
        "max_length": 2048, "tokenizer_sha256": file_hash(snapshot / "tekken.json"), "files": files, "stats": stats,
        "draft": draft, "curation_sha256": file_hash(CURATION / "decisions.json"), "mining": metadata,
        "label_source": "assistant-authored/reviewed; no independent human labels", "test_data_used": False,
        "limitation": "Distinct scene instances share skill patterns; new evaluation is development, not a locked test"})
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "mine", "finalize"))
    args = parser.parse_args()
    {"prepare": prepare, "mine": mine, "finalize": finalize}[args.mode]()
