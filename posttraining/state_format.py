"""Frozen scene-value rendering ablation; no training or default prompt changes."""

import argparse
import copy
import html
import json
from pathlib import Path
import random
import re

from posttraining.runtime import Runtime, file_hash


ROOT = Path("reports/ministral-state-format-v1")
PREVIOUS = Path("reports/ministral-bilingual-control-v2")
FIXTURES = Path("posttraining/state_format_fixtures.json")
ADAPTER = Path("checkpoints/ministral-v2-csft-v1/epoch-02")
ARMS = ("prose", "entity")
FIELDS = ("role_error", "contradiction", "omission", "agency", "language_ok",
          "coherence", "relevance", "grammar")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_rows(path, records):
    with Path(path).open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False)+"\n")


def replace_value(messages, assignments):
    """Only edit the scene-established JSON value, preserving all other bytes."""
    if not assignments or any(not isinstance(x, str) or " = " not in x for x in assignments):
        raise ValueError("Explicit assignment list required.")
    result = copy.deepcopy(messages)
    prefix, raw = messages[0]["content"].rsplit("\n", 1)
    state = json.loads(raw)
    facts = [f for f in state["facts"] if f["subject"] == "scene" and f["predicate"] == "established"]
    if len(facts) != 1 or not isinstance(facts[0]["value"], str):
        raise ValueError("Expected exactly one original prose scene fact.")
    original = facts[0]["value"]
    facts[0]["value"] = assignments
    result[0]["content"] = prefix + "\n" + json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    # Reversibility also proves no unrelated JSON serialization/instruction changes.
    facts[0]["value"] = original
    assert prefix + "\n" + json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(",", ":")) == messages[0]["content"]
    assert result[1:] == messages[1:]
    return result, original


def protected_files():
    paths = [Path(__file__), FIXTURES, Path("STATE_FORMAT_PROTOCOL.md"), Path("configs/ministral_p1.json"),
             PREVIOUS / "requests.jsonl", PREVIOUS / "review_criteria.json", PREVIOUS / "protocol.json",
             PREVIOUS / "csft/manifest.json", PREVIOUS / "csft/generations.jsonl",
             ADAPTER / "adapter_config.json", ADAPTER / "adapter_model.safetensors", ADAPTER / "manifest.json",
             Path("posttraining/runtime.py"), Path("posttraining/lora_data.py")]
    return {p.as_posix(): file_hash(p) for p in paths}


def prepare():
    from posttraining.preferences_v2 import encoder
    previous = read(PREVIOUS / "protocol.json")
    assert file_hash(PREVIOUS / "requests.jsonl") == previous["requests_sha256"]
    config = read("configs/ministral_p1.json")
    assert config == previous["config"]
    enc, snapshot, _ = encoder()
    fixtures = read(FIXTURES)
    records, audit = [], {}
    originals = rows(PREVIOUS / "requests.jsonl")
    assert len(originals) == 48 and {r["family"] for r in originals} == set(fixtures)
    for original in originals:
        entity, prose_value = replace_value(original["messages"], fixtures[original["family"]][original["language"]])
        key = original["family"]+"-"+original["language"]
        audit[key] = {"original_value": prose_value, "entity_value": fixtures[original["family"]][original["language"]],
                      "audit": "Assistant manually checked coverage, negation and actor reference before generation; not independent."}
        record = {k: original[k] for k in ("id", "family", "language", "expected", "seed")}
        record["conditions"] = {}
        for arm, messages in (("prose", original["messages"]), ("entity", entity)):
            ids = enc(messages)
            if len(ids) > config["max_context_tokens"]:
                raise ValueError("No truncation allowed.")
            if arm == "prose":
                assert ids == original["input_ids"]
            record["conditions"][arm] = {"messages": messages, "input_ids": ids}
        records.append(record)
    ROOT.mkdir(exist_ok=False, parents=True)
    write_rows(ROOT / "requests.jsonl", records)
    write(ROOT / "fact_audit.json", audit)
    write(ROOT / "protocol.json", {"experiment": ROOT.name, "config": config, "adapter": str(ADAPTER),
          "protected_files": protected_files(), "requests_sha256": file_hash(ROOT / "requests.jsonl"),
          "fact_audit_sha256": file_hash(ROOT / "fact_audit.json"), "tokenizer_sha256": file_hash(snapshot / "tekken.json"),
          "review_criteria": read(PREVIOUS / "review_criteria.json"), "fields": FIELDS,
          "development_only": True, "test_data_used": False})
    print("Frozen 48 paired requests; all original native inputs reproduced; 16 value mappings audited.", flush=True)


def verify():
    protocol = read(ROOT / "protocol.json")
    for path, expected in protocol["protected_files"].items():
        if file_hash(path) != expected:
            raise ValueError(f"Protected file changed: {path}")
    assert file_hash(ROOT / "requests.jsonl") == protocol["requests_sha256"]
    assert file_hash(ROOT / "fact_audit.json") == protocol["fact_audit_sha256"]
    return protocol


def generate():
    protocol = verify()
    # Claim the run before allocating GPU; completed/partial runs cannot be overwritten.
    write(ROOT / "started.json", {"status": "started", "protocol_sha256": file_hash(ROOT / "protocol.json")})
    runtime = Runtime(protocol["config"], adapter_path=ADAPTER)
    preflight = runtime.preflight()
    with (ROOT / "generations.jsonl").open("x", encoding="utf-8") as handle:
        for i, request in enumerate(rows(ROOT / "requests.jsonl")):
            for arm in ARMS if i % 2 == 0 else tuple(reversed(ARMS)):
                condition = request["conditions"][arm]
                result = runtime.generate(condition["messages"], request["seed"])
                assert result["input_ids"] == condition["input_ids"]
                handle.write(json.dumps({"id": request["id"], "arm": arm, **result}, ensure_ascii=False)+"\n")
                handle.flush()
                print(f"Pair {i+1}/48 | {arm} | {result['tokens_per_second']:.1f} tok/s | "
                      f"reserved {runtime.memory()['reserved_gib']:.2f} GiB", flush=True)
    verify()
    write(ROOT / "manifest.json", {"status": "complete", "adapter": runtime.adapter, "preflight": preflight,
          "memory": runtime.memory(), "protocol_sha256": file_hash(ROOT / "protocol.json"),
          "generations_sha256": file_hash(ROOT / "generations.jsonl")})


def outputs():
    verify()
    manifest = read(ROOT / "manifest.json")
    assert manifest["status"] == "complete"
    assert manifest["protocol_sha256"] == file_hash(ROOT / "protocol.json")
    assert manifest["generations_sha256"] == file_hash(ROOT / "generations.jsonl")
    records = rows(ROOT / "generations.jsonl")
    indexed = {(r["id"], r["arm"]): r for r in records}
    requests = rows(ROOT / "requests.jsonl")
    assert len(records) == len(indexed) == 96
    assert set(indexed) == {(r["id"], a) for r in requests for a in ARMS}
    for request in requests:
        for arm in ARMS:
            assert indexed[request["id"], arm]["input_ids"] == request["conditions"][arm]["input_ids"]
    return requests, indexed


def pair():
    requests, generated = outputs()
    rng = random.Random(791032)
    blind, key, articles = [], {}, []
    historical = {r["id"]: r for r in rows(PREVIOUS / "csft/generations.jsonl")}
    reproduced = 0
    for request in requests:
        names = list(ARMS)
        rng.shuffle(names)
        key[request["id"]] = dict(zip("AB", names))
        blind.append({**{k: request[k] for k in ("id", "family", "language", "expected")},
                      "original_messages": request["conditions"]["prose"]["messages"],
                      **{letter: generated[request["id"], arm]["text"] for letter, arm in zip("AB", names)}})
        reproduced += generated[request["id"], "prose"]["output_ids"] == historical[request["id"]]["output_ids"]
        columns = []
        for arm in ARMS:
            columns.append("<section><h3>"+arm+"</h3><pre>"+html.escape(generated[request["id"], arm]["text"])+
                           "</pre><details><summary>Exact input</summary><pre>"+
                           html.escape(json.dumps(request["conditions"][arm]["messages"], ensure_ascii=False, indent=2))+"</pre></details></section>")
        articles.append("<article><h2>"+html.escape(request["id"])+"</h2><p>"+
                        html.escape(request["conditions"]["prose"]["messages"][-1]["content"])+
                        "</p><div class='columns'>"+"".join(columns)+"</div></article>")
    write_rows(ROOT / "blind_review.jsonl", blind)
    write(ROOT / "blind_key.json", key)
    write(ROOT / "pair_manifest.json", {"historical_prose_exact_output_matches": reproduced, "requests": 48,
          "blind_sha256": file_hash(ROOT / "blind_review.jsonl"), "key_sha256": file_hash(ROOT / "blind_key.json")})
    (ROOT / "comparison.html").write_text("""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'"><title>State format comparison</title>
<style>body{font:16px/1.6 system-ui;max-width:1300px;margin:32px auto;padding:16px;background:#101820;color:#e8edf2}.columns{display:grid;grid-template-columns:1fr 1fr;gap:16px}section{background:#1d2b36;padding:16px;border-radius:10px}pre{white-space:pre-wrap;overflow-wrap:anywhere}article{border-top:1px solid #607080;padding:20px 0}summary{cursor:pointer;color:#99d9ef}@media(max-width:850px){.columns{grid-template-columns:1fr}}</style>
<h1>Same model, two scene formats</h1><p>Continued-SFT epoch 02, identical history, user inputs and sampling settings. Only the scene fact value changes from prose to explicit entity assignments. 48 paired development requests; no training or default changes. Any assistant ratings are not independent human review.</p>
"""+"\n".join(articles)+"</html>", encoding="utf-8")
    print(f"Paired 48 requests; original prose reproduces {reproduced}/48 historical outputs token-for-token.")


def summarize():
    requests, generated = outputs()
    pairing = read(ROOT / "pair_manifest.json")
    assert pairing["blind_sha256"] == file_hash(ROOT / "blind_review.jsonl")
    assert pairing["key_sha256"] == file_hash(ROOT / "blind_key.json")
    annotations = rows(ROOT / "assistant_review.jsonl")
    reviews = {r["id"]: r for r in annotations}
    assert len(reviews) == len(annotations) == 48
    assert set(reviews) == {r["id"] for r in requests}
    for review in reviews.values():
        assert review["winner"] in ("A", "B", "tie") and review["note"].strip()
        for letter in "AB":
            values = review[letter]
            assert len(values) == 8 and all(type(x) is int for x in values)
            assert all(x in (0, 1) for x in values[:5]) and all(1 <= x <= 5 for x in values[5:])
            assert not values[0] or values[1]
    write(ROOT / "review_frozen.json", {"sha256": file_hash(ROOT / "assistant_review.jsonl"), "fields": FIELDS,
          "reviewer": "Single assistant; authored mapping; not independent; familiar historical outputs limit blinding"})
    key = read(ROOT / "blind_key.json")
    scored = []
    for request in requests:
        review = reviews[request["id"]]
        scored.append({**{k: request[k] for k in ("id", "family", "language")},
                       "winner": "tie" if review["winner"] == "tie" else key[request["id"]][review["winner"]],
                       "scores": {key[request["id"]][letter]: dict(zip(FIELDS, review[letter])) for letter in "AB"}})

    def group(selected):
        result = {"n": len(selected), "arms": {}, "wins": {a: sum(r["winner"] == a for r in selected) for a in (*ARMS, "tie")}}
        for arm in ARMS:
            scores = [r["scores"][arm] for r in selected]
            result["arms"][arm] = {f: sum(s[f] for s in scores) for f in FIELDS[:5]}
            result["arms"][arm].update({"mean_"+f: sum(s[f] for s in scores)/len(scores) for f in FIELDS[5:]})
            result["arms"][arm]["all_constraints_satisfied"] = sum(not s["contradiction"] and not s["omission"]
                and not s["agency"] and s["language_ok"] for s in scores)
        return result

    groups = {"all": group(scored), **{lang: group([r for r in scored if r["language"] == lang]) for lang in ("en", "fr")}}
    gates = {}
    for lang in ("en", "fr"):
        p, e = (groups[lang]["arms"][a] for a in ARMS)
        gates[lang] = e["role_error"] < p["role_error"] and all(e[f] <= p[f] for f in ("contradiction", "omission", "agency")) and all(
            e[f] >= p[f] for f in ("language_ok", "mean_coherence", "mean_relevance", "mean_grammar"))
    technical = {}
    for arm in ARMS:
        samples = [generated[r["id"], arm] for r in requests]
        repeats = []
        for sample in samples:
            words = re.findall(r"\w+", sample["text"].lower())
            grams = [tuple(words[i:i+4]) for i in range(max(0, len(words)-3))]
            repeats.append((len(grams)-len(set(grams)))/len(grams) if grams else 0)
        technical[arm] = {"empty": sum(not r["text"].strip() for r in samples), "eos": sum(r["stopped_on_eos"] for r in samples),
                          "capped": sum(len(r["output_ids"]) == 192 for r in samples),
                          "mean_tokens": sum(len(r["output_ids"]) for r in samples)/48,
                          "mean_repeated_4gram_fraction": sum(repeats)/48,
                          "mean_input_tokens": sum(len(r["input_ids"]) for r in samples)/48}
    result = {"experiment": ROOT.name, "groups": groups, "language_gates": gates, "passes_local_gate": all(gates.values()),
              "families": {f: group([r for r in scored if r["family"] == f]) for f in sorted({r["family"] for r in scored})},
              "technical": technical, "pairing": pairing, "manifest": read(ROOT / "manifest.json"),
              "review": read(ROOT / "review_frozen.json"), "protocol_sha256": file_hash(ROOT / "protocol.json"),
              "limits": "Familiar eight-family development panel; non-independent assistant judgments; no default promotion"}
    write(ROOT / "summary.json", result)
    write(Path("results") / (ROOT.name+".json"), result)
    print(json.dumps({"groups": groups, "language_gates": gates, "technical": technical}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "generate", "pair", "summarize"))
    args = parser.parse_args()
    {"prepare": prepare, "generate": generate, "pair": pair, "summarize": summarize}[args.mode]()
