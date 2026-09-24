"""Freeze and compare a direct-response rule on familiar and fresh bilingual scenes."""

import argparse
import copy
import html
import json
from pathlib import Path
import random
import re

from posttraining.build_state_fixtures import fact
from posttraining.natural_reference import read, rows, write, write_rows
from posttraining.runtime import Runtime, file_hash
from posttraining.state import build_context


ROOT = Path("reports/ministral-response-refinement-v1")
PREVIOUS = Path("reports/ministral-natural-reference-v1")
SCENES = Path("posttraining/response_refinement_scenes.json")
ADAPTER = Path("checkpoints/ministral-v2-csft-v1/epoch-02")
ARMS = ("natural", "direct")
FIELDS = ("role_error", "contradiction", "omission", "agency", "language_ok",
          "coherence", "relevance", "grammar", "third_person_self")
RULES = {
    "en": "Speak as the character in first person and answer every part of the user's question using the current facts, correcting any false premise.",
    "fr": "Parle à la première personne en incarnant le personnage et réponds à chaque partie de la question de l'utilisateur à partir des faits actuels, en corrigeant toute prémisse fausse.",
}
VOICE_IDS = {"r2-telescope_cover-en-s43", "r2-ceramics_partner-en-s43", "r2-ceramics_partner-en-s44",
             "r2-ceramics_partner-fr-s44", "r2-equipment_booking-en-s43", "r2-painting_collection-en-s42"}


def refine(messages, language):
    result = copy.deepcopy(messages)
    prefix, state = messages[0]["content"].rsplit("\n", 1)
    json.loads(state)  # Refuse a non-state prompt rather than adding in an arbitrary place.
    if RULES[language] in prefix:
        raise ValueError("Response rule already present.")
    result[0]["content"] = prefix + "\n" + RULES[language] + "\n" + state
    assert result[0]["content"].replace("\n"+RULES[language], "", 1) == messages[0]["content"]
    return result


def fresh_scenes():
    for scene in read(SCENES):
        for language in ("en", "fr"):
            role, setting, user, expected = scene[language]
            history = []
            if "history_"+language in scene:
                old_user, old_reply = scene["history_"+language]
                history = [{"role": "user", "content": old_user}, {"role": "assistant", "content": old_reply}]
            history.append({"role": "user", "content": user})
            turn = (len(history)+1)//2
            state = {"schema_version": 1, "scene_id": "rr1-"+scene["id"], "revision": turn,
                     "language": language, "persona": {"character_id": "character", "voice": "Natural",
                     "traits": [], "goals": []}, "entities": {"character": role, "scene": "Scene", "user": "User"},
                     "facts": [fact("character", "role", role, scope="persona")],
                     "events": [{"turn": turn, "set": [fact("scene", "established", setting, turn=turn)], "remove": []}]}
            messages, _ = build_context(role, language, history, state, turn, lambda m: 0, 2048)
            for seed in (42, 43, 44):
                yield {"id": f"rr1-{scene['id']}-{language}-s{seed}", "family": "rr1-"+scene["id"],
                       "panel": "fresh", "language": language, "seed": seed, "messages": messages,
                       "expected": expected, "criteria": scene["criteria"]}


def archive_anchors():
    from posttraining.natural_reference import outputs
    outputs()
    assert file_hash(PREVIOUS / "assistant_review.jsonl") == read(PREVIOUS / "review_frozen.json")["sha256"]
    key = read(PREVIOUS / "blind_key.json")
    reviews = {r["id"]: r for r in rows(PREVIOUS / "assistant_review.jsonl")}
    generated = rows(PREVIOUS / "generations.jsonl")
    voiced = {(r["id"], r["text"]) for r in generated if r["arm"] == "natural" and r["id"] in VOICE_IDS}
    assert len(voiced) == 6
    anchors = {(r["id"], r["text"]): r for r in rows(PREVIOUS / "rating_anchors.jsonl")}
    for r in generated:
        letter = next(k for k, arm in key[r["id"]].items() if arm == r["arm"])
        score = reviews[r["id"]][letter]
        slot = (r["id"], r["text"])
        if slot in anchors:
            assert score == anchors[slot]["scores"]
        anchors[slot] = {"id": r["id"], "text": r["text"], "scores": score, "note": reviews[r["id"]]["note"]}
    return [{**r, "scores": r["scores"]+[int(slot in voiced)]} for slot, r in anchors.items()]


def protected():
    paths = [Path(__file__), SCENES, Path("RESPONSE_REFINEMENT_PROTOCOL.md"), Path("configs/ministral_p1.json"),
             Path("posttraining/runtime.py"), Path("posttraining/state.py"), Path("posttraining/build_state_fixtures.py"),
             Path("posttraining/natural_reference.py"), Path("posttraining/preferences_v2.py"), Path("posttraining/lora_data.py")]
    paths.extend(ADAPTER / p for p in ("manifest.json", "adapter_model.safetensors", "adapter_config.json"))
    paths.extend(PREVIOUS / p for p in ("protocol.json", "requests.jsonl", "generations.jsonl", "manifest.json",
        "rating_anchors.jsonl", "assistant_review.jsonl", "review_frozen.json", "blind_key.json"))
    return {p.as_posix(): file_hash(p) for p in paths}


def prepare():
    from posttraining.preferences_v2 import encoder
    enc, snapshot, config = encoder()
    previous = read(PREVIOUS / "protocol.json")
    assert config == previous["config"]
    assert file_hash(PREVIOUS / "requests.jsonl") == previous["requests_sha256"]
    anchors = archive_anchors()
    requests = []
    for r in rows(PREVIOUS / "requests.jsonl"):
        requests.append({**{k: r[k] for k in ("id", "family", "language", "seed", "expected")}, "panel": "familiar",
                         "messages": r["conditions"]["natural"]["messages"],
                         "archived_input_ids": r["conditions"]["natural"]["input_ids"],
                         "criteria": previous["review_criteria"]["criteria"][r["family"]]})
    fresh = list(fresh_scenes())
    assert not {r["family"] for r in requests} & {r["family"] for r in fresh}
    requests.extend(fresh)
    assert len(requests) == len({r["id"] for r in requests}) == 84
    for r in requests:
        baseline = r.pop("messages")
        r["conditions"] = {}
        for arm, messages in (("natural", baseline), ("direct", refine(baseline, r["language"]))):
            ids = enc(messages)
            assert len(ids) <= config["max_context_tokens"]
            if arm == "natural" and r["panel"] == "familiar":
                assert ids == r["archived_input_ids"]
            r["conditions"][arm] = {"messages": messages, "input_ids": ids}
    ROOT.mkdir(parents=True, exist_ok=False)
    write_rows(ROOT / "requests.jsonl", requests)
    write_rows(ROOT / "rating_anchors.jsonl", anchors)
    write(ROOT / "protocol.json", {"config": config, "adapter": str(ADAPTER), "rules": RULES, "fields": FIELDS,
          "protected": protected(), "requests_sha256": file_hash(ROOT / "requests.jsonl"),
          "anchors_sha256": file_hash(ROOT / "rating_anchors.jsonl"), "tokenizer_sha256": file_hash(snapshot / "tekken.json"),
          "panels": {"familiar": 48, "fresh": 36}, "test_data_used": False})
    print("Frozen 84 paired requests, six new bilingual scenes, exact rule and anchored scores.", flush=True)


def verify():
    protocol = read(ROOT / "protocol.json")
    for path, expected in protocol["protected"].items():
        if file_hash(path) != expected:
            raise ValueError(f"Protected file changed: {path}")
    assert file_hash(ROOT / "requests.jsonl") == protocol["requests_sha256"]
    assert file_hash(ROOT / "rating_anchors.jsonl") == protocol["anchors_sha256"]
    return protocol


def generate():
    protocol = verify()
    write(ROOT / "started.json", {"protocol_sha256": file_hash(ROOT / "protocol.json")})
    runtime = Runtime(protocol["config"], adapter_path=ADAPTER)
    preflight = runtime.preflight()
    requests = rows(ROOT / "requests.jsonl")
    with (ROOT / "generations.jsonl").open("x", encoding="utf-8") as handle:
        for i, r in enumerate(requests):
            for arm in ARMS if i % 2 == 0 else tuple(reversed(ARMS)):
                result = runtime.generate(r["conditions"][arm]["messages"], r["seed"])
                assert result["input_ids"] == r["conditions"][arm]["input_ids"]
                handle.write(json.dumps({"id": r["id"], "arm": arm, **result}, ensure_ascii=False)+"\n")
                handle.flush()
                print(f"Pair {i+1}/{len(requests)} | {r['panel']} | {arm} | {result['tokens_per_second']:.1f} tok/s | "
                      f"reserved {runtime.memory()['reserved_gib']:.2f} GiB", flush=True)
    verify()
    write(ROOT / "manifest.json", {"status": "complete", "adapter": runtime.adapter, "preflight": preflight,
          "memory": runtime.memory(), "protocol_sha256": file_hash(ROOT / "protocol.json"),
          "generations_sha256": file_hash(ROOT / "generations.jsonl")})


def outputs():
    verify()
    manifest = read(ROOT / "manifest.json")
    assert manifest["status"] == "complete" and manifest["protocol_sha256"] == file_hash(ROOT / "protocol.json")
    assert manifest["generations_sha256"] == file_hash(ROOT / "generations.jsonl")
    requests, records = rows(ROOT / "requests.jsonl"), rows(ROOT / "generations.jsonl")
    indexed = {(r["id"], r["arm"]): r for r in records}
    assert len(records) == len(indexed) == len(requests)*2
    assert set(indexed) == {(r["id"], a) for r in requests for a in ARMS}
    for r in requests:
        for arm in ARMS:
            assert indexed[r["id"], arm]["input_ids"] == r["conditions"][arm]["input_ids"]
    return requests, indexed


def pair():
    requests, generated = outputs()
    anchors = {(r["id"], r["text"]): r["scores"] for r in rows(ROOT / "rating_anchors.jsonl")}
    historical = {r["id"]: r for r in rows(PREVIOUS / "generations.jsonl") if r["arm"] == "natural"}
    rng = random.Random(791034)
    blind, key, articles, reproduced = [], {}, [], 0
    for r in requests:
        names = list(ARMS)
        rng.shuffle(names)
        key[r["id"]] = dict(zip("AB", names))
        entry = {k: r[k] for k in ("id", "family", "panel", "language", "expected", "criteria")}
        entry["original_messages"] = r["conditions"]["natural"]["messages"]
        entry.update({letter: generated[r["id"], a]["text"] for letter, a in zip("AB", names)})
        entry["anchored_scores"] = {letter: anchors.get((r["id"], entry[letter])) for letter in "AB"}
        blind.append(entry)
        if r["panel"] == "familiar":
            assert generated[r["id"], "natural"]["output_ids"] == historical[r["id"]]["output_ids"]
            reproduced += 1
        columns = []
        for a in ARMS:
            columns.append("<section><h3>"+a+"</h3><pre>"+html.escape(generated[r["id"], a]["text"])+
                           "</pre><details><summary>Exact input</summary><pre>"+
                           html.escape(json.dumps(r["conditions"][a]["messages"], ensure_ascii=False, indent=2))+"</pre></details></section>")
        articles.append("<article><h2>"+html.escape(r["id"])+"</h2><p>Panel: "+r["panel"]+"</p><p>"+
                        html.escape(r["conditions"]["natural"]["messages"][-1]["content"])+"</p><div class='columns'>"+
                        "".join(columns)+"</div><details><summary>Required facts</summary><p>"+html.escape(r["criteria"])+"</p></details></article>")
    assert reproduced == 48
    write_rows(ROOT / "blind_review.jsonl", blind)
    write(ROOT / "blind_key.json", key)
    write(ROOT / "pair_manifest.json", {"reproduced": reproduced,
          "anchored_scores": sum(r["anchored_scores"][a] is not None for r in blind for a in "AB"),
          "blind_sha256": file_hash(ROOT / "blind_review.jsonl"), "key_sha256": file_hash(ROOT / "blind_key.json")})
    (ROOT / "comparison.html").write_text("""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'"><title>Direct character replies</title>
<style>body{font:16px/1.6 system-ui;max-width:1300px;margin:32px auto;padding:16px;background:#101820;color:#e8edf2}.columns{display:grid;grid-template-columns:1fr 1fr;gap:16px}section{background:#1d2b36;padding:16px;border-radius:10px}pre{white-space:pre-wrap;overflow-wrap:anywhere}article{border-top:1px solid #607080;padding:20px 0}summary{cursor:pointer;color:#99d9ef}@media(max-width:850px){.columns{grid-template-columns:1fr}}</style>
<h1>Direct and complete character replies</h1><p>Natural references versus the same inputs plus one first-person/completeness rule. Same fixed CSFT model. Familiar: 48 pairs; fresh authored scenes: 36 pairs. Assistant ratings are not independent human review.</p>
"""+"\n".join(articles)+"</html>", encoding="utf-8")
    print("Paired 84 requests. All 48 familiar natural controls reproduce historical tokens.", flush=True)


def group(records):
    result = {"n": len(records), "arms": {}, "wins": {a: sum(r["winner"] == a for r in records) for a in (*ARMS, "tie")}}
    for arm in ARMS:
        scores = [r["scores"][arm] for r in records]
        result["arms"][arm] = {f: sum(s[f] for s in scores) for f in (*FIELDS[:5], FIELDS[8])}
        result["arms"][arm].update({"mean_"+f: sum(s[f] for s in scores)/len(scores) for f in FIELDS[5:8]})
        passes = [not s["contradiction"] and not s["omission"] and not s["agency"] and s["language_ok"] for s in scores]
        result["arms"][arm]["semantic_pass"] = sum(passes)
        result["arms"][arm]["semantic_and_voice_pass"] = sum(p and not s["third_person_self"] for p, s in zip(passes, scores))
    return result


def summarize():
    requests, generated = outputs()
    pairing = read(ROOT / "pair_manifest.json")
    assert pairing["blind_sha256"] == file_hash(ROOT / "blind_review.jsonl")
    assert pairing["key_sha256"] == file_hash(ROOT / "blind_key.json")
    blind = {r["id"]: r for r in rows(ROOT / "blind_review.jsonl")}
    annotations = rows(ROOT / "assistant_review.jsonl")
    reviews = {r["id"]: r for r in annotations}
    assert len(reviews) == len(annotations) == len(requests) == 84
    assert set(reviews) == set(blind)
    for review in reviews.values():
        assert review["winner"] in ("A", "B", "tie") and review["note"].strip()
        for a in "AB":
            s = review[a]
            assert len(s) == 9 and all(type(v) is int for v in s)
            assert all(s[i] in (0, 1) for i in (0, 1, 2, 3, 4, 8)) and all(1 <= s[i] <= 5 for i in (5, 6, 7))
            assert not s[0] or s[1]
            anchor = blind[review["id"]]["anchored_scores"][a]
            assert anchor is None or s == anchor
        if blind[review["id"]]["A"] == blind[review["id"]]["B"]:
            assert review["A"] == review["B"] and review["winner"] == "tie"
    write(ROOT / "review_frozen.json", {"sha256": file_hash(ROOT / "assistant_review.jsonl"), "fields": FIELDS,
          "reviewer": "Single non-independent assistant; anchored historical scores; familiar texts limit blinding"})
    key = read(ROOT / "blind_key.json")
    scored = [{**{k: r[k] for k in ("id", "panel", "family", "language")},
               "winner": "tie" if reviews[r["id"]]["winner"] == "tie" else key[r["id"]][reviews[r["id"]]["winner"]],
               "scores": {key[r["id"]][a]: dict(zip(FIELDS, reviews[r["id"]][a])) for a in "AB"}} for r in requests]
    panels = {panel: {"all": group([r for r in scored if r["panel"] == panel]),
              **{lang: group([r for r in scored if r["panel"] == panel and r["language"] == lang]) for lang in ("en", "fr")}}
              for panel in ("familiar", "fresh")}
    gates = {}
    for panel, groups in panels.items():
        for lang in ("en", "fr"):
            n, d = (groups[lang]["arms"][a] for a in ARMS)
            gates[panel+"-"+lang] = all(d[f] <= n[f] for f in ("role_error", "contradiction", "omission", "agency", "third_person_self")) and all(
                d[f] >= n[f] for f in ("semantic_pass", "semantic_and_voice_pass", "language_ok", "mean_coherence", "mean_relevance", "mean_grammar"))
    n, d = (panels["familiar"]["all"]["arms"][a] for a in ARMS)
    gates["familiar_strict_improvement"] = d["omission"] < n["omission"] and d["third_person_self"] < n["third_person_self"]
    technical = {}
    for panel in panels:
        technical[panel] = {}
        for arm in ARMS:
            samples = [generated[r["id"], arm] for r in requests if r["panel"] == panel]
            repeat = []
            for r in samples:
                words = re.findall(r"\w+", r["text"].lower())
                grams = [tuple(words[i:i+4]) for i in range(max(0, len(words)-3))]
                repeat.append((len(grams)-len(set(grams)))/len(grams) if grams else 0)
            technical[panel][arm] = {"empty": sum(not r["text"].strip() for r in samples), "eos": sum(r["stopped_on_eos"] for r in samples),
                "caps": sum(len(r["output_ids"]) == 192 for r in samples), "mean_repeated_4gram_fraction": sum(repeat)/len(repeat),
                "mean_input_tokens": sum(len(r["input_ids"]) for r in samples)/len(samples),
                "mean_output_tokens": sum(len(r["output_ids"]) for r in samples)/len(samples)}
    result = {"experiment": ROOT.name, "panels": panels, "gates": gates, "passes_gate": all(gates.values()),
              "families": {f: group([r for r in scored if r["family"] == f]) for f in sorted({r["family"] for r in scored})},
              "technical": technical, "pairing": pairing, "manifest": read(ROOT / "manifest.json"),
              "review": read(ROOT / "review_frozen.json"), "protocol_sha256": file_hash(ROOT / "protocol.json")}
    write(ROOT / "summary.json", result)
    write(Path("results") / (ROOT.name+".json"), result)
    print(json.dumps({"panels": panels, "gates": gates}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "generate", "pair", "summarize"))
    args = parser.parse_args()
    {"prepare": prepare, "generate": generate, "pair": pair, "summarize": summarize}[args.mode]()
