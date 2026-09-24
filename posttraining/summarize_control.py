"""Unblind a frozen three-arm review and report scene-clustered comparisons."""

import json
from pathlib import Path
import random
import re

from posttraining.compare_control import ROOT, ARMS
from posttraining.compare_dpo import read_rows
from posttraining.lora_train import write_json
from posttraining.runtime import file_hash


FIELDS = ("contradiction", "omission", "agency", "language_ok", "coherence", "relevance", "grammar")


def main():
    criteria_hash = file_hash(ROOT / "review_criteria.json")
    if criteria_hash != json.loads((ROOT / "review_criteria_fingerprint.json").read_text())["sha256"]:
        raise ValueError("Pre-review semantic criteria changed.")
    blinded = read_rows(ROOT / "blind_review.jsonl")
    annotations = read_rows(ROOT / "assistant_review.jsonl")
    reviews = {r["id"]: r for r in annotations}
    if len(reviews) != len(annotations) or set(reviews) != {r["id"] for r in blinded}:
        raise ValueError("Review every request exactly once.")
    for review in reviews.values():
        flat = [letter for group in review["rank"] for letter in group]
        if sorted(flat) != list("ABC") or any(not group for group in review["rank"]):
            raise ValueError("Rank all three anonymous arms, with ties grouped.")
        for letter in "ABC":
            values = review[letter]
            if (len(values) != 7 or any(type(x) is not int for x in values)
                    or any(x not in (0, 1) for x in values[:4]) or any(not 1 <= x <= 5 for x in values[4:])):
                raise ValueError("Invalid anonymous score.")
    review_hash = file_hash(ROOT / "assistant_review.jsonl")
    frozen = ROOT / "review_frozen.json"
    if frozen.exists() and json.loads(frozen.read_text())["sha256"] != review_hash:
        raise ValueError("Review changed after unblinding.")
    if not frozen.exists():
        write_json(frozen, {"sha256": review_hash, "fields": FIELDS,
            "reviewer": "Single assistant, also data author/curator; not independent human review"})
    key = json.loads((ROOT / "blind_key.json").read_text())
    rows = []
    for prompt in blinded:
        review = reviews[prompt["id"]]
        ranks = {key[prompt["id"]][letter]: i for i, group in enumerate(review["rank"]) for letter in group}
        rows.append({"id": prompt["id"], "family": prompt["family"], "language": prompt["language"], "ranks": ranks,
            "scores": {key[prompt["id"]][letter]: dict(zip(FIELDS, review[letter])) for letter in "ABC"}})

    def group(records):
        summary = {"n": len(records), "arms": {}, "dpo_vs_csft": {"dpo_wins": 0, "ties": 0, "csft_wins": 0}}
        for arm in ARMS:
            scores = [r["scores"][arm] for r in records]
            summary["arms"][arm] = {field: sum(s[field] for s in scores) for field in FIELDS[:4]}
            summary["arms"][arm].update({"mean_"+field: sum(s[field] for s in scores)/len(scores) for field in FIELDS[4:]})
            summary["arms"][arm]["all_constraints_satisfied"] = sum(
                not s["contradiction"] and not s["omission"] and not s["agency"] and s["language_ok"] for s in scores)
        for row in records:
            a, b = row["ranks"]["dpo"], row["ranks"]["csft"]
            summary["dpo_vs_csft"]["dpo_wins" if a < b else "csft_wins" if a > b else "ties"] += 1
        pref = summary["dpo_vs_csft"]
        pref["dpo_win_share"] = (pref["dpo_wins"]+0.5*pref["ties"])/len(records)
        return summary

    families = sorted({r["family"] for r in rows})
    by_family = {f: [r for r in rows if r["family"] == f] for f in families}
    rng = random.Random(20260930)
    bootstrap = []
    for _ in range(10000):
        selected = [r for family in rng.choices(families, k=len(families)) for r in by_family[family]]
        bootstrap.append(sum(1 if r["ranks"]["dpo"] < r["ranks"]["csft"] else
                             0.5 if r["ranks"]["dpo"] == r["ranks"]["csft"] else 0 for r in selected)/len(selected))
    bootstrap.sort()
    result = {"experiment": "ministral-bilingual-control-v2", "review_sha256": review_hash,
        "review_criteria_sha256": criteria_hash,
        "reviewer": "Single non-independent assistant", "scope": "Exposure/update matched; not equal compute",
        "groups": {"all": group(rows), **{lang: group([r for r in rows if r["language"] == lang]) for lang in ("en", "fr")}},
        "families": {f: group(rs) for f, rs in by_family.items()},
        "cluster_bootstrap": {"families": len(families), "resamples": 10000,
                              "dpo_win_share_95_percentile_interval": [bootstrap[249], bootstrap[9749]], "descriptive_only": True},
        "generation": {}}
    for arm in ARMS:
        generated = read_rows(ROOT / arm / "generations.jsonl")
        repeats = []
        for row in generated:
            words = re.findall(r"\w+", row["text"].lower())
            grams = [tuple(words[i:i+4]) for i in range(max(0, len(words)-3))]
            repeats.append((len(grams)-len(set(grams)))/len(grams) if grams else 0)
        manifest = json.loads((ROOT / arm / "manifest.json").read_text())
        result["generation"][arm] = {"mean_output_tokens": sum(len(r["output_ids"]) for r in generated)/len(generated),
            "empty": sum(not r["text"].strip() for r in generated),
            "eos": sum(r["stopped_on_eos"] for r in generated),
            "capped": sum(len(r["output_ids"]) == 192 for r in generated),
            "mean_repeated_4gram_fraction": sum(repeats)/len(repeats),
            "seconds": sum(r["seconds"] for r in generated), "manifest": manifest}
    write_json(ROOT / "summary.json", result)
    write_json(Path("results/ministral-bilingual-control-v2.json"), result)
    print(json.dumps({"groups": result["groups"], "bootstrap": result["cluster_bootstrap"]}, indent=2))


if __name__ == "__main__":
    main()
