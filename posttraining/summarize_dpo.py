"""Aggregate a completed, frozen randomized A/B assistant review."""

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import re

from posttraining.compare_dpo import read_rows
from posttraining.runtime import file_hash


FIELDS = ("fact_bad", "fact_missing", "agency_violation", "language_ok", "coherence", "relevance", "grammar")


def summarize(root):
    requests = read_rows(root / "blind_review.jsonl")
    reviews = read_rows(root / "assistant_review.jsonl")
    by_id = {r["id"]: r for r in reviews}
    if len(by_id) != len(reviews) or set(by_id) != {r["id"] for r in requests}:
        raise ValueError("Review must cover each request exactly once.")
    for request in requests:
        review = by_id[request["id"]]
        if review["preferred"] not in ("A", "B", "tie"):
            raise ValueError("Invalid preference.")
        for side in ("A", "B"):
            values = review[side]
            if (len(values) != 7 or any(not isinstance(v, int) for v in values)
                    or min(values[:2]) < 0 or sum(values[:2]) > len(request["fact_slots"])
                    or any(v not in (0, 1) for v in values[2:4])
                    or any(not 1 <= v <= 5 for v in values[4:])):
                raise ValueError(f"Invalid scores: {request['id']} {side}")
    # Validate and freeze review before opening the hidden model key.
    freeze = root / "review_frozen.json"
    fingerprint = file_hash(root / "assistant_review.jsonl")
    if freeze.exists():
        if json.loads(freeze.read_text())["review_sha256"] != fingerprint:
            raise ValueError("Review changed after unblinding.")
    else:
        freeze.write_text(json.dumps({"review_sha256": fingerprint, "fields": FIELDS,
            "reviewer": "Single non-independent AI reviewer; authored synthetic preference curriculum"}, indent=2))
    key = json.loads((root / "blind_key.json").read_text())
    scored = []
    for request in requests:
        review = by_id[request["id"]]
        scored.append({**request, "winner": "tie" if review["preferred"] == "tie" else key[request["id"]][review["preferred"]],
            "scores": {key[request["id"]][side]: dict(zip(FIELDS, review[side])) for side in ("A", "B")}})

    def group(rows):
        counts = Counter(r["winner"] for r in rows)
        opportunities = sum(len(r["fact_slots"]) for r in rows)
        metrics = {}
        for arm in ("sft", "dpo"):
            sums = {f: sum(r["scores"][arm][f] for r in rows) for f in FIELDS}
            metrics[arm] = {**sums, "fact_opportunities": opportunities,
                "mean_coherence": sums["coherence"] / len(rows),
                "mean_relevance": sums["relevance"] / len(rows),
                "mean_grammar": sums["grammar"] / len(rows)}
        return {"n": len(rows), "dpo_wins": counts["dpo"], "ties": counts["tie"], "sft_wins": counts["sft"],
                "dpo_win_share": (counts["dpo"] + 0.5*counts["tie"]) / len(rows), "metrics": metrics}

    groups = {"all": scored, "en": [r for r in scored if r["language"] == "en"],
              "fr": [r for r in scored if r["language"] == "fr"],
              "first_turn": [r for r in scored if r["turn"] == 1],
              "followup": [r for r in scored if r["turn"] > 1]}
    families = sorted({r["family"] for r in scored})
    grouped = {f: [r for r in scored if r["family"] == f] for f in families}
    rng = random.Random(20260926)
    boot = []
    for _ in range(10000):
        sample = [r for f in rng.choices(families, k=len(families)) for r in grouped[f]]
        boot.append(sum(1 if r["winner"] == "dpo" else 0.5 if r["winner"] == "tie" else 0 for r in sample) / len(sample))
    boot.sort()
    result = {"review_sha256": fingerprint, "reviewer": "Single non-independent AI reviewer",
              "groups": {name: group(rows) for name, rows in groups.items()},
              "families": {name: group(rows) for name, rows in grouped.items()},
              "cluster_bootstrap": {"families": len(families), "resamples": 10000,
                  "dpo_win_share_95_percentile_interval": [boot[249], boot[9749]],
                  "interpretation": "Descriptive only; small familiar development panel and one training seed"},
              "generation": {}}
    for arm in ("sft", "dpo"):
        rows = read_rows(root / arm / "generations.jsonl")
        repetitions = []
        for row in rows:
            words = re.findall(r"\w+", row["text"].lower())
            grams = [tuple(words[i:i+4]) for i in range(max(0, len(words)-3))]
            repetitions.append((len(grams)-len(set(grams)))/len(grams) if grams else 0)
        manifest = json.loads((root / arm / "manifest.json").read_text())
        result["generation"][arm] = {"empty": sum(not r["text"].strip() for r in rows),
            "ended_with_eos": sum(r["output_ids"][-1:] == [2] for r in rows),
            "reached_192_tokens": sum(len(r["output_ids"]) == 192 for r in rows),
            "mean_output_tokens": sum(len(r["output_ids"]) for r in rows)/len(rows),
            "mean_repeated_word_4gram_fraction": sum(repetitions)/len(rows),
            "manifest": manifest}
    (root / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"groups": result["groups"], "cluster_bootstrap": result["cluster_bootstrap"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    summarize(parser.parse_args().output_dir)
