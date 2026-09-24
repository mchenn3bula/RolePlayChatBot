# Natural character references: strong local improvement, coverage gate still fails

Completed `ministral-natural-reference-v1` after the primary-paper review in
[NATURAL_REFERENCE_RESEARCH.md](NATURAL_REFERENCE_RESEARCH.md). Explicit character
references in otherwise unchanged natural sentences eliminated **observed explicit
role-attribution errors on this panel** and improved task completion. They did
not eliminate omissions or third-person copying. The prespecified no-omission-
regression gate fails in both languages, so the chat default remains unchanged.
No training, adapter modification or advanced method was used.

## What was controlled

Same `ministral-v2-csft-v1/epoch-02` adapter, original 48 v2 development requests,
native Mistral tokenizer, history, current inputs, instructions and decoding.
Only the existing scene fact's string value changed. Exact reversible spans
replace second-person character references with the existing role description
and necessary grammatical agreement. Sentence order, punctuation, facts, negation
and chronology are preserved. No new character names, instructions or examples.

Example: “You still carry the eyepiece case.” becomes “The astronomy club host
still carries the eyepiece case.” The user and Paz keep their original references.

Six scene families changed, giving 36 paired requests. Screening and song-sharing
had no targeted character references; their 12 requests are identical negative
controls. All 48 fresh prose outputs reproduced their historical native output
tokens exactly, and all 12 unchanged controls matched across conditions.

The research motivates a controlled attribution test, not a guaranteed fix.
[SPASM (ACL 2026)](https://aclanthology.org/2026.findings-acl.412/) studies
perspective-aware history; our experiment changes scene references instead and
does not reproduce its method. See the research note for five references,
including contrary evidence about evaluation-driven prompt sensitivity.

## Results

Scores are single-assistant judgments, not independent human ratings. To prevent
review drift, 68 of the 96 reply ratings were copied from pre-frozen scores for
identical text at the same prompt ID. The 28 new texts were reviewed under
randomized condition labels using the same semantic criteria. All annotations
were fixed before unblinding. Familiar outputs and displayed rating anchors mean
this is not strong blinding.

| Per-response metric | Original prose, 48 | Natural references, 48 |
| --- | ---: | ---: |
| Explicit role/ownership errors | 13 | 0 |
| Any explicit factual contradiction | 15 | 0 |
| Required information omitted | 4 | 7 |
| All required constraints satisfied | 30 | 41 |
| Enacted user-agency violations | 0 | 0 |
| Correct response language | 48 | 48 |
| Mean coherence, 1–5 | 3.77 | 4.52 |
| Mean relevance, 1–5 | 4.02 | 4.56 |
| Mean grammar, 1–5 | 4.96 | 5.00 |

Natural references won 18 pairs, original prose won four, and 26 tied. Twelve
ties are unchanged controls. On the **36 actually changed requests**, all-
constraint passes increased **18 → 29**, with 18 natural wins / four prose wins /
14 ties. All observed error-count differences come from this changed subset.

| Language | Role errors | Contradictions | Omissions | All constraints satisfied |
| --- | ---: | ---: | ---: | ---: |
| English, 24 requests | 6 → 0 | 7 → 0 | 1 → 3 | 16 → 21 |
| French, 24 requests | 7 → 0 | 8 → 0 | 3 → 4 | 14 → 20 |

Both gates fail specifically because omissions increased. The gate was frozen
before generation and has not been relaxed in light of the larger overall gain.
Five replies acquired an omission flag and two prior omissions were resolved:
three of those five previously had contradictions, while two were previously
fully correct. Thus some failures became less harmful, but two genuine coverage
regressions remain. This paired breakdown is explanatory, not a new endpoint.

The previous assignment-format run had six role errors, eight contradictions,
17 omissions and 24/48 all-constraint passes. These are **archival context**, not
a newly randomized third arm. It used the same model and requests, and the same
per-text rating anchors apply here. The natural-sentence result is more promising
than that prior serialization without satisfying the full acceptance criterion.

## What improved, and what remains

- Telescope ownership: all-constraint passes rose 3/6 → 6/6. French seed 43 now
  says “La housse est à Paz et je porte toujours la mallette des oculaires.”
  The original incorrectly assigned carrying the case to the user.
- Equipment reservation: passes rose 1/6 → 5/6. English seed 42 now refuses
  borrowing and states Vic's Wednesday-evening deadline rather than extending a
  booking for the requesting user.
- Ceramics: passes rose only 2/6 → 3/6. English seed 42 correctly identifies the
  speaker as kiln operator but still fails to correct “daughter” to “partner”.
  French seed 42 loses that correction from an originally correct response.
- Painting: passes rose 2/6 → 3/6. English seed 44 newly evades the known location
  by saying the character does not track framed items. French seed 42 corrects
  payment status but gives only “still there”, omitting the required location.
- Some outputs recite “the ceramics studio owner…” instead of naturally saying
  “I…”. Correctly naming the role is not an attribution error, but this is still
  a conversational-quality limitation. Ratings and notes record that distinction;
  zero attribution errors must not be read as perfect role-play.

Original versus natural mean input length was 360.44 versus 367.06 tokens. Mean
output length including EOS was 20.94 versus 22.44. Both arms had 48 EOS endings,
zero empty replies and zero output caps. Mean repeated word 4-gram fraction rose
from 0 to 0.00932 (0.93%), consistent with the observed repeated role descriptions.
There is no evidence here for a universally repetition-free or generally reliable
prompt. The intervention changes lexical references and length as well as
grammatical person; it does not isolate a pure pronoun mechanism.

## Reproducibility and artifacts

- Full gallery with exact paired inputs: `reports/ministral-natural-reference-v1/comparison.html`.
- Pre-run research and protocol: `NATURAL_REFERENCE_RESEARCH.md`, `NATURAL_REFERENCE_PROTOCOL.md`.
- Reversible edits and runner: `posttraining/natural_reference_edits.json`, `posttraining/natural_reference.py`.
- Frozen `requests.jsonl`, `fact_audit.json`, `rating_anchors.jsonl`, protocol/source/
  adapter hashes, native IDs, randomized review, annotation hash and manifests are
  together under the report directory. Existing experiment artifacts remain intact.
- Aggregate scores: `results/ministral-natural-reference-v1.json`, including
  per-language, per-family and changed/unchanged subsets.
- Original execution command: `.\evaluate_natural_reference.ps1`; the launcher
  prints visible technical progress and creates a unique log in `reports/logs/`.
  Completed or partial experiment directories cannot be silently overwritten.

Validation: full GPU-enabled suite ran 79 tests (78 passed, one native-only skip
in the baseline environment). Native encoding and serving IDs matched on all 96
generated requests in the LoRA environment. Ruff, Bash syntax and PowerShell
parsing passed. Preflight logits were finite, parameters frozen, adapter hashes
unchanged. Peak reserved VRAM was 9.22 GiB; device-free memory at completion was
11.94 GiB. No GPU work remains. Raw source conversations were not inspected.

## Decision

Retain natural references as the leading **experimental candidate**, not a promoted
default. This familiar development panel contains only eight families and three
sampling seeds; results are not independent confirmations or evidence of universal
zero-error behavior. Before deployment, prioritize fresh authored English/French
scenes with multi-fact questions, independent/user review, and explicit checks of
coverage and conversational voice. Any further prompting experiment should change
one factor and retain this result; do not tune this frozen run after seeing its
failures. Additional training or advanced losses are not justified by this result
alone, and none has been launched.
