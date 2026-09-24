# State-format experiment v1: fewer role errors, worse fact coverage

Completed `ministral-state-format-v1`. Explicit entity assignments reduce role
confusion in both languages, but increase omissions and reduce the number of
responses satisfying every required constraint. **Do not adopt this rendering as
the default.** Both language-specific gates in the frozen protocol fail.
No training, adapter changes, chat-default changes or advanced method was used.

## Matched comparison

The fixed `checkpoints/ministral-v2-csft-v1/epoch-02` adapter generated both arms.
Eight already-inspected development scene families × English/French × three
seeds produced 48 paired requests / 96 replies. This is an iteration diagnostic,
not a new held-out test. The fresh prose control reproduced all 48 historical
CSFT outputs token-for-token.

Only `scene.established.value` changed: its original prose string became a list
of explicit entity/property assignments. Instructions, persona, entity registry,
fact identity/source, history, current user input, model and decoding were fixed.
All 16 mappings were manually audited for coverage, negations and time qualifiers
before generation. Both variants retained the same semantic assertions. The new
format uses the ledger's existing `character` and `user` identifiers; it does not
add a persistent schema or automatic state extraction.

This intervention combines explicit actor references and list/assignment syntax.
It does not isolate the causal effect of pronoun removal alone. Average native
input length increased from 360.44 to 385.81 tokens; nothing was truncated.

## Frozen paired review

One assistant reviewed every reply under randomized A/B labels, with original
scene context available but the transformed prompt and condition key hidden.
Annotations were frozen before unblinding. This assistant also authored the
fixtures, and familiar historical outputs limit blinding. These are **not human
or independent judgments**. There are only eight scene families, not 48 independent
replications. No significance or general-superiority claim is made.

| Per-response metric, 48 requests | Prose | Entity assignments |
| --- | ---: | ---: |
| Role/ownership confusion | 13 | 6 |
| Any factual contradiction | 15 | 8 |
| Required information omitted | 4 | 17 |
| All required constraints satisfied | 30 | 24 |
| Enacted user-agency violation | 0 | 0 |
| Correct response language | 48 | 48 |
| Mean coherence, 1–5 | 3.77 | 4.08 |
| Mean relevance, 1–5 | 4.02 | 3.96 |
| Mean grammar, 1–5 | 4.96 | 4.98 |

Entity assignments won 13 paired comparisons, prose won 11, and 24 tied. A win
can mean a less-bad reply: it does not require satisfying every constraint.
Contradiction and omission flags can overlap, and role confusion is a subset of
contradiction. Wrong values are not also scored as missing values of that same
slot; a separately absent requirement can still count as an omission.

| Language | Role errors, prose → entity | Contradictions | Omissions | All constraints satisfied |
| --- | ---: | ---: | ---: | ---: |
| English, 24 | 6 → 3 | 7 → 4 | 1 → 7 | 16 → 13 |
| French, 24 | 7 → 3 | 8 → 4 | 3 → 10 | 14 → 11 |

This is a fresh review with an added explicit role-error flag. Some borderline
omission/agency decisions differ from the previous three-arm review, despite
identical prose outputs. The previous annotations and results remain unchanged;
compare arms within this review rather than mixing these counts with older ones.
Every borderline decision has a note in `assistant_review.jsonl`. For example,
beginning to listen now is allowed as a new character action; past payment
accusations are factual errors rather than newly enacted user actions. Suggestions
that preserve choice may pass the constraint check despite low relevance scores.

## What changed in replies

- Ceramics, English seed 42: prose assigned the relationship and kiln operation
  to the user. The entity version answered, “No, she doesn't. I'm the one who
  does.” This fixes the operator but leaves the false daughter premise uncorrected.
  Across this family, role errors fell 4 → 1 while omissions rose 0 → 6.
- Screening, French seed 42: prose answered “Non, c'est déplacé à samedi en
  salle 4.” The entity version said only “Non, la séance de vendredi est annulée.”
  It omitted the new time/place despite both being present in the input. The
  screening family's fully correct replies fell 6 → 2.
- Telescope, English seed 42: the original correctly gave the cover to Paz;
  the entity version instead claimed “I have the telescope cover”. Entity
  assignments do not eliminate attribution errors.
- Painting, English seed 42: the new format fixed whose workbench holds the
  painting, but introduced a claim implying wrapping was unfinished. Correct
  role reference alone does not ensure complete state fidelity.

The entity arm produced shorter answers on average, 17.83 versus 20.94 native
output tokens including EOS. Both arms had 48/48 EOS endings, no empty replies,
no length caps and zero repeated word 4-grams. Shorter answers coincided with more
omissions; this experiment does not establish length as their cause.

## Artifacts and verification

- User gallery: `reports/ministral-state-format-v1/comparison.html`, including
  each arm's exact input and all generated replies.
- Frozen source protocol: `STATE_FORMAT_PROTOCOL.md`.
- Fixture mapping: `posttraining/state_format_fixtures.json`.
- Request/native-token freeze and manual coverage audit: `requests.jsonl`,
  `fact_audit.json`, and `protocol.json` inside the report directory.
- Randomized review/key, frozen annotation hash, generation hashes and technical
  manifests are retained in the same directory. Source, model adapter, previous
  control outputs and protocol fingerprints are verified before and after inference.
- Aggregate result: `results/ministral-state-format-v1.json`.
- Visible launcher used: `.\evaluate_state_format.ps1`; a second invocation
  refuses to overwrite the existing experiment. Unique technical log is under
  `reports/logs/ministral-state-format-v1-*.log`; replies are not printed there.

Full GPU-enabled test suite: 77 tests, 76 passed and one native-only tokenizer test
skipped in the baseline environment. All seven LoRA/native tests then passed in
the LoRA environment. Native encoding was also exercised on every paired prompt;
runtime serving inputs matched those frozen IDs. Ruff, Bash syntax and PowerShell
parsing passed. Peak GPU reservation was 9.22 GiB, with 11.79 GiB device-free at
completion and no memory-guard failure. All adapter files remained unchanged.

## Next decision

Keep the current default. The results support investigating explicit references,
but do not justify promoting assignment-style state or training on it.

The next narrowly scoped experiment should retain natural sentences and their
original order, replacing only ambiguous second-person references with explicit
character references. This would separate referent clarity from assignment syntax
more cleanly. Preserve this failed format run, freeze any follow-up before new
outputs, and require both role-error and omission improvements. A promising
development result should then be checked on fresh authored EN/FR scene families
and user review before applying it to chat or further SFT.
