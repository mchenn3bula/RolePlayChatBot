# Explicit persona and scene state (P1)

P1 supplies a versioned, **user-authored** state ledger alongside the current
conversation. It uses the same pinned Ministral weights, FP16 runtime, decoding
settings, 2,048-token input budget and 192-token output cap as P0. No weights or
training data change. P0 remains selectable as a separate control.

The coding assistant does not read, quote or assess dialogue or generated replies.
New inference runs save replies for the user, with technical progress only in logs.
Past AI reviews are historical artifacts, not the workflow for future runs.

## Start a conversation

From the project directory in PowerShell:

```powershell
.\ministral.ps1 -Mode Chat -Profile P1 -Language en
.\ministral.ps1 -Mode Chat -Profile P1 -Language fr
```

Each session creates a unique `reports/chat-<timestamp>-<id>/` directory containing:

- `replies.html`: readable replies; open in a browser and refresh after each turn.
- `generations.jsonl`: exact messages, replies, token IDs, state snapshots and timings.
- `state.json`: your editable copy of the benign starter persona/scene.
- `manifest.json`: pinned configuration and finite-logit preflight metadata.

The terminal prints these paths. Personal state edits and transcripts stay under
ignored `reports/`; do not edit tracked example files with private dialogue.

To use an existing state file:

```powershell
.\ministral.ps1 -Mode Chat -Profile P1 -Language en -StateFile .\reports\my-state.json
```

`/state` shows the state file location. `/reload` validates and loads your edits;
increase `revision` whenever changing the file. `/reset` clears conversation
history and resets the user-turn counter, so scheduled events replay from turn 1.
Start a new chat for a different `scene_id` or language. `/quit` exits. A custom
state file is used directly; it is not overwritten by the chat runner.

**Dialogue alone does not update the ledger.** Record changes in the state file
and use `/reload` before sending the next message. Model replies never become
authoritative facts automatically. Each saved reply includes the exact snapshot
used, including any intentional corrections made with a newer revision.

## State schema

See `posttraining/personas/lantern_en.json` and `lantern_fr.json` for complete
authored examples. Every object has an explicit schema; unknown fields are rejected.

| Field | Purpose |
| --- | --- |
| `schema_version` | Currently `1` |
| `scene_id`, `revision`, `language` | Stable scene identity, nonnegative revision, `en` or `fr` |
| `persona` | Registered `character_id`, `voice`, `traits`, `goals` |
| `entities` | Stable IDs mapped to display names |
| `facts` | Initial persona and scene facts, sourced at turn `0` |
| `events` | Ordered changes applied at explicit user turns |

A fact has `id`, `subject`, `predicate`, `value`, `scope`, `status`, and `source`.
For example, this benign scene change moves the character at user turn 2:

```json
{
  "turn": 2,
  "set": [{
    "id": "character.location",
    "subject": "character",
    "predicate": "location",
    "value": "Beside the window",
    "scope": "scene",
    "status": "known",
    "source": {"kind": "user", "turn": 2}
  }],
  "remove": []
}
```

Add it to `events`, increase `revision`, and `/reload` before submitting turn 2.
Events take effect **through** their numbered user turn; future events are never
included in an earlier snapshot. The counter advances after a successfully saved
reply, not for commands or rejected inputs. Initial persona/setup facts use source
turn `0`; event facts use source kind `user` and their event turn.

`scope: persona` facts cannot be edited or removed by scene events. Changing the
persona itself requires an explicitly edited higher-revision file. Entity references
use `{"entity_id": "registered_id"}`. Unknown facts use `status: unknown` and
`value: null`. Claims use `status: claim` with
`value: {"by": "registered_id", "text": "Attributed statement"}`; they are
not promoted to established truth. Known values support text, booleans, entity
references and nonempty lists of text. Use one fact ID per subject/predicate/scope.

The entity registry is available from the beginning: keep unrevealed names out of
it, using generic labels until an explicit revision reveals them. Similarly, all
initial facts are visible from turn 1. Put timed scene changes in `events`.

## Context and limits

The native Mistral tokenizer counts the entire chat template. P1 keeps persona,
current state and latest user message intact, dropping only oldest complete
user/assistant exchanges when necessary. Saved audit metadata identifies dropped
turns. If those required components alone exceed the budget, the request is
rejected rather than silently truncating state. Full session history remains in
local artifacts; the input context is not unlimited.

This is a prompting control, not a guarantee that the model will follow every
fact. No generated-text quality claim is made without the user's own review.
Automatic extraction, summarization, retrieval, SFT and preference training are
separate future experiments.

## Development runs

```powershell
.\ministral.ps1 -Mode Evaluate -Profile P1 -RunName ministral-p1-my-run
.\ministral.ps1 -Mode Evaluate -Profile P0 -RunName ministral-p0-my-control
```

P1 uses `configs/ministral_p1.json`, the existing English/French development inputs
and `posttraining/eval_states_v1.json`. The state fixtures are authored controls,
not information extracted from outputs. Seeds and sampling settings match P0.
No test split is used. Output directories are exclusive and completed runs are
never overwritten. New runs produce a reply gallery and technical summary only;
`posttraining.report` locates a gallery without reading its content or scoring it.

The code tests use synthetic fixtures for turn replay, immutable persona facts,
unknowns/claims, invalid updates, context trimming, HTML escaping and content-free
progress. Real GPU checks are restricted to finite logits, successful execution,
saved artifact counts, timing and memory. Replies are for the user to inspect.

Verified September 24: all 56 tests and affected-file Ruff/shell checks passed.
The real P1 run `reports/ministral-p1-v1-20260924/` saved 108 replies (54 EN / 54 FR)
with finite logits and frozen weights. Peak GPU reservation was 9.17 GiB;
aggregate generation speed was 29.56 tokens/sec. The panel fit without dropping
history; budget trimming was verified with synthetic fixtures. Technical metadata
is in `results/ministral-p1-v1.json`. Generated replies were not read or judged.
