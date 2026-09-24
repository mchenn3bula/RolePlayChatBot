# Entity-explicit scene-format ablation v1

Freeze this protocol, fixtures, runner and review criteria before generation.
This is a development diagnostic on the already-inspected eight bilingual v2
scene families, not a new held-out test or an independent confirmation.

## Single intervention

Use the unchanged `ministral-v2-csft-v1/epoch-02` adapter and pinned P1 inference
configuration. Freshly generate both arms for the same 48 requests (eight scenes,
two languages, seeds 42/43/44). The prose arm must reproduce the original inputs.
The entity arm changes only the value of the existing `scene.established` fact
from second-person prose to a list of explicit entity/property assignments.
`character` refers to the existing persona.character_id; `user` is the existing
user entity. Local object identifiers are descriptive labels, not new people.
Keep every original assertion, negation and temporal qualifier. Do not add the
expected answer, new scene facts, new instructions or a corrective dialogue turn.
Keep system instructions, persona, entity registry, source attribution, current
user input and previous exchanges byte-identical outside this one JSON value.

The authored mapping is manually audited against all 16 original scene values
before generation. Save both prompts, native token IDs and the value-level audit.
This tests a bundle of explicit reference and assignment-style serialization;
it does not isolate pronouns from list structure, token length or lexical choice.
It is a rendering experiment, not a new persistent ledger schema or extractor.

## Fixed execution

Both conditions: FP16, SDPA, unchanged model revision and adapter hashes, native
Mistral template, context 2048, max output 192, temperature .7, top-p .9, top-k 50,
repetition penalty 1.0. Reset the same sampling seed per paired request. Assert
each arm's native input equals its frozen encoding; no context truncation.
Alternate condition order per request to reduce timing/order confounding.
Use existing GPU preflight and 18 GiB reserved / 4 GiB free guards. Log technical
progress only. No training, checkpoint selection, tuning or automatic promotion.

## Frozen scoring

Reuse the v2 minimum semantic criteria without alteration. Primary outcome is
explicit role/ownership confusion per reply: assigning the character's possessions,
relationships, knowledge, actions or responsibilities to the user (or another wrong
entity), including reversed actor/observer roles. Add this as a separate binary
flag; also retain total contradiction, omission, agency, language correctness and
1–5 coherence/relevance/grammar. A role error is a contradiction; enacted user
actions may also be an agency violation. Mere second-person wording is not an
error when it correctly refers to the user. General unknown-state mistakes without
wrong entity attribution are contradictions, not necessarily role confusion.

Randomize A/B labels per request; review outputs against original prose/expected
semantics without the transformed prompt or key. Freeze all annotations before
unblinding. Rate all 96 replies, allow ties; report English/French and family-level
counts, paired wins/ties/losses, and technical empty/EOS/cap/repetition statistics.
Judgments are by the same assistant who authored the mapping, not independent
human review. Familiar historical outputs limit blinding. Do not infer significance
from 48 correlated requests; eight scene families and three decoding seeds do not
measure broad generalization. Show full outputs for the user's own review.

For a promising local result require strictly fewer role errors in each language,
no increase in omissions or total contradictions in either language, no increase
in agency/language failures and no reduction in mean coherence/relevance/grammar.
If it fails, preserve the negative result and propose a separate targeted follow-up;
do not revise prompts after seeing outputs within this experiment. Even a passing
result requires fresh scene validation before changing the default chat path.
