# Natural-sentence character-reference ablation v1

Research basis: `NATURAL_REFERENCE_RESEARCH.md`, completed before implementation
and generation. Freeze this protocol, reference edits, code and historical rating
anchors before viewing any new output. Preserve the failed assignment-format run.

## Conditions

Same fixed CSFT epoch-02 adapter and 48 authored v2 development requests as the
state-format experiment: eight families × EN/FR × seeds 42/43/44. Generate both
original `prose` and `natural` conditions afresh, alternating execution order.
Use unchanged P1 FP16/SDPA/native tokenizer and decoding (2048 context, 192 output,
temperature .7, top-p .9, top-k 50, repetition penalty 1).

Change only `scene.established.value`, retaining its string type, sentence order
and punctuation. Replace second-person character references with definite noun
phrases derived from the already supplied role; change verbs/possessives only as
grammar requires. Every edit must match exactly once and reverse exactly. Retain
all other state fields, instructions, persona, history, user input, named entities,
facts, negations and time qualifiers. Do not add claims or an answer template.
No names, new identity mapping, first-person directives or training are added.

Six families change (36 requests); film_session and song_sharing have no targeted
references, so both language versions remain byte-identical (12 requests). These
are deterministic negative controls, not evidence of improved performance.
Require original native IDs and outputs to reproduce the archived prose control,
and require unchanged-condition outputs to match exactly. Fail the experiment's
validity check if these invariants fail. All prompts must fit with no truncation.
Measure native token lengths; role noun phrase repetition remains a lexical/length
confound, so this does not prove a pure grammatical-person mechanism.

## Review and endpoints

Use unchanged minimum semantic criteria and eight score fields from the frozen
state-format experiment: role_error, contradiction, omission, agency, language_ok,
coherence, relevance, grammar. Role errors imply contradictions. Accept semantic
equivalents and retain the previous documented borderline interpretations.

Before generation, freeze the previous item-specific text/score pairs. Reuse their
scores for any identical text at the same prompt ID, regardless of condition.
Reject conflicts between archived ratings of identical item/text. Present each
new pair with randomized A/B condition labels and original context. Review new
texts against the frozen rubric, fill only unanchored scores, and rank pairs with
ties allowed. All labels and notes must be complete before unblinding. Require
identical output texts within a pair to have identical scores and a tied ranking.
This is still a non-independent assistant review; familiar texts and displayed
historical anchors limit blinding. No claim of independent or human assessment.

Report all 48 and the 36 changed requests separately, plus EN/FR/family counts and
the 12 unchanged controls. Primary role-error counts must decrease in each
language. In each language omissions, total contradictions and agency violations
must not increase; all-constraint passes, correct-language count and mean quality
ratings must not decrease. This is the local candidate gate, not statistical
significance. Report paired wins/ties/losses and empty/EOS/cap/repetition measures.
The earlier assignment condition is archival context, not a newly randomized arm.

No optimization over prompt variants after observing outputs. No default promotion
or extra training in this run. A positive pilot requires fresh authored bilingual
scenes and user/independent review before adoption. A failed run remains preserved
and must not be relabeled or repaired within the same experiment directory.
