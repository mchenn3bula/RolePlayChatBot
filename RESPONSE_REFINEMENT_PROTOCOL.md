# Direct, complete character replies: refinement v1

Freeze protocol, exact rule, authored new scenes, semantic criteria, source hashes
and historical rating anchors before generation. This continues the natural-
reference experiment and its literature rationale, not a new training method.

## One bundled response-rule intervention

Baseline `natural`: unchanged natural-reference prompts. Candidate `direct`: add
one sentence immediately before the state JSON in the system message:

EN: Speak as the character in first person and answer every part of the user's
question using the current facts, correcting any false premise.

FR: Parle à la première personne en incarnant le personnage et réponds à chaque
partie de la question de l'utilisateur à partir des faits actuels, en corrigeant
toute prémisse fausse.

All facts, persona, original instructions, history and current user input stay
identical. This intentionally bundles first-person voice, question coverage and
premise correction; it does not identify the separate effect of each clause.
No answer exemplars, reasoning trace, new memory system or training are added.

Same CSFT epoch-02 adapter, P1 native tokenizer, FP16/SDPA, 2048 context and 192
output tokens, temperature .7, top-p .9, top-k 50, repetition penalty 1, seeds
42/43/44. Alternate arm order; verify frozen native input IDs and model hashes.
No truncation, output editing or retries chosen by reply quality.

## Familiar diagnostic and fresh validation

All 48 previous natural-reference requests remain the familiar diagnostic set.
Their natural-arm outputs must reproduce the archive token-for-token. Add six new
authored scene families, each EN/FR × three seeds, for 36 fresh requests: 84 pairs,
168 total replies. Fresh scenes combine handoffs, multi-part questions, relational
roles, revised appointments, uncertain quantities, multiple-item constraints and
demonstration/observation. Freeze both panels before any outputs; no adaptive
prompt changes after seeing familiar results. These are new scene instances with
related skills, not an independent benchmark or untouched project test split.
Report panels separately; improvements on familiar scenes cannot mask fresh
regressions. User's private source dialogue remains unread.

Fresh minimum requirements are frozen in each fixture's `criteria`:
all requested possession/relationship/time/location/epistemic facts must be
answered; paraphrases are valid. Known false premises must be corrected, and
explicitly unknown values may not be fabricated. Compatible character actions
are permitted, but do not enact the user's actions/decisions/feelings.

## Frozen review

Retain the previous eight fields and per-item/text scores without changing any
historical judgments. Add a ninth binary field, `third_person_self`, marking
explicit self-description via the character's role noun phrase instead of first-
person character speech (including mixed first/third person). A correct mention
of another person is not this error. Correct third-person self-description is a
voice issue, not automatically a role-attribution contradiction. Do not require
the literal token I/je when an answer naturally omits a subject. Full state/role
recitation referring to the speaking character also counts. Other evasiveness,
grammatical and attribution problems remain covered by the original fields.

Before generation, add voice labels to historical anchors: the six known natural
replies with third-person self-description are telescope EN43, ceramics EN43/EN44/
FR44, equipment EN43 and painting EN42. All other previously reviewed texts have
no explicit third-person self-description. These labels are frozen before new
outputs and refer to exact archived texts, not all outputs sharing those IDs.
Reuse identical item/text ratings under either new arm. Randomize A/B labels,
provide baseline semantic context and criteria, and freeze all new scores and
rankings before unblinding. Identical text requires identical scores and a tie.
Judgments remain from the authoring assistant; familiar anchors limit blinding.

Report contradiction, omission, role_error, agency, language, coherence,
relevance, grammar, all semantic constraints satisfied, third_person_self, and
semantic-plus-voice pass counts. Also report paired preference, lengths, EOS,
caps, empties and repetition. Third-person voice is separate from factual quality.

Candidate gate: in each language of each panel, no increase in role errors,
contradictions, omissions, agency errors or third-person self-description; no
decrease in semantic passes, semantic-plus-voice passes, correct-language count
or mean quality scores. Additionally, familiar-panel total omissions and third-
person self-description must both strictly decrease. Fresh results cannot be
pooled with familiar results to rescue a failed gate. This is a conservative
local rule, not significance testing. No automatic default promotion, expanded
prompt search or training follows from this experiment.
