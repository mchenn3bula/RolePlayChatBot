# References for the natural-sentence character-reference experiment

Literature search completed before implementing or generating the new conditions.
Scope: primary papers on role attribution, prompt representation and evaluation;
this is a focused review, not a systematic survey. Publication details below come
from the papers/venue pages, not search-engine relative dates.

## Most directly related: attribution rather than personality imitation

**Luo and Laban, SPASM: Stable Persona-driven Agent Simulation for Multi-turn
Dialogue Generation, Findings of ACL 2026, July.**
[Paper and publication record](https://aclanthology.org/2026.findings-acl.412/).
Read the introduction, §2.1 perspective projection, the history-construction ablation and
limitations. Their method retains absolute speaker identities and projects history
into each agent's self/partner perspective. This addresses role confusion/echoing
in two-agent simulations without changing weights. Their stated limitations
include mostly English instruction-tuned models and untested transfer to smaller
models/languages. Their persona construction itself uses second-person prose:
the paper does **not** establish that second person is inherently wrong, or that
our third-person scene edit will work. Application here is an inference: test
reference clarity while preserving content and other conditions. We are not
implementing or claiming to reproduce their history-projection method.

**Lu et al., RoleMRC: A Fine-Grained Composite Benchmark for Role-Playing and
Instruction-Following, Findings of ACL 2025.**
[Paper](https://aclanthology.org/2025.findings-acl.1082/).
Read the task formulation, role/knowledge-boundary examples and benchmark setup.
It treats role identity, instruction following, and role-specific answerability as
distinct capabilities. Our design implication is to keep attribution, factual
coverage and unknown-information handling separate: correcting the speaker must
not receive full credit if a requested fact disappears. Its benchmark/training
recipe is not evidence for a particular English/French scene serialization.

## Prompt effects and measurement

**Sclar et al., Quantifying Language Models' Sensitivity to Spurious Features in
Prompt Design or: How I learned to start worrying about prompt formatting,
arXiv:2310.11324 (2023; version 2 read).**
[Full paper](https://arxiv.org/html/2310.11324v2).
Read the meaning-preserving format experiments and atomic-variation analysis.
The paper demonstrates substantial few-shot format sensitivity and weak transfer
of format rankings between models. This motivates holding the model and semantic
content fixed, measuring token lengths and retaining a reproducible control.
Its results concern different models/tasks; they do not imply that natural prose
universally beats structured facts. We do not run FormatSpread or a prompt search.

**Hua et al., Flaw or Artifact? Rethinking Prompt Sensitivity in Evaluating LLMs,
EMNLP 2025.**
[Paper](https://aclanthology.org/2025.emnlp-main.1006/).
Read the evaluation-method comparison and conclusions. Across their tasks,
rigid matching and likelihood-based evaluation overstate some prompt sensitivity
relative to semantic judging. This qualifies the preceding paper rather than
disproving all format effects. We will judge meaning, accept valid paraphrases,
and preserve item-specific historical ratings for identical replies. That last
procedure is our experimental-control choice, not a method claimed by this paper.
Single-assistant judgments remain fallible and non-independent.

**Peng and Chen, Rethinking Role-Playing Evaluation: Anonymous Benchmarking and
A Systematic Study of Personality Effects, SIGDIAL 2026, August.**
[Paper](https://aclanthology.org/2026.sigdial-1.15/).
Read the anonymous-benchmark motivation and evaluation setup. The paper probes
whether known character names supply memorized cues that inflate apparent
role-playing ability. We retain our authored, unnamed role scenarios and reuse
the existing occupation labels rather than introducing famous character names.
This removes one obvious new confound; it does not make our small familiar panel
independent or establish generalization to unseen roles.

## Design decision

Proceed with one separately frozen natural-reference ablation. Preserve the
original prose string, sentence boundaries, order and all assertions. Replace
only second-person character references with the existing role's definite noun
phrase, with necessary possessive syntax/verb agreement. Leave user references
and already explicit third-party names alone. Do not add examples, role rules,
reasoning prompts, memory projection, a critic, training or new character names.

Two scene families contain no targeted references. Keep them byte-identical as
negative controls and require identical seeded outputs. Reuse the same fixed
CSFT adapter and 48-request development panel; show the 36 changed requests
separately from the 12 unchanged controls. Freeze previous per-reply scores before
generation and reuse them whenever the same prompt ID produces identical text.
This addresses the review drift seen in the previous experiment.

No reviewed paper directly validates this exact intervention on Ministral 3B in
English and French. This run tests a local hypothesis, not a literature-backed
guarantee. A positive result still needs new scene families and independent/user
review before changing the chat default.
