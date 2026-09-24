# Research reading notes: roleplay on a single AMD GPU

Reviewed September 23, 2026. A targeted review, not an exhaustive survey.
Sources below are original papers or official documentation. Read the relevant
methods, experiment settings, and limitations; no author code or results were
reproduced. Preprint claims remain provisional. The application decisions are
our hypotheses and are operationalized in [RESEARCH_PLAN.md](RESEARCH_PLAN.md).

## 1. Data selection: Edu-QuRating

[Edu-QuRating: Multi-Dimensional Educational Data Curation with Distilled Pairwise
Judgements](https://arxiv.org/abs/2609.09425), September 8, 2026; revised September
10. Preprint. Reviewed the v1 full text and v2 abstract/version record.

The authors distill pairwise educational judgments into scorers for multiple
quality dimensions, then use them for pretraining selection and GRPO rewards.
Their pretraining evidence is matched single-run comparisons, with uneven gains
across tasks. Agreement with a model judge is not equivalent to human ground truth.

**Our use:** independently audit relevance, continuity, grammar, and voice rather
than selecting solely for fluent prose. Begin with manual audits; their corpus
scale and scorer-training pipeline are not required for our pilot. Educational
rubrics and reported improvements do not transfer automatically to roleplay.

## 2. Preference optimization: Se-DPO

[Se-DPO: Self-Evolving Token Credit for Direct Preference
Optimization](https://arxiv.org/html/2608.09568v1), August 10, 2026. Reviewed as a
preprint, including objective, algorithm, and appendices.

It learns token credits using implicit rewards and reference entropy. Tests include
small-model LoRA, but assistant benchmarks do not establish multi-turn persona
fidelity. The derivation approximates partition-function terms; mean normalization
does not itself guarantee a fixed maximum/minimum credit ratio.

**Our use:** optional after DPO. Credit is not a factual-importance label. Check
output length and quality; measure runtime locally.

## 3. Evaluation: anonymous roleplay

[Rethinking Role-Playing Evaluation: Anonymous Benchmarking and A Systematic Study
of Personality Effects](https://aclanthology.org/2026.sigdial-1.15/), SIGDIAL,
August 2026. Reviewed full-paper evaluation setup and findings.

Character-name anonymization exposes reliance on pretrained character knowledge;
personality descriptions can help. Its basic anonymization replaces the target
character name while some related names remain. That does not fully eliminate
recognition, and the evaluated languages do not establish French performance.

**Our use:** paired name changes across all entities, invented character cards,
and explicit facts that differ from familiar canon. This stronger manipulation
is our extension. Keep related variants together when splitting and computing
uncertainty; evaluate French separately.

## 4. Pretraining: domain repetition

[Scaling Domain Data Repetition in LLM
Pretraining](https://arxiv.org/pdf/2608.14071), August 14, 2026. Preprint.
Reviewed setup, repetition analyses, and theory scope.

The study varies repeated domain data mixed with fresh web data, holding token
budgets fixed within each model size and scaling tokens with parameters across
sizes. Useful repetition differs by domain; results do not support one universal
epoch count. Its theoretical account uses a simplified regression setting.

**Our use:** measure unique examples and token presentations separately; tune
mixtures/repetition on validation if we revisit scratch training. Their general
pretraining setup differs from our reply-masked overlapping dialogue windows.
The paper neither diagnoses our gibberish nor justifies another identical run.

## 5. Memory and roleplay: PsyMem

[PsyMem: Fine-grained Psychological Alignment and Explicit Memory Control for
Advanced Role-Playing LLMs](https://aclanthology.org/2026.tacl-1.24/), TACL,
April 2026. Reviewed memory construction, two-stage training, and ablations.

PsyMem combines psychological profiles with memory-conditioned training, including
a Qwen2.5-7B model. Its memory ablation improves memory adherence while showing
some tradeoff in attribute alignment. Simply inserting retrieved text is not the
same intervention as training a model to use it. Some evaluation relies on model
judges, and novel-derived characters differ from user-created scenes.

**Our use:** test explicit grounded state first and then memory-conditioned SFT.
A small fact ledger is an economical starting hypothesis; we are not reproducing
its graph construction, 26 psychological indicators, or published scores.

## 6. Distillation: OPCD

[On-Policy Context Distillation for Language
Models](https://arxiv.org/html/2602.12275v1), February 12, 2026. Preprint.
Reviewed objective, teacher configurations, and task settings.

The student generates its own trajectories and is trained toward a teacher that
receives additional context using reverse KL. Experiments cover experiential
knowledge and system-prompt distillation; the implementation approximates KL using
student top-k tokens. The teacher may share a model family or underlying weights,
but rollout and teacher-scoring work remains. It is not ordinary cached-output SFT.

**Our use:** a conditional alternative when extra scaffolding measurably helps.
Roleplay transfer is unproven. Dynamic facts still belong in runtime context;
test-persona facts must not be distilled into weights. Compare quality/diversity
and end-to-end cost against simpler SFT before adopting it.

## 7. RL for role awareness: VeriRole

[VeriRole: Verifiable Role-Awareness through Hint-Guided Reinforcement
Learning](https://proceedings.iclr.cc/paper_files/paper/2026/file/12df08e9280061b4877f97598c644c3e-Paper-Conference.pdf),
ICLR 2026. Reviewed the published paper's hint mechanism and reward design;
used the proceedings PDF because OpenReview access was challenged.

The method extracts profile/history hints and uses role-awareness rewards with
GRPO. Some response checks use model evaluation; the entire creative output is
not made objectively verifiable. Main experimental backbones include 14B/32B
models, so the results do not establish feasibility or gains on our 3B AMD setup.

**Our use:** separate factual constraints from subjective quality and test reward
exploits before RL. Correct copied hints can coexist with a bad final answer.
Retain independent dialogue evaluation and defer online RL until the baseline,
data, reward audit, and rollout memory budget pass.

## Runtime and scope decisions

The [official Ministral BF16 model card](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-BF16)
supports continuing the existing Mistral choice, but an inference fit estimate is
not an adapter-training measurement. The current
[bitsandbytes installation guide](https://huggingface.co/docs/bitsandbytes/main/en/installation)
lists AMD ROCm support; actual WSL compatibility needs a local preflight.

The earlier plan's [PersonaForge](https://aclanthology.org/2026.findings-acl.386/)
and [DeepSeek-V4.1-Flash](https://arxiv.org/abs/2609.19969) remain background
references. Their records were checked, but they are not implementation
dependencies or full reproductions in this review. Do not add psychological
subsystems, a giant-model training recipe, or KV-cache compression just because
they are recent.

The practical priority is evaluation and grounded state, followed by curated SFT
and ordinary [DPO](https://arxiv.org/abs/2305.18290). Advanced optimization must
earn its place through a matched improvement on the failures we actually observe.
