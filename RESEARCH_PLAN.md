# Persona-consistent dialogue with token-aware preference training

Working project specification, September 22, 2026.
Target role: ML / LLM engineer. Status: proposed implementation and experiments;
no new model, dataset, or result described below has been produced yet.
Backbone preference: Mistral, selected by the user for relevance to roles in France.

Build a useful roleplay chatbot and investigate whether token-aware preference
training reduces character contradictions without degrading fluency or diversity.
The deliverable includes a data pipeline, training code, reproducible comparisons,
an evaluation report, and a multi-turn demo.

## Research basis

The main recent method is [Se-DPO: Self-Evolving Token Credit for Direct Preference
Optimization](https://arxiv.org/html/2608.09568v1), published August 10, 2026. It
adapts token credit using evolving policy/reference signals and a small calibration
network. Implement the paper's objective and compare it with ordinary DPO. This
project changes the model and domain, so describe it as an adaptation and controlled
evaluation, not a reproduction of the published benchmark scores.

[PersonaForge](https://aclanthology.org/2026.findings-acl.386/) (July 2026) motivates
evaluation of personality consistency and conversational drift. Use its evaluation
concerns to inform the rubric; do not claim to reproduce its full architecture.

[DeepSeek-V4.1-Flash](https://arxiv.org/html/2609.19969v1#S5) (September 2026)
emphasizes systematic task synthesis and data quality in post-training. Apply
that principle through explicit character constraints and auditable preference
examples. This is methodological inspiration, not a DeepSeek architecture replica.

Supporting foundations: [DPO](https://arxiv.org/abs/2305.18290) and
[QLoRA](https://arxiv.org/abs/2305.14314).
The selected backbone is described in [Ministral 3](https://arxiv.org/abs/2601.08584)
(January 2026).

## Existing experiment

First retain and evaluate the corrected T4 decoder as specified in
[BASELINE.md](BASELINE.md). Its conventional architecture is the from-scratch
starting point; no architecture replacement or new training run is required to
establish it. Complete its held-out generation evaluation before attributing the
observed failure to a specific mechanism or reporting an improvement.

The user reported a completed 16.3M-parameter decoder run under
`/content/drive/MyDrive/RolePlayChatBot/runs/t4-batch8-v1/`. Its validation
perplexities were 100.6102, 79.2765, and 75.6797 across three epochs. The supplied
generation showed severe repetition and weak grammatical coherence. These are
user-reported Colab results; the trained checkpoint has not been inspected locally.

Preserve that run as project history and an educational reference. It is not the
causal control for preference-training claims: it differs in model, tokenizer,
initialization, and training corpus. Do not compare perplexity directly across
the GPT-2 tokenizer and the new model's tokenizer.

## Model and runtime

Use [mistralai/Ministral-3-3B-Instruct-2512-BF16](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-BF16)
as the initial backbone. It is an instruction-tuned pretrained checkpoint; the
project adds persona-specific SFT and preference training. Pin its exact revision
when implementation starts. Use its native processor/tokenizer and conversation
template consistently in training and inference, including explicit system persona
and scene information. Do not carry over Qwen-specific thinking switches.

The official model card lists a 3.4B language model and a 0.4B vision encoder under
Apache 2.0. This project initially uses text inputs and adapts language-model
modules only; keep the vision components frozen and verify their memory overhead.
Use the supported model loader rather than assuming this multimodal checkpoint
has exactly the same loading path as a text-only causal LM.

Use the BF16 source checkpoint for on-load 4-bit quantization. The source weight
format does not require BF16 training compute: use FP16 compute on T4. The default
Instruct repository distributes FP8 weights, so it must not be substituted without
checking quantization/training compatibility. Published inference memory estimates
do not establish the training-memory budget.

Initial feasibility settings, subject to profiling:

- Frozen 4-bit backbone and LoRA adapters; initially rank 16 or 32.
- Maximum total sequence length 1,024 for smoke tests, then 2,048 if feasible.
- Micro-batch 1; accumulate to a fixed effective batch for all compared methods.
- FP16 compute on T4; device-appropriate BF16 only where natively supported.
- Gradient checkpointing and checkpoint/resume support.
- Identical trainable modules, adapter rank, precision, data, and generation
  settings across preference-method comparisons.

Do not reuse the custom decoder's optimizer settings or tokenizer files blindly.
Run an equal-size validation-only learning-rate search for each method. Record the
search budget as well as the final configuration.

For the frozen SFT reference, cache per-token selected-token log probabilities
and reference entropy with exact token IDs and masks. Se-DPO needs more than the
sequence-level reference sums sufficient for standard DPO. Compute entropy in
chunks to avoid retaining full-vocabulary logits for entire datasets. Key caches
by model/adapter revision, tokenizer, template, truncation settings, and data hash.

## Data pipeline

Start with a small manually audited pilot: approximately 200 preference pairs and
20 evaluation scenarios. Scale only after training and scoring are verified.
Initial full-study targets are 5,000-10,000 SFT examples and 2,000-5,000 preference
pairs, with the final counts determined by data quality and compute measurements.
These counts are scope proposals, not sufficient-data guarantees.

Each record includes character ID, scene ID, provenance, an explicit persona,
conversation history, and the response or preference pair. Track dataset sources,
versions, licenses, construction method, and rejected-example counts.

Construct chosen/rejected pairs around specific failures: established fact
contradictions, inappropriate character voice, scene discontinuity, user-agency
violations, and repetition. Include natural model failures as well as controlled
edits. Avoid allowing every rejected example to be identifiable by length,
punctuation, a stock phrase, or a single synthetic editing pattern.

Split by character and scene before augmentation; remove near-duplicates across
splits. Author persona facts independently or extract them only from available
context, never from the target reply or future test turns. Keep annotation rules
and a reviewed sample of disagreements.

Bluemoon provides narrative material, but it does not automatically supply reliable
explicit persona labels or preference pairs. Audit a subset before reusing it.
Do not simply declare every historical reply a preferred answer.

## Experiments

| Experiment | Initialization | Purpose |
| --- | --- | --- |
| A: pretrained model | Pinned Ministral Instruct checkpoint | Measure the starting product quality |
| B: persona SFT | Same pinned checkpoint | Test domain-specific supervised adaptation |
| C: SFT + DPO | Exact B checkpoint | Standard preference-training control |
| D: SFT + Se-DPO | Exact B checkpoint | Test the recent method |
| E: static-credit ablation | Exact B checkpoint | Test whether evolving credit matters |

Use identical preference pairs and token budgets for C/D/E and report actual GPU
time and memory. Run three preference-training seeds for the main comparison when
the pilot budget permits; disclose fewer seeds if compute is limited. Training an
additional seed must not silently change the SFT checkpoint or dataset.

Keep retrieval, external memory, and automated response rewriting disabled for
the core comparison. Afterward, those may be separate product features with their
own evaluation. Improvements from them must not be attributed to Se-DPO.

## Evaluation

Lock a held-out set of roughly 200 scenarios spanning unseen characters and scenes.
Include single-turn responses and 5-10-turn scripted conversations. Track exactly
which facts are still inside the context window; forgetting truncated information
is a different failure from contradicting visible information.

Include French and English scenarios and report results separately by language.
Keep translations and paraphrases of the same scenario in one split. Have a fluent
reviewer check French examples and judgements; model branding alone does not
demonstrate French-language product quality. Treat bilingual evaluation as a
concrete product capability, not evidence of a hiring advantage.

Primary endpoint: violation rate of explicit, applicable character/scene facts.
Report its definition, denominator, severity rubric, and uncertainty. Verify
automated labels on a blinded, manually reviewed subset. For free-form dialogue,
do not present an automated judge as objective ground truth.

Secondary endpoints: blinded preference win/tie/loss rates, fluency and relevance,
repetition, output length, and diversity. Also report inference latency, generation
throughput, peak allocated/reserved VRAM, and total training GPU time. Compare
generated lengths and resample output order to check judge position/length bias.

Cluster bootstrap intervals by scenario or character rather than treating every
turn as independent. Tune only on validation; run the locked test after method and
checkpoint selection. Report failures and negative findings alongside successes.

## Implementation and acceptance gates

1. Establish pretrained inference, persona formatting, model revision manifests,
   and the evaluation runner. Confirm that the selected backbone can produce
   coherent replies before adaptation.
2. Build and audit the pilot datasets. Add split-leakage and schema validation.
3. Add QLoRA SFT and standard DPO, including adapter-aware reference handling,
   useful logs, and resumable checkpoints. Profile both on the actual Colab GPU.
4. Implement Se-DPO from the paper, with masking, gradient-flow, reference-freezing,
   and numerical checks. Verify that uniform credit recovers standard DPO and that
   padding does not change the objective. Compare with independent loss calculations.
5. Run the matched experiments and the static-credit ablation; generate the report
   from recorded artifacts rather than manually entered summary numbers.
6. Package the selected model in a simple persona-editor/chat demo with visible
   model/version information and reproducible example conversations.

Suggested additions live under a new `posttraining/` package, with versioned data
manifests, configs, an evaluation CLI, and a separate Colab runner. Keep the existing
custom-decoder path usable. Add a model card, dataset card, and limitations section.

## Compute decision

A Ministral 3 3B QLoRA pilot on T4 is the initial engineering target, not a verified
memory or speed claim. Account for the 3.4B language model and 0.4B vision component.
Measure peak memory and examples/second on representative maximum-length batches,
including both chosen/rejected sequences and reference statistics. If T4 is too
constrained, profile an available L4 or A100 before reducing experiment quality.
Keep the 3B backbone for the first complete study; consider Ministral 3 8B only
after its cost is justified by the smaller model's results.

Consider paid Colab when the pilot demonstrates that repeated experiments or
longer contexts need more runtime/compute. An available L4 or A100 can be used to
repeat the same controlled study faster or expand it after profiling.
[Colab's FAQ](https://research.google.com/colaboratory/faq.html) states that paid
plans also have variable hardware availability; Pro does not guarantee an A100.

Estimate full-run hours from the pilot before committing paid compute, and account
for dataset generation and judging costs separately. No subscription purchase,
cloud allocation, external teacher call, or full training launch is part of this
planning step.

## Portfolio outcome

Publish a useful multi-turn demo, an auditable dataset pipeline, paper-linked
implementation notes, matched baseline/ablation results, and a reproducible report.
The central claim is a measured answer to the research question, whether positive
or negative. Do not claim new-algorithm novelty or reproduce the paper's reported
gains as this project's own results.

Possible final resume structure, populated only after experiments:
"Implemented and evaluated token-aware preference optimization for persona-based
dialogue; compared SFT, DPO, and Se-DPO on held-out characters, measuring [result]
and [quality/cost tradeoff] with a reproducible single-GPU training pipeline."
