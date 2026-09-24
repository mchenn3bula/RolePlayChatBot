# RolePlayChatBot handoff

## Latest work: response-rule refinement and fresh scenes completed

`RESPONSE_REFINEMENT_RESULTS.md` records the frozen comparison of natural
references versus one added first-person/completeness/premise-correction rule.
Same fixed CSFT epoch-02, decoding and facts; no training or default changes.
Familiar 48 requests: semantic passes 41 -> 39, omissions 7 -> 8, role errors
0 -> 1, third-person self-description 6 -> 2. Fresh 36 authored EN/FR requests:
passes 10 -> 14, omissions 21 -> 17, role errors 5 -> 5, third-person self 13 -> 12.
Fresh language gates pass, familiar gates fail; overall candidate is rejected.
Do not pool panels or claim the earlier zero role errors generalized.

All 48 familiar baseline outputs reproduced exactly; 168 native serving inputs
verified. 72 historical ratings reused, 96 new; review frozen before unblinding.
Single non-independent assistant judgments. Private source dialogue unread.
81 tests (80 passed, one native-only skip); Ruff/launcher checks passed;
9.22 GiB peak reservation. No active GPU work. Gallery:
`reports/ministral-response-refinement-v1/comparison.html`; compact aggregate:
`results/ministral-response-refinement-v1.json`. Preserve protocol, sources,
fixtures, anchors, outputs and review hashes. Visible launcher:
`evaluate_response_refinement.ps1`.

Next hypothesis, not started: isolate stale-state failures using a controlled
obsolete-history ablation on new authored EN/FR handoffs, preserving current
state. This is diagnostic, not a production history-removal proposal. The generic
response rule did not fix returned-object attribution. Coverage supervision is
a possible separately scoped later experiment; no automatic training follows.
Older next-step notes below are historical and superseded by this section.

## Latest work: papers reviewed; natural-reference experiment completed

User requested literature research before testing explicit character references
in natural sentences. Research is `NATURAL_REFERENCE_RESEARCH.md` (SPASM 2026,
RoleMRC 2025, formatting sensitivity, semantic-evaluation counterevidence and
anonymous role evaluation). Protocol frozen before generation:
`NATURAL_REFERENCE_PROTOCOL.md`. Same CSFT epoch-02; only reversible scene-value
reference/grammar edits. 36 requests changed, 12 unchanged negative controls.
48/48 prose outputs reproduced historical tokens; 12/12 negative controls matched.
68 historical per-item/text scores reused unchanged; 28 new replies rated before
unblinding. Single non-independent assistant review, familiar development panel.

Results: role errors 13 -> 0, contradictions 15 -> 0, omissions 4 -> 7, all-required-
constraint passes 30 -> 41/48 (changed subset 18 -> 29/36). Natural 18 wins, prose
four, 26 ties. EN omissions 1 -> 3, FR 3 -> 4: both prespecified gates FAIL, despite
strong overall improvement. Natural references remain experimental; no default or
training changes. Third-person role recitation and repeated descriptions remain.
Do not claim perfect role-play or general zero-error behavior. Preserve all anchors
and annotations. Private source dialogue remains unread.

`NATURAL_REFERENCE_RESULTS.md`; gallery
`reports/ministral-natural-reference-v1/comparison.html`; aggregate
`results/ministral-natural-reference-v1.json`. Code: `posttraining/natural_reference.py`
and `natural_reference_edits.json`. Visible launcher `evaluate_natural_reference.ps1`.
79 tests OK (one native-only skip), all native serving inputs verified, Ruff and
shell syntax checks passed; 9.22 GiB peak. No active GPU work. Suggested next work,
NOT started: fresh authored EN/FR scene validation plus coverage/conversational-
voice review. No advanced method or further training automatically follows.

## Latest work: controlled state-format experiment completed

User authorized fixing role confusion through a controlled state-format experiment.
`STATE_FORMAT_PROTOCOL.md` was frozen before generation; results are in
`STATE_FORMAT_RESULTS.md`. Same CSFT epoch-02, same 48 development requests,
history, seeds and decoding; only the scene fact value changes from prose to
entity/property assignment strings. No training or model/prompt default change.
Fresh prose reproduced all 48 historical outputs token-for-token. Frozen A/B
assistant review: role errors 13 -> 6, contradictions 15 -> 8, omissions 4 -> 17,
all constraints satisfied 30 -> 24. Entity 13 wins / prose 11 / 24 ties.
Both EN and FR fail the prespecified no-omission-regression gate. Do not promote.
All judgments are non-independent assistant ratings; this familiar panel is not
a fresh test. Borderline re-review decisions differ from prior ratings; old reviews
remain intact. Private source conversations were not read.

Gallery: `reports/ministral-state-format-v1/comparison.html`; aggregate:
`results/ministral-state-format-v1.json`. Frozen inputs, coverage audit, source and
adapter fingerprints, A/B key and annotations stay together in the report folder.
Runner `posttraining/state_format.py`; visible launcher `evaluate_state_format.ps1`.
77-test suite OK (one native-only skip), native encodings verified, 9.22 GiB peak
inference reservation. No GPU work remains. Suggested next experiment, NOT started:
keep natural sentence structure/order and replace only ambiguous second-person
references with explicit character references; validate any promising result on
new authored scenes before adoption. Advanced optimization remains deferred.

## Latest work: strengthened bilingual preferences and matched continued SFT

User authorized strengthening bilingual preferences and the continued-SFT control.
`BILINGUAL_CONTROL_PROTOCOL.md` is frozen before mining/training/new evaluation.
The data at `data/ministral-preferences-v2/` has 80 EN/FR training pairs from 40
scene families, 24 validation pairs from 12 and 16 evaluation prompts from eight.
One alias per D1 template is retained; 24 new training scenes add substantive
ownership/update/relationship/unknown/speaker/agency contrasts. Of 64 S1 samples
reviewed on new train/validation prompts, 32 clear mistakes were retained as
negatives (24 train, 8 validation). All labels remain assistant-curated.
Review artifact: `reports/ministral-preferences-v2-curation/review.html`.

Both new controls initialize from immutable S1 epoch-02: `ministral-v2-csft-v1`
and `ministral-v2-dpo-v1`. They match chosen exposure, order, LR and 20 optimizer
updates, not compute. `train_bilingual_pair.ps1` runs them sequentially with
visible logs; completed outputs cannot be overwritten. Check their `complete.json`
or `failure.json` for execution status. Final epoch-02 is prespecified for both.
New three-arm evaluation is `reports/ministral-bilingual-control-v2/`, with 48
identical-input requests using authored history. See `BILINGUAL_CONTROL_RESULTS.md`
for the completed comparison when available. No model default or advanced method
is automatically promoted. Old protocols, annotations and checkpoints stay intact.

Both controls and evaluation completed. CSFT: 96.3 seconds total / 41.4 optimization;
DPO: 133.3 / 81.9; 20 updates / 160 chosen presentations each, no overflows,
9.17 GiB peak training reservation. New 48-request randomized three-arm review:
all constraints satisfied S1 25, CSFT 29, DPO 24. CSFT beats DPO 7 / 1 with 40
ties. Contradictory replies: S1 15, CSFT 15, DPO 20. CSFT improves French coverage
but still regresses one English contradiction, so no default promotion. All
annotations are single-assistant judgments. Results file:
`results/ministral-bilingual-control-v2.json`; gallery:
`reports/ministral-bilingual-control-v2/comparison.html`. Main residual failure is
copying second-person state prose into user-role replies. Prefer investigating
unambiguous entity-based state wording and independent review over advanced loss.

## Latest decision: conditional advanced-method request

The user requested one advanced method only if justified. Assessment complete:
advanced training is deferred on present evidence. See `ADVANCED_METHOD_DECISION.md`.
Se-DPO remains the sole candidate; token weighting has not been shown to cause
the observed failures. No new training, adapter or default change occurred.
Prioritize more varied, independently reviewed bilingual preference data and a
matched continued-SFT control. This is an evidence-based decision, not an awaiting-
permission state. Do not automatically launch an advanced run from the old plan.

## Latest instruction: standard DPO comparison

The user now permits reading generated evaluation replies to compare standard DPO
with SFT. This supersedes older no-output-review language below **for this
comparison only**. Private source conversations remain unread. Synthetic authored
fixtures may be inspected. Ratings must be labeled as a single non-independent
AI review, never human preferences. Logs remain content-free.

`ministral-d1-standard-v1` completed 40 updates in 234.8 seconds, with no overflow
retries and 9.17 GiB peak reservation. It starts from the immutable S1 epoch-02
adapter and uses that exact adapter as the frozen reference. The selected DPO
checkpoint is `checkpoints/ministral-d1-standard-v1/epoch-02`. This is original
sigmoid DPO, beta 0.1, LR 5e-6, 160 train / 40 held-out synthetic preference pairs
with 16 / 4 distinct template families. They are not human-labeled examples.

See `DPO_COMPARISON.md` and the frozen `DPO_COMPARISON_PROTOCOL.md` for results,
commands and limits. Matched outputs and review artifacts are under
`reports/ministral-sft-vs-dpo-v1/`. Preserve both SFT and DPO checkpoint directories.

Comparison complete: identical native inputs for 108 requests; fresh SFT reproduced
all historical outputs. Randomized, frozen AI review: DPO 27 wins / 67 ties / 14
losses. Required-fact contradictions stay 27/174; omissions fall 61 to 49. French
contradictions increase 18 to 21, so do not promote DPO as a reliable bilingual
upgrade. SFT validation NLL is essentially unchanged (2.688166 versus 2.687299).
Results: `results/ministral-d1-vs-sft-v1.json`. Full suite: 71 tests, one native-only
skip; all seven LoRA/native tests pass separately. No chat default was changed.

## September 24: whole-conversation LoRA S1 authorized

The user requested LoRA training with curated examples, selected as random
complete conversations. The no-reading/no-generated-text-review instruction
remains in force. Curation is structural only; see `LORA_TRAINING.md`.
Selected complete available source threads: 116 train / 17 validation, native
length <=2,048, no truncation. Historical splits were verified by hashes.
Exact dialogue files under `data/ministral-s1-whole-v1/` are user-only: do not
open them. No semantic/language review or invented persona/state labels.

A separate `$HOME/.venvs/roleplay-lora` environment contains pinned PEFT 0.20.0
and the existing ROCm/Transformers stack. The rank-16 q/k/v/o adapter trains
9,371,648 FP32 parameters over an FP16 frozen backbone, with checkpointing,
reply-only native-EOS loss, micro-batch 1, accumulation 8, two epochs and a
six-hour cap. Source: `posttraining/lora_train.py`; config:
`configs/ministral_s1_lora.json`; launcher: `train_lora.ps1`.

Synthetic 1,024/2,048-token backward/optimizer and exact checkpoint-resume tests
passed. After a real-data allocator-reservation guard stop (before any updates),
unused cache is released between variable-length micro-batches. Final synthetic
peak reservation: 11.41 GiB. Preserve the failed run directory. Fresh training:
`checkpoints/ministral-s1-whole-lora-v1-r2/`. Check its technical `complete.json`
or `failure.json` for status; do not inspect dialogue. Logs print metadata only.
Adapter inference is explicit via `ministral.ps1 -Adapter <checkpoint directory>`;
it keeps the P1 state wrapper and validates model revision plus adapter hashes.

Training completed: `ministral-s1-whole-lora-v1-r2` finished 30 updates / two epochs
in 233.9 seconds, no FP16 overflow retries, peak reservation 11.41 GiB. Original
held-out reply NLL 2.88195; epoch 1 2.69375; epoch 2 2.68817. Selected adapter:
`checkpoints/ministral-s1-whole-lora-v1-r2/epoch-02/`. These are technical loss
measurements, not assistant judgments of dialogue or generated quality.
Final full-suite log: `reports/lora-tests-cache-fix.log` (63 tests, one native-only
skip); all seven LoRA tests separately passed in the LoRA environment.

Adapter reload and native inference passed. The user-only gallery is
`reports/ministral-s1-whole-lora-v1-user-panel/replies.html`, with all 108 replies
saved (54 EN / 54 FR). Do not read the gallery or raw generations. The content-free
summary and `results/ministral-s1-whole-lora-v1.json` are safe technical metadata.
Use `ministral.ps1 -Mode Chat -Profile P1 -Language en -Adapter
.\checkpoints\ministral-s1-whole-lora-v1-r2\epoch-02` (one command) to chat with S1.

## September 24 update: authored persona/scene state and user-only review

The user's latest instruction prohibits the coding assistant from reading,
quoting, scoring or analyzing dialogue/generated replies. Do not open transcripts,
galleries, historical example replies or AI-review artifacts. Supply paths instead.
Use authored benign fixtures and technical execution metadata for verification.
This supersedes earlier instructions to judge generations or collect AI ratings.

P1 is implemented in `posttraining/state.py`, `chat.py` and `evaluate.py` with
`configs/ministral_p1.json`. It adds a validated, revisioned authored ledger;
immutable persona facts, stable entity references, known/unknown/claim status,
source turns, point-in-time scene events and whole-exchange context trimming.
No model-derived state extraction or training occurs. P0 stays separately selectable.
The Windows launcher now defaults to P1. Chat writes private session files under
ignored `reports/`; `/state` shows the editable state path, `/reload` validates a
higher revision, `/reset` restarts history and the event clock. Replies go only to
`replies.html` and `generations.jsonl`. Progress/summary/preflight are technical only.
The old AI-review entry point now only locates the saved user gallery.

Read `PERSONA_SCENE_STATE.md` for commands/schema/limitations. State must be updated
explicitly; ordinary dialogue and model replies do not update the ledger. Initial
facts/entity names are visible from the first turn, so timed information belongs
in events. Generated quality remains unassessed pending the user's own review.

P1 technical run completed: `reports/ministral-p1-v1-20260924/` contains
`replies.html` and `generations.jsonl` for the user. Do not read either artifact.
All 108 replies were saved (54 EN / 54 FR), with finite logits and frozen weights;
peak reserved VRAM was 9.17 GiB, aggregate generation 29.56 tokens/sec, and total
elapsed time 213 seconds. No history trimming was needed in this panel; synthetic
tests exercise trimming and over-budget rejection. The full 56-test suite, Ruff,
PowerShell parser and Bash syntax checks passed. Content-free results are in
`results/ministral-p1-v1.json`. No generated quality assessment was performed.

## Ministral P0 inference — September 24, 2026

The user requested local Ministral 3B inference and English/French evaluation.
Completed under WSL2 in separate `$HOME/.venvs/roleplay-ministral`, preserving
the baseline environment. Pinned HF revision is
`b6d637bef2393152b3da2b2fde72eecdee30557e` of the selected BF16 source repository;
loaded the full 3,849,090,048-parameter model in FP16, with native Mistral Common
formatting and no adapters, quantization, or offloading. Dependencies are locked.

`MINISTRAL_EVALUATION.md` is the current report. Twenty authored development
scenarios (10 paired EN/FR families) x three seeds, including four five-turn
scenarios, produced 108 replies. Throughput: 35.22 tokens/sec; peak reserved
9.17 GiB; all EOS, none empty, no repeated word 4-grams. AI review of every reply
found continuing factual and scene drift: first-turn acceptable coherence AND
relevance 26/30 English, 13/30 French. Combined 65% fails the plan's 80% gate.
Scores are non-blinded assistant judgments; human review remains blank.

Three failure-prompt probes had closely agreeing FP16 default/math and FP32 CPU
first-token logits. BF16 forwards were finite; BF16 training remains untested.
These limited checks do not prove all inference correct or diagnose quality.
The greeting-only numerical warmup answered in Spanish; it is not a quality pass.

Evidence: `reports/ministral-p0-v1-20260924-r2/`; public aggregate:
`results/ministral-p0-v1.json`. The initial directory without `-r2` records an
offline snapshot-completeness error before generation, fixed by matching the
download allowlist. Completed run source snapshots preserve the evaluated code.
The original custom-model checkpoints and datasets remain untouched.

Commands: `./ministral.ps1 -Mode Chat -Language en` (or `fr`), and
`./ministral.ps1 -Mode Evaluate -RunName <unique-name>`. Evaluation logs stream
to the terminal and `reports/logs/`. Chat supports `/reset` and `/quit`.
All 47 offline tests passed in the old ROCm environment; six harness tests also
passed in the new environment. Next: diagnose P0's failed quality gate using
separately named prompt/authored-state controls, especially French, before SFT.
This work is local; no new push was performed.

## Research revision — September 23, 2026

The user requested a review of recent NLP/ML/pretraining/post-training/RL papers
and an updated game plan. `LITERATURE_REVIEW.md` reviews seven 2026 papers with
primary-source links, limitations, and project-specific implications.
`RESEARCH_PLAN.md` now prioritizes frozen Ministral inference, explicit persona
and scene-state controls, curated SFT, then ordinary DPO. Se-DPO, on-policy
context distillation, and verifiable RL are conditional branches, not a required
stack. The completed scratch model stays intact; its short-prompt failure is
still undiagnosed. A bounded diagnostic session can precede further scratch work.

Next implementation milestone: a P0/P1/P2 generation-evaluation harness and
Ministral inference/memory preflight on this PC. Use separate runtime dependencies,
native templates, exact revision manifests, and authored versus extracted memory
controls. The plan specifies bilingual evaluation, data audit, acceptance gates,
an initial 18 GiB reservation target with device-wide headroom, and six-hour
resumable runs after profiling. These are proposals, not measured Ministral
performance. This revision downloaded no weights and launched no training.

The source project was pushed to `mchenn3bula/RolePlayChatBot` main at `3e10818`
before this research revision. Research updates are local until separately pushed.
The older transfer-era sections below are historical where superseded.

## Completed run and generation evaluation — September 23, 2026

`modern-rope-swiglu-v1` completed all four epochs: validation perplexities
52.96, 43.03, 38.68, 38.10; 22,210 successful updates, six FP16 overflow skips.
The final epoch is best. Keep its complete checkpoint directory intact.

The user requested coherence, relevance, and repetition evaluation. The fixed
50-validation-prompt x 3-seed panel is complete, using the exact `BASELINE.md`
settings and FP32 inference. See `GENERATION_EVALUATION.md` for findings/evidence.
All 150 replies were nonempty; 104 stopped on EOS, 46 hit the 128-token limit,
and the mean repeated word 4-gram fraction was 0.41%. Qualitative AI review of
the first 20 seed-42 replies found weak relevance and coherence (means 2.05/5
and 1.95/5); only three scored >=3 on both. Character substitutions and scene
drift are common. These are assistant ratings, not human annotations.

The original inn-door prompt generated gibberish in all three seeds, including
a subword loop that word-level repetition misses. Prompt/EOS round-trip and
the chat API were verified; CPU/GPU and math/default GPU first-token logits
agree closely. Root cause is still unproven. The old trained T4 checkpoint is
still absent, preventing a matched repetition comparison. The test split was
not used. Next useful work is diagnosing short-prompt/context tracking failures
before committing another long training run. Historical preflight notes below
are superseded by these completed results.

## Current selected experiment: RoPE + SwiGLU

The user adopted the proposed RoPE, SwiGLU, and reply-position-only vocabulary
projection, and requested visible training logs plus preflight before training.
Implemented as architecture version 3, named `modern-rope-swiglu-v1`, with
`configs/overnight_rope_swiglu.json`. Use `train_amd.ps1 -Profile Modern`.
See `MODERN_BASELINE.md` for the complete recipe and measured evidence.

The model has 123,551,232 parameters: 12 layers, width 768, 12 full attention
heads, SwiGLU width 2,048, RoPE theta 10,000, tied GPT-2 embeddings, maximum
length 1,024. The existing reply-only objective, EOS boundaries, tokenizer,
prepared data, and thread splits are unchanged. Full-conversation pretraining
was only proposed for separate investigation and has not been implemented.

All 41 tests passed on the ROCm GPU runtime, including FP16 causal/padding
checks, selected/full-projection loss and gradient equivalence, exact resume
with dropout, time-budget recovery, perplexity scoring, and generation panels.
The actual full-size smoke run passed eight optimizer updates with no skipped
updates, finite gradients/weights/optimizer state, and verified resume/reload.
Smoke peak reservation was 9.37 GiB. The subsequent 2,048-example benchmark
measured 40.19 examples/sec, 7.54 GiB allocated / 10.60 GiB reserved, 64 successful
updates and zero skipped updates. Four epochs project to 4.91 training-only hours
plus validation/checkpoint overhead. The full run subsequently completed;
see the September 23 results above.

The source now supports both versions; old v2 model defaults and checkpoint
keys remain compatible. Same-seed v2 initialization, forward output with dropout,
and gradients matched the pre-edit source exactly. Original v2 configs are
unchanged. Saved model config and variant settings guard architecture/resume
compatibility, including RoPE theta.

The launcher prints unbuffered progress to the terminal and tees stdout/stderr
to unique `reports/logs/` files. Training reports every 25 successful updates,
including loss, LR, examples/sec, elapsed time, epoch ETA, skipped updates, and
VRAM reservation; validation and checkpoint writes also print progress.
Keep Windows awake. Use `-Mode Train -RunName modern-rope-swiglu-v1` to start,
or `-Mode Resume` with that same name to continue the six-hour/four-epoch recipe.
The older PC/laptop notes below are retained as history.

## PC continuation — September 22, 2026

The transfer was extracted to `C:\Users\N3BULA\Documents\ChatGPT\RolePlayChatBot`;
all 63 payload checksums matched before edits. The architecture review is complete:
retain the small dense baseline. See `ARCHITECTURE_REVIEW.md`.

AMD preparation is now validated on this Windows 11 PC: Ubuntu 24.04 WSL2,
ROCm 7.2.1 / ROCDXG 1.2.2, Python 3.12, AMD PyTorch 2.9.1. All 26 tests passed,
and a real-data FP16 train/resume/reload smoke run passed on the RX 7900 XTX.
The benchmark measured 135.6 examples/sec with 5.75 GiB peak allocation at
micro-batch 8 / accumulation 4. See `AMD_TRAINING.md` for commands and evidence.

`training.choose_precision` now handles ROCm separately from NVIDIA; ROCm auto
defaults to FP16. The benchmark now uses the same precision policy and
target-token weighting as the trainer. Baseline model/config/data-format source
files still match the laptop snapshot. The packaged data needs the pinned
`datasets==4.4.2`; an initial 3.x selection could not read its List feature schema.

The user subsequently requested a robust overnight baseline and specified about
six hours. The new `overnight-124m-v1` experiment has 124,337,664 parameters:
12 layers, width 768, 12 heads, FF 3,072, max length 1,024, tied GPT-2 embeddings,
pre-RMSNorm, dense causal SDPA. It uses the same tested architecture-version-2
implementation, data, and target-only objective; the historical config is intact.
See `OVERNIGHT_BASELINE.md` and `configs/overnight_124m.json`.

The selected recipe is four planned epochs, FP16, micro-batch 8 / accumulation 4,
LR 1.5e-4, warm-up 500, seed 42, with gradient checkpointing disabled after
profiling. Benchmark throughput was 36.28 examples/sec (5.44 hours projected
training-only for four epochs). The actual 124M model passed eight optimizer
updates plus checkpoint resume/reload with no skipped updates; peak reservation
was 15.28 GiB. All 28 tests passed on ROCm, including deterministic time-budget
stop/resume equivalence. Ruff and launcher syntax checks passed.

`train.py --max-hours` saves at an optimizer boundary and preserves the planned
learning-rate schedule. The `Overnight` profile sets a six-hour per-invocation
soft cap; validation/checkpoint IO may overrun it. Start with
`train_amd.ps1 -Profile Overnight -Mode Train -RunName overnight-124m-v1` and
continue with the same command using `-Mode Resume`. Windows must remain awake.

No full PC training run has been launched. The completed Colab checkpoint is
still absent locally. Next work is launching/evaluating the new experiment when
requested, or retrieving/evaluating the existing Colab run. Mistral work remains
proposed. The original laptop handoff below is preserved as history; its pending
AMD setup notes are superseded.

Snapshot: September 22, 2026. Read this with `BASELINE.md` and `TRANSFER_README.md`.

## User goal and decisions

Build a useful, evaluated roleplay chatbot as an ML/LLM engineering portfolio
project. Preserve a reasonable from-scratch baseline that trains on Colab T4,
then develop improvements supported by papers and controlled experiments.
The user prefers a Mistral pretrained backbone for the later project and already
owns or can access an AMD Radeon RX 7900 XTX (24 GB). The destination operating
system has not yet been established. School projects may also need GPU compute.

## Implemented and verified

- Converted the original notebook workflow into Python scripts. The original
  notebook, PDF, and diagram are retained as historical artifacts.
- Removed future-token leakage from sequence-compressed attention, removed the
  duplicate next-token/MTP loss, repaired optional MoE balance gradients, and
  corrected padding, initialization, target alignment, and sampling issues.
- The baseline in `configs/colab_t4.json` is a standard dense causal decoder,
  architecture version 2, 16,275,968 parameters, 4 layers, width 256, 4 heads,
  feed-forward width 1,024, and 1,024-token maximum sequence length. MoE is off.
- Data: Bluemoon roleplay threads, three preceding messages -> next reply;
  thread-disjoint train/validation/test, context 768 tokens and reply 256 including
  EOS. GPT-2 tokenizer only; model weights start randomly. Split sizes: 177,698 /
  28,018 / 44,508. Preserve the packaged data rather than regenerate it casually.
- Training: mixed precision, exact target-token-weighted accumulation, AdamW,
  cosine schedule, validation selection, and complete checkpoint/resume state.
- Generation evaluation: `evaluate_generation.py`, exact prepared context IDs,
  50 validation prompts selected with seed 42, generation seeds 42/43/44,
  max-new-tokens 128, temperature .8, top-k 50, top-p .9, repetition penalty 1.05.
  Produces a manifest, JSONL generations, automatic metrics, and human review CSV.
- The 21 existing tests passed after the generation refactor. Three new evaluation
  tests also passed after fixing a test fixture's generation length. Ruff checks
  passed for changed Python files. The Colab notebook's cells/embedded Python
  compiled, and archive checksums/source contents were verified.
- Local NVIDIA RTX 4060 Laptop FP16/checkpoint tests passed. Earlier BF16 benchmark:
  36.3 examples/sec on the small baseline; not an AMD or Colab speed measurement.

## Completed training: user-reported Colab result

Run: `/content/drive/MyDrive/RolePlayChatBot/runs/t4-batch8-v1/`.
Settings: three epochs, micro-batch 8, accumulation 4, FP16, peak LR 1.5e-4,
warm-up 500 updates, seed 42. There were 16,657 successful optimizer updates and
5 skipped FP16 updates. Validation perplexities: 100.6102, 79.2765, 75.6797.

The sample prompt was: `The traveler knocks on the inn door. "Is anyone there?"`.
The reply was grammatically weak and repeatedly generated "door". This is a valid
poor-quality baseline result, not proof of one specific remaining architecture bug.
The completed weights are not available on this source PC and are not in the ZIP.
To evaluate on the destination, copy the entire run directory from the user's Drive
to `checkpoints/t4-batch8-v1/`, including `config.json` and `tokenizer/`.

The latest notebook selects this run and sets `RUN_TRAINING = False` by default.
To replicate from scratch, use a new run name and enable training explicitly.
The last Colab error was `torch.cuda.is_available() == False` in setup, before
checkpoint loading. The runtime needed a GPU assignment; it was not a model error.

## Repetition investigation

Working hypotheses: limited learned language capability plus self-reinforcing
generation loops, potentially amplified by decoding and context mismatch.
Do not claim exposure bias alone is proven, or that the corrected baseline still
has the original future-token leakage. No diagnostic probability tracing of the
completed checkpoint has been run.

A deterministic sample of 5,000 training examples (Python random seed 42) had
median raw context length 306 words, versus 12 in the example test prompt. Only
0.56% of sampled raw replies contained three identical consecutive words. Mean
within-reply repeated word 4-gram fraction was about 0.154%. These are sample
statistics, not a full deduplication/data-quality audit.

Useful sources:
- [DITTO, NeurIPS 2022](https://arxiv.org/abs/2206.02369): sentence self-reinforcement.
- [Unlikelihood training](https://arxiv.org/abs/1908.04319): repetition and likelihood.
- [Exposure bias versus self-recovery](https://arxiv.org/abs/1905.10617): do not
  assume the teacher-forcing discrepancy necessarily causes escalating errors.
- [FOCUS & RePAIR, 2026](https://arxiv.org/abs/2608.26676): loop entry and escape
  probability in pruned models; hypothesis inspiration, not proof for this model.

## Historical transfer-era next work (superseded above)

1. Establish the OS, driver, and supported ROCm/PyTorch versions for RX 7900 XTX.
   Create a new environment. The laptop `.venv` and NVIDIA wheel pins are unsuitable.
2. Review backend detection. `training.choose_precision` currently interprets
   `torch.cuda.get_device_capability` using NVIDIA compute-capability thresholds.
   Replace this assumption with a tested backend-aware check before AMD training.
   The model mainly uses ordinary PyTorch operations; AMD execution is still untested.
3. Run the offline tests, then a short FP16 real-data smoke test and record
   throughput, peak memory, finite loss/gradients, and checkpoint/reload behavior.
   Keep this distinct from the full baseline. Do not start a long run automatically.
4. Retrieve the completed Drive checkpoint and run the fixed validation panel.
   Review grammar, relevance, coherence, repetition, and response length together.
5. Implement the Mistral pilot after the baseline evaluation. `RESEARCH_PLAN.md`
   describes proposed SFT -> DPO -> token-aware preference training. The selected
   model is `mistralai/Ministral-3-3B-Instruct-2512-BF16`; verify its current loader,
   native template, revision, and quantization compatibility before implementation.
   It is a multimodal checkpoint; account for frozen vision components even for
   text-only work. No memory/runtime result for this exact pipeline exists yet.

Native Linux was recommended for the broadest AMD training support; Windows/WSL
may also be viable, depending on the actual package combination. Consult current
[AMD training documentation](https://www.amd.com/en/developer/resources/technical-articles/2026/train-and-run-models-on-amd-gpus-with-unsloth.html)
and [bitsandbytes installation docs](https://huggingface.co/docs/bitsandbytes/installation).

## Git and Codex state

Original remote: `https://github.com/mchenn3bula/RolePlayChatBot.git`.
The original transfer was based on commit `824b351` (`Update README.md`). The
modernized code was subsequently pushed at `3e10818`; see the current notes above.
The archive records source
file hashes and Git HEAD for provenance. It contains the current working files,
including previously uncommitted changes, but not `.git` or Git credentials.

`AGENTS.md` and this document are the portable Codex context. The Codex application,
login state, personal settings, other projects, and private conversation database
are not copied. Open the extracted project in Codex on the destination and ask it
to read these documents. This handoff summarizes the relevant decisions rather
than reproducing the complete chat history.
