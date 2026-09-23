# RolePlayChatBot handoff

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

## Next work on the AMD PC

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
The modernized code is local work based on commit
`824b351` (`Update README.md`); it has not been pushed. The archive records source
file hashes and Git HEAD for provenance. It contains the current working files,
including previously uncommitted changes, but not `.git` or Git credentials.

`AGENTS.md` and this document are the portable Codex context. The Codex application,
login state, personal settings, other projects, and private conversation database
are not copied. Open the extracted project in Codex on the destination and ask it
to read these documents. This handoff summarizes the relevant decisions rather
than reproducing the complete chat history.
