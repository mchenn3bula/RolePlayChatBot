# Six-hour RoPE + SwiGLU experiment

**Completed September 23, 2026:** all four epochs finished, best validation
perplexity 38.10. The fixed 150-reply generation panel is also complete.
Replies still have weak coherence/relevance and the original short inn prompt
fails. See [GENERATION_EVALUATION.md](GENERATION_EVALUATION.md) before another run.

Experiment: `modern-rope-swiglu-v1`. This is the user's selected next
from-scratch baseline candidate. It is a separate architecture-version-3 run;
the historical 16.3M v2 baseline and proposed 124.3M GELU experiment are preserved.

## Architecture and objective

| Component | Selected configuration |
| --- | --- |
| Parameters | 123,551,232 (123.6M) |
| Decoder | 12 dense pre-RMSNorm blocks, width 768 |
| Attention | 12 query/key/value heads, 64 dimensions each, causal SDPA |
| Positions | Interleaved RoPE, theta 10,000; no learned position table |
| Feed-forward | SwiGLU, intermediate width 2,048, bias-free projections |
| Residual initialization | Output projections scaled by 1 / sqrt(2 x layers) |
| Dropout | 0.1, attention and feed-forward branches |
| Embedding/output | Tied GPT-2 vocabulary matrix, random initialization |
| Context | 1,024 total tokens; existing 768 context / 256 reply preparation |
| Training projection | Only hidden states immediately before supervised reply targets |
| Gradient checkpointing | Off |

The SwiGLU block uses three matrices with 3 x 768 x 2,048 parameters, matching
the previous GELU block's 2 x 768 x 3,072 parameters. Removing learned positions
reduces the total by 786,432. RoPE rotates queries and keys in FP32 before
casting them back to the attention dtype. Position indices count valid tokens
so left and right padding preserve real-token positions.

The vocabulary projection selects states BEFORE supervised targets, including
the final context state that predicts the first reply token. EOS remains a
target. Context and padding losses remain excluded. The decoder still processes
context tokens, and their states still receive gradients through attention.
This optimization preserves the mathematical loss; matrix-kernel rounding can
differ. Both losses and all parameter gradients are compared in FP32 and FP16.
Generation still uses the complete next-token distribution.

No full-conversation pretraining phase, tokenizer change, GQA, MoE, KV cache,
or context extension is included in this experiment. The earlier discussion of
full-conversation pretraining was a separate objective experiment to investigate.
RoPE does not imply reliable generation beyond the trained context length.

## Recipe and visible logging

Use FP16 with gradient scaling, FP32 master weights, micro-batch 8 x accumulation
4, AdamW (0.9, 0.95), weight decay 0.01, peak LR 1.5e-4, 500 warm-up updates,
cosine decay over four planned epochs, gradient clipping 1.0, seed 42.

The six-hour per-invocation limit saves at an optimizer boundary. An ongoing
validation or checkpoint write may exceed the limit. Resume preserves the
original four-epoch schedule; it does not restart warm-up. Checkpoints are saved
every 250 successful updates, retaining latest and previous snapshots, with
best-model selection after each full validation pass.

Run these commands in PowerShell from the project directory:

```powershell
# Start the selected full experiment:
.\train_amd.ps1 -Profile Modern -Mode Train -RunName modern-rope-swiglu-v1

# Continue an interrupted/time-limited run:
.\train_amd.ps1 -Profile Modern -Mode Resume -RunName modern-rope-swiglu-v1

# Optional repeat of the short preflight with an automatically unique name:
.\train_amd.ps1 -Profile Modern -Mode Smoke
```

The terminal prints a first-update report and then reports every 25 successful
updates: epoch, update, examples, batch target NLL, next-step learning rate,
average examples/sec, elapsed time, estimated training time left in the epoch,
skipped FP16 updates, and current reserved VRAM. Timing includes setup/checkpoint
overhead and is an estimate. Checkpoint start/completion and validation progress
are also visible; validation reports every 100 batches. All stdout/stderr is
unbuffered and copied to a unique `reports/logs/<run-name>-<mode>-*.log` file.
Resume gets a new log without overwriting prior logs. Pipeline failures propagate
to the PowerShell exit code.

Checkpoints are under `checkpoints/modern-rope-swiglu-v1/`. Keep Windows awake
and the terminal open. The launcher does not change Windows power settings.
`-Profile Small` and `-Profile Overnight` still select their original configs.

## Preflight and interpretation

41 tests passed on ROCm, including causal prefix equivalence, future-gradient
isolation, left/right padding, RoPE relative-position behavior, supervised
projection loss/gradient equivalence, exact dropout checkpoint/time-stop resume,
RoPE-theta resume rejection, prefix-scored perplexity, and generation evaluation.
The historical v2 implementation also matched the pre-edit source exactly for
same-seed initialization, dropout forward output, and parameter gradients.

Evidence: `reports/modern-tests.log`, `reports/v2-compatibility.json`, and the
smoke/benchmark reports recorded below. Syntax checks and Ruff passed.

| Actual RX 7900 XTX measurement | Result |
| --- | --- |
| Full-model smoke train / resume / reload | Passed, 8 successful updates, 0 skipped |
| Smoke peak allocated / reserved | 7.49 / 9.37 GiB |
| Benchmark | 2,048 examples, 256 micro-batches, 64 successful updates, 0 skipped |
| Longest benchmark sequence | 1,024 tokens |
| Benchmark throughput | 40.19 examples/sec |
| Benchmark peak allocated / reserved | 7.54 / 10.60 GiB |
| Projected training-only time | 1.23 hours/epoch; 4.91 hours for four epochs |

Use the larger observed reservation, **10.60 GiB**, when budgeting memory.
The six-hour limit leaves about an hour for validation/checkpoint overhead and
variation, but completion time is not guaranteed by a short benchmark. The
previous GELU/learned-position candidate measured 36.28 examples/sec; the new
measurement is about 11% faster, not a controlled attribution to any one change.

Results: `checkpoints/modern-rope-swiglu-smoke/result.json` and
`reports/modern-rope-swiglu-b8.json`. The smoke log is
`reports/logs/modern-rope-swiglu-smoke-Smoke-iC5jPlpl.log`; the benchmark log is
`reports/logs/modern-rope-swiglu-b8-Benchmark-SiUvDG0X.log`. Benchmark weights
were discarded. The full four-epoch experiment subsequently completed; see the
status and generation evaluation linked above.

This validates implementation and runtime feasibility. It does not establish
better roleplay quality than GELU/learned positions. Retain the fixed validation
generation panel in `BASELINE.md`; reserve the test split for final evaluation.
Training still has only 24.57M supervised reply-token targets per pass, repeated
across epochs. Compare validation perplexity, relevance, grammar, coherence,
repetition, and response length after training.
