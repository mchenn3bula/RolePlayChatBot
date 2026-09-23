# Architecture review — September 22, 2026

Conclusion: the current architecture is reasonable as a small from-scratch
baseline. No architecture replacement is warranted by the code review.
It is not yet a useful open-ended roleplay chatbot: the recorded Colab
generation is poor, and the completed checkpoint still needs its fixed
validation generation panel.

## What was reviewed

- `configs/colab_t4.json` and `mini_deepseek.py`: 4 decoder blocks, width 256,
  4 heads (64 dimensions each), dense 1,024-wide GELU feed-forward blocks,
  RMSNorm before residual branches, learned absolute positions, dropout 0.1,
  tied input/output embeddings, one next-token output head. No MoE in this run.
- Attention projects each position independently before applying a lower
  triangular mask and key-padding mask. There is no sequence compression
  mixing future positions into earlier keys/values. PyTorch SDPA receives
  a boolean mask with allowed positions set to true.
- Loss shifts exactly once, masks context and padding with -100, and includes
  the first reply token and final EOS. Context EOS remains a real input even
  when the same token ID is used to fill padded positions.
- Position indices count valid tokens, keeping real-token positions consistent
  under padding. Generation shares the same message and EOS boundary encoding.
- Initialization uses small tied embeddings and scaled residual projections.
  Gradient checkpointing recomputes the same causal blocks.
- The trainer weights gradient accumulation by the total target-token count,
  clips gradients, uses AdamW/cosine scheduling, and records optimizer,
  scheduler, scaler, RNG, and data cursor for continuation. Validation selects
  checkpoints. The real test split is reserved for final evaluation.

## Capacity and design tradeoffs

The 16,275,968 parameters include 12,865,792 in the shared 50,257 x 256
token embedding/output matrix (79.1%). Only about 3.41M parameters remain
for positions, decoder blocks, and normalization. This limits language-model
capacity but is not a correctness defect. Tying the matrix already avoids
paying for a second vocabulary projection.

The model learns language from the roleplay reply objective with random weights;
it does not inherit GPT-2's language knowledge by using its tokenizer. Four
small blocks and this dataset/budget can establish an educational baseline,
but coherent general dialogue is an empirical goal, not an expected guarantee.

The 1,024-token window and three-message input limit longer-term character
consistency. Truncating replies at 256 tokens can teach artificial endings.
There are no explicit persona fields or persistent memory in this baseline.
Generation recomputes the entire prefix because there is no KV cache; this
affects latency, not causal correctness. Full-vocabulary logits are materialized
for context positions as well as replies, which increases training memory.

Keep these choices fixed for the baseline. A smaller tokenizer, larger decoder,
RoPE, new data/objective, or pretrained Mistral backbone should be a separately
named experiment. The current poor sample does not establish which change will
help, or prove that the previous leakage bug has returned.

## Runtime changes

The identified AMD issue was in precision selection, not the model architecture:
NVIDIA compute-capability thresholds were being applied to ROCm devices.
ROCm now defaults to FP16 with gradient scaling and handles explicit BF16
separately. NVIDIA T4 behavior is preserved. The benchmark now shares the
trainer's precision policy, optimizer settings, and token-weighted accumulation.

Verification covers causal invariance, future-position gradients, padding,
EOS alignment, independently scored perplexity, synthetic learning, and exact
CPU continuation. GPU verification adds FP16 causal/padding checks and
scaler/optimizer resume; `smoke_amd.py` exercises the actual baseline and
prepared data. See `AMD_TRAINING.md` and the saved validation logs/results.

All 26 tests passed in the configured ROCm environment. The real baseline
completed eight FP16 updates with checkpoint resume and reload, finite final
gradients/weights/optimizer state, and no skipped updates. A separate 64-update
benchmark also completed without skipped updates. Checks on 1,024 sampled
training and 1,024 sampled validation records verified vocabulary bounds,
EOS boundaries, and the 768/256 token limits. The actual model parameter count
and tied matrix match the baseline specification.

The imported source manifest remains the checksum record of the laptop
snapshot. Modified files are expected to differ from those original hashes.
