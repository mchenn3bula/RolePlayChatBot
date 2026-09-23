# T4 causal-decoder baseline

Status: architecture version 2, retained as the project's from-scratch baseline.
This is a deliberately small, conventional autoregressive Transformer. It is a
reasonable baseline architecture; useful open-ended dialogue is an empirical
goal, not a prerequisite for calling the architecture valid.

## Fixed model

Source: `mini_deepseek.py`; configuration: `configs/colab_t4.json`.

| Component | Baseline |
| --- | --- |
| Tokenizer | GPT-2, 50,257 tokens; no pretrained model weights |
| Decoder blocks | 4, pre-normalization with RMSNorm and residual connections |
| Hidden width / heads | 256 / 4; head dimension 64 |
| Attention | Standard causal multi-head self-attention; padding keys masked |
| Feed-forward | Dense 256 -> 1,024 -> 256, GELU |
| Positions | Learned absolute positions, maximum 1,024 tokens |
| Output | One next-token head, tied to token embeddings |
| Dropout | 0.1 |
| Parameters | 16,275,968, including 12,865,792 shared embedding/output parameters |
| Optional features | MoE disabled; no sequence compression or duplicate MTP head |

The vocabulary consumes a large fraction of parameters, but the shared matrix
does useful input/output modeling. This ratio is a design tradeoff, not evidence
of incorrectness. A smaller vocabulary or larger decoder would be a separate
experiment. RoPE, SwiGLU, MoE, and MLA are not required to establish this baseline.
The lack of a KV cache affects generation speed, not causal correctness.

## Fixed data and training

Use the existing version-2 prepared Bluemoon datasets and GPT-2 tokenizer.
Entire conversation threads stay within one split. Each example predicts a reply
from three previous messages, with up to 768 context tokens and 256 reply tokens.
The loss covers reply tokens, including EOS, and excludes context and padding.
This is conditional next-token training from scratch, not broad-corpus pretraining.

The user's completed Colab run is `t4-batch8-v1`: FP16 with gradient scaling,
micro-batch 8, accumulation 4, effective batch 32, AdamW, peak learning rate
1.5e-4, 500 warm-up updates, cosine decay over three epochs, and seed 42.
Keep its checkpoint, saved config, tokenizer, and progress together. Its validation
perplexities were 100.6102, 79.2765, and 75.6797. The supplied generation was
repetitive and grammatically weak. These are user-reported results; the completed
checkpoint has not been inspected locally.

This completed run establishes T4 feasibility for that setup. The updated upload
notebook uses 8 x 4 and selects `t4-batch8-v1`, with training disabled by default
to evaluate the completed checkpoint. A fresh replication needs a new output
directory and `RUN_TRAINING = True`. Resume an existing run only with its original
settings. Older 2 x 16 runs require those settings; equal effective batch sizes
do not guarantee bitwise-identical training trajectories.

No retraining is required merely to adopt the completed run as the baseline.
Three epochs define this experiment's budget, not a claim of convergence. Changes
to objective, vocabulary, model dimensions, data, or schedule must get a distinct
run name and be reported explicitly.

## Verification and evaluation before extending the model

The existing offline tests check future-token isolation, padding isolation,
target/EOS alignment, independent prefix-only perplexity scoring, learning and
generation on synthetic examples, checkpoint loading, and training resumption.
Run them with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Use the notebook's generation panel or `evaluate_generation.py` to evaluate the
completed checkpoint on a fixed sample of 50 validation
contexts, selected with seed 42. Generate with sampling seeds 42, 43, and 44.
Use temperature 0.8, top-k 50, top-p 0.9, repetition penalty 1.05, and a maximum
of 128 new tokens to match the initial sample's decoding settings. Use the same
tokenization, context truncation, and reply boundary as training. Save the exact
prompts, outputs, seeds, checkpoint identity, and settings for later comparisons.

Report validation perplexity alongside output length, empty-response frequency,
within-response repeated word 4-gram fraction, and manual ratings of grammar,
relevance, and coherence. Short or empty replies must not receive a favorable
quality judgment just because they contain no repeated 4-grams. Compare full
training-style contexts with shortened contexts as a separate diagnostic. These
generation evaluations remain pending on the user's completed checkpoint. The
script produces a manifest with checkpoint/data/source hashes, exact context and
output token IDs, automatic metrics, and a CSV for human review. Its default
inference is FP32 to match `chat.py`; this does not change FP16 training settings.

Reserve the test split for the final evaluation. Do not select decoding settings
using test results. Future scratch-model experiments should hold data, tokenizer,
training budget, and evaluation prompts fixed where the comparison permits.

## Relationship to the Mistral project

Keep this model as the historical from-scratch baseline. It can establish the
practical benefit of moving to a pretrained backbone, but that comparison changes
architecture, tokenizer, data exposure, and initialization simultaneously.

For claims about a particular post-training method, the controlled baseline must
be the same Mistral checkpoint with ordinary SFT. Compare later DPO or token-aware
training against that SFT baseline. Compare models with different tokenizers using
the same prompts and generation-quality rubric, not raw token perplexity.
