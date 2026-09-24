# Whole-conversation LoRA pilot

**Completed September 24, 2026:** the fresh run
`checkpoints/ministral-s1-whole-lora-v1-r2/` finished two epochs / 30 updates in
233.9 seconds, with zero FP16 overflow retries and 11.41 GiB peak reservation.
Held-out reply NLL was 2.88195 initially, 2.69375 after epoch 1 and 2.68817 after
epoch 2. The selected trained adapter is `epoch-02`; no generated-text review
was performed. This is a small pilot, not a six-hour-scale or production-quality claim.

The selected adapter was reloaded through the actual inference launcher and passed
finite-logit/frozen-parameter checks. All 108 paired English/French development
replies were saved for the user at
`reports/ministral-s1-whole-lora-v1-user-panel/replies.html` (raw records beside it).
Do not open or score either artifact with assistant tools. Technical results are
in `results/ministral-s1-whole-lora-v1.json`.

The user authorized a randomly ordered complete-conversation training pilot.
Dialogue is processed locally by software, but the coding assistant does not read,
quote or judge it. Curation here is **structural**, not a semantic quality audit.
No generated-text quality improvement is claimed from this experiment alone.

## Data

`posttraining/lora_data.py` uses the pinned original Bluemoon Parquet revision
`f8cf6b0cbd69294b084d502e2806dc60b9f9c4a0`. It reconstructs available whole threads
before applying filters, preserving the historical title-level split with seed 42.
Train and validation selection use deterministic random order with seed 20260924.

The 2,048-token pilot contains **116 train / 17 validation conversations**, with
448 / 64 supervised assistant replies and 69,033 / 12,702 target tokens per pass.
These are all eligible threads at this budget; the requested sample ceilings of
256 / 32 could not be filled. Selection was not based on reading or rating content.
At 1,024 tokens there were only 31 training conversations and no validation
conversations; no shortened-thread fallback was used.

Filters require complete available records, unambiguous chronological order,
two consistently alternating authors, at least two exchanges, an assistant-ending
thread, an unambiguous source thread URL, exact duplicate exclusion and native
template compatibility. Every retained thread fits without trimming, packing or
splitting. “Complete” means all messages available in this source thread; it does
not establish that a scraped story reached its narrative conclusion. Long threads,
odd-length threads and conversations with more participants are excluded, so this
small pilot has selection bias and should not be treated as representative.

Authors are mapped consistently: the first chronological participant is `user`,
the second is `assistant`. This is a training perspective, not a claim that the
source contained a deployed assistant. No source usernames or thread titles are
printed in logs. The exact selected conversations are saved for the user at:

```
data/ministral-s1-whole-v1/conversations_train.jsonl
data/ministral-s1-whole-v1/conversations_validation.jsonl
```

Do not open these files with assistant tools. The manifest contains only provenance,
hashes, counts and structural rejection statistics. Programmatic fingerprint checks
matched all 548 train and 77 validation sliding windows to their historical prepared
split. Test examples are not tokenized, trained on or scored. Global exact-thread
deduplication uses source hashes only to prevent contamination across splits.

Language distribution, prose quality and content suitability have not been reviewed.
No explicit persona/state labels exist in this source; none were inferred or invented.
The existing authored P1 state wrapper remains available at inference. This run is
ordinary dialogue LoRA SFT, not supervised training of a state extractor.

## Fixed recipe

| Setting | Value |
| --- | --- |
| Backbone | Same pinned Ministral 3 3B Instruct BF16-source revision as P0/P1 |
| Frozen weights / compute | Unquantized FP16 on ROCm |
| Adapter weights / optimizer | FP32 LoRA / AdamW |
| Modules | Language decoder attention q/k/v/o projections only |
| Rank / alpha / dropout | 16 / 32 / 0.05 |
| Trainable parameters | 9,371,648; vision, embeddings, output head and backbone frozen |
| Sequence / micro-batch / accumulation | 2,048 / 1 complete conversation / 8 |
| Objective | Native-template assistant-only next-token loss, including every reply EOS |
| Loss weighting | Supervised-token weighted across each accumulation group |
| Epochs / planned updates | 2 / 30 |
| LR | Peak 1e-4; 3-update warm-up, cosine decay to 1e-5 |
| Optimizer | AdamW, betas 0.9/0.95, weight decay 0.01, clip norm 1.0 |
| Memory / time guard | Reserved <=18 GiB, device-wide free >=4 GiB; 6-hour soft cap |

There is no cross-thread packing or padding in this micro-batch-1 pilot. The native
Mistral tokenizer supplies all boundaries; prefix equivalence is verified before
constructing loss masks. Any ambiguous tokenization rejects the entire thread.
FP16 overflow retries the same accumulation group without advancing progress.
Eight failed retries stop the run. The frozen base model is never saved over or merged.

Variable-length conversations require cache management: the first real-data
validation hit the 18 GiB reservation guard before any training updates. The
trainer now releases unused allocator cache before each micro-batch. This does
not change tokens, loss, gradients or the memory limits. The failed run is retained
at `checkpoints/ministral-s1-whole-lora-v1/`; the fresh run is
`checkpoints/ministral-s1-whole-lora-v1-r2/`.

The target module selection follows the explicit-module mechanism in
[PEFT's LoRA documentation](https://huggingface.co/docs/peft/en/package_reference/lora).
The actual module set and trainable dtypes were checked on the local model.

## Launch, checkpoints and adapter use

Separate environment: `/home/n3bula/.venvs/roleplay-lora`. Dependencies are recorded
in `requirements-lora-rocm.lock.txt`; inference and custom-baseline environments
remain unchanged. Use `setup_lora_env.sh` to recreate the environment. The pinned
model and source dataset must already be in the local HF cache before offline runs.

```powershell
# A new technical smoke test, then a distinctly named new training run:
.\train_lora.ps1 -Mode Smoke -RunName ministral-s1-my-smoke
.\train_lora.ps1 -Mode Train -RunName ministral-s1-my-run

# Resume an unfinished run using one of its exact checkpoint directory names:
.\train_lora.ps1 -Mode Resume -RunName ministral-s1-my-run -Checkpoint step-00005
```

Visible progress and unique logs are under `reports/logs/`. Checkpoints are saved
every five updates and after each validation. Each immutable checkpoint directory
contains adapter weights/config, optimizer/scaler/RNG/progress and compatibility
metadata with file fingerprints. `latest.json` points to the newest checkpoint.
Interrupted temporary checkpoint directories are kept; completed runs are never
overwritten. Resume requires the original code, config, data and pinned base model.
The cap stops at an optimizer boundary; validation/checkpoint writing may add time.

`complete.json` records the best trained epoch by held-out reply NLL. It also records
the original frozen-model validation loss. Selecting the better trained epoch does
not by itself show an improvement over the original model or better generated replies.

Use a selected adapter explicitly; it does not silently replace P1:

```powershell
.\ministral.ps1 -Mode Chat -Profile P1 -Language en -Adapter .\checkpoints\ministral-s1-whole-lora-v1-r2\epoch-02
.\ministral.ps1 -Mode Evaluate -Profile P1 -RunName ministral-s1-user-panel -Adapter .\checkpoints\ministral-s1-whole-lora-v1-r2\epoch-02
```

Replace `epoch-02` with the checkpoint named in `complete.json` if a different epoch
was selected. Adapter inference uses the LoRA environment and validates backbone
revision/precision and adapter fingerprints. Chat still saves replies to a local
HTML gallery and accepts explicit state edits through `/reload`.

## Verification

The synthetic actual-model smoke passed backward and optimizer steps at 1,024 and
2,048 tokens, at 11.41 GiB peak reservation after allocator-cache cleanup. Adapters changed, backbone parameters
remained frozen, and a checkpoint/resume reproduced the uninterrupted next update
exactly (maximum parameter difference zero). Evidence:
`checkpoints/ministral-s1-smoke-v3/smoke.json`.

The project suite ran 63 tests successfully with one native-tokenizer test skipped
in the baseline environment. All seven LoRA tests, including that native-tokenizer
test, passed in the separate training environment. These use authored benign
fixtures only. Affected-file Ruff and launcher syntax checks also passed.
