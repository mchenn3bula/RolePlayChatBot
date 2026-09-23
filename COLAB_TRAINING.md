# Train on a Colab T4

The upload bundle contains the corrected Python source, local GPT-2 tokenizer,
and the prepared tokenized train/validation/test datasets. It does not depend on
the older code currently on GitHub. The notebook is only a launch interface;
model and training implementation remain in Python files.

## Upload and run

1. In Google Drive, create a folder named `RolePlayChatBot` under **My Drive**.
2. Upload `dist/RolePlayChatBot_colab.zip` into that folder.
3. Open Colab and upload `dist/RolePlayChatBot_Colab.ipynb` as the notebook.
4. Select **Runtime → Change runtime type → T4 GPU**, then run the cells in order.
5. Authorize the Drive mount when prompted. The default run is the completed
   `My Drive/RolePlayChatBot/runs/t4-batch8-v1/`. Training is skipped by default;
   the notebook evaluates the saved checkpoint.

For a new training run, change `RUN_NAME` to a fresh folder and set
`RUN_TRAINING = True` in the first setup cell. To resume an unfinished run, set
that flag and keep its exact original batch and schedule settings.

No Hugging Face token or data regeneration is needed. Installation needs internet
access. The dependency cell preserves Colab's preinstalled CUDA-enabled PyTorch;
do not use the Windows `requirements-cuda.txt` there. PyTorch 2.3+ is required.
The dataset archive is copied to local VM storage before extraction and training.

## Default parameters

| Setting | Value |
| --- | --- |
| Model | Dense causal decoder, 16,275,968 parameters |
| Layers / hidden size / heads | 4 / 256 / 4 |
| Context | 1,024 tokens: 768 context + 256 reply |
| Precision | FP16 with dynamic gradient scaling |
| Micro-batch / gradient accumulation | 8 / 4 (effective batch 32) |
| Optimizer | AdamW, betas 0.9/0.95, weight decay 0.01 |
| Learning rate | 0.00015 peak, cosine decay |
| Warm-up | 500 optimizer updates |
| Planned training | 3 epochs; validation after each epoch |
| Dropout / gradient clipping | 0.1 / norm 1.0 |
| Gradient checkpointing | Enabled |
| Checkpoint / log interval | 250 / 10 successful optimizer updates |
| Seed | 42 |

These are conservative starting settings for a 16 GB T4. FP16 is used because
the T4 lacks native BF16. There are 177,698 training examples, about 5,554
optimizer updates per epoch before any FP16 overflow skips. Three epochs are an
initial experiment, not a guarantee of useful chat quality. Select by validation
and inspect generated replies; reserve the test set for a final evaluation.

The user completed this 8 x 4 FP16 configuration on a Colab T4. Its full checkpoint
has not been inspected locally, and no systematic T4 throughput or peak-memory
profile is available. Local FP16 smoke tests also passed on an RTX 4060 Laptop GPU.
Colab GPU assignment,
availability, and runtime limits vary; see the [Colab FAQ](https://research.google.com/colaboratory/faq.html).

## Disconnects and continuation

Reconnect to a GPU and rerun the same notebook with the same `RUN_NAME` and
parameters, enabling `RUN_TRAINING` if continuing training. It finds `latest.pt`
automatically and restores model weights, AdamW,
learning-rate schedule, FP16 scaler, shuffled data position, and RNG state.
Uncheckpointed progress must be repeated. Checkpoints are written directly to
Drive, with the prior complete snapshot retained as `previous.pt`. A partially
written `.tmp` is never used. Drive must remain mounted and have free space;
allow at least 2 GB for the bundle and one run.

If `latest.pt` is missing, the notebook tries `previous.pt`. If it exists but is
corrupt, set `RESUME_FILE = 'previous.pt'` manually. Configuration, dataset, and
training schedule must match the original run. To change these, choose a new
`RUN_NAME`; extending `EPOCHS` changes the cosine schedule and is not a resume.

For a deliberate short session, set `STOP_AFTER_UPDATES` to a total update count,
such as 100. It saves and stops at that optimizer boundary. Reset this option to
`None` on continuation. To stop immediately with Colab's interrupt button, resume
later from the last complete periodic checkpoint.

`best.pt` contains weights selected by validation; `ckpt_epN.pt` contains each
completed epoch's weights. These are for inference, not optimizer resumption.
Keep `config.json` and `tokenizer/` alongside the weights. `progress.json` records
saved progress, and `metrics.json` records completed epochs. The notebook includes
an optional quick-chat cell, a validation generation panel, and an optional
final-test cell. Quick chat and final-test evaluation are disabled by default.

## Evaluate the baseline

After setup, run the checkpoint-selection and validation generation cells.
The default is 50 validation contexts selected with seed 42, each sampled using
seeds 42, 43, and 44. Exact prepared context IDs preserve training truncation and
the EOS reply boundary. The model architecture and checkpoint weights are unchanged.

Reports go to a new timestamped `evaluations/validation-.../` folder inside the
run directory. The manifest records checkpoint, source, and dataset hashes;
`generations.jsonl` stores prompts, reference replies, generated replies, and token
IDs. `summary.json` reports lengths, empty replies, EOS termination, and repeated
word 4-grams. `human_review.csv` has blank grammar/relevance/coherence rating fields.
The evaluation uses FP32 inference, matching the existing `chat.py` behavior.
Training still uses FP16. Identical sampling seeds are not a guarantee of identical
outputs across different GPUs or library versions.

Use `SAMPLES = 5` for a quick check, then 50 for the report. A runtime disconnect
leaves completed JSONL/CSV rows but no completed summary; rerun into a new folder.
Read low repetition together with response length and human judgments. Empty or
very short replies are not evidence of improved dialogue quality. The full baseline
specification and comparison constraints are in `BASELINE.md`.

If setup fails before even the initial checkpoint is saved, use a new run name
after fixing the error; the trainer refuses to overwrite a nonempty fresh run.

## Rebuild the upload files after editing source

From the prepared local repository:

```powershell
.\.venv\Scripts\python.exe build_colab_bundle.py
```

The builder includes only an explicit source allowlist, the saved tokenizer, and
the three tokenized datasets. It omits `.git`, `.venv`, raw datasets, local reports,
existing checkpoints, and the archived original notebook. SHA-256 checksums are
verified during extraction. Rebuild before uploading if the Python code changes.
