# Training on this laptop

This workspace is configured for the NVIDIA RTX 4060 Laptop GPU (8 GB VRAM),
using the small dense decoder: 16,275,968 parameters and a 1,024-token context.

Verified on September 21, 2026: CUDA/BF16 training passes. A 512-micro-batch
real-data benchmark measured about **36.3 examples/second**, **0.96 GiB peak tensor
memory**, and **1.75 GiB peak PyTorch reserved GPU memory**. It included full
1,024-token examples. The estimated training time is **about 1.4 hours per epoch**,
before validation and saving; this is a short-run extrapolation, not a completed
epoch or a guarantee under sustained laptop load.

## Environment

The project uses `.venv` and CUDA-enabled PyTorch 2.14.0 with the CUDA 13.0
runtime. The installed NVIDIA 591.59 driver supports it. A separate CUDA Toolkit
installation is not required for this project's prebuilt PyTorch operations.

To recreate the environment from the repository directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-cuda.txt
```

The CUDA wheel comes from the [official PyTorch package index](https://download.pytorch.org/whl/cu130).
This requirements file targets NVIDIA/Windows; use the generic requirements for
other environments.

## Data

The local dataset is prepared under `data/`, with raw and tokenized train,
validation, and test splits. It uses 768 context tokens and 256 reply tokens,
including the EOS boundaries. There are **177,698 training**, **28,018 validation**,
and **44,508 test** examples. To create it in a fresh directory:

```powershell
.\.venv\Scripts\python.exe prepare_data.py --output-dir data --context-length 768 --target-length 256
```

Do not repeat this command once that output directory is populated. Use another
output directory if you intentionally want to rebuild the data.

## Start training

From this repository directory:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\train_local.ps1
```

This runs **one epoch** on CUDA with micro-batch 1, accumulation 32 (effective
batch 32), BF16 when supported, and gradient checkpointing. It prints training loss
every 10 optimizer updates. Validation follows the training epoch. Every run gets
a new timestamped directory under `checkpoints/`.

For a longer run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\train_local.ps1 -Epochs 5
```

Keep the laptop plugged in and prevent sleep during a run. The launcher verifies
CUDA and never silently switches to CPU. Training competes with other apps for
memory and compute; batch 1 is the conservative starting point.

Weights are saved after each completed epoch as `ckpt_ep1.pt`, etc. `best.pt` is
selected by validation perplexity. The folder also contains the tokenizer, model
configuration, and per-epoch metrics. Back up the complete output folder.
The trainer also saves full training state every 250 optimizer updates as
`latest.pt`, retaining `previous.pt` as a fallback. Resume with the Python command
below, replacing `local-RUN` and keeping the original epoch count and settings:

```powershell
.\.venv\Scripts\python.exe train.py --config configs/small.json --data-dir data/bluemoon_train_tok_ds --validation-dir data/bluemoon_validation_tok_ds --output-dir checkpoints/local-RUN --epochs 1 --micro-batch 1 --grad-accum 32 --device cuda --log-every 10 --resume checkpoints/local-RUN/latest.pt
```

The PowerShell launcher starts a new run; use this direct command for continuation.
Only progress since the last complete checkpoint is lost after an interruption.

## Inspect a completed run

Replace `local-RUN` with the directory printed by the launcher:

```powershell
.\.venv\Scripts\python.exe evaluate.py --checkpoint checkpoints/local-RUN/best.pt --data-dir data/bluemoon_test_tok_ds --device cuda --batch-size 1
.\.venv\Scripts\python.exe chat.py --checkpoint checkpoints/local-RUN/best.pt --device cuda --prompt "Once upon a time, a cat found a mysterious key."
```

## Local speed/memory check

```powershell
.\.venv\Scripts\python.exe benchmark.py
```

By default this measures 256 real-data micro-batches after warm-up, with the same small model,
batch size, precision, and checkpointing used by the launcher. It performs actual
backpropagation and optimizer updates, then discards its experimental weights.
It saves measurements to `reports/local-benchmark.json`. Epoch time is an estimate
from that short sample and excludes validation, saving, and changes in laptop
temperature or competing workloads.

The initial setup includes this short benchmark; a full training run is started
separately with the launcher above.

To repeat the longer setup measurement, use `benchmark.py --steps 512 --warmup-steps 16`.
For a quick launcher/GPU/data check without training, add `-CheckOnly` to the
PowerShell launch command.
