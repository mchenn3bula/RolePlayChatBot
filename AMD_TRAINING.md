# RX 7900 XTX training on this PC

The selected next experiment uses **RoPE + SwiGLU** via `-Profile Modern`;
see [MODERN_BASELINE.md](MODERN_BASELINE.md). All launcher modes now show live
terminal output and save unique logs under `reports/logs/`.

For the new 124M-parameter, six-hour experiment, see
[OVERNIGHT_BASELINE.md](OVERNIGHT_BASELINE.md) and use `-Profile Overnight`.
The commands below default to the historical 16.3M `Small` profile.

The project uses Ubuntu 24.04 under WSL2, ROCm 7.2.1, AMD's PyTorch
2.9.1 Radeon wheel, and Python 3.12. The Windows host has Adrenalin 26.8.1
(driver store version 32.0.31041.1004). The WSL distribution is `Ubuntu-24.04`;
training runs as `n3bula`, with the environment at
`/home/n3bula/.venvs/roleplay-rocm` on the Linux filesystem.

The source, datasets, reports, and checkpoints stay in the Windows project
folder. The launcher resolves its location automatically. Installing the
environment does not launch a full training run.

## Verified results on this PC

September 22, 2026:

- All **26 regression tests passed** under ROCm, including FP16 causal/padding
  isolation and optimizer/scaler checkpoint continuation (`reports/amd-tests.log`).
- The real-data smoke run passed: 256 training examples, 16 validation examples,
  8 successful optimizer updates, 0 skipped updates, finite final gradients,
  weights and optimizer tensors, and matching checkpoint-reload logits.
  Result: `checkpoints/rx7900xtx-smoke-20260922/result.json`.
- Peak smoke training memory: **5.75 GiB allocated / 9.35 GiB reserved**.
- Benchmark: **135.6 examples/second**, 256 measured micro-batches (2,048
  examples), 64 successful optimizer updates, 0 skipped updates, sequences up
  to 1,024 tokens, **5.75 GiB allocated / 7.19 GiB reserved**.
  Result: `reports/rx7900xtx-benchmark-256-20260922.json`.
- This 15-second measurement projects about **22 minutes of training per
  epoch**, or 66 minutes for three epochs, **excluding validation, checkpoint
  IO, and sustained-run variation**. A complete long run has not been launched.
- The resolved environment is recorded in `reports/amd-environment-freeze.txt`.

The smoke model starts from random weights and receives only eight updates;
its perplexity is not the completed Colab model's score or a quality result.

## Commands from PowerShell

```powershell
# Offline regression suite, including GPU tests when available:
.\train_amd.ps1 -Mode Test

# Default: small real-data train / resume / validate / reload check:
.\train_amd.ps1

# Measure throughput with the fixed baseline and effective batch 32:
.\train_amd.ps1 -Mode Benchmark

# Explicitly start a NEW, full three-epoch baseline replication:
.\train_amd.ps1 -Mode Train -RunName rx7900xtx-batch8-v1

# Resume that same run after an interruption:
.\train_amd.ps1 -Mode Resume -RunName rx7900xtx-batch8-v1
```

The launcher selects the RX 7900 XTX by name, so the integrated Radeon GPU
cannot accidentally become the training target. PyTorch calls AMD devices
`cuda` too; `torch.version.hip` identifies the ROCm backend.

Training settings match the completed Colab experiment: architecture v2,
4 layers, width 256, 4 heads, FFN width 1,024, context 1,024, tied embeddings,
gradient checkpointing, FP16 with gradient scaling, micro-batch 8,
accumulation 4, AdamW, peak LR 0.00015, 500 warm-up updates, three epochs,
seed 42. GPU/runtime differences mean this is a replication, not a claim
of bitwise equivalence with Colab. Use a new run name and keep the complete
checkpoint directory together.

The smoke run uses copies of 256 shuffled training examples and 16 validation
examples. It stops after two successful optimizer updates, loads the complete
checkpoint into a fresh model, finishes the short epoch, and checks finite
gradients/weights/optimizer state and matching reloaded logits. It writes
`checkpoints/<run-name>/result.json`. Its throughput includes validation and
checkpoint IO, so use the separate benchmark for steady training throughput.
Neither command uses the real test split. Smoke outputs do not measure trained
chatbot quality.

## Rebuild the environment

WSL2 and a compatible Windows AMD driver must already be available. Install an
Ubuntu 24.04 distribution if needed. This PC's Ubuntu image was obtained from
Microsoft's WSL distribution manifest and verified with SHA-256 before import.

From PowerShell in the project directory:

```powershell
$project = (wsl -d Ubuntu-24.04 -u root --exec wslpath -a $PWD.Path).Trim()
wsl -d Ubuntu-24.04 -u root --cd $project --exec bash ./setup_amd_system.sh
wsl -d Ubuntu-24.04 -u n3bula --cd $project --exec bash ./setup_amd_env.sh
.\train_amd.ps1 -Mode Test
.\train_amd.ps1 -Mode Smoke
```

System setup installs user-space ROCm and ROCDXG 1.2.2 inside the dedicated
Ubuntu distribution. It uses the existing Windows display driver. Environment
setup creates a Linux venv and installs `requirements-rocm.txt`; it does not
modify the Windows Python installation. `ROLEPLAY_VENV` can override the venv
location inside WSL. The `n3bula` Linux account has no configured login password;
WSL can launch it directly. Administrative provisioning uses explicit
`wsl -u root`, not a stored password.

For this ROCm version, `run_amd.sh` sets `HSA_ENABLE_DXG_DETECTION=1` per process.
It also uses the packaged tokenizer/datasets offline. ROCm `auto` precision
defaults to FP16; explicit BF16 remains an opt-in workload experiment.

Do not install `requirements-cuda.txt` on AMD or substitute a normal CPU/CUDA
PyTorch wheel. The CPU-only `.venv-cpu` created for initial code validation
reuses existing Windows packages; GPU training uses the separate Linux venv.

## Sources checked September 22, 2026

- [AMD WSL and ROCDXG training support](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2.1/docs/install/installrad/wsl/howto_wsl.html)
- [AMD Radeon PyTorch wheels](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2.1/docs/install/installrad/native_linux/install-pytorch.html)
- [AMD native Windows limitations](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2.1/docs/limitations/limitationsrad.html): the documented Windows package excludes ML training.
- [ROCDXG releases](https://github.com/ROCm/librocdxg/releases/tag/v1.2.2)
- [Installed Windows driver release notes](https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-WIN-26-8-1.html)

The versioned stack was deliberately selected from AMD's documented Radeon
combination. It is not a claim that ROCm 7.2.1 is the newest release.
