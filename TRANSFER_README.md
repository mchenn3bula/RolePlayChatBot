# Move this project to another PC

Extract `RolePlayChatBot_transfer.zip`. Its top-level `RolePlayChatBot/` directory
is the project folder. Open that folder in your editor or Codex.

## Included

- Current Python source, tests, configs, project documentation, and archived
  original notebook/PDF/diagram.
- Raw and tokenized train/validation/test datasets under `data/`.
- The matching GPT-2 tokenizer under `tokenizer/`.
- The updated `dist/RolePlayChatBot_Colab.ipynb`, plus the builder needed to
  recreate its separate Colab ZIP.
- `AGENTS.md`, `CODEX_HANDOFF.md`, and a checksummed `transfer_manifest.json`.

This is a personal project snapshot, including work not yet pushed to GitHub.
The raw data also supports `baseline.py` and future data audits. The existing
Colab ZIP is not nested in this archive because the same datasets/source are
already included; run `build_colab_bundle.py` if you need a fresh Colab ZIP.

## Environment on the destination

Create a fresh Python environment; do not reuse the old laptop's `.venv`.
For AMD RX 7900 XTX, first choose a supported ROCm/PyTorch/Python combination
from current AMD documentation. Install that ROCm PyTorch build, then the other
project dependencies while ensuring pip retains the chosen PyTorch build.
Do not run `requirements-cuda.txt`: it installs NVIDIA-specific binaries.
The generic requirements are compatibility ranges, not an AMD-tested lockfile.

AMD runtime adaptation/testing remains pending, specifically the NVIDIA-specific
precision detection in `training.py`. See `CODEX_HANDOFF.md` for the next steps.
The archive transfers the working project; it does not claim AMD validation.

Once the environment is configured, run:

```text
python -m unittest discover -s tests -v
```

For the NVIDIA laptop or Colab, use their separate existing setup documents.
To rebuild the Colab upload archive using the packaged tokenizer without a new
download:

```text
python build_colab_bundle.py --tokenizer tokenizer
```

## Restore the completed model

The full trained model lives in Google Drive, not on this source PC. Copy the
entire `MyDrive/RolePlayChatBot/runs/t4-batch8-v1/` folder to the destination's
`checkpoints/t4-batch8-v1/`. Keep the tokenizer and config alongside the weights.
The ZIP does not contain pretrained Mistral weights or trained adapters either.

## Continue with Codex

Use Codex installed and signed in on the destination. Project instructions are
supplied through the documented
[AGENTS.md convention](https://learn.chatgpt.com/docs/agent-configuration/agents-md).
The ZIP includes a project handoff, not Codex application binaries or account data.

Suggested first message:

> Read AGENTS.md, CODEX_HANDOFF.md, and BASELINE.md. I have an RX 7900 XTX on this
> PC. Inspect this machine's OS and GPU setup, prepare a compatible ROCm environment,
> adapt the NVIDIA-specific precision check, and validate the existing baseline
> with tests and a short smoke run. Preserve the baseline architecture and saved
> checkpoints. Do not launch full training until the smoke run is reviewed.

## Integrity

`transfer_manifest.json` lists SHA-256 hashes for all packaged payload files.
The build script verifies these hashes inside the ZIP before reporting success.
It intentionally excludes environments, caches, `.git`, account credentials,
local smoke checkpoints, and personal Codex session/configuration files.
