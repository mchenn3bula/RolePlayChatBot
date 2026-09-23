"""Build a portable Colab notebook and a ZIP of source, tokenizer, and prepared data."""

import argparse
import hashlib
import json
import textwrap
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SPLITS = ("train", "validation", "test")
SOURCE_FILES = (
    "train.py",
    "training.py",
    "mini_deepseek.py",
    "conversation.py",
    "workflow.py",
    "evaluate.py",
    "evaluate_generation.py",
    "chat.py",
    "prepare_data.py",
    "requirements-colab.txt",
    "configs/colab_t4.json",
    "COLAB_TRAINING.md",
    "BASELINE.md",
)


def notebook():
    cells = []

    def add(kind, source):
        cell = {
            "cell_type": kind,
            "id": f"cell-{len(cells):02d}",
            "metadata": {},
            "source": textwrap.dedent(source).strip().splitlines(keepends=True),
        }
        if kind == "code":
            compile("".join(cell["source"]), "colab-cell", "exec")
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add(
        "markdown",
        """
    # RolePlayChatBot: train on a Colab T4

    1. Upload `RolePlayChatBot_colab.zip` to **My Drive / RolePlayChatBot**.
    2. In Colab, select **Runtime → Change runtime type → T4 GPU**.
    3. Run the cells in order and authorize your Drive mount.

    This notebook runs the corrected Python scripts. The ZIP already contains the
    prepared train/validation/test data and GPT-2 tokenizer. Data is extracted to
    the VM's local disk; complete training checkpoints are saved to Drive.

    **Defaults:** 16.3M parameters, 1,024 tokens (768 context + 256 reply), FP16,
    micro-batch 8 × accumulation 4 = effective batch 32, learning rate 0.00015,
    500 warm-up updates, cosine decay, three planned epochs, seed 42.
    This is a small model trained from scratch; assess generated replies as well as
    validation perplexity. Colab does not guarantee enough runtime for all epochs.

    **Existing baseline:** this notebook opens `t4-batch8-v1` and skips training
    by default. Run setup, then evaluate your completed checkpoint. To train a fresh
    replication, choose a new run name and set `RUN_TRAINING = True`.
    Architecture: four dense causal decoder blocks, width 256, four heads, GELU
    feed-forward width 1,024, RMSNorm, learned positions, and tied embeddings.
    See the bundled `BASELINE.md` for the specification and comparison protocol.

    **After a disconnect:** reconnect to a GPU and run the same notebook again.
    Keep the same `RUN_NAME`, dataset, and training settings. It resumes the last
    complete checkpoint (saved every 250 optimizer updates and at epoch boundaries).
    Changes to batch size, epochs, or learning rate require a new run name.
    """,
    )
    add(
        "code",
        """
    from pathlib import Path
    from google.colab import drive

    drive.mount('/content/drive')
    DRIVE_ROOT = Path('/content/drive/MyDrive/RolePlayChatBot')
    ARCHIVE = DRIVE_ROOT / 'RolePlayChatBot_colab.zip'
    RUN_NAME = 't4-batch8-v1'
    RUN_TRAINING = False  # Evaluate the completed baseline without retraining.
    assert RUN_NAME not in ('', '.', '..') and Path(RUN_NAME).name == RUN_NAME, 'Use a simple folder name.'
    OUTPUT = DRIVE_ROOT / 'runs' / RUN_NAME
    PROJECT = Path('/content/RolePlayChatBot')
    assert ARCHIVE.is_file(), f'Upload the prepared ZIP to {ARCHIVE} first.'
    print('Checkpoints:', OUTPUT)
    """,
    )
    add(
        "markdown",
        """
    ## Copy the bundle to local disk
    This may take a few minutes. Verification checks every packaged file before use.
    Re-running this cell refreshes the source and data; checkpoints stay in Drive.
    """,
    )
    add(
        "code",
        """
    import hashlib
    import json
    import shutil
    import zipfile

    local_archive = Path('/content/RolePlayChatBot_colab.zip')
    shutil.copy2(ARCHIVE, local_archive)
    PROJECT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(local_archive) as archive:
        for member in archive.infolist():
            destination = (PROJECT / member.filename).resolve()
            if not destination.is_relative_to(PROJECT.resolve()):
                raise ValueError(f'Unsafe ZIP entry: {member.filename}')
        archive.extractall(PROJECT)
    manifest = json.loads((PROJECT / 'bundle_manifest.json').read_text())
    for relative, expected in manifest['sha256'].items():
        digest = hashlib.sha256()
        with (PROJECT / relative).open('rb') as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                digest.update(block)
        assert digest.hexdigest() == expected, f'Bundle verification failed: {relative}'
    print(f"Verified {len(manifest['sha256'])} files.")
    """,
    )
    add(
        "markdown",
        """
    ## Install dependencies and check the GPU
    Keep Colab's CUDA-enabled PyTorch. If Colab requests a session restart after
    installation, restart it and run the notebook again. Do not install the Windows
    CUDA requirements in Colab.
    """,
    )
    add(
        "code",
        """
    import subprocess
    import sys

    def run_command(command):
        # Stream child errors as well as progress into Colab's cell output.
        with subprocess.Popen(command, cwd=PROJECT, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
            try:
                for line in process.stdout:
                    print(line, end='', flush=True)
                code = process.wait()
            except KeyboardInterrupt:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
        if code:
            raise RuntimeError(f'Command exited with code {code}; see the error above.')

    run_command([
        sys.executable, '-m', 'pip', 'install', '-q',
        '-r', str(PROJECT / 'requirements-colab.txt'),
    ])
    run_command([sys.executable, '-c', '''
    import torch
    from datasets import load_from_disk
    from conversation import validate_dataset_metadata
    from workflow import load_tokenizer, read_config
    from training import choose_precision
    assert torch.cuda.is_available(), 'Select a GPU runtime in Colab.'
    assert tuple(map(int, torch.__version__.split('+')[0].split('.')[:2])) >= (2, 3)
    print('GPU:', torch.cuda.get_device_name())
    print('VRAM GiB:', round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1))
    print('PyTorch:', torch.__version__, '| precision:', choose_precision('cuda', 'fp16'))
    tokenizer = load_tokenizer('tokenizer')
    model = read_config('configs/colab_t4.json').build(len(tokenizer))
    print('Parameters:', sum(p.numel() for p in model.parameters()))
    for split in ('train', 'validation', 'test'):
        path = f'data/bluemoon_{split}_tok_ds'
        validate_dataset_metadata(path, tokenizer)
        print(split, 'examples:', len(load_from_disk(path)))
    '''])
    """,
    )
    add(
        "markdown",
        """
    ## Train or resume
    Training is disabled by default because the baseline has already completed.
    Enable `RUN_TRAINING` in the setup cell only when training or resuming.
    Run this cell again after a disconnect. Initial FP16 scale adjustment may skip
    a few optimizer updates; the trainer records these in `progress.json`.
    The first checkpoint contains the initialized model; later checkpoints also
    contain AdamW moments, scheduler, loss scaler, shuffled data position, and RNG.

    To pause deliberately at a saved boundary, set `STOP_AFTER_UPDATES` to a total
    update count (for example 100). On continuation set it back to `None`.
    To change the experiment, use a new `RUN_NAME` above. Three epochs describe the
    full learning-rate schedule, even if they span multiple Colab sessions.

    If `latest.pt` is damaged, set `RESUME_FILE = 'previous.pt'` to recover the prior
    complete snapshot. Ignore unfinished `.tmp` files. Keep the entire run folder.
    """,
    )
    add(
        "code",
        """
    EPOCHS = 3
    MICRO_BATCH = 8
    GRAD_ACCUM = 4
    LEARNING_RATE = 1.5e-4
    WARMUP_UPDATES = 500
    STOP_AFTER_UPDATES = None
    RESUME_FILE = None  # None selects latest.pt, or previous.pt if latest is missing.

    if not RUN_TRAINING:
        print('Training skipped. Evaluating existing checkpoints in:', OUTPUT)
    else:
        candidates = [OUTPUT / RESUME_FILE] if RESUME_FILE else [
            OUTPUT / 'latest.pt', OUTPUT / 'previous.pt'
        ]
        resume = next((path for path in candidates if path.is_file()), None)
        if RESUME_FILE and resume is None:
            raise FileNotFoundError(candidates[0])
        command = [
            sys.executable, '-u', str(PROJECT / 'train.py'),
            '--config', 'configs/colab_t4.json', '--tokenizer', 'tokenizer',
            '--data-dir', 'data/bluemoon_train_tok_ds',
            '--validation-dir', 'data/bluemoon_validation_tok_ds',
            '--output-dir', str(OUTPUT), '--device', 'cuda', '--precision', 'fp16',
            '--epochs', str(EPOCHS), '--micro-batch', str(MICRO_BATCH),
            '--grad-accum', str(GRAD_ACCUM), '--lr', str(LEARNING_RATE),
            '--warmup-updates', str(WARMUP_UPDATES), '--seed', '42',
            '--save-every', '250', '--log-every', '10', '--cpu-threads', '2',
        ]
        if resume:
            command += ['--resume', str(resume)]
            print('Resuming:', resume)
        if STOP_AFTER_UPDATES is not None:
            command += ['--max-updates', str(STOP_AFTER_UPDATES)]
        run_command(command)
    """,
    )
    add(
        "markdown",
        """
    ## Inspect progress and an optional generated reply
    `best.pt` is selected by validation perplexity after completed epochs. Before
    then, this cell uses the latest training snapshot. Early replies may be poor.
    The single short prompt is optional; the held-out panel below is the baseline
    evaluation. Run the setup cells first after every runtime reset.
    """,
    )
    add(
        "code",
        """
    progress = OUTPUT / 'progress.json'
    if progress.exists():
        print(progress.read_text())
    checkpoint = next((OUTPUT / name for name in ('best.pt', 'latest.pt', 'previous.pt')
                       if (OUTPUT / name).is_file()), None)
    assert checkpoint is not None, f'No checkpoint in {OUTPUT}. Check RUN_NAME or enable training for a new run.'
    print('Using checkpoint:', checkpoint)
    RUN_QUICK_CHAT = False
    if RUN_QUICK_CHAT:
        run_command([
            sys.executable, '-u', str(PROJECT / 'chat.py'), '--checkpoint', str(checkpoint),
            '--device', 'cuda', '--prompt', 'The traveler knocks on the inn door. "Is anyone there?"',
            '--max-new-tokens', '128', '--temperature', '0.8', '--top-p', '0.9',
            '--top-k', '50', '--repetition-penalty', '1.05',
        ])
    """,
    )
    add(
        "markdown",
        """
    ## Fixed validation generation panel
    Generate from **50 held-out contexts × 3 seeds** using the exact prepared
    token IDs, truncation, and reply boundary. No training or test examples are used.
    This can take several minutes; progress is printed after each context.
    Use `SAMPLES = 5` for a quick check, then 50 for the baseline report.

    Each run saves `manifest.json`, `generations.jsonl`, `summary.json`, and
    `human_review.csv` to a new evaluation folder in Drive. Interrupted runs retain
    completed rows but have no final summary; rerun for a complete panel.
    Repetition scores alone do not establish quality. In the CSV, rate grammar,
    relevance to context, and internal coherence from 1 (poor) to 5 (strong),
    recording notes. Empty/short replies must not count as successful anti-repetition.
    The reference is one possible reply, not the only valid continuation.
    """,
    )
    add(
        "code",
        """
    from datetime import datetime, timezone

    SAMPLES = 50
    GENERATION_SEEDS = [42, 43, 44]
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    EVALUATION = OUTPUT / 'evaluations' / f'validation-{stamp}'
    run_command([
        sys.executable, '-u', str(PROJECT / 'evaluate_generation.py'),
        '--checkpoint', str(checkpoint), '--device', 'cuda',
        '--data-dir', 'data/bluemoon_validation_tok_ds',
        '--output-dir', str(EVALUATION), '--samples', str(SAMPLES),
        '--selection-seed', '42', '--seeds', *map(str, GENERATION_SEEDS),
        '--max-new-tokens', '128', '--temperature', '0.8',
        '--top-k', '50', '--top-p', '0.9', '--repetition-penalty', '1.05',
    ])
    print('Report:', EVALUATION)
    with (EVALUATION / 'generations.jsonl').open(encoding='utf-8') as handle:
        for line in list(handle)[:3]:
            sample = json.loads(line)
            print('Seed:', sample['seed'], '| Context index:', sample['dataset_index'])
            print('Context:', sample['context'])
            print('Reply:', sample['generation'])
            print()
    """,
    )
    add(
        "markdown",
        """
    ## Optional final test evaluation
    Use validation and generated replies while choosing the model. Enable this
    only when you have selected your final checkpoint; it evaluates all 44,508
    held-out test examples and can take a while. It is disabled for **Run all**.
    """,
    )
    add(
        "code",
        """
    RUN_FINAL_TEST = False
    if RUN_FINAL_TEST:
        assert (OUTPUT / 'best.pt').is_file(), 'Complete validation first.'
        run_command([
            sys.executable, str(PROJECT / 'evaluate.py'),
            '--checkpoint', str(OUTPUT / 'best.pt'),
            '--data-dir', 'data/bluemoon_test_tok_ds', '--batch-size', '2',
            '--device', 'cuda',
        ])
    """,
    )
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": "RolePlayChatBot_Colab.ipynb", "provenance": []},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dist")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--tokenizer", default="gpt2", help="Tokenizer name or local directory.")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    from conversation import validate_dataset_metadata
    from workflow import load_tokenizer

    tokenizer = load_tokenizer(args.tokenizer)
    tokenizer_dir = output / "colab_tokenizer"
    tokenizer.save_pretrained(tokenizer_dir)
    sources = {name: ROOT / name for name in SOURCE_FILES}
    for path in tokenizer_dir.iterdir():
        if path.is_file():
            sources[f"tokenizer/{path.name}"] = path
    for split in SPLITS:
        directory = args.data_dir / f"bluemoon_{split}_tok_ds"
        validate_dataset_metadata(directory, tokenizer)
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                sources[
                    f"data/{directory.name}/{path.relative_to(directory).as_posix()}"
                ] = path
    manifest = {
        "format_version": 1,
        "sha256": {name: sha256(path) for name, path in sources.items()},
    }
    notebook_path = output / "RolePlayChatBot_Colab.ipynb"
    notebook_path.write_text(json.dumps(notebook(), indent=2) + "\n", encoding="utf-8")
    destination = output / "RolePlayChatBot_colab.zip"
    temporary = destination.with_suffix(".zip.tmp")
    with zipfile.ZipFile(
        temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=4
    ) as archive:
        for name, path in sources.items():
            archive.write(path, name)
        archive.writestr("bundle_manifest.json", json.dumps(manifest, indent=2) + "\n")
    with zipfile.ZipFile(temporary) as archive:
        if bad_file := archive.testzip():
            raise RuntimeError(f"Archive integrity failure: {bad_file}")
    temporary.replace(destination)
    print(f"Notebook: {notebook_path}")
    print(f"Bundle: {destination} ({destination.stat().st_size / 1024**2:.1f} MiB)")
    print(
        f"Packaged and verified {len(sources)} files. No weights or environment included."
    )


if __name__ == "__main__":
    main()
