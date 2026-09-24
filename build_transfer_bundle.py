"""Build a personal portable snapshot with datasets and Codex project context."""

import argparse
import json
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from build_colab_bundle import ROOT, sha256

ROOT_FILES = (
    ".gitignore",
    "AGENTS.md",
    "CODEX_HANDOFF.md",
    "TRANSFER_README.md",
    "README.md",
    "BASELINE.md",
    "COLAB_TRAINING.md",
    "LOCAL_TRAINING.md",
    "AMD_TRAINING.md",
    "ARCHITECTURE_REVIEW.md",
    "OVERNIGHT_BASELINE.md",
    "MODERN_BASELINE.md",
    "GENERATION_EVALUATION.md",
    "RESEARCH_PLAN.md",
    "LITERATURE_REVIEW.md",
    "MINISTRAL_EVALUATION.md",
    "PERSONA_SCENE_STATE.md",
    "LORA_TRAINING.md",
    "DPO_COMPARISON.md",
    "ADVANCED_METHOD_DECISION.md",
    "BILINGUAL_CONTROL_PROTOCOL.md",
    "BILINGUAL_CONTROL_RESULTS.md",
    "STATE_FORMAT_PROTOCOL.md",
    "STATE_FORMAT_RESULTS.md",
    "NATURAL_REFERENCE_RESEARCH.md",
    "NATURAL_REFERENCE_PROTOCOL.md",
    "NATURAL_REFERENCE_RESULTS.md",
    "RESPONSE_REFINEMENT_PROTOCOL.md",
    "RESPONSE_REFINEMENT_RESULTS.md",
    "evaluate_response_refinement.ps1",
    "run_response_refinement.sh",
    "evaluate_natural_reference.ps1",
    "run_natural_reference.sh",
    "evaluate_state_format.ps1",
    "run_state_format.sh",
    "train_bilingual_control.ps1",
    "train_bilingual_pair.ps1",
    "run_bilingual_control.sh",
    "DPO_COMPARISON_PROTOCOL.md",
    "DPO_PROTOCOL_ADDENDUM.md",
    "train_dpo.ps1",
    "run_dpo.sh",
    "train_lora.ps1",
    "run_lora.sh",
    "setup_lora_env.sh",
    "requirements-lora-rocm.txt",
    "requirements-lora-rocm.lock.txt",
    "ministral.ps1",
    "setup_ministral_env.sh",
    "run_ministral.sh",
    "chat_ministral.sh",
    "requirements-ministral-rocm.txt",
    "requirements-ministral-rocm.lock.txt",
    "baseline.py",
    "benchmark.py",
    "build_colab_bundle.py",
    "build_transfer_bundle.py",
    "chat.py",
    "conversation.py",
    "evaluate.py",
    "evaluate_generation.py",
    "markov_baseline.py",
    "mini_deepseek.py",
    "prepare_data.py",
    "train.py",
    "training.py",
    "workflow.py",
    "train_local.ps1",
    "train_amd.ps1",
    "run_amd.sh",
    "setup_amd_system.sh",
    "setup_amd_env.sh",
    "amd_run.py",
    "smoke_amd.py",
    "requirements-rocm.txt",
    "requirements.txt",
    "requirements-cuda.txt",
    "requirements-colab.txt",
    "mini_deepseek.pdf",
    "chatbot_roleplay.drawio.png",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "dist/RolePlayChatBot_transfer.zip"
    )
    args = parser.parse_args()
    sources = {name: ROOT / name for name in ROOT_FILES}
    for directory, pattern in (
        ("configs", "*.json"),
        ("tests", "test_*.py"),
        ("notebooks", "*.ipynb"),
        ("posttraining", "*.py"),
        ("posttraining", "*.json"),
        ("posttraining/personas", "*.json"),
    ):
        for path in sorted((ROOT / directory).glob(pattern)):
            sources[path.relative_to(ROOT).as_posix()] = path
    sources["dist/RolePlayChatBot_Colab.ipynb"] = (
        ROOT / "dist/RolePlayChatBot_Colab.ipynb"
    )
    # Personal transfer only: keep whole-thread LoRA and synthetic DPO data reproducible.
    # Dialogue stays excluded from Git, and is never printed or interpreted here.
    for data_name in ("ministral-s1-whole-v1", "ministral-d1-preferences-v1", "ministral-preferences-v2"):
        lora_data = ROOT / "data" / data_name
        if not lora_data.is_dir():
            continue
        manifest_path = lora_data / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        sources[manifest_path.relative_to(ROOT).as_posix()] = manifest_path
        for name, expected in manifest["files"].items():
            path = lora_data / name
            if not path.resolve().is_relative_to(lora_data.resolve()) or sha256(path) != expected:
                raise ValueError("Prepared adapter dataset path/fingerprint mismatch.")
            sources[path.relative_to(ROOT).as_posix()] = path
    # Preserve v2 curation provenance in personal transfers, including unused candidates.
    for directory, pattern in (
        ("data/ministral-preferences-v2", "draft*"),
        ("data/ministral-preferences-v2/source_snapshot", "*"),
        ("reports/ministral-preferences-v2-curation", "*.json"),
        ("reports/ministral-preferences-v2-curation", "*.jsonl"),
        ("reports/ministral-preferences-v2-curation", "*.html"),
        ("reports/ministral-state-format-v1", "*.json"),
        ("reports/ministral-state-format-v1", "*.jsonl"),
        ("reports/ministral-state-format-v1", "*.html"),
        ("reports/ministral-natural-reference-v1", "*.json"),
        ("reports/ministral-natural-reference-v1", "*.jsonl"),
        ("reports/ministral-natural-reference-v1", "*.html"),
        ("reports/ministral-response-refinement-v1", "*.json"),
        ("reports/ministral-response-refinement-v1", "*.jsonl"),
        ("reports/ministral-response-refinement-v1", "*.html"),
    ):
        for path in sorted((ROOT / directory).glob(pattern)):
            sources[path.relative_to(ROOT).as_posix()] = path
    for path in sorted((ROOT / "dist/colab_tokenizer").glob("*")):
        if path.is_file() and path.suffix in (".json", ".txt", ".model"):
            sources[f"tokenizer/{path.name}"] = path
    if "tokenizer/tokenizer_config.json" not in sources:
        raise FileNotFoundError(
            "Run build_colab_bundle.py first to prepare the tokenizer."
        )
    for split in ("train", "validation", "test"):
        for suffix in ("ds", "tok_ds"):
            directory = ROOT / "data" / f"bluemoon_{split}_{suffix}"
            if not directory.is_dir():
                raise FileNotFoundError(directory)
            state = json.loads((directory / "state.json").read_text(encoding="utf-8"))
            names = {"state.json", "dataset_info.json"}
            names.update(item["filename"] for item in state["_data_files"])
            if suffix == "tok_ds":
                names.add("roleplay_format.json")
            for name in sorted(names):
                path = directory / name
                if not path.resolve().is_relative_to(directory.resolve()):
                    raise ValueError(f"Dataset references a nonlocal file: {name}")
                sources[path.relative_to(ROOT).as_posix()] = path
    for name, path in sources.items():
        if (
            not path.is_file()
            or path.is_symlink()
            or not path.resolve().is_relative_to(ROOT)
        ):
            raise ValueError(f"Missing or nonlocal payload: {name}")
    output = args.output.resolve()
    if output in (path.resolve() for path in sources.values()):
        raise ValueError("Output must not replace a payload file.")
    try:
        git = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        git_commit = git.stdout.strip() if git.returncode == 0 else None
    except FileNotFoundError:
        git_commit = None
    manifest = {
        "format_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "original_remote": "https://github.com/mchenn3bula/RolePlayChatBot.git",
        "base_git_commit": git_commit,
        "snapshot": "Current working files, including uncommitted changes; not a Git history export.",
        "checkpoint_status": "Completed baseline weights are on the user's Google Drive and are not included.",
        "sha256": {name: sha256(path) for name, path in sources.items()},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".zip.tmp")
    with zipfile.ZipFile(
        temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=4
    ) as archive:
        for index, (name, path) in enumerate(sources.items(), 1):
            print(f"Packing {index}/{len(sources)}: {name}", flush=True)
            archive.write(path, f"RolePlayChatBot/{name}")
        archive.writestr(
            "RolePlayChatBot/transfer_manifest.json",
            json.dumps(manifest, indent=2) + "\n",
        )
    print("Verifying every packaged file...", flush=True)
    import hashlib

    with zipfile.ZipFile(temporary) as archive:
        for name, expected in manifest["sha256"].items():
            digest = hashlib.sha256()
            with archive.open(f"RolePlayChatBot/{name}") as handle:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != expected:
                raise RuntimeError(f"Archive verification failed: {name}")
    temporary.replace(output)
    print(f"Verified {len(sources)} payload files.", flush=True)
    print(
        f"Transfer ZIP: {output} ({output.stat().st_size / 1024**2:.1f} MiB)",
        flush=True,
    )


if __name__ == "__main__":
    main()
