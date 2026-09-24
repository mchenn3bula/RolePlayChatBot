"""Download pinned HF-format weights only; no duplicate consolidated weights."""

import argparse
import json
from pathlib import Path


def main():
    from huggingface_hub import snapshot_download

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/ministral_p0.json"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    print(f"Downloading {config['model_id']} @ {config['revision']}", flush=True)
    path = snapshot_download(
        config["model_id"], revision=config["revision"], max_workers=3,
        allow_patterns=["*.json", "*.jinja", "*.txt", "README.md", "model-*.safetensors"],
    )
    print(f"Pinned model cache: {path}", flush=True)


if __name__ == "__main__":
    main()
