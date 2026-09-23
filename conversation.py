"""One prompt/token boundary convention shared by training and generation."""

import hashlib
import json
from pathlib import Path

DATA_FORMAT_VERSION = 2


def format_messages(messages):
    if not messages or any(
        not isinstance(message, str) or not message.strip() for message in messages
    ):
        raise ValueError("A conversation must contain nonempty text messages.")
    return "\n\n".join(message.strip() for message in messages)


def encode_context(text, tokenizer, max_length=None):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("The prompt must contain nonempty text.")
    if tokenizer.eos_token_id is None:
        raise ValueError("The tokenizer must define an EOS token.")
    ids = tokenizer(text.strip(), add_special_tokens=False)["input_ids"]
    # EOS separates the last context message from the answer. It is a real input
    # token even when the tokenizer also uses EOS as its padding ID.
    ids.append(tokenizer.eos_token_id)
    if max_length is not None:
        if max_length < 2:
            raise ValueError("Context length must allow text plus a boundary token.")
        ids = ids[-max_length:]  # preserve the most recent context
    return ids


def tokenizer_fingerprint(tokenizer):
    payload = {
        "vocab": tokenizer.get_vocab(),
        "eos": tokenizer.eos_token_id,
        "pad": tokenizer.pad_token_id,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def write_dataset_metadata(path, tokenizer):
    metadata = {
        "format_version": DATA_FORMAT_VERSION,
        "tokenizer_fingerprint": tokenizer_fingerprint(tokenizer),
    }
    (Path(path) / "roleplay_format.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def validate_dataset_metadata(path, tokenizer):
    metadata_path = Path(path) / "roleplay_format.json"
    if not metadata_path.is_file():
        raise ValueError(
            "Dataset is missing format metadata. Rebuild it using prepare_data.py."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format_version") != DATA_FORMAT_VERSION:
        raise ValueError(
            "Incompatible dataset format. Rebuild it using prepare_data.py."
        )
    if metadata.get("tokenizer_fingerprint") != tokenizer_fingerprint(tokenizer):
        raise ValueError(
            "Dataset tokenizer differs from the model tokenizer; rebuild the dataset."
        )
