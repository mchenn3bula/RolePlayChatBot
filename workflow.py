"""Shared configuration and checkpoint loading for the command-line scripts."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from transformers import AutoTokenizer

from mini_deepseek import RolePlayTransformer

ARCHITECTURE_VERSION = 2


@dataclass
class ModelConfig:
    """Versioned causal decoder configuration; dense unless MoE is requested."""

    architecture_version: int = ARCHITECTURE_VERSION
    max_len: int = 3072
    d_model: int = 768
    n_layers: int = 10
    n_heads: int = 12
    d_ff: int = 3072
    dropout: float = 0.1
    moe_start: int | None = None
    lambda_lb: float = 0.01
    gradient_checkpointing: bool = True
    position_encoding: str = "learned"
    ffn_type: str = "gelu"
    rope_theta: float = 10000.0
    target_only_projection: bool = False

    def build(self, vocab: int) -> RolePlayTransformer:
        self.validate()
        config = asdict(self)
        config.pop("architecture_version")
        return RolePlayTransformer(vocab=vocab, **config)

    def validate(self):
        if self.architecture_version not in (2, 3):
            raise ValueError("Unsupported model architecture; expected version 2 or 3.")
        if self.architecture_version == 2 and (
            self.position_encoding != "learned"
            or self.ffn_type != "gelu"
            or self.target_only_projection
            or self.rope_theta != 10000.0
        ):
            raise ValueError(
                "Version 2 preserves learned positions and GELU; use version 3 for experiments."
            )
        if self.architecture_version == 3 and (
            self.position_encoding != "rope"
            or self.ffn_type != "swiglu"
            or self.moe_start is not None
        ):
            raise ValueError("Version 3 requires dense RoPE + SwiGLU.")

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")


def resolve_device(name: str = "auto") -> str:
    return ("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else name


def load_tokenizer(name: str = "gpt2"):
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def read_config(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("architecture_version") not in (2, 3):
        raise ValueError(
            "Legacy or unsupported architecture. Old compressed-attention weights cannot be used; retrain the corrected model."
        )
    config = ModelConfig(**payload)
    config.validate()
    return config


def load_checkpoint(checkpoint: str, device: str = "auto"):
    """Load corrected weights with their exact saved configuration and tokenizer."""
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}. Run train.py first.")
    config_path = path.parent / "config.json"
    if not config_path.is_file():
        raise ValueError(
            "Checkpoint config.json is missing. Legacy checkpoints require retraining."
        )
    config = read_config(config_path)
    tokenizer_path = path.parent / "tokenizer"
    if not tokenizer_path.is_dir():
        raise ValueError(
            "Checkpoint tokenizer directory is missing; restore the complete checkpoint folder."
        )
    tokenizer = load_tokenizer(str(tokenizer_path))
    device = resolve_device(device)
    model = config.build(len(tokenizer))
    state = torch.load(path, map_location="cpu", weights_only=True)
    if "checkpoint_version" in state:
        from training import CHECKPOINT_VERSION

        if state["checkpoint_version"] != CHECKPOINT_VERSION:
            raise ValueError("Unsupported training checkpoint version.")
        state = state["model"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, tokenizer, device
