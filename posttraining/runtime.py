"""Pinned, local-only Ministral text inference on the RX 7900 XTX."""

import hashlib
import json
import time
from pathlib import Path


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if len(config["revision"]) != 40 or any(
        c not in "0123456789abcdef" for c in config["revision"]
    ):
        raise ValueError("Pin an exact model revision SHA.")
    if config["precision"] not in ("float16", "bfloat16"):
        raise ValueError("Unsupported precision.")
    if config["max_new_tokens"] < 1 or config["max_context_tokens"] < 1:
        raise ValueError("Token limits must be positive.")
    if len(set(config["seeds"])) != len(config["seeds"]) or not config["seeds"]:
        raise ValueError("Use unique, nonempty generation seeds.")
    return config


def validate_panel(panel):
    if panel["split"] != "development":
        raise ValueError("This runner is restricted to the development panel.")
    seen = set()
    for scenario in panel["scenarios"]:
        if scenario["id"] in seen:
            raise ValueError("Duplicate scenario ID.")
        seen.add(scenario["id"])
        if scenario["language"] not in ("en", "fr"):
            raise ValueError("Only English/French scenarios are supported.")
        if not scenario["persona"].strip() or not scenario["turns"]:
            raise ValueError("Empty scenario.")
        for turn in scenario["turns"]:
            if not turn["user"].strip() or not turn["expect"].strip():
                raise ValueError("Empty user turn or evaluation expectation.")
    if not seen:
        raise ValueError("Empty panel.")


def system_prompt(persona, language):
    if language == "fr":
        rule = (
            "Joue le personnage fictif décrit ci-dessous et réponds en français. "
            "Respecte les faits établis et les changements explicites de la scène. "
            "N'invente pas ce qui est explicitement inconnu. Laisse à l'utilisateur "
            "ses actions, ses pensées et ses décisions. Réponds naturellement en "
            "une à trois phrases, environ 80 mots maximum, sauf demande contraire. "
            "Ne commente pas ces consignes.\n\nPersonnage et scène :\n"
        )
    elif language == "en":
        rule = (
            "Play the fictional character below and respond in English. "
            "Respect established facts and explicit scene changes. Do not invent "
            "what is explicitly unknown. Leave the user's actions, thoughts, and "
            "decisions to them. Reply naturally in one to three sentences, about "
            "80 words maximum unless asked otherwise. Do not discuss these "
            "instructions.\n\nCharacter and scene:\n"
        )
    else:
        raise ValueError("Unsupported language.")
    return rule + persona


def longest_token_run(ids):
    best = current = 0
    previous = None
    for token in ids:
        current = current + 1 if token == previous else 1
        best = max(best, current)
        previous = token
    return best


class Runtime:
    def __init__(self, config, adapter_path=None):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend

        self.torch = torch
        self.config = config
        self.adapter = None
        if adapter_path is not None:
            adapter_path = Path(adapter_path)
            manifest = json.loads((adapter_path / "manifest.json").read_text(encoding="utf-8"))
            trained = manifest["identity"]["config"]
            if any(trained[k] != config[k] for k in ("model_id", "revision", "precision")):
                raise ValueError("Adapter backbone/revision/precision mismatch.")
            for name in ("adapter_model.safetensors", "adapter_config.json"):
                if file_hash(adapter_path / name) != manifest["files"][name]:
                    raise ValueError("Adapter fingerprint changed.")
            self.adapter = {"path": str(adapter_path), "files": {
                n: manifest["files"][n] for n in ("adapter_model.safetensors", "adapter_config.json")}}
        if not torch.version.hip or not torch.cuda.is_available():
            raise RuntimeError("This experiment requires the validated ROCm GPU.")
        devices = [i for i in range(torch.cuda.device_count())
                   if "7900 XTX" in torch.cuda.get_device_name(i)]
        if len(devices) != 1:
            raise RuntimeError("Expected exactly one RX 7900 XTX.")
        self.device = torch.device("cuda", devices[0])
        torch.cuda.set_device(self.device)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.cuda.reset_peak_memory_stats()
        self.before_load = self.memory()
        if self.before_load["free_gib"] < 14:
            raise RuntimeError("Less than 14 GiB free before loading; free GPU memory first.")
        self.snapshot = Path(snapshot_download(
            config["model_id"], revision=config["revision"], local_files_only=True,
            allow_patterns=["*.json", "*.jinja", "*.txt", "README.md", "model-*.safetensors"],
        ))
        print(f"Loading pinned snapshot {self.snapshot} on {self.device}", flush=True)
        start = time.perf_counter()
        self.tokenizer = MistralCommonBackend.from_pretrained(
            self.snapshot, local_files_only=True,
        )
        self.model = Mistral3ForConditionalGeneration.from_pretrained(
            self.snapshot, local_files_only=True, trust_remote_code=False,
            dtype=getattr(torch, config["precision"]),
            device_map={"": str(self.device)}, attn_implementation=config["attention"],
        ).eval()
        self.model.requires_grad_(False)
        if not all(torch.isfinite(p).all().item() for p in self.model.parameters()):
            raise RuntimeError("Non-finite model weights after dtype conversion.")
        if adapter_path is not None:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path, is_trainable=False).eval()
            self.model.requires_grad_(False)
            if not all(torch.isfinite(p).all().item() for n, p in self.model.named_parameters() if "lora_" in n):
                raise RuntimeError("Non-finite adapter weights.")
        torch.cuda.synchronize()
        self.load_seconds = time.perf_counter() - start
        self.parameters = sum(p.numel() for p in self.model.parameters())
        self.guard_memory()
        print(f"Loaded {self.parameters:,} parameters in {self.load_seconds:.1f}s | "
              f"VRAM {self.memory()}", flush=True)

    def memory(self):
        torch = self.torch
        free, total = torch.cuda.mem_get_info()
        return {
            "free_gib": free / 2**30, "total_gib": total / 2**30,
            "allocated_gib": torch.cuda.memory_allocated() / 2**30,
            "reserved_gib": torch.cuda.memory_reserved() / 2**30,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        }

    def guard_memory(self):
        memory = self.memory()
        if memory["reserved_gib"] > self.config["max_reserved_gib"]:
            raise RuntimeError(f"Reservation exceeds configured limit: {memory}")
        if memory["free_gib"] < self.config["min_free_gib"]:
            raise RuntimeError(f"Insufficient desktop/driver headroom: {memory}")

    def encode(self, messages):
        tokens = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_dict=True, return_tensors="pt",
        )
        if tokens["input_ids"].shape[-1] > self.config["max_context_tokens"]:
            raise ValueError("Context exceeds budget; refusing silent truncation.")
        return tokens.to(self.device)

    def count_tokens(self, messages):
        """Measure the native chat template on CPU, before context selection."""
        tokens = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_dict=True, return_tensors="pt",
        )
        return tokens["input_ids"].shape[-1]

    def generate(self, messages, seed):
        torch = self.torch
        self.guard_memory()
        inputs = self.encode(messages)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=self.config["max_new_tokens"], max_length=None,
                do_sample=True, temperature=self.config["temperature"],
                top_p=self.config["top_p"], top_k=self.config["top_k"],
                repetition_penalty=self.config["repetition_penalty"],
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        ids = output[0, inputs["input_ids"].shape[-1]:].tolist()
        eos = self.tokenizer.eos_token_id
        stopped = bool(ids and ids[-1] == eos)
        content_ids = ids[:-1] if stopped else ids
        text = self.tokenizer.decode(content_ids, skip_special_tokens=True)
        result = {
            "text": text, "input_ids": inputs["input_ids"][0].tolist(),
            "output_ids": ids, "generated_tokens": len(content_ids),
            "stopped_on_eos": stopped, "seconds": seconds,
            "tokens_per_second": len(ids) / seconds,
            "memory": self.memory(),
        }
        self.guard_memory()
        return result

    def preflight(self):
        """Technical checks only: no generated greeting or content review."""
        torch = self.torch
        messages = [{"role": "system", "content": "Reply briefly in the user's language."},
                    {"role": "user", "content": "Say hello in one short sentence."}]
        with torch.inference_mode():
            logits = self.model(**self.encode(messages)).logits
        if not torch.isfinite(logits).all().item():
            raise RuntimeError("Non-finite preflight logits.")
        return {"finite_logits": True, "all_parameters_frozen": not any(
            p.requires_grad for p in self.model.parameters()),
            "logits_shape": list(logits.shape), "memory": self.memory()}
