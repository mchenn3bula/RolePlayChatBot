"""Causal decoder language model with an optional sparse mixture of experts.

The pre-v2 sequence-compressed attention and duplicate MTP head are deliberately
removed. Old checkpoints are not compatible with this architecture.
"""

import math

import torch
from datasets import load_from_disk
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from conversation import encode_context, format_messages, validate_dataset_metadata


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Accumulate squares in float32 even during mixed-precision training.
        normalized = x.float() * torch.rsqrt(
            x.float().square().mean(-1, keepdim=True) + self.eps
        )
        return normalized.to(x.dtype) * self.weight


class MultiHeadAttention(nn.Module):
    """Standard self-attention; no mixing across positions before masking."""

    def __init__(self, dim, heads, p=0.1, rotary=None):
        super().__init__()
        if heads < 1 or dim % heads:
            raise ValueError("d_model must be divisible by n_heads.")
        self.h = heads
        self.dh = dim // heads
        self.dropout = p
        self.rotary = rotary
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, mask, positions=None):
        batch, length, dim = x.shape
        q, k, v = (
            self.qkv(x).view(batch, length, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        )
        if self.rotary is not None:
            q, k = self.rotary(q, positions), self.rotary(k, positions)
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.o_proj(output.transpose(1, 2).reshape(batch, length, dim))


class RotaryEmbedding(nn.Module):
    """Interleaved RoPE with positions counting valid tokens, including left padding."""

    def __init__(self, head_dim, max_len, theta=10000.0):
        super().__init__()
        if head_dim % 2 or not math.isfinite(theta) or theta <= 1:
            raise ValueError("RoPE requires an even head dimension and theta > 1.")
        frequencies = theta ** (-torch.arange(0, head_dim, 2).float() / head_dim)
        angles = torch.arange(max_len).float()[:, None] * frequencies[None, :]
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x, positions):
        # FP32 rotations preserve accuracy during FP16 autocast.
        cos = self.cos[positions][:, None].float()
        sin = self.sin[positions][:, None].float()
        even, odd = x.float()[..., 0::2], x.float()[..., 1::2]
        rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), -1)
        return rotated.flatten(-2).to(x.dtype)


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden, p=0.1):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        hidden = self.dropout(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return self.dropout(self.down_proj(hidden))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden, p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden, bias=False),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(hidden, dim, bias=False),
            nn.Dropout(p),
        )

    def forward(self, x):
        return self.net(x)


class MoEBlock(nn.Module):
    """Top-k routing with a differentiable, padding-aware balance objective."""

    def __init__(self, dim, hidden, n_experts=4, top_k=2, p=0.1):
        super().__init__()
        if not 1 <= top_k <= n_experts:
            raise ValueError("Require 1 <= top_k <= n_experts.")
        self.n_experts = n_experts
        self.top_k = top_k
        self.router = nn.Linear(dim, n_experts, bias=False)
        self.experts = nn.ModuleList(
            [FeedForward(dim, hidden, p) for _ in range(n_experts)]
        )

    def forward(self, x, token_mask=None):
        batch, length, dim = x.shape
        flat = x.reshape(-1, dim)
        valid = (
            torch.ones(batch * length, dtype=torch.bool, device=x.device)
            if token_mask is None
            else token_mask.reshape(-1).bool()
        )
        probabilities = self.router(flat).float().softmax(-1)
        weights, indices = probabilities.topk(self.top_k, dim=-1)
        if self.top_k > 1:
            weights = weights / weights.sum(-1, keepdim=True)
        output = torch.zeros_like(flat)
        for expert_index, expert in enumerate(self.experts):
            tokens, slots = ((indices == expert_index) & valid[:, None]).nonzero(
                as_tuple=True
            )
            if tokens.numel():
                routed = expert(flat[tokens]) * weights[tokens, slots, None].to(x.dtype)
                output = output.index_add(0, tokens, routed)
        # Hard assignment counts are constants; soft router probabilities keep gradients.
        assignment = F.one_hot(indices, self.n_experts).float().sum(1)
        valid_float = valid.float().unsqueeze(-1)
        count = valid_float.sum().clamp_min(1)
        importance = (probabilities * valid_float).sum(0) / count
        load = (assignment * valid_float).sum(0) / (count * self.top_k)
        balance = self.n_experts * (importance * load.detach()).sum()
        return output.view(batch, length, dim), balance


class TransformerBlock(nn.Module):
    def __init__(
        self, dim, heads, hidden, use_moe, p=0.1, rotary=None, ffn_type="gelu"
    ):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = MultiHeadAttention(dim, heads, p, rotary)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = (
            MoEBlock(dim, hidden, p=p)
            if use_moe
            else (
                SwiGLU(dim, hidden, p)
                if ffn_type == "swiglu"
                else FeedForward(dim, hidden, p)
            )
        )
        self.use_moe = use_moe

    def forward(self, x, mask, token_mask, positions=None):
        x = x + self.attn(self.attn_norm(x), mask, positions)
        if self.use_moe:
            output, balance = self.ffn(self.ffn_norm(x), token_mask)
        else:
            output, balance = self.ffn(self.ffn_norm(x)), x.new_zeros(())
        return x + output, balance


def next_token_loss(logits, labels):
    """Summed target-only negative log likelihood and its exact token count."""
    if logits.shape[:2] != labels.shape or labels.shape[1] < 2:
        raise ValueError(
            "Logits and labels must have matching batch/sequence shapes of length >= 2."
        )
    targets = labels[:, 1:]
    count = (targets != -100).sum()
    if count.item() == 0:
        raise ValueError("Batch contains no target tokens.")
    loss = F.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    return loss, count


class RolePlayTransformer(nn.Module):
    def __init__(
        self,
        vocab,
        max_len=3072,
        d_model=768,
        n_layers=10,
        n_heads=12,
        d_ff=3072,
        dropout=0.1,
        moe_start=None,
        lambda_lb=0.01,
        gradient_checkpointing=True,
        position_encoding="learned",
        ffn_type="gelu",
        rope_theta=10000.0,
        target_only_projection=False,
    ):
        super().__init__()
        if min(vocab, max_len, d_model, n_layers, n_heads, d_ff) < 1:
            raise ValueError("Model dimensions must be positive.")
        if d_model % n_heads or not 0 <= dropout < 1 or lambda_lb < 0:
            raise ValueError("Invalid head dimensions, dropout, or balance weight.")
        if moe_start is not None and not 0 <= moe_start < n_layers:
            raise ValueError("moe_start must be None or a zero-based layer index.")
        if position_encoding not in {"learned", "rope"} or ffn_type not in {
            "gelu",
            "swiglu",
        }:
            raise ValueError("Unsupported position encoding or feed-forward type.")
        if moe_start is not None and ffn_type != "gelu":
            raise ValueError("SwiGLU experiments use dense blocks; MoE is unsupported.")
        self.vocab = vocab
        self.max_len = max_len
        self.target_only_projection = target_only_projection
        self.variant = {
            "position_encoding": position_encoding,
            "ffn_type": ffn_type,
            "rope_theta": rope_theta,
            "target_only_projection": target_only_projection,
        }
        self.gc = gradient_checkpointing
        self.lambda_lb = lambda_lb
        self.embed = nn.Embedding(vocab, d_model)
        self.pos = (
            nn.Parameter(torch.empty(1, max_len, d_model))
            if position_encoding == "learned"
            else None
        )
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    n_heads,
                    d_ff,
                    moe_start is not None and i >= moe_start,
                    dropout,
                    RotaryEmbedding(d_model // n_heads, max_len, rope_theta)
                    if position_encoding == "rope"
                    else None,
                    ffn_type,
                )
                for i in range(n_layers)
            ]
        )
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab, bias=False)
        self.apply(self._initialize)
        if self.pos is not None:
            nn.init.normal_(self.pos, std=0.02)
        # Tied embeddings need a small initialization, not unit-variance embeddings.
        self.lm_head.weight = self.embed.weight
        for block in self.layers:
            nn.init.normal_(
                block.attn.o_proj.weight, std=0.02 / math.sqrt(2 * n_layers)
            )
            experts = block.ffn.experts if block.use_moe else [block.ffn]
            for expert in experts:
                nn.init.normal_(
                    (
                        expert.down_proj
                        if isinstance(expert, SwiGLU)
                        else expert.net[3]
                    ).weight,
                    std=0.02 / math.sqrt(2 * n_layers),
                )

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, ids, attention_mask=None, labels=None):
        if ids.ndim != 2 or not 1 <= ids.size(1) <= self.max_len:
            raise ValueError(
                f"Expected (batch, sequence) IDs with 1..{self.max_len} tokens."
            )
        valid = (
            torch.ones_like(ids, dtype=torch.bool)
            if attention_mask is None
            else attention_mask.to(device=ids.device, dtype=torch.bool)
        )
        if valid.shape != ids.shape:
            raise ValueError("attention_mask must have the same shape as ids.")
        positions = (valid.long().cumsum(-1) - 1).clamp_min(0)
        x = self.embed(ids)
        if self.pos is not None:
            x = x + self.pos[0, positions]
        length = ids.size(1)
        causal = torch.ones(length, length, dtype=torch.bool, device=ids.device).tril()
        mask = causal[None, None] & valid[:, None, None, :]
        balance = x.new_zeros(())
        moe_layers = 0
        for block in self.layers:
            if self.gc and self.training:
                x, auxiliary = checkpoint(
                    block, x, mask, valid, positions, use_reentrant=False
                )
            else:
                x, auxiliary = block(x, mask, valid, positions)
            balance = balance + auxiliary
            moe_layers += int(block.use_moe)
        balance = balance / max(moe_layers, 1)
        if labels is not None:
            if labels.shape != ids.shape or ids.size(1) < 2:
                raise ValueError("Labels must match IDs with sequence length >= 2.")
            targets = labels[:, 1:]
            selected = targets != -100
            count = selected.sum()
            if count.item() == 0:
                raise ValueError("Batch contains no target tokens.")
            # Select states BEFORE each target, preserving the one-token shift.
            logits = self.lm_head(self.norm(x[:, :-1][selected]))
            loss = F.cross_entropy(logits.float(), targets[selected], reduction="sum")
            return {"loss_sum": loss, "target_count": count, "lb": balance}
        return {"main": self.lm_head(self.norm(x)), "lb": balance}

    def forward_loss(self, ids, attention_mask, labels):
        if self.target_only_projection:
            return self(ids, attention_mask=attention_mask, labels=labels)
        output = self(ids, attention_mask=attention_mask)
        loss, count = next_token_loss(output["main"], labels)
        return {"loss_sum": loss, "target_count": count, "lb": output["lb"]}

    @staticmethod
    def make_collate_fn(tokenizer):
        if tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must define a padding token.")

        def collate(batch):
            inputs, labels = [], []
            for example in batch:
                context = torch.tensor(example["input_ids"], dtype=torch.long)
                target = torch.tensor(example["labels"], dtype=torch.long)
                if not context.numel() or not target.numel():
                    raise ValueError(
                        "Every example needs nonempty context and target tokens."
                    )
                inputs.append(torch.cat([context, target]))
                labels.append(torch.cat([torch.full_like(context, -100), target]))
            padded = nn.utils.rnn.pad_sequence(
                inputs, batch_first=True, padding_value=tokenizer.pad_token_id
            )
            labels = nn.utils.rnn.pad_sequence(
                labels, batch_first=True, padding_value=-100
            )
            lengths = torch.tensor([len(sequence) for sequence in inputs])
            mask = torch.arange(padded.size(1))[None] < lengths[:, None]
            return padded, mask, labels

        return collate

    def train_model(self, *args, **kwargs):
        """Train with mixed precision and resumable checkpoints; see training.py."""
        from training import train_model

        return train_model(self, *args, **kwargs)

    def _save_weights(self, path):
        temporary = path.with_suffix(".pt.tmp")
        torch.save(self.state_dict(), temporary)
        temporary.replace(path)

    @torch.no_grad()
    def evaluate_perplexity(
        self, dataset_path, tokenizer, device=None, batch_size=8, label="Test"
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        device = device or next(self.parameters()).device
        self.to(device)
        validate_dataset_metadata(dataset_path, tokenizer)
        loader = DataLoader(
            load_from_disk(dataset_path),
            batch_size=batch_size,
            collate_fn=self.make_collate_fn(tokenizer),
        )
        was_training = self.training
        total_loss, total_tokens = 0.0, 0
        self.eval()
        try:
            print(f"{label}: evaluating {len(loader.dataset)} examples...", flush=True)
            for batch_index, (ids, mask, labels) in enumerate(loader, 1):
                ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
                output = self.forward_loss(ids, mask, labels)
                loss, count = output["loss_sum"], output["target_count"]
                total_loss += loss.item()
                total_tokens += count.item()
                if batch_index % 100 == 0 or batch_index == len(loader):
                    print(
                        f"{label}: batch {batch_index}/{len(loader)} | target NLL {total_loss / total_tokens:.4f}",
                        flush=True,
                    )
        finally:
            self.train(was_training)
        if not total_tokens:
            raise ValueError("Evaluation dataset contains no target tokens.")
        mean_loss = total_loss / total_tokens
        perplexity = math.exp(mean_loss) if mean_loss < 709 else math.inf
        print(
            f"{label} perplexity: {perplexity:.4f} | target NLL: {mean_loss:.4f} | tokens: {total_tokens}"
        )
        return perplexity

    @staticmethod
    def top_k_top_p_filtering(logits, top_k=0, top_p=1.0):
        if top_k < 0 or not 0 < top_p <= 1:
            raise ValueError("Require top_k >= 0 and 0 < top_p <= 1.")
        filtered = logits.clone()
        if top_k:
            threshold = filtered.topk(min(top_k, filtered.size(-1)), dim=-1).values[
                ..., -1, None
            ]
            filtered = filtered.masked_fill(filtered < threshold, -torch.inf)
        if top_p < 1:
            sorted_logits, sorted_indices = filtered.sort(dim=-1, descending=True)
            remove = sorted_logits.float().softmax(-1).cumsum(-1) > top_p
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            # Scatter independently for every row; flattening mixes batch indices.
            remove = torch.zeros_like(remove).scatter(-1, sorted_indices, remove)
            filtered = filtered.masked_fill(remove, -torch.inf)
        return filtered

    @staticmethod
    def apply_repetition_penalty(logits, previous_ids, penalty):
        if not math.isfinite(penalty) or penalty <= 0:
            raise ValueError("repetition_penalty must be finite and positive.")
        result = logits.clone()
        for row in range(logits.size(0)):
            seen = previous_ids[row].unique()
            scores = result[row, seen]
            result[row, seen] = torch.where(
                scores < 0, scores * penalty, scores / penalty
            )
        return result

    @torch.no_grad()
    def generate(
        self,
        input_text,
        max_new_tokens=50,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        repetition_penalty=1.0,
        stop_token=None,
        device=None,
        tokenizer=None,
        return_full_text=False,
    ):
        tokenizer = (
            tokenizer
            if tokenizer is not None
            else AutoTokenizer.from_pretrained("gpt2")
        )
        stop_token = tokenizer.eos_token_id if stop_token is None else stop_token
        tokens = self.generate_token_ids(
            encode_context(input_text, tokenizer),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stop_token=stop_token,
            device=device,
        )
        visible = tokens[:-1] if tokens and tokens[-1] == stop_token else tokens
        completion = tokenizer.decode(visible, skip_special_tokens=True)
        return input_text + "\n\n" + completion if return_full_text else completion

    @torch.no_grad()
    def generate_token_ids(
        self,
        input_ids,
        max_new_tokens=50,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        repetition_penalty=1.0,
        stop_token=None,
        device=None,
    ):
        """Generate from exact context IDs; return new IDs, including EOS if emitted.

        Prepared evaluation contexts already include the reply boundary and
        truncation. This avoids decoding and re-tokenizing them before generation.
        """
        if max_new_tokens < 1 or not math.isfinite(temperature) or temperature < 0:
            raise ValueError(
                "Require positive max_new_tokens and finite nonnegative temperature."
            )
        if (
            top_k < 0
            or not 0 < top_p <= 1
            or not math.isfinite(repetition_penalty)
            or repetition_penalty <= 0
        ):
            raise ValueError("Invalid sampling settings.")
        if stop_token is not None and not 0 <= stop_token < self.vocab:
            raise ValueError("stop_token is outside the vocabulary.")
        if not input_ids or any(
            not isinstance(token, int) or not 0 <= token < self.vocab
            for token in input_ids
        ):
            raise ValueError("input_ids must be a nonempty list of vocabulary IDs.")
        device = device or next(self.parameters()).device
        ids = torch.tensor([input_ids], dtype=torch.long, device=device)
        if ids.size(1) + max_new_tokens > self.max_len:
            raise ValueError("Prompt plus requested output exceeds the context length.")
        prompt_length = ids.size(1)
        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                logits = self(ids)["main"][:, -1].float()
                logits = self.apply_repetition_penalty(logits, ids, repetition_penalty)
                if temperature == 0:
                    next_token = logits.argmax(-1, keepdim=True)
                else:
                    filtered = self.top_k_top_p_filtering(
                        logits / temperature, top_k, top_p
                    )
                    next_token = torch.multinomial(filtered.softmax(-1), 1)
                ids = torch.cat([ids, next_token], dim=-1)
                if next_token.item() == stop_token:
                    break
            return ids[0, prompt_length:].tolist()
        finally:
            self.train(was_training)

    def generate_chat(self, prompt, context=None, **generation_options):
        if isinstance(prompt, list):
            if context is not None:
                raise ValueError("Supply either a message list or prompt plus context.")
            messages = prompt
        else:
            messages = list(context or []) + [prompt]
        return self.generate(format_messages(messages), **generation_options)
