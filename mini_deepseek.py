import math
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
# -----------------------------------------------------------------------------
#  Normalisation & Attention
# -----------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, heads: int, p: float = 0.1):
        super().__init__()
        assert dim % heads == 0, "d_model must divide n_heads"
        self.h = heads; self.dh = dim // heads; self.scale = self.dh ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.drop   = nn.Dropout(p)  

    def forward(self, x, mask):  # mask (B,1,T,T)
        B, T, _ = x.size()
        qkv = self.qkv(x).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv
        att = (q @ k.transpose(-2, -1)) * self.scale
        att = att.masked_fill(~mask, -1e9)
        att = F.softmax(att, dim=-1)
        att = self.drop(att) 
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)

class MLA(nn.Module):
    """
    Linformer-style latent attention:
    - Projects K & V with a learned (T→L) matrix, giving O(T·L) complexity.
    - Same API as before: __init__(d_model, n_heads, dk_per_head=64, target_L=64, p=0.1)
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dk_per_head: int = 64,
        target_L: int = 64,
        p: float = 0.1,
        max_len: int = 3072,                # maximum supported sequence length
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must divide n_heads"

        self.h        = n_heads
        self.dk       = dk_per_head
        self.dq       = d_model // n_heads
        self.target_L = target_L
        self.max_len  = max_len

        d_latent = self.dk * n_heads
        # --- standard projections ---
        self.kv_proj = nn.Linear(d_model, 2 * d_latent, bias=False)
        self.q_proj  = nn.Linear(d_model, n_heads * self.dq, bias=False)
        self.o_proj  = nn.Linear(n_heads * self.dq, d_model, bias=False)
        self.scale   = self.dk ** -0.5
        self.drop    = nn.Dropout(p)

        # --- Linformer projections over sequence length ---
        #   P_k, P_v: (target_L × max_len) mapping T→target_L
        self.P_k = nn.Parameter(torch.empty(target_L, max_len))
        self.P_v = nn.Parameter(torch.empty(target_L, max_len))
        nn.init.xavier_uniform_(self.P_k)
        nn.init.xavier_uniform_(self.P_v)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        x:    (B, T, d_model)
        mask: (B, 1, T, T) causal/padding mask
        returns: (B, T, d_model)
        """
        B, T, _ = x.size()
        # 1) Project & split K/V
        kv = self.kv_proj(x)                   # (B, T, 2*d_latent)
        kv = kv.permute(0, 2, 1)               # (B, 2*d_latent, T)
        k, v = kv.chunk(2, dim=1)              # each (B, d_latent, T)

        # 2) Learnable Linformer compression: matmul with P_k^T, P_v^T
        #    P_k[:, :T] is (L, T), so k @ P_k.T -> (B, d_latent, L)
        Pk = self.P_k[:, :T]                   # (L, T)
        Pv = self.P_v[:, :T]                   # (L, T)
        k = torch.matmul(k, Pk.t())            # (B, d_latent, L)
        v = torch.matmul(v, Pv.t())            # (B, d_latent, L)

        # 3) Reshape into heads
        k = k.view(B, self.h, self.dk, self.target_L)            # (B, h, dk, L)
        v = v.view(B, self.h, self.dk, self.target_L)            # (B, h, dk, L)
        v = v.permute(0, 1, 3, 2)                                # (B, h, L, dk)

        # 4) Queries
        q = self.q_proj(x) \
             .view(B, T, self.h, self.dq) \
             .transpose(1, 2)                                 # (B, h, T, dq)

        # 5) Scaled dot-product attention
        att = (q @ k) * self.scale                             # (B, h, T, L)
        if mask is not None:
            # mask: (B,1,T,T).  We need a (B,1,T, target_L) mask.
            # First slice to the true seq-length:
            mask_k = mask[..., :T]                            # (B,1,T,T)
            # If target_L > T, pad the last dim with False:
            if self.target_L > T:
                pad_size = self.target_L - T
                mask_k = F.pad(mask_k, (0, pad_size), value=False)  # (B,1,T,target_L)
            else:
                mask_k = mask_k[..., :self.target_L]           # (B,1,T,target_L)
            causal = mask_k.expand(B, self.h, T, self.target_L)
            att = att.masked_fill(~causal, -1e9)
        att = F.softmax(att, dim=-1)
        att = self.drop(att)

        # 6) Aggregate & output
        out = att @ v                                           # (B, h, T, dk)
        out = out.transpose(1, 2).reshape(B, T, -1)             # (B, T, h*dk)
        return self.o_proj(out)                                 # (B, T, d_model)
    
# -----------------------------------------------------------------------------
#  Feed‑Forward & MoE
# -----------------------------------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden: int, p: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden, bias=False),
            nn.GELU(),
            nn.Dropout(p),         # pre-projection
            nn.Linear(hidden, dim, bias=False),
            nn.Dropout(p),         # post-projection
        )
    def forward(self, x):
        return self.net(x)


class MoEBlock(nn.Module):
    """
    Top-k (default k=2) gating Mixture-of-Experts block.
    - Differentiable gating (no hard argmax)
    - Vectorized dispatch
    - Built-in dropout in experts
    - Returns (output, load_balancing_loss)
    """
    def __init__(
        self,
        dim: int,
        hidden: int,
        n_experts: int = 4,
        top_k: int = 2,
        p: float = 0.1,
    ):
        super().__init__()
        self.dim       = dim
        self.hidden    = hidden
        self.n_experts = n_experts
        self.top_k     = top_k
        self.p         = p

        # router: maps each token to E logits
        self.router = nn.Linear(dim, n_experts, bias=False)

        # experts: each is a small MLP with dropout
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden, bias=False),
                nn.GELU(),
                nn.Dropout(p),
                nn.Linear(hidden, dim, bias=False),
                nn.Dropout(p),
            )
            for _ in range(n_experts)
        ])

    @staticmethod
    def _lb_loss(gate_p: torch.Tensor, expert_mask: torch.Tensor) -> torch.Tensor:
        # importance and load distributions over experts
        imp  = gate_p.sum(dim=(0,1))
        load = expert_mask.sum(dim=(0,1))
        imp  = imp  / (imp.sum()  + 1e-9)
        load = load / (load.sum() + 1e-9)
        # Switch-Transformer load balance: E · ⟨imp, load⟩
        return (imp * load).sum() * gate_p.size(-1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, T, D)
        returns:
          out: (B, T, D)
          lb : scalar load-balancing loss
        """
        B, T, D = x.shape
        # 1) router logits → probabilities
        logits = self.router(x)                        # (B,T,E)
        if self.training:
            noise = torch.randn_like(logits) / self.n_experts
            logits = logits + noise
        probs = F.softmax(logits, dim=-1)              # (B,T,E)

        # 2) top-k gating & renormalize
        top_p, top_i = probs.topk(self.top_k, dim=-1)  # both (B,T,K)
        top_p = top_p / (top_p.sum(dim=-1, keepdim=True) + 1e-9)

        # 3) dispatch to experts
        out = torch.zeros_like(x)
        x_flat    = x.view(-1, D)                        # (B·T, D)
        # flatten the top-k indices & weights along the token dimension
        flat_top_i = top_i.reshape(-1, self.top_k)       # (B·T, K)
        flat_top_p = top_p.reshape(-1, self.top_k)       # (B·T, K)
        mask_flat  = torch.zeros(B*T, self.n_experts, device=x.device)

        for e_idx, expert in enumerate(self.experts):
            # find tokens where this expert was one of the top-k
            token_mask = (flat_top_i == e_idx).any(dim=-1)  # (B·T,) bool
            if not token_mask.any():
                continue

            routed_in = x_flat[token_mask]                # (n, D)
            # pick out the ONE gate weight per token where flat_top_i == e_idx
            gate_vals = flat_top_p[token_mask]            # (n, K)
            sel       = (flat_top_i[token_mask] == e_idx) # (n, K) bool
            gate_vals = gate_vals.masked_select(sel)      # (n,) scalar per token

            routed_out = expert(routed_in) * gate_vals.unsqueeze(-1)  # (n, D)
            out_flat   = out.view(-1, D)
            out_flat[token_mask] += routed_out
            mask_flat[token_mask, e_idx] = 1.0

        # 4) load-balancing loss (detach gates so only router sees grad)
        lb = self._lb_loss(probs.detach(), mask_flat.view(B, T, -1).detach())

        return out, lb


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, hidden: int, use_moe: bool, p: float = 0.1):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = MLA(dim, heads, p=p)
        self.ffn_norm = RMSNorm(dim)
        self.use_moe = use_moe
        self.ffn = MoEBlock(dim, hidden, p=p) if use_moe else FeedForward(dim, hidden, p)

    def forward(self, x, mask):
        x = x + self.attn(self.attn_norm(x), mask)
        if self.use_moe:
            ff, lb = self.ffn(self.ffn_norm(x))
            return x + ff, lb
        return x + self.ffn(self.ffn_norm(x)), torch.tensor(0.0, device=x.device)

# -----------------------------------------------------------------------------
#  RolePlayTransformer with 4‑expert MoE from layer 4 onward
# -----------------------------------------------------------------------------
class RolePlayTransformer(nn.Module):
    def __init__(
        self,
        vocab: int,
        max_len: int = 3072,
        d_model: int = 768,
        n_layers: int = 10,
        n_heads: int = 12,
        d_ff: int = 3072,
        dropout: float = 0.1,
        moe_start: int = 5,          # first 5 layers dense, rest MoE
        lambda_lb: float = 0.01,
        use_mtp: bool = True,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.vocab = vocab; self.use_mtp = use_mtp; self.gc = gradient_checkpointing; self.lambda_lb = lambda_lb

        self.embed = nn.Embedding(vocab, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, use_moe=(i>=moe_start), p=dropout)
            for i in range(n_layers)
        ])

        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab, bias=False)
        if use_mtp:
            self.mtp_head = nn.Linear(d_model, vocab, bias=False)
            self.mtp_head.weight = self.embed.weight
        self.lm_head.weight = self.embed.weight

    # --------------------------------------------------------------
    def _mask(self, B, T, dev):
        return torch.tril(torch.ones(T, T, device=dev, dtype=torch.bool)).unsqueeze(0).unsqueeze(0).expand(B,1,T,T)

    @staticmethod
    def _lb_loss(gate_p: torch.Tensor, expert_mask: torch.Tensor) -> torch.Tensor:
        """
        Load-balancing from the Switch-Transformer paper:
        L = E_gate · E_mask  (expectations over tokens)
        """
        imp = gate_p.sum(dim=(0, 1))        # importance - sum of gates
        load = expert_mask.sum(dim=(0, 1))  # actual load (# tokens)
        imp = imp / imp.sum()
        load = load / load.sum()
        return (imp * load).sum() * gate_p.size(-1)

    def forward(self, ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        ids            : (B, T) int64  token ids
        attention_mask : (B, T) bool / int {0,1} — optional padding mask
        returns dict { "main": logits, "lb": Σ load-balancing, "mtp": logits? }
        """
        B, T = ids.size()
        dev   = ids.device

        x = self.embed(ids) + self.pos[:, :T, :]          # (B,T,D)

        # causal mask (B,1,T,T)
        mask = self._mask(B, T, dev)
        if attention_mask is not None:
            pad = attention_mask.unsqueeze(1).unsqueeze(2).bool()  # (B,1,1,T)
            mask = mask & pad

        lb_total = 0.0
        for block in self.layers:
            if self.gc and self.training:
                x, lb = torch.utils.checkpoint.checkpoint(block, x, mask)
            else:
                x, lb = block(x, mask)
            lb_total = lb_total + lb

        x = self.norm(x)
        logits = self.lm_head(x)

        out = {"main": logits, "lb": lb_total}
        if self.use_mtp:
            out["mtp"] = self.mtp_head(x)
        return out
    
    @staticmethod
    def make_collate_fn(tokenizer):
        pad_id = tokenizer.pad_token_id
        def coll(batch):
            inp_seq, lbl_seq = [], []
            for ex in batch:
                ctx = torch.tensor(ex["input_ids"], dtype=torch.long)
                tgt = torch.tensor(ex["labels"],    dtype=torch.long)
                full   = torch.cat([ctx, tgt[:-1]])            # inputs
                labels = torch.cat([
                    torch.full_like(ctx, -100),                # ignore ctx
                    tgt                                         # predict tgt
                ])
                inp_seq.append(full)
                lbl_seq.append(labels)
            inp_pad = nn.utils.rnn.pad_sequence(
                inp_seq, batch_first=True, padding_value=pad_id
            )
            lbl_pad = nn.utils.rnn.pad_sequence(
                lbl_seq, batch_first=True, padding_value=-100
            )
            attn_mask = (inp_pad != pad_id).long()
            return inp_pad, attn_mask, lbl_pad
        return coll
# -----------------------------------------
    def train_model(
        self,
        dataset_path: str,
#       tokenizer_name: str = "deepseek-ai/DeepSeek-V3",
        tokenizer_name: str = "gpt2",   
        epochs: int = 1,
        micro_batch: int = 8,
        grad_accum: int = 4,
        lr: float = 1e-4,
        warmup_updates: int = 1000,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        save_path: Optional[str] = None,
    ):
        tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        # --- make sure GPT-2 has a pad token ---------------------------------
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        ds  = load_from_disk(dataset_path)

        # --- dataloader -----------------------------------------------------
        coll_fn = self.make_collate_fn(tok)
        loader = DataLoader(
            ds,
            batch_size = micro_batch,
            shuffle    = True,
            collate_fn = coll_fn,
        )

        # --- optimiser & scheduler -----------------------------------------
        self.to(device)
        opt   = torch.optim.AdamW(self.parameters(), lr=lr, betas=(0.9,0.95),
                                  weight_decay=0.01)
        total_updates = math.ceil(len(loader) * epochs / grad_accum)
        sched = get_cosine_schedule_with_warmup(
            opt, num_warmup_steps = warmup_updates, num_training_steps=total_updates
        )
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        # --- training loop --------------------------------------------------
        self.train()
        step_global = 0
        opt.zero_grad()

        for ep in range(epochs):
            for step, (iid, attn, lab) in enumerate(loader):
                iid, attn, lab = iid.to(device), attn.to(device), lab.to(device)

                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    out    = self(iid, attention_mask=attn)
                    logits = out["main"]                         # (B, L_in, V)

                    # align logits and labels to the same effective length
                    L_in   = logits.size(1)
                    L_lab  = lab.size(1)
                    seq_len = min(L_in - 1, L_lab - 1)           # number of training steps

                    # slice both to [0..seq_len)
                    log_slice = logits[:, :seq_len]              # (B, seq_len, V)
                    lab_slice = lab    [:, 1:seq_len+1]          # (B, seq_len)

                    # main CE
                    loss = loss_fn(
                        log_slice.reshape(-1, self.vocab),
                        lab_slice.reshape(-1)
                    )

                    # optional MTP branch
                    if self.use_mtp:
                        mtp = out["mtp"][:, :seq_len]
                        loss += 0.1 * loss_fn(
                            mtp.reshape(-1, self.vocab),
                            lab_slice.reshape(-1)
                        )
 
                     # MoE load-balancing
                    loss += self.lambda_lb * out["lb"]
 
                     # scale for gradient accumulation
                    loss = loss / grad_accum
                loss.backward()
 
                 # optimizer update every grad_accum mini-batches
                if (step + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                    opt.step()
                    opt.zero_grad()
                    sched.step()


                    step_global += 1
                    if step_global % 100 == 0:
                        print(f"epoch {ep+1} | update {step_global} | "
                              f"loss {loss.item() * grad_accum:.4f}")  # true CE

            if save_path:
                torch.save(self.state_dict(), f"{save_path}/ckpt_ep{ep+1}.pt")

    def evaluate_perplexity(
        self,
        model, 
        dataset_path: str, 
        tokenizer, 
        device: str = "cuda", 
        batch_size: int = 8,
    ):
        # 1) Load test split
        ds = load_from_disk(dataset_path)

        # 2) Collate: same as train but no grad
        coll_fn = self.make_collate_fn(tokenizer)
        loader = DataLoader(
            ds, 
            batch_size   = batch_size, 
            shuffle      = False, 
            collate_fn   = coll_fn
        )


        model.eval()
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
        total_loss = 0
        total_tokens = 0

        with torch.no_grad():
            for (ids, attn_mask, labels) in loader:
                ids, attn_mask, labels = ids.to(device), attn_mask.to(device), labels.to(device)
                out    = model(ids, attention_mask=attn_mask)
                logits = out["main"]              # (B, L, V)

                # align lengths
                L_in  = logits.size(1)
                L_lab = labels.size(1)
                seq_len = min(L_in-1, L_lab-1)
                log_slice = logits[:, :seq_len].reshape(-1, model.vocab)
                lab_slice = labels[:, 1:seq_len+1].reshape(-1)

                # sum of CE, ignoring -100
                batch_loss = loss_fn(log_slice, lab_slice)
                total_loss += batch_loss.item()
                total_tokens += (lab_slice != -100).sum().item()

        # perplexity = exp(avg loss per token)
        ppl = torch.exp(torch.tensor(total_loss / total_tokens))
        print(f"Test perplexity: {ppl:.2f}")
        return ppl

    @staticmethod
    def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    ) -> torch.Tensor:
        """
        Filter a distribution of logits using top-k and/or nucleus (top-p) filtering.
        """
        logits = logits.clone()
        # Top-k
        if top_k > 0:
            top_k = min(max(top_k, 1), logits.size(-1))
            threshold = torch.topk(logits, top_k)[0][..., -1, None]
            logits = torch.where(logits < threshold, filter_value, logits)
        # Nucleus (top-p)
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            to_remove = cumulative_probs > top_p
            to_remove[..., 1:] = to_remove[..., :-1].clone()
            to_remove[..., 0] = False
            remove_indices = sorted_indices[to_remove]
            logits.flatten()[remove_indices] = filter_value
        return logits

    @staticmethod
    def _build_attention_mask(ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
        """
        Create attention mask (1 for real tokens, 0 for padding).
        """
        return (ids != pad_token_id).int()

    @torch.no_grad()
    def generate(
        self,
        input_text: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        stop_token: Optional[int] = None,
        device: Optional[str] = None,
    ) -> str:
        """
        Generate text step-by-step with top-k/top-p filtering and optional repetition penalty.
        """
        was_training = self.training
        self.eval()
        device = device or next(self.parameters()).device

        tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        stop_token = stop_token or tokenizer.eos_token_id

        # Encode prompt
        input_ids = tokenizer(
            input_text, return_tensors="pt", padding=False
        ).input_ids.to(device)
        output_ids = input_ids.clone()

        for _ in range(max_new_tokens):
            # Build attention mask
            attn_mask = self._build_attention_mask(output_ids, pad_id).to(device)
            # Forward
            outputs = self(output_ids, attention_mask=attn_mask)
            logits = outputs["main"]
            next_logits = logits[:, -1, :]

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(output_ids.view(-1).tolist()):
                    next_logits[:, token_id] /= repetition_penalty

            # Choose next token
            if temperature == 0:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                scaled = next_logits / temperature
                filtered = self.top_k_top_p_filtering(scaled, top_k=top_k, top_p=top_p)
                probs = F.softmax(filtered, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            if next_token.item() == stop_token:
                break

            output_ids = torch.cat([output_ids, next_token], dim=-1)

        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        if was_training:
            self.train()
        return text

    @torch.no_grad()
    def generate_chat(
        self,
        prompt,
        context: Optional[List[str]] = None,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        stop_token: Optional[int] = None,
        device: Optional[str] = None,
    ) -> str:
        """
        Chat generation: accepts either a prompt string with context list or a single list of messages.
        """
        if isinstance(prompt, list) and context is None:
            history = prompt
            context = history[:-1]
            prompt = history[-1]

        history = (context.copy() if context else []) + [prompt]
        full_input = "\n".join(history)
        return self.generate(
            input_text=full_input,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stop_token=stop_token,
            device=device,
        )

