"""Original sigmoid DPO: summed completion log-probabilities, fixed SFT reference."""


def completion_labels(prompt_ids, full_ids, eos_id):
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("Completion changed the native prompt prefix.")
    if len(full_ids) <= len(prompt_ids) + 1 or full_ids[-1] != eos_id:
        raise ValueError("Completion requires nonempty content and native EOS.")
    return [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]


def summed_log_probs(logits, labels):
    """Independent reference implementation, including EOS and excluding prompt/pad."""
    import torch.nn.functional as functional
    shifted = labels[:, 1:]
    valid = shifted != -100
    safe = shifted.masked_fill(~valid, 0)
    logp = functional.log_softmax(logits[:, :-1].float(), dim=-1)
    return (logp.gather(-1, safe.unsqueeze(-1)).squeeze(-1) * valid).sum(-1)


def dpo_loss(chosen, rejected, ref_chosen, ref_rejected, beta):
    import torch.nn.functional as functional
    if beta <= 0:
        raise ValueError("DPO beta must be positive.")
    margin = (chosen - rejected) - (ref_chosen - ref_rejected)
    return -functional.logsigmoid(beta * margin), margin


def disable_dropout(model):
    import torch
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
        if hasattr(module, "attention_dropout"):
            module.attention_dropout = 0.0


def sequence_logp(runtime, row):
    import torch
    from posttraining.lora_train import batch
    # HF causal CE averages nonmasked shifted targets. Undo that normalization
    # to obtain the sequence sum used by standard DPO, not IPO/SimPO.
    with torch.autocast("cuda", dtype=torch.float16):
        mean_nll = runtime.model(**batch(row, runtime.device), use_cache=False).loss
    return -mean_nll.float() * row["target_tokens"]


def preference_metrics(runtime, pairs, beta):
    import torch
    runtime.model.eval()
    records = []
    with torch.inference_mode():
        for pair in pairs:
            torch.cuda.empty_cache()
            runtime.guard_memory()
            chosen = sequence_logp(runtime, pair["chosen"])
            rejected = sequence_logp(runtime, pair["rejected"])
            loss, margin = dpo_loss(chosen, rejected, pair["ref_chosen"], pair["ref_rejected"], beta)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite preference validation.")
            records.append({"loss": loss.item(), "margin": margin.item(),
                            "chosen_preferred": float(chosen > rejected),
                            "relative_margin_positive": float(margin > 0),
                            "chosen_nll": -chosen.item() / pair["chosen"]["target_tokens"]})
            del chosen, rejected, loss, margin
            runtime.guard_memory()
    return {key: sum(r[key] for r in records) / len(records) for key in records[0]}


def train_group(runtime, pairs, optimizer, scaler, config):
    import torch
    runtime.model.train()
    disable_dropout(runtime.model)
    for attempt in range(8):
        optimizer.zero_grad(set_to_none=True)
        loss_sum = margin_sum = 0.0
        for pair in pairs:
            torch.cuda.empty_cache()
            runtime.guard_memory()
            chosen = sequence_logp(runtime, pair["chosen"])
            rejected = sequence_logp(runtime, pair["rejected"])
            loss, margin = dpo_loss(chosen, rejected, pair["ref_chosen"], pair["ref_rejected"], config["beta"])
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite DPO loss.")
            loss_sum += loss.detach().item()
            margin_sum += margin.detach().item()
            # Equal preference-pair weight, not a target-token-weighted SFT objective.
            scaler.scale(loss / len(pairs)).backward()
            del chosen, rejected, loss, margin
            runtime.guard_memory()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_([p for p in runtime.model.parameters() if p.requires_grad],
                                             config["max_grad_norm"])
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= old_scale:
            if not torch.isfinite(norm):
                raise RuntimeError("Nonfinite DPO gradient norm.")
            optimizer.zero_grad(set_to_none=True)
            runtime.guard_memory()
            return loss_sum / len(pairs), margin_sum / len(pairs), float(norm), attempt
        print(f"FP16 overflow: retrying DPO group ({attempt+1}/8)", flush=True)
    raise RuntimeError("Repeated DPO overflow; group was not advanced.")
