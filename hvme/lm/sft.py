"""Hvme-LM SFT: learn Gym CoT prompt -> solution trajectory. The loss is applied only to solution tokens.

- Teacher: solution-template trajectories (hvme.lm.solutions, the engine's computation steps) plus, optionally, 9B rejection-sampling trajectories (hvme.lm.sample)
- The input format matches inference: the solution and EOS are appended after the chat template's generation prompt (enable_thinking=False)
- Trained with LoRA (hvme.fast.lora); afterwards the adapter is merged into the weights and saved as a plain model
"""
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import WEIGHTS, pick, template_examples  # noqa: F401 (also imported via sft in notebooks)
from .tools import TOOLS

def encode(tok, ex: dict, max_len: int) -> tuple[list[int], list[int]]:
    """(input_ids, labels). Labels at prompt positions are -100 (excluded from the loss). If ex has messages, uses encode_messages."""
    if "messages" in ex:
        return encode_messages(tok, ex, max_len)
    prompt = tok.apply_chat_template([{"role": "user", "content": ex["prompt"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    p = tok(prompt, add_special_tokens=False).input_ids
    r = tok(ex["response"] + tok.eos_token, add_special_tokens=False).input_ids
    return (p + r)[:max_len], ([-100] * len(p) + r)[:max_len]


def encode_messages(tok, ex: dict, max_len: int) -> tuple[list[int], list[int]]:
    """Multi-turn conversation (including tool calls). The loss is applied only to assistant turns (content, tool calls, <|im_end|>). Turns with "train": False (wrongly chosen tool calls) are used only as context.

    The head of each assistant turn is the string built with add_generation_prompt, as at inference, and each piece is tokenized separately (matching the token boundaries at inference).
    """
    msgs, tools = ex["messages"], TOOLS if ex.get("tools") else None
    render = lambda m, gen: tok.apply_chat_template(m, tools=tools, tokenize=False, add_generation_prompt=gen, enable_thinking=False)
    ids, labels, prev = [], [], ""
    for i, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        prompt, full = render(msgs[:i], True), render(msgs[:i + 1], False)
        assert prompt.startswith(prev) and full.startswith(prompt)
        p = tok(prompt[len(prev):], add_special_tokens=False).input_ids
        r = tok(full[len(prompt):], add_special_tokens=False).input_ids
        ids, labels, prev = ids + p + r, labels + [-100] * len(p) + (r if m.get("train", True) else [-100] * len(r)), full
    return ids[:max_len], labels[:max_len]


def collate(batch, pad_id: int, device):
    n = max(len(i) for i, _ in batch)
    ids = torch.tensor([i + [pad_id] * (n - len(i)) for i, _ in batch], device=device)
    labels = torch.tensor([l + [-100] * (n - len(l)) for _, l in batch], device=device)
    mask = torch.tensor([[1] * len(i) + [0] * (n - len(i)) for i, _ in batch], device=device)
    return ids, mask, labels


def train_sft(model, tok, examples: list[dict], epochs: float = 1.0, lr: float = 1e-4, batch_size: int = 2, grad_accum: int = 16,
              max_len: int = 3072, warmup: float = 0.03, log_every: int = 20, seed: int = 0, log=print,
              ckpt: str | Path | None = None, ckpt_every: int = 40) -> pd.DataFrame:
    """Takes a model with LoRA attached (hvme.fast.lora.add_lora). Examples of similar length are batched together to reduce padding.

    If ckpt is given, LoRA, optimizer and scheduler are saved every ckpt_every steps, and training resumes from that step if the file exists
    (batch order is determined by seed, so the same examples and seed see the same batches as an uninterrupted run)."""
    model.train()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    enc = [encode(tok, ex, max_len) for ex in examples]
    rng = np.random.default_rng(seed)
    order = np.concatenate([rng.permutation(len(enc)) for _ in range(math.ceil(epochs))])[: int(len(enc) * epochs)]
    chunks = [order[i:i + batch_size * 64] for i in range(0, len(order), batch_size * 64)]  # sort by length within groups of 64 batches
    batches = [sorted(c, key=lambda i: len(enc[i][0]))[j:j + batch_size] for c in chunks for j in range(0, len(c), batch_size)]
    batches = [batches[i] for i in rng.permutation(len(batches))]
    steps = len(batches) // grad_accum
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, int(warmup * steps))) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / max(steps, 1))))
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    dev, hist, running, t0, start = model.device, [], [], time.time(), 0
    lora = [n for n, p in model.named_parameters() if p.requires_grad]
    if ckpt and Path(ckpt).exists():
        st = torch.load(ckpt, map_location=dev, weights_only=False)
        model.load_state_dict(st["lora"], strict=False)
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        hist, start = st["hist"], st["step"]
        t0 -= hist[-1]["sec"] if hist else 0
        log(f"checkpoint {ckpt}: resuming at step {start}/{steps}")
    for step in range(start, steps):
        for b in batches[step * grad_accum:(step + 1) * grad_accum]:
            ids, mask, labels = collate([enc[i] for i in b], pad, dev)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss / grad_accum
            loss.backward()
            running.append(loss.item() * grad_accum)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if (step + 1) % log_every == 0 or step + 1 == steps:
            hist.append({"step": step + 1, "loss": float(np.mean(running)), "lr": sched.get_last_lr()[0], "sec": round(time.time() - t0, 1)})
            running = []
            log(hist[-1])
        if ckpt and (step + 1) % ckpt_every == 0 and step + 1 < steps:
            state = dict(model.named_parameters())
            torch.save({"lora": {n: state[n].detach() for n in lora}, "opt": opt.state_dict(), "sched": sched.state_dict(), "hist": hist, "step": step + 1}, f"{ckpt}.tmp")
            os.replace(f"{ckpt}.tmp", ckpt)
    model.eval()
    model.config.use_cache = True
    return pd.DataFrame(hist)
