"""Hvme-Fast training: fine-tune with cross-entropy (a proper scoring rule) between a softmax restricted to the answer tokens and the target distribution.

- Hard labels are one-hot; soft labels use the distribution itself as the target
- Weights stay in fp32 and compute runs under bf16 autocast (with bf16 weights, small learning-rate updates vanish to rounding)
- After training, fit a temperature per option kind on the validation set (temperature scaling)
"""
import math
import time

import numpy as np
import pandas as pd
import torch

from .calibrate import apply_temperature, fit_temperature, metrics
from .judge import FastJudge, kind_of, temp_key


def batch_tensors(judge: FastJudge, items: list[dict]):
    """Items can have different numbers of options, so option ids and targets are padded to the longest and a mask is kept."""
    k = max(len(d["labels"]) for d in items)
    ids = [judge.label_ids(d["labels"]) for d in items]
    dev = judge.model.device
    label_ids = torch.tensor([x + [0] * (k - len(x)) for x in ids], device=dev)
    mask = torch.tensor([[True] * len(x) + [False] * (k - len(x)) for x in ids], device=dev)
    target = torch.tensor([d["target"] + [0.0] * (k - len(d["target"])) for d in items], device=dev)
    return judge.encode([d["prompt"] for d in items]), label_ids, mask, target


def loss_fn(judge: FastJudge, items: list[dict]) -> torch.Tensor:
    enc, label_ids, mask, target = batch_tensors(judge, items)
    with torch.autocast(judge.model.device.type, dtype=torch.bfloat16, enabled=judge.model.device.type == "cuda"):
        logits = judge.last_logits(enc)
    z = logits.gather(1, label_ids).masked_fill(~mask, float("-inf"))
    logp = torch.log_softmax(z, -1).masked_fill(~mask, 0.0)
    return -(target * logp).sum(-1).mean()


def train(judge: FastJudge, items: list[dict], epochs: float = 1.0, lr: float = 1e-5, batch_size: int = 8, grad_accum: int = 4,
          warmup: float = 0.03, eval_items: list[dict] | None = None, eval_every: int = 200, seed: int = 0, log=print,
          gradient_checkpointing: bool = True) -> pd.DataFrame:
    """Iterate over items for epochs (a fractional value runs that fraction). Returns the loss and evaluation history."""
    model = judge.model
    model.train()
    model.config.use_cache = False
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    rng = np.random.default_rng(seed)
    order = np.concatenate([rng.permutation(len(items)) for _ in range(math.ceil(epochs))])[: int(len(items) * epochs)]
    steps = len(order) // (batch_size * grad_accum)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, int(warmup * steps))) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / max(steps, 1))))
    hist, t0, running = [], time.time(), []
    for step in range(steps):
        for j in range(grad_accum):
            idx = order[(step * grad_accum + j) * batch_size:(step * grad_accum + j + 1) * batch_size]
            loss = loss_fn(judge, [items[i] for i in idx]) / grad_accum
            loss.backward()
            running.append(loss.item() * grad_accum)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        last = step + 1 == steps
        if (step + 1) % eval_every == 0 or last:
            row = {"step": step + 1, "loss": float(np.mean(running)), "lr": sched.get_last_lr()[0], "sec": round(time.time() - t0, 1)}
            row |= {f"eval_{k}": v for k, v in evaluate(judge, eval_items)[["acc", "ece", "nll"]].mean().items()} if eval_items else {}
            hist.append(row)
            running = []
            log(row)
            model.train()
    model.eval()
    model.config.use_cache = True
    return pd.DataFrame(hist)


def predict(judge: FastJudge, items: list[dict]) -> pd.DataFrame:
    """Compute logits per option group (labels). Returns the uncalibrated logits together with probabilities under the current temperatures."""
    df = pd.DataFrame(items).assign(key=[tuple(d["labels"]) for d in items])
    out = []
    for key, g in df.groupby("key", sort=False):
        z = judge.logits(g.prompt.tolist(), list(key))
        p = [apply_temperature([zi], judge.temperature_for(q, key))[0] for zi, q in zip(z, g.prompt)]
        out.append(g.assign(logits=list(z), probs=p, target_idx=[int(np.argmax(t)) for t in g.target]))
    res = pd.concat(out).loc[df.index]
    return res.assign(pred=[int(np.argmax(p)) for p in res.probs], kind=[kind_of(k) for k in res.key],
                      tkey=[temp_key(q, k) for q, k in zip(res.prompt, res.key)])


def evaluate(judge: FastJudge, items: list[dict], by=("kind",)) -> pd.DataFrame:
    res = items if isinstance(items, pd.DataFrame) else predict(judge, items)
    return pd.DataFrame([dict(zip(by, k if isinstance(k, tuple) else (k,))) | metrics(list(g.probs), g.target_idx.to_numpy())
                         for k, g in res.groupby(list(by))])


def calibrate(judge: FastJudge, items: list[dict]) -> dict:
    """For each temperature key (option kind, with a 'knowledge:' prefix for domain-knowledge judgments), fit the temperature minimizing validation NLL and store it in judge.temperature."""
    judge.temperature = {}
    res = predict(judge, items)
    judge.temperature = {k: fit_temperature(list(g.logits), g.target_idx.to_numpy()) for k, g in res.groupby("tkey")}
    return judge.temperature


def rescale(res: pd.DataFrame, temperature: dict) -> pd.DataFrame:
    """Apply different temperatures to predict output (uncalibrated logits) without another forward pass."""
    probs = [apply_temperature([z], temperature.get(t, temperature.get(k, 1.0)))[0] for z, k, t in zip(res.logits, res.kind, res.tkey)]
    return res.assign(probs=probs, pred=[int(np.argmax(p)) for p in probs])
