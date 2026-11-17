"""Hume-LM SFT (Step 9-1): Gym CoT 프롬프트 → 풀이 궤적을 학습한다. 손실은 풀이 토큰에만 준다.

- 교사: 풀이 템플릿 궤적(humm.lm.solutions, 엔진 계산 과정) + 선택적으로 9B rejection sampling 궤적(humm.lm.sample)
- 입력 형식은 추론과 같다: 채팅 템플릿(enable_thinking=False)의 생성 프롬프트 뒤에 풀이와 EOS를 붙인다
- LoRA(humm.fast.lora)로 학습하고, 끝나면 가중치에 합쳐 일반 모델로 저장한다
"""
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import WEIGHTS, pick, template_examples  # noqa: F401 (노트북에서 sft로도 불러온다)
from .tools import TOOLS

def encode(tok, ex: dict, max_len: int) -> tuple[list[int], list[int]]:
    """(input_ids, labels). 프롬프트 위치의 labels는 -100(손실 제외). ex에 messages가 있으면 encode_messages."""
    if "messages" in ex:
        return encode_messages(tok, ex, max_len)
    prompt = tok.apply_chat_template([{"role": "user", "content": ex["prompt"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    p = tok(prompt, add_special_tokens=False).input_ids
    r = tok(ex["response"] + tok.eos_token, add_special_tokens=False).input_ids
    return (p + r)[:max_len], ([-100] * len(p) + r)[:max_len]


def encode_messages(tok, ex: dict, max_len: int) -> tuple[list[int], list[int]]:
    """다중 턴(도구 호출 포함) 대화. 어시스턴트 턴(내용·도구 호출·<|im_end|>)에만 손실을 준다. "train": False인 턴(잘못 고른 도구 호출)은 문맥으로만 쓴다.

    각 어시스턴트 턴의 앞부분은 추론 때와 같이 add_generation_prompt로 만든 문자열이고, 조각마다 따로 토큰화한다(추론 때 토큰 경계와 같다).
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
    """LoRA를 붙인 모델(humm.fast.lora.add_lora)을 받는다. 길이가 비슷한 예제끼리 묶어 패딩을 줄인다.

    ckpt를 주면 ckpt_every step마다 LoRA·옵티마이저·스케줄러를 저장하고, 파일이 있으면 그 step부터 이어 학습한다
    (배치 순서는 seed로 정해지므로 같은 예제·seed면 끊기지 않은 학습과 같은 배치를 본다)."""
    model.train()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    enc = [encode(tok, ex, max_len) for ex in examples]
    rng = np.random.default_rng(seed)
    order = np.concatenate([rng.permutation(len(enc)) for _ in range(math.ceil(epochs))])[: int(len(enc) * epochs)]
    chunks = [order[i:i + batch_size * 64] for i in range(0, len(order), batch_size * 64)]  # 64배치 묶음 안에서 길이순 정렬
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
        log(f"체크포인트 {ckpt}: {start}/{steps} step에서 이어 학습")
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
