"""Hume-LM GRPO (Step 9-4): SFT v2.1에서 시작해 검증 가능 보상(엔진 정답)으로 도구 사용 풀이를 다듬는다.

- 문항마다 G개 rollout(도구 실행 포함, humm.lm.sample.generate_tools) → 보상 → 묶음 안에서 표준화한 이점(advantage)
- 보상: 정답 1, 'Final answer' 없음 −0.2, 반복 루프 −0.1, 4,000자 넘는 길이 최대 −0.1. 도구 오류 뒤 고친 재호출은 깎지 않는다
- 2차(proc=True): 이진·선택형 구조 유형(dsep·backadj·identify·corr2cause)은 답만 맞힌 풀이도 1을 받아 운이 강화됐다(1차 리포트 3-1).
  정답 보상에 과정 점수(엔진 풀이 humm.lm.solutions의 3단계 이후 핵심 줄 중 모델 출력에 있는 비율, 줄 안 나열 순서는 무시)를 곱한다
- 보상이 모두 같은 묶음은 기울기가 없으므로 버린다(DAPO의 동적 샘플링)
- 손실: 어시스턴트 토큰(도구 결과 제외)에 −A·log π를 토큰 평균한다. rollout마다 한 번 갱신하는 on-policy라 비율 클리핑은 없다.
  기준 모델(LoRA를 끈 SFT v2.1)과의 KL(k3 추정)을 β로 더한다. kl_target을 주면 β를 적응형으로 조절한다(KL이 목표의 2배를 넘으면 ×1.5, 절반 밑이면 ÷1.5, 시작 β 밑으로는 내리지 않는다.
  2차에서 하한 없이 돌렸더니 초반 KL이 작아 β가 3e-6까지 떨어져 제약이 꺼졌다)
- 3차(SFT v2.8에서 시작): 오류를 받고도 같은 인자로 다시 부르는 행동을 다룬다(v2.10은 복구 궤적 SFT로 바뀌지 않았다).
  문항에 'prefix'(trajectories.error_prefix: 실수를 넣은 첫 호출과 엔진 오류)가 있으면 그 뒤부터 rollout하고, 앞부분 턴은 손실에서 뺀다.
  rep>0이면 오류 바로 뒤 같은 인자 재호출마다 rep를 깎는다(앞부분 안의 재호출은 세지 않는다). dev_cot은 도구 없는 풀이(render 'cot')의 퇴보 감시용이다
- dev를 주면 eval_every step마다 greedy로 풀어 정답률·구조 유형 정답률·과정 점수를 기록한다(체크포인트 고르기)
- rollout은 두 GPU에서 나눠 돌린다: 학습 모델과, 매 step LoRA 가중치를 받는 생성 전용 하위 프로세스(Roller).
  fla의 Triton 자동 튜너가 스레드 안전하지 않아 스레드 대신 프로세스를 쓴다. Roller.close()로 끝내면 그 GPU 메모리도 풀린다
"""
import os
import queue
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import torch.multiprocessing as mp

import numpy as np
import pandas as pd
import torch

from ..fast.lora import LoRALinear, add_lora
from ..gym.formats import parse_final, render, reward
from .data import STRUCT, error_repeats, looping, process
from .sample import generate_tools, split_turns
from .sft import encode_messages


def prefix_calls(d: dict) -> int:
    return d.get("prefix", "").count("<tool_call>")


def uses_process(d: dict, out: str) -> bool:
    """과정 점수는 손풀이 구조 유형과 corr2cause 도구 인자에만 매긴다(causal_graph 도구 풀이에는 손풀이 줄이 없다)."""
    return d["query_type"] in STRUCT and ("call" in d["meta"] or "<tool_call>" not in out)


def score(d: dict, out: str, proc: bool = False, rep: float = 0.0) -> float:
    r = reward(d, out) * (process(d, out) if proc and uses_process(d, out) else 1.0)
    return (r - 0.2 * (parse_final(out) is None) - 0.1 * looping(out) - 0.1 * min(1.0, max(0, len(out) - 4000) / 4000)
            - rep * error_repeats(out, prefix_calls(d)))


def lora_params(model) -> dict:
    return {n: p for n, p in model.named_parameters() if p.requires_grad}


@contextmanager
def lora_off(model):
    """LoRA 보정을 끈 기준 모델(SFT v2.1)."""
    mods = [m for m in model.modules() if isinstance(m, LoRALinear)]
    scales = [m.scale for m in mods]
    for m in mods:
        m.scale = 0.0
    yield
    for m, s in zip(mods, scales):
        m.scale = s


def token_logps(model, ids: list[int], labels: list[int]) -> torch.Tensor:
    """labels가 -100이 아닌 위치의 log π(토큰). 그 위치의 로짓만 계산한다(어휘 248k라 전체 로짓은 크다)."""
    dev = model.device
    lab = torch.tensor(labels[1:], device=dev)
    pos = (lab != -100).nonzero().squeeze(1)
    with torch.autocast(dev.type, dtype=torch.bfloat16):
        logits = model(input_ids=torch.tensor([ids], device=dev), logits_to_keep=pos).logits[0].float()
    return logits.log_softmax(-1).gather(1, lab[pos, None]).squeeze(1)


def _roll_worker(model_path: str, device: str, r: int, alpha: float, gen_batch: int, inq, outq):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map=device).eval()
    add_lora(model, r=r, alpha=alpha)
    prov, mp_ = SimpleNamespace(tok=tok, model=model, batch_size=gen_batch), dict(model.named_parameters())
    outq.put("ready")
    parent = os.getppid()
    while True:
        try:
            job = inq.get(timeout=30)
        except queue.Empty:
            if os.getppid() != parent:  # 부모(커널)가 비정상 종료되면 GPU 메모리를 들고 남지 않게 끝낸다
                return
            continue
        if job is None:
            return
        sd, prompts, kw = job
        with torch.no_grad():
            for n, v in sd.items():
                mp_[n].copy_(v)
        outq.put(generate_tools(prov, prompts, **kw))


class Roller:
    """다른 GPU의 생성 전용 사본(하위 프로세스). submit(LoRA 가중치, 프롬프트) → result()."""

    def __init__(self, model_path: str, device: str = "cuda:0", r: int = 16, alpha: float = 32.0, gen_batch: int = 64):
        ctx = mp.get_context("spawn")
        self.inq, self.outq = ctx.Queue(), ctx.Queue()
        self.proc = ctx.Process(target=_roll_worker, args=(model_path, device, r, alpha, gen_batch, self.inq, self.outq), daemon=True)
        self.proc.start()
        assert self._get() == "ready"

    def _get(self):
        while True:
            try:
                return self.outq.get(timeout=60)
            except queue.Empty:
                assert self.proc.is_alive(), f"Roller 프로세스가 죽었다(exit {self.proc.exitcode})"

    def submit(self, policy, prompts: list[str], **kw):
        self.inq.put(({n: p.detach().cpu() for n, p in lora_params(policy).items()}, prompts, kw))

    def result(self) -> list[str]:
        return self._get()

    def close(self):
        self.inq.put(None)
        self.proc.join()


def rollout(policy, tok, roller, items: list[dict], group: int, gen_batch: int, do_sample: bool = True, mode: str = "tool", **kw) -> list[str]:
    """문항마다 group개를 생성한다. roller가 있으면 앞 절반을 그쪽 GPU에서 동시에 돌린다. 'prefix'가 있는 문항은 그 뒤부터 생성한다."""
    kw |= {"do_sample": do_sample, "tools": mode == "tool"}
    prompts = [render(d, mode) for d in items for _ in range(group)]
    pre = [d.get("prefix", "") for d in items for _ in range(group)]
    half = len(items) // 2 * group if roller is not None else 0
    if half:
        roller.submit(policy, prompts[:half], prefixes=pre[:half], **kw)
    mine = generate_tools(SimpleNamespace(tok=tok, model=policy, batch_size=gen_batch), prompts[half:], prefixes=pre[half:], **kw)
    return (roller.result() if half else []) + mine


def train_grpo(policy, roller, tok, items: list[dict], steps: int = 60, prompts_per_step: int = 16, group: int = 8, lr: float = 2e-5,
               beta: float = 0.02, temperature: float = 0.8, max_new_tokens: int = 2048, max_len: int = 6144, gen_batch: int = 64,
               ckpt_dir: Path | None = None, ckpt_every: int = 10, seed: int = 0, start: int = 0, proc: bool = False, kl_target: float | None = None,
               dev: list[dict] | None = None, rep: float = 0.0, dev_cot: list[dict] | None = None, cot_tokens: int = 8192, log=print) -> pd.DataFrame:
    """policy: LoRA를 붙인 학습 모델. roller: 다른 GPU의 Roller(없으면 None). items: 보상을 줄 Gym 문항 풀.
    start: 체크포인트(lora_step{start}.pt)를 불러온 뒤 이어 돌릴 때. 같은 문항 순서를 따르고, opt_step{start}.pt가 있으면 옵티마이저 상태와 β도 잇는다.
    proc: 구조 유형 과정 점수 보상. kl_target: 적응형 β. dev: ckpt_every step마다(그리고 시작 전) greedy 평가할 문항('prefix'가 있는 문항은 dev_fix로 따로 낸다).
    rep: 오류 뒤 같은 인자 재호출 벌점. dev_cot: 같은 때 도구 없이(cot, cot_tokens) 풀 문항."""
    rng, beta0 = np.random.default_rng(seed), beta
    params = list(lora_params(policy).values())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0, betas=(0.9, 0.99))
    policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    csv = ckpt_dir / "grpo_history.csv" if ckpt_dir else None
    hist = pd.read_csv(csv).query(f"step <= {start}").to_dict("records") if start else []
    if start and (ckpt_dir / f"opt_step{start}.pt").exists():
        st = torch.load(ckpt_dir / f"opt_step{start}.pt")
        opt.load_state_dict(st["opt"])
        beta = st["beta"]

    def evaluate():
        policy.eval()
        policy.config.use_cache = True
        outs = rollout(policy, tok, roller, dev, 1, gen_batch, do_sample=False, max_new_tokens=max_new_tokens)
        acc = np.array([reward(d, o) for d, o in zip(dev, outs)])
        st, pre = np.array([d["query_type"] in STRUCT for d in dev]), np.array(["prefix" in d for d in dev])
        res = {"dev_acc": acc[~pre].mean(), "dev_struct": acc[st & ~pre].mean(),
               "dev_proc": np.mean([process(d, o) for d, o, p in zip(dev, outs, pre) if uses_process(d, o) and not p] or [1.0]),
               "dev_rep": np.mean([error_repeats(o, prefix_calls(d)) > 0 for d, o in zip(dev, outs)])}
        if pre.any():
            res["dev_fix"] = acc[pre].mean()
        if dev_cot:
            res["dev_cot"] = np.mean([reward(d, o) for d, o in zip(dev_cot, rollout(policy, tok, roller, dev_cot, 1, gen_batch, do_sample=False, mode="cot",
                                                                                      max_new_tokens=cot_tokens, max_rounds=1))])
        return res

    if dev and not start:
        hist.append({"step": 0, **evaluate()})
        log(hist[-1])
    for step in range(steps):
        batch = [items[i] for i in rng.choice(len(items), prompts_per_step, replace=False)]
        if step < start:
            continue
        t0 = time.time()
        policy.eval()
        policy.config.use_cache = True
        outs = rollout(policy, tok, roller, batch, group, gen_batch, temperature=temperature, max_new_tokens=max_new_tokens)
        t1 = time.time()
        r = np.array([score(d, o, proc, rep) for d, o in zip((d for d in batch for _ in range(group)), outs)]).reshape(len(batch), group)
        acc = np.array([reward(d, o) for d, o in zip((d for d in batch for _ in range(group)), outs)])
        adv = (r - r.mean(1, keepdims=True)) / (r.std(1, keepdims=True) + 1e-4)
        keep = [(i, j) for i in range(len(batch)) for j in range(group) if r[i].std() > 0]
        def turns(d, out):  # 주어진 앞부분의 어시스턴트 턴은 문맥으로만 쓴다
            ms, k = split_turns(out), prefix_calls(d)
            return [m | {"train": False} if m["role"] == "assistant" and n < 2 * k else m for n, m in enumerate(ms)]

        enc = {(i, j): encode_messages(tok, {"messages": [{"role": "user", "content": render(batch[i], "tool")}] + turns(batch[i], outs[i * group + j]),
                                             "tools": True}, max_len) for i, j in keep}
        ntok = sum(sum(l != -100 for l in enc[k][1][1:]) for k in keep)
        policy.train()
        policy.config.use_cache = False
        kls = []
        for i, j in keep:
            ids, labels = enc[i, j]
            with torch.no_grad(), lora_off(policy):
                ref = token_logps(policy, ids, labels)
            lp = token_logps(policy, ids, labels)
            kl = torch.exp(ref - lp) - (ref - lp) - 1
            ((-adv[i, j] * lp + beta * kl).sum() / ntok).backward()
            kls.append(kl.mean().item())
        if keep:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
        kl = float(np.mean(kls)) if kls else 0.0
        pre = np.array(["prefix" in d for d in batch for _ in range(group)])
        reps = [error_repeats(o, prefix_calls(d)) > 0 for d, o in zip((d for d in batch for _ in range(group)), outs)]
        hist.append({"step": step + 1, "score": r.mean(), "acc": acc.mean(), "fix": acc[pre].mean() if pre.any() else np.nan, "rep": np.mean(reps), "groups_kept": len({i for i, _ in keep}) / len(batch), "tokens": ntok,
                     "kl": kl, "beta": beta, "chars": float(np.mean([len(o) for o in outs])), "gen_sec": round(t1 - t0), "train_sec": round(time.time() - t1)})
        if kl_target and kls:
            beta = max(beta0, beta * (1.5 if kl > 2 * kl_target else 1 / 1.5 if kl < kl_target / 2 else 1.0))
        if ckpt_dir and ((step + 1) % ckpt_every == 0 or step + 1 == steps):
            torch.save({n: p.detach().cpu() for n, p in lora_params(policy).items()}, ckpt_dir / f"lora_step{step + 1}.pt")
            torch.save({"opt": opt.state_dict(), "beta": beta}, ckpt_dir / f"opt_step{step + 1}.pt")
            if dev:
                hist[-1] |= evaluate()
        log(hist[-1])
        if csv:
            pd.DataFrame(hist).to_csv(csv, index=False)
    policy.eval()
    policy.config.use_cache = True
    return pd.DataFrame(hist)


def load_lora(model, path: Path):
    """lora_step*.pt를 LoRA를 붙인 모델에 불러온다."""
    sd = torch.load(path)
    mp = dict(model.named_parameters())
    with torch.no_grad():
        for n, v in sd.items():
            mp[n].copy_(v)
    return model

