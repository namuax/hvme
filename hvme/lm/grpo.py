"""Hvme-LM GRPO: starting from the SFT model, refine tool-using solutions with a verifiable reward (the engine answer).

- G rollouts per item (including tool execution, hvme.lm.sample.generate_tools) -> reward -> advantage standardized within the group
- Reward: 1 for a correct answer, -0.2 for a missing 'Final answer', -0.1 for a repetition loop, up to -0.1 for length over 4,000 characters. A corrected re-call after a tool error is not penalized
- proc=True: on binary/multiple-choice structure types (dsep, backadj, identify, corr2cause) a solution with only the right answer also got 1, which reinforced luck.
  The correct-answer reward is multiplied by a process score (fraction of the engine solution's (hvme.lm.solutions) key lines after step 3 that appear in the model output; order within a line is ignored)
- Groups whose rewards are all equal have no gradient and are dropped (DAPO dynamic sampling)
- Loss: -A * log pi averaged over assistant tokens (tool results excluded). It is on-policy with one update per rollout, so there is no ratio clipping.
  The KL to the reference model (the SFT model with LoRA disabled; k3 estimate) is added with weight beta. If kl_target is given, beta adapts (x1.5 when KL exceeds twice the target, /1.5 when below half, never below the initial beta;
  without that floor, beta fell to 3e-6 because early KL was small, which switched the constraint off)
- Error recovery: handles re-calling with the same arguments after receiving an error.
  If an item has a 'prefix' (trajectories.error_prefix: a first call containing a mistake and the engine error), rollout starts after it and the prefix turns are excluded from the loss.
  If rep>0, each same-argument re-call right after an error is penalized by rep (re-calls inside the prefix are not counted). dev_cot monitors regression of tool-free solving (render 'cot')
- If dev is given, every eval_every steps it solves greedily and records accuracy, structure-type accuracy and process score (for choosing checkpoints)
- Rollouts run on two GPUs: the training model, and a generation-only subprocess (Roller) that receives the LoRA weights every step.
  Processes are used instead of threads because fla's Triton autotuner is not thread-safe. Ending with Roller.close() also frees that GPU's memory
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
    """The process score applies only to hand-solved structure types and to corr2cause tool arguments (causal_graph tool solutions have no hand-solution lines)."""
    return d["query_type"] in STRUCT and ("call" in d["meta"] or "<tool_call>" not in out)


def score(d: dict, out: str, proc: bool = False, rep: float = 0.0) -> float:
    r = reward(d, out) * (process(d, out) if proc and uses_process(d, out) else 1.0)
    return (r - 0.2 * (parse_final(out) is None) - 0.1 * looping(out) - 0.1 * min(1.0, max(0, len(out) - 4000) / 4000)
            - rep * error_repeats(out, prefix_calls(d)))


def lora_params(model) -> dict:
    return {n: p for n, p in model.named_parameters() if p.requires_grad}


@contextmanager
def lora_off(model):
    """Reference model (the SFT model with the LoRA correction disabled)."""
    mods = [m for m in model.modules() if isinstance(m, LoRALinear)]
    scales = [m.scale for m in mods]
    for m in mods:
        m.scale = 0.0
    yield
    for m, s in zip(mods, scales):
        m.scale = s


def token_logps(model, ids: list[int], labels: list[int]) -> torch.Tensor:
    """log pi(token) at positions where labels != -100. Only the logits at those positions are computed (the vocabulary is 248k, so full logits are large)."""
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
            if os.getppid() != parent:  # if the parent (kernel) dies abnormally, exit so this process does not keep holding GPU memory
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
    """Generation-only copy on another GPU (subprocess). submit(LoRA weights, prompts) -> result()."""

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
                assert self.proc.is_alive(), f"Roller process died (exit {self.proc.exitcode})"

    def submit(self, policy, prompts: list[str], **kw):
        self.inq.put(({n: p.detach().cpu() for n, p in lora_params(policy).items()}, prompts, kw))

    def result(self) -> list[str]:
        return self._get()

    def close(self):
        self.inq.put(None)
        self.proc.join()


def rollout(policy, tok, roller, items: list[dict], group: int, gen_batch: int, do_sample: bool = True, mode: str = "tool", **kw) -> list[str]:
    """Generate `group` samples per item. If a roller exists, the first half runs on its GPU concurrently. Items with a 'prefix' are generated from after the prefix."""
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
    """policy: training model with LoRA attached. roller: Roller on another GPU (None if absent). items: pool of Gym items to reward.
    start: when resuming after loading a checkpoint (lora_step{start}.pt). Follows the same item order and, if opt_step{start}.pt exists, restores optimizer state and beta too.
    proc: structure-type process-score reward. kl_target: adaptive beta. dev: items evaluated greedily every ckpt_every steps (and before the start); items with a 'prefix' are passed separately as dev_fix.
    rep: penalty for re-calling with the same arguments after an error. dev_cot: items solved at the same times without tools (cot, cot_tokens)."""
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
        def turns(d, out):  # assistant turns in the given prefix are used only as context
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
    """Load lora_step*.pt into a model with LoRA attached."""
    sd = torch.load(path)
    mp = dict(model.named_parameters())
    with torch.no_grad():
        for n, v in sd.items():
            mp[n].copy_(v)
    return model

