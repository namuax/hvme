"""GPU job runner: launches teacher sampling and evaluation as per-GPU subprocesses so two GPUs run concurrently.

    python -m humm.lm.jobs sample   --model Qwen/Qwen3.5-9B --device cuda:1 --items a.jsonl --out b.jsonl [--mode cot --n 2]
    python -m humm.lm.jobs paraphrase --model Qwen/Qwen3.5-9B --items a.jsonl --out p.jsonl --max-new-tokens 1024 [--style story]   (context rewriting)
    python -m humm.lm.jobs gym      --model <path> --device cuda:0 --items a.jsonl --mode tool --out c.csv
    python -m humm.lm.jobs cladder  --model <path> --device cuda:1 --mode tool|causalcot [--per-rung 50] --out d.csv
    python -m humm.lm.jobs corr2cause --model <path> --mode tool|causalcot --n 300 --out e.csv   (n items from the Corr2Cause test set)
    python -m humm.lm.jobs caugym   --model <path> --mode tool|cot --per-type 15 --out g.csv                (the 6 CauGym test types, --per-type items per type)
    python -m humm.lm.jobs calm     --model <path> --mode tool|basic --per-type 30 --out h.csv                (CaLM-lite, 45 English tasks, original basic prompt and original scoring)
    python -m humm.lm.jobs math|kmmlu --model <path> --out f.csv                                  (general ability: all of MATH-500, KMMLU with --per-subject items per subject (default 7); --split train collects trajectories)
    --lora <path>/lora_step20.pt: merges a GRPO LoRA checkpoint into the model before evaluating (the model column in the results is the checkpoint name)

In notebooks, start jobs with launch() and wait with wait(). Logs go to the --log file.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from ..paths import ROOT

PY = sys.executable


def launch(cmd: str, *args, log: str | Path, **opts) -> subprocess.Popen:
    """Launch python -m humm.lm.jobs <cmd> ... in the background."""
    argv = [PY, "-m", "humm.lm.jobs", cmd, *map(str, args)] + [x for k, v in opts.items() for x in (f"--{k.replace('_', '-')}", str(v))]
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(argv, stdout=open(log, "w"), stderr=subprocess.STDOUT, cwd=ROOT, env=os.environ | {"PYTHONPATH": str(ROOT)})


def wait(*procs, every: int = 600):
    """Wait until all finish, printing progress every `every` seconds. Stops if any job fails."""
    t = nxt = time.time()
    while any(p.poll() is None for p in procs):
        time.sleep(30)
        if time.time() >= nxt + every:
            nxt = time.time()
            print(f"{(nxt - t) / 60:.0f} min elapsed, {sum(p.poll() is None for p in procs)} jobs remaining", flush=True)
    bad = [p.args for p in procs if p.returncode]
    assert not bad, f"failed jobs: {bad}"
    return round(time.time() - t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sample", "paraphrase", "gym", "cladder", "corr2cause", "caugym", "calm", "math", "kmmlu"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--items")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="tool")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--lora")
    ap.add_argument("--quant", default="", choices=["", "int8", "nf4"], help="bitsandbytes weight quantisation (appended to the model column of the results)")
    ap.add_argument("--style", default="context", help="paraphrase: context (rewrite the context only) | story (rewrite the whole item in another domain) | story2 (rotate domains and text formats)")
    ap.add_argument("--per-subject", type=int, default=7)
    ap.add_argument("--per-type", type=int, default=15)
    ap.add_argument("--per-rung", type=int, default=50, help="cladder: items per rung (200 gives the 600-item set)")
    ap.add_argument("--split", default="test", help="kmmlu: test (evaluation) or train (collect Korean general-ability trajectories)")
    ap.add_argument("--lm-url", help="URL of an OpenAI-compatible server (vLLM). When set, --model is only used as the tokenizer path")
    ap.add_argument("--served", default="humm-lm-9b", help="model name served at --lm-url")
    a = ap.parse_args()

    import pandas as pd

    from ..gym.dataset import read_jsonl, write_jsonl
    from ..gym.formats import TOOL, parse_final, render, reward
    from ..providers.hf import HFProvider
    from .sample import generate_tools, sample_tools

    if a.lm_url:
        from ..providers.remote import RemoteProvider
        lm = RemoteProvider(a.lm_url, a.served, tokenizer=a.model, batch_size=a.batch)
    else:
        lm = HFProvider(a.model, device=a.device, batch_size=a.batch, quant=a.quant)
    name = Path(a.model).name + (f"@{a.quant}" if a.quant else "")
    if a.lora:
        from ..fast.lora import add_lora, merge_lora
        from .grpo import load_lora
        add_lora(lm.model, r=16, alpha=32)
        merge_lora(load_lora(lm.model, a.lora))
        name = f"{name}+{Path(a.lora).stem}"
    t = time.time()
    if a.cmd == "sample":
        items = read_jsonl(a.items)
        df = sample_tools(lm, items, n=a.n, mode=a.mode, max_new_tokens=a.max_new_tokens, max_rounds=3 if a.mode == "tool" else 1)
        write_jsonl(df.to_dict("records"), a.out)
        print(df.groupby("task").agg(n=("reward", "size"), reward=("reward", "mean"), loop=("loop", "mean")).round(3))
    elif a.cmd == "paraphrase":  # rewrite the Gym context as natural prose (humm.fast.distill)
        from ..fast.distill import paraphrase_prompt, story2_prompt, story_prompt
        items = read_jsonl(a.items)
        fn = {"context": lambda d, i: paraphrase_prompt(d), "story": lambda d, i: story_prompt(d), "story2": story2_prompt}[a.style]
        outs = generate_tools(lm, [fn(d, i) for i, d in enumerate(items)], tools=False, do_sample=True, max_new_tokens=a.max_new_tokens, max_rounds=1)
        write_jsonl([{"id": d["id"], "output": o} for d, o in zip(items, outs)], a.out)
    elif a.cmd == "gym":
        items = read_jsonl(a.items)
        outs = generate_tools(lm, [render(d, a.mode) for d in items], tools=a.mode == "tool", max_new_tokens=a.max_new_tokens,
                              max_rounds=3 if a.mode == "tool" else 1)
        df = pd.DataFrame(items).assign(output=outs, answer=[parse_final(o) for o in outs], correct=[bool(reward(d, o)) for d, o in zip(items, outs)],
                                        mode=a.mode, model=name)
        df.drop(columns=["meta"]).to_csv(a.out, index=False)
        print(df.groupby("query_type").correct.mean().round(3), "\noverall", round(df.correct.mean(), 3))
    elif a.cmd in ("math", "kmmlu"):
        from ..bench import datasets as D, general as G
        df = D.load_math500() if a.cmd == "math" else D.load_kmmlu(a.per_subject, split=a.split)
        tmpl = G.MATH_PROMPT if a.cmd == "math" else G.KMMLU_PROMPT
        outs = generate_tools(lm, [tmpl.format(**r) for r in df.to_dict("records")], tools=False, max_new_tokens=a.max_new_tokens, max_rounds=1)
        df = df.assign(output=outs, model=name, bench=a.cmd)
        df["correct"] = ([G.math_correct(o, x) for o, x in zip(outs, df.answer)] if a.cmd == "math" else
                         [G.parse_choice(o) == x for o, x in zip(outs, df.answer)])
        df.to_csv(a.out, index=False)
        print("accuracy", round(df.correct.mean(), 3))
    elif a.cmd == "caugym":
        from ..bench import datasets as D, general as G
        df = pd.concat([D.load_caugym(s, a.per_type) for s in D.CAUGYM_SETS], ignore_index=True)
        suffix = f"\n\n{TOOL['en'].format(fmt='the JSON object asked for above')}" if a.mode == "tool" else ""
        outs = generate_tools(lm, [p + suffix for p in df.problem], tools=a.mode == "tool", max_new_tokens=a.max_new_tokens, max_rounds=3 if a.mode == "tool" else 1)
        df = df.drop(columns=["Reason"]).assign(output=outs, correct=[G.caugym_correct(o, x) for o, x in zip(outs, df.answer)], mode=a.mode, model=name, bench="caugym")
        df.to_csv(a.out, index=False)
        print(df.pivot_table(index="Type", columns="set", values="correct").round(2), "\noverall", round(df.correct.mean(), 3))
    elif a.cmd == "calm":
        from ..bench import calm as C
        items = [d | {"task": t} for t in C.TASKS for d in C.questions(t, n=a.per_type)]
        use = [a.mode == "tool" or a.mode == "route" and C.needs_tools(d["question"]) for d in items]  # route: use tools only for problems that give causal relations or probabilities
        suffix = f"\n\n{TOOL['en'].format(fmt='the answer format asked for above')}"
        outs = [None] * len(items)
        for flag in (True, False):
            idx = [i for i, u in enumerate(use) if u == flag]
            gen = generate_tools(lm, [items[i]["question"] + suffix * flag for i in idx], tools=flag, max_new_tokens=a.max_new_tokens, max_rounds=3 if flag else 1) if idx else []
            for i, o in zip(idx, gen):
                outs[i] = o
        df = pd.DataFrame({"task": [d["task"] for d in items], "index": [d["index"] for d in items], "tools": use, "output": outs,
                           "correct": [C.correct(d["task"], d, o) for d, o in zip(items, outs)],
                           "correct_tail": [C.correct(d["task"], d, C.tail(o)) for d, o in zip(items, outs)]}).assign(mode=a.mode, model=name, bench="calm")
        df.to_csv(a.out, index=False)
        print(df.groupby("task")[["correct", "correct_tail"]].mean().round(2), "\noverall", df[["correct", "correct_tail"]].mean().round(3).to_dict())
    elif a.cmd == "corr2cause":
        from ..bench import datasets as D, prompts as P, runner as R
        cc = D.load_corr2cause(n=a.n, seed=0)
        if a.mode == "tool":
            prompts = [f"{P.CORR2CAUSE_NOUL.format(input=x)}\n\n{TOOL['en'].format(fmt='yes or no')}" for x in cc.input]
            outs = generate_tools(lm, prompts, tools=True, max_new_tokens=a.max_new_tokens)
        else:
            outs = generate_tools(lm, [P.corr2cause_cot(r) for _, r in cc.iterrows()], tools=False, max_new_tokens=a.max_new_tokens, max_rounds=1)
        df = cc.assign(output=outs, pred=[R.parse_yes_no(o) for o in outs], mode=f"cot-{a.mode}", model=name, bench="corr2cause")
        df.to_csv(a.out, index=False)
        print(R.summarize(df)[["acc", "f1", "majority", "parse_fail"]].round(3))
    else:
        from ..bench import datasets as D, prompts as P, runner as R
        cl = D.load_cladder(n_per_rung=200, seed=0).groupby("rung").head(a.per_rung)
        if a.mode == "tool":
            prompts = [f"{p}\n\n{TOOL['en'].format(fmt='yes or no')}" for p in cl.prompt]
            outs = generate_tools(lm, prompts, tools=True, max_new_tokens=a.max_new_tokens)
        else:
            outs = generate_tools(lm, [P.cladder_cot(r) for _, r in cl.iterrows()], tools=False, max_new_tokens=a.max_new_tokens, max_rounds=1)
        df = cl.assign(output=outs, pred=[R.parse_yes_no(o) for o in outs], mode=f"cot-{a.mode}", model=name, bench="cladder")
        df.to_csv(a.out, index=False)
        print(R.summarize(df, by=["rung"])[["rung", "acc", "parse_fail"]].round(3), "\noverall", round((df.pred == df.label).mean(), 3))
    print(f"{time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
