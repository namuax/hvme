"""Train and evaluate one arm (config and seed) on a single GPU.

The caller pickles the training items and eval sets; this module is then launched as a subprocess per arm to spread arms across GPUs.
fla's Triton autotuner is not thread-safe, so one process never uses several GPUs.

    python -m humm.fast.seed_worker --inputs results/fast_v2_seeds/inputs.pkl --cfg story-gold --seed 1 --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from humm.paths import RESULTS, ROOT

OUT, LORAS, V1 = RESULTS / "fast_v2_seeds", ROOT / "models" / "fast_seeds", ROOT / "models" / "humm-fast-4b"


def run_arm(inputs: dict, cfg: str, seed: int, device: str, out: Path = OUT, loras: Path = LORAS) -> str:
    import torch

    from humm.fast.judge import FastJudge
    from humm.fast.lora import add_lora, merge_lora
    from humm.fast.train import calibrate, predict, train

    name = f"{cfg}-s{seed}"
    if (out / f"pred_{name}.csv").exists():
        return name
    judge = FastJudge.load(str(V1), device=device, batch_size=16)
    judge.temperature = {}
    torch.manual_seed(seed)
    add_lora(judge.model, r=16, alpha=32)
    hist = train(judge, inputs["data"][cfg], epochs=1, lr=1e-4, batch_size=8, grad_accum=4, eval_items=inputs["eval_sub"], eval_every=100, seed=seed)
    hist.assign(arm=name).to_csv(out / f"history_{name}.csv", index=False)
    torch.save({k: v.detach().cpu() for k, v in judge.model.state_dict().items() if "lora_" in k}, loras / f"{name}.pt")
    judge.model = merge_lora(judge.model).to(torch.bfloat16)
    temps = calibrate(judge, inputs["dev_items"])
    (out / f"temperature_{name}.json").write_text(json.dumps(temps, indent=1))
    res = pd.concat([predict(judge, its).assign(set=s) for s, its in inputs["sets"].items()], ignore_index=True)
    tmp = out / f"pred_{name}.csv.tmp"  # pred_*.csv appears only after the write completes, so it marks a finished arm
    res.assign(model=name, logits=res.logits.map(lambda z: json.dumps(np.round(z, 4).tolist())), probs=res.probs.map(lambda p: json.dumps(np.round(p, 6).tolist())))[
        ["model", "set", "id", "task", "kind", "tkey", "logits", "probs", "target_idx", "pred"]].to_csv(tmp, index=False)
    tmp.rename(out / f"pred_{name}.csv")
    return name


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args(argv)
    inputs = pickle.loads(Path(a.inputs).read_bytes())
    print(run_arm(inputs, a.cfg, a.seed, a.device), "done", flush=True)


if __name__ == "__main__":
    main()
