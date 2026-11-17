import sys
import nbformat as nbf

cells = [
"""md:# 09. Hvme-Fast — a single-pass causal judgment model (Noul · Choice · Score)

Hvme-Fast produces probabilities over answer tokens (yes/no, option letters, step numbers) in a single forward pass, without generating any reasoning text.

| Section | What it does | GPU |
|---|---|---|
| 1 | Build the training, calibration and evaluation items (Hvme-Gym + bnlearn edge direction and counterfactual verification) | none |
| 2 | Zero-shot baselines of the base models, and the effect of temperature calibration alone | yes |
| 3 | Fine-tune Qwen3.5-4B (gold labels, cross-entropy = a proper scoring rule) → temperature calibration | yes |
| 4 | Evaluation: accuracy, ECE, Brier, AUROC, and a confidence-bin table | none |
| 5 | Latency | yes |
| 6 | Routing: accuracy versus escalation rate when only low-confidence items go to the slow path (9B CoT) | none |
| 7 | Graph-construction integration: replace pairwise direction judgments (notebook 06) and L3 counterfactual verification (notebook 07) with Fast, and compare the number of 9B calls | yes |

Completion criteria: ECE of at most 0.05 on the Hvme-Gym test set, and lower LLM call cost during graph construction.

> Data split: Gym uses its own train (training), dev (calibration) and test (evaluation) splits. The bnlearn networks are split into asia, sachs and alarm (training), insurance (calibration), and child, hepar2, win95pts and hailfinder (evaluation; the held-out networks of notebook 06).""",
"""import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # set before importing torch (reduces memory fragmentation)

import gc
import json
import time

import numpy as np
import pandas as pd

from hvme.fast import from_gym, metrics, network_items, reliability, routing_curve
from hvme.graph.tabular.networks import DEV, HELDOUT
from hvme.gym.dataset import read_jsonl
from hvme.paths import GYM, RESULTS, ROOT
from hvme.validate.benchmark import sample_bnlearn

N_TRAIN = 40_000  # how many of the 100k Gym train items to use; adjust based on throughput (first log in section 3)
TRAIN_NETS, CALIB_NETS = ["asia", "sachs", "alarm"], ["insurance"]
graphs = {n: sample_bnlearn(n, 10)[0] for n in DEV + HELDOUT}
rng = np.random.default_rng(0)
gym_train = read_jsonl(GYM / "train.jsonl")
train_items = from_gym([gym_train[i] for i in rng.choice(len(gym_train), N_TRAIN, replace=False)]) + network_items(TRAIN_NETS, graphs) * 4
dev_items = from_gym(read_jsonl(GYM / "dev.jsonl")) + network_items(CALIB_NETS, graphs)
test_items = from_gym(read_jsonl(GYM / "test.jsonl")) + network_items(HELDOUT, graphs)
group = lambda d: d["source"] if d["source"] == "gym" else d["task"]
pd.DataFrame([{"split": s, "group": group(d), "kind": d["kind"]} for s, its in [("train", train_items), ("dev", dev_items), ("test", test_items)] for d in its]
             ).value_counts().unstack("split").fillna(0).astype(int)""",
"""md:## 2. (GPU) Zero-shot baselines

Use the base models as-is as Fast judges. Compare uncalibrated (T=1) with a per-item-kind temperature fitted on dev. Temperature calibration does not change accuracy; it only changes how sharp the probabilities are.""",
"""gpu:import torch

from hvme.fast.judge import FastJudge
from hvme.fast.train import calibrate, evaluate, predict, rescale, train

MODELS = {
    "Qwen/Qwen3.5-4B": "cuda:0",
    "Qwen/Qwen3.5-9B": "cuda:1",
}
preds, temps = {}, {}
for model_id, device in MODELS.items():
    name = model_id.split("/")[-1]
    j = FastJudge.load(model_id, device=device, batch_size=16)
    temps[name] = calibrate(j, dev_items)
    res = predict(j, test_items)
    preds[f"{name} zero-shot T=1"], preds[f"{name} zero-shot +T"] = rescale(res, {}), res
    del j
    gc.collect()
    torch.cuda.empty_cache()
temps""",
"""md:## 3. (GPU) Fine-tuning Qwen3.5-4B

- Loss: cross-entropy between the restricted softmax over answer tokens and the gold (one-hot) label. It is a proper scoring rule, so training itself encourages calibration.
- The default is **LoRA** (`FULL_FT = False`): the base weights stay frozen in bf16 and only rank-16 adapters on all linear layers are trained, in fp32. It needs about 20 GB of GPU memory. After training, the adapters are merged into the weights and saved as a regular model.
- `FULL_FT = True` is full fp32 fine-tuning. Weights, gradients and AdamW state alone take about 64 GB, and with activations it approaches 90 GB, so choose it only when the GPU is exclusively yours. Check with `nvidia-smi` that no other process is using it first.
- Use the `sec` column of the first evaluation log (step 200) to estimate the total time. If it takes too long, reduce `N_TRAIN`. Installing `flash-linear-attention` and `causal-conv1d` speeds up Qwen3.5's linear attention.""",
"""gpu:from hvme.fast.lora import add_lora, merge_lora

FULL_FT = False
judge = FastJudge.load("Qwen/Qwen3.5-4B", device="cuda:0", dtype=torch.float32 if FULL_FT else torch.bfloat16, batch_size=16)
n_params = sum(p.numel() for p in judge.model.parameters()) if FULL_FT else add_lora(judge.model, r=16, alpha=32)
print(f"Trainable parameters: {n_params:,} total", "(full)" if FULL_FT else "(LoRA)")
eval_sub = [dev_items[i] for i in np.random.default_rng(1).choice(len(dev_items), 600, replace=False)]
hist = train(judge, train_items, epochs=1, lr=1e-5 if FULL_FT else 1e-4, batch_size=8, grad_accum=4, eval_items=eval_sub, eval_every=200)
hist.to_csv(RESULTS / "fast_history.csv", index=False)
hist""",
"""gpu:judge.model = judge.model if FULL_FT else merge_lora(judge.model)
judge.model.to(torch.bfloat16)
judge.name = "hvme-fast-4b"
temps["hvme-fast-4b"] = calibrate(judge, dev_items)
res = predict(judge, test_items)
preds["hvme-fast-4b T=1"], preds["hvme-fast-4b +T"] = rescale(res, {}), res
judge.save(ROOT / "models" / "hvme-fast-4b", {"base": "Qwen/Qwen3.5-4B", "method": "full" if FULL_FT else "lora r=16", "n_train": len(train_items), "train_nets": TRAIN_NETS})
temps""",
"""md:## 4. Evaluation

`group` is gym (all of Hvme-Gym), pairwise (edge-direction Choice items on the held-out networks) or verify (counterfactual-dependence Noul items on the held-out networks). Per-item probabilities are saved to `results/fast_predictions.csv` so they can be re-analyzed later without a GPU.

> Do not judge by ECE alone. Even a chance-level model gets an ECE near 0 if the temperature is raised enough to push all probabilities toward 0.5. Read the 'ECE of at most 0.05' criterion **only when accuracy, Brier and AUROC are better than zero-shot**.""",
"""gpu:pred_df = pd.concat([r.assign(model=m, group=[group(d) for d in r.to_dict("records")]) for m, r in preds.items()], ignore_index=True)
pred_df.assign(probs=pred_df.probs.map(lambda p: json.dumps(np.round(p, 5).tolist())), logits=pred_df.logits.map(lambda z: json.dumps(np.round(z, 4).tolist())))[
    ["id", "model", "group", "task", "kind", "lang", "source", "target_idx", "pred", "probs", "logits"]].to_csv(RESULTS / "fast_predictions.csv", index=False)
summ = pd.concat([evaluate(None, g, by=("group",)).assign(model=m) for m, g in pred_df.groupby("model", sort=False)])
summ.to_csv(RESULTS / "fast_eval.csv", index=False)
summ.pivot_table(index="model", columns="group", values=["acc", "ece"], sort=False).round(3)""",
"""gpu:gym = pred_df[pred_df.group == "gym"]
by_task = pd.concat([evaluate(None, g, by=("task",)).assign(model=m) for m, g in gym.groupby("model", sort=False)])
by_task.pivot_table(index="task", columns="model", values="acc", sort=False).round(3)""",
"""gpu:best = pred_df[(pred_df.model == "hvme-fast-4b +T") & (pred_df.group == "gym")]
print("Gym test overall:", {k: round(v, 4) for k, v in metrics(list(best.probs), best.target_idx).items()})
reliability(list(best.probs), best.target_idx.to_numpy())""",
"""md:## 5. (GPU) Latency

Measure forward-pass time per item at batch size 1 and with batches of 32. The 9B figure is measured the same way (option logits); 9B CoT generates thousands of tokens per item and takes tens of seconds.""",
"""gpu:def latency(j, items, batch, reps=5):
    prompts = [d["prompt"] for d in items[: batch * reps]]
    j.batch_size = batch
    j.logits(prompts[:batch], items[0]["labels"])  # warm-up
    torch.cuda.synchronize()
    out = []
    for r in range(reps):
        t = time.time()
        j.logits(prompts[r * batch:(r + 1) * batch], items[0]["labels"])
        torch.cuda.synchronize()
        out.append((time.time() - t) / batch * 1000)
    return out

sample = [d for d in test_items if d["source"] == "gym" and d["labels"] == ["Yes", "No"]][:200]
nine = FastJudge.load("Qwen/Qwen3.5-9B", device="cuda:1")
lat = pd.DataFrame([{"model": n, "batch": b, "ms_per_item_p50": np.percentile(v, 50), "ms_per_item_p95": np.percentile(v, 95)}
                    for n, j in [("hvme-fast-4b", judge), ("Qwen3.5-9B", nine)] for b in (1, 32) for v in [latency(j, sample, b)]])
del nine
gc.collect()
torch.cuda.empty_cache()
judge.batch_size = 16
lat.to_csv(RESULTS / "fast_latency.csv", index=False)
lat.round(1)""",
"""md:## 6. Routing (only low-confidence items go to the slow path)

The slow path is the 600 Gym test items solved by 9B CoT in the Gym baseline run. Replace Fast's answer with the CoT answer only for items whose top option probability is below the threshold, and look at the escalation rate and the final accuracy. CoT parse failures are treated as a uniform distribution.""",
"""cot = pd.read_csv(RESULTS / "gym_baseline_qwen35.csv").query("mode == 'cot' and model == 'Qwen/Qwen3.5-9B'")
fast = pd.read_csv(RESULTS / "fast_predictions.csv").query("model == 'hvme-fast-4b +T'").set_index("id")
cot = cot[cot.id.isin(fast.index)]
f_probs = [np.array(json.loads(fast.loc[i, "probs"])) for i in cot.id]
def cot_onehot(answer, kind, k):
    idx = {"noul": {"yes": 0, "no": 1}.get(answer), "choice": "ABCDEFGH".find(answer) if isinstance(answer, str) else None,
           "score": int(answer) - 1 if isinstance(answer, str) and answer.isdigit() else None}[kind]
    return np.eye(k)[idx] if idx is not None and 0 <= idx < k else np.full(k, 1 / k)
s_probs = [cot_onehot(a, k, len(p)) for a, k, p in zip(cot.answer, cot.kind, f_probs)]
target = fast.loc[cot.id, "target_idx"].to_numpy()
pad = lambda ps: np.array([np.r_[p, np.zeros(8 - len(p))] for p in ps])  # pad to a common number of options so they form one matrix
curve = pd.DataFrame(routing_curve(pad(f_probs), pad(s_probs), target, np.round(np.arange(0.5, 1.001, 0.05), 2)))
print("Fast only:", round(float(np.mean([p.argmax() == t for p, t in zip(f_probs, target)])), 3), "| 9B CoT only:", round(float(np.mean([p.argmax() == t for p, t in zip(s_probs, target)])), 3))
curve.to_csv(RESULTS / "fast_routing.csv", index=False)
curve.round(3)""",
"""md:## 7. (GPU) Graph-construction integration

**Tabular (notebook 06)**: on the four held-out networks, replace the pairwise direction judgment with (a) 9B, (b) Hvme-Fast, or (c) Fast first with 9B only when confidence is low (`Routed`), and compare SID and the number of 9B calls. The data and settings are the same as in section 7 of notebook 06 (5,000 rows, hybrid mode). 9B responses are replayed from the notebook 06 recordings rather than regenerated.

**Text (notebook 07)**: replace only counterfactual verification (L3) with Fast. Extraction and normalization replay the 9B recordings, and verification queries are answered by Fast first via `Routed`.""",
"""gpu:from hvme.bench.metrics import graph_report
from hvme.fast.router import Routed
from hvme.graph.tabular import NETWORKS, tabular_to_hcg
from hvme.graph.text import evaluate as text_eval, gold_ids, make_corpus, text_to_hcg
from hvme.providers.cache import CachedProvider
from hvme.providers.hf import HFProvider

fast = CachedProvider(judge, RESULTS / "llm_cache" / "hvme-fast-4b.json")
slow = CachedProvider(HFProvider("Qwen/Qwen3.5-9B", device="cuda:1", batch_size=8), RESULTS / "llm_cache" / "Qwen3.5-9B.json")
held = {n: sample_bnlearn(n, 5000, seed=0) for n in HELDOUT}
rows = []
for n in HELDOUT:
    true, df = held[n]
    for label, prov in [("data only", None), ("9B", slow), ("Fast", fast), ("Fast→9B@0.8", Routed(fast, slow, 0.8))]:
        t = time.time()
        r = tabular_to_hcg(df, n, NETWORKS[n]["domain"], provider=prov, mode="hybrid", n_boot=0, validate_kw={"n_perm": 0})
        calls = {"data only": 0, "9B": len(r.pairwise), "Fast": 0}.get(label, getattr(prov, "n_slow", 0))
        rows.append({"network": n, "prior": label, "pairs": len(r.pairwise), "slow_calls": calls, "sec": round(time.time() - t, 1), **graph_report(true, r.dag)})
pipe = pd.DataFrame(rows)
pipe.to_csv(RESULTS / "fast_pipeline_tabular.csv", index=False)
pipe.pivot_table(index="prior", columns="network", values=["sid", "slow_calls"], sort=False)""",
"""gpu:from hvme.graph.text import TEXT_VOCAB

trows = []
for d in TEXT_VOCAB:
    for lang in ("ko", "en"):
        c = make_corpus(d, lang, seed=1)  # same corpus as notebook 07: extraction and normalization are replayed from the 9B recordings
        for label, prov in [("9B verify", slow), ("Fast verify→9B@0.8", Routed(fast, slow, 0.8))]:
            res = text_to_hcg(c.docs, f"{d}-{lang}", d, prov)
            gid = gold_ids(res.mapping, c)
            on_true = [(gid.get(a), gid.get(b)) in set(c.graph.edges) for a, b in zip(res.edges.src, res.edges.dst)]
            trows.append({"domain": d, "lang": lang, "verify": label, "verify_calls": 2 * len(res.candidates),
                          "slow_verify_calls": getattr(prov, "n_slow", 2 * len(res.candidates)),
                          "p_fwd_mean_true": res.edges[on_true].p_fwd.mean(),
                          **text_eval(list(zip(res.edges.src, res.edges.dst)), res.mapping, c)})
fast.save()
slow.save()
tpipe = pd.DataFrame(trows)
tpipe.to_csv(RESULTS / "fast_pipeline_text.csv", index=False)
tpipe.groupby("verify")[["verify_calls", "slow_verify_calls", "p_fwd_mean_true", "f1"]].mean().round(3)""",
"""md:## 8. (GPU) Per-task temperature and external generalization

Two issues showed up in sections 1-7.

- When one temperature per option kind is shared by the computational Gym items and the domain-knowledge items (pairwise direction, counterfactual verification), the ECE on the knowledge items worsens to 0.17. Knowledge items (prompts starting with `You are an expert in …`) therefore get a separate temperature (`knowledge:*`).
- We need to check whether the Gym test accuracy of 0.957 only reflects fitting the generator's sentence patterns. External generalization is measured on CLadder and Corr2Cause Noul items, the same ones used in the baseline benchmarks.

You can run just this section after running the cells of section 1. It loads the saved model and rewrites only the temperatures in `hvme_fast.json`.""",
"""gpu:import torch

from hvme.bench import datasets as D, prompts as P, runner as R
from hvme.fast.judge import FastJudge
from hvme.fast.train import calibrate

fast8 = FastJudge.load(str(ROOT / "models" / "hvme-fast-4b"), device="cuda:0", batch_size=16)
temps8 = calibrate(fast8, dev_items)
fast8.save_temperature(ROOT / "models" / "hvme-fast-4b")
(RESULTS / "fast_temperature.json").write_text(json.dumps(temps8, indent=1))
cladder = D.load_cladder(n_per_rung=200, seed=0)
c2c, c2c_ref = D.load_corr2cause("default"), D.load_corr2cause("refactorization", n=1000, seed=0)
ext = pd.concat([R.run_noul(fast8, cladder, P.cladder_noul).assign(bench="cladder"),
                 R.run_noul(fast8, c2c, P.corr2cause_noul).assign(bench="corr2cause"),
                 R.run_noul(fast8, c2c_ref, P.corr2cause_noul).assign(bench="corr2cause_ref")], ignore_index=True)
R.save(ext, "fast_external")
temps8""",
"""from hvme.bench import runner as R
from hvme.fast.calibrate import apply_temperature

base = pd.read_csv(RESULTS / "baseline_qwen35.csv").query("mode == 'noul'")
ext = pd.read_csv(RESULTS / "fast_external.csv")
R.summarize(pd.concat([base, ext]), by=["bench"]).pivot_table(index="model", columns="bench", values=["acc", "majority", "ece"]).round(3)""",
"""R.summarize(pd.concat([base, ext]).query("bench == 'cladder'"), by=["rung"]).pivot_table(index="model", columns="rung", values="acc").round(3)""",
"""temps8 = json.loads((RESULTS / "fast_temperature.json").read_text())
pr = pd.read_csv(RESULTS / "fast_predictions.csv").query("model == 'hvme-fast-4b T=1'")
pr = pr.assign(logits=pr.logits.map(lambda s: np.array(json.loads(s))))
key = lambda g, k: f"knowledge:{k}" if g in ("pairwise", "verify") else k
pr["probs"] = [apply_temperature([z], temps8.get(key(g, k), temps8.get(k, 1.0)))[0] for z, g, k in zip(pr.logits, pr.group, pr.kind)]
pd.DataFrame([{"group": g, **metrics(list(d.probs), d.target_idx)} for g, d in pr.groupby("group")]).round(3)""",
"""md:## Result files

Files written by this notebook.

| File | Required | Notes |
|---|---|---|
| `results/fast_predictions.csv` | yes | Sections 2-4: per-item probabilities and logits per model (before and after calibration) |
| `results/fast_eval.csv`, `fast_history.csv` | yes | Section 4 summary, section 3 training history |
| `results/fast_latency.csv`, `fast_routing.csv` | yes | Sections 5 and 6 |
| `results/fast_pipeline_tabular.csv`, `fast_pipeline_text.csv` | yes | Section 7: graph-construction integration |
| `results/llm_cache/hvme-fast-4b.json`, `Qwen3.5-9B.json` | yes | Section 7 recorded responses (for replay without a GPU) |
| `results/fast_external.csv`, `fast_temperature.json` | yes | Section 8: CLadder/Corr2Cause results, per-task temperatures |
| `models/hvme-fast-4b/` | no | The trained model (about 8 GB); too large to keep with the results |""",
]

n = nbf.v4.new_notebook()
out = []
for c in cells:
    tag = next((t for t in ("md", "gpu") if c.startswith(t + ":")), None)
    body = c[len(tag) + 1:] if tag else c
    cell = nbf.v4.new_markdown_cell(body) if tag == "md" else nbf.v4.new_code_cell(body)
    cell.metadata["tags"] = ["gpu"] if tag == "gpu" else []
    out.append(cell)
n.cells = out
n.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(n, sys.argv[1])
