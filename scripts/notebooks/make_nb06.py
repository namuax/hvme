import sys
import nbformat as nbf

cells = [
"""md:# 06. Tabular data → causal graph — PC + LLM priors

This notebook builds a causal graph (HCG) from tabular data.

| Stage | What it does | Module |
|---|---|---|
| S1 | Variable profile (type, categories, missingness) | `profile` |
| S2 | LLM prior: the **order** strategy asks once for a causal ordering (tiers) of all variables; the **pairwise** strategy (default) asks for the direction of each adjacent pair in the PC skeleton | `llm_order`, `llm_pairwise` |
| S3 | PC algorithm (conditional independence → skeleton → v-structures → Meek rules) + background knowledge | `pc` |
| S4 | With the order strategy, only the edges still undirected are queried pairwise | `llm_pairwise` |
| S5 | Bootstrap edge frequencies → HCG evidence → validation | `tabular_to_hcg` |

There are two ways to apply the LLM prior.
- **soft**: first fix the directions the data determines (v-structures, Meek rules), and use the prior only for the edges that remain undirected.
- **hard**: apply the prior before the data.

The question this notebook answers is how accurate an LLM prior must be to help, and which mode is safer. Sections 1-3 run on CPU; sections 4-5 need a GPU.""",
"""import json

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from humm.bench.metrics import graph_report
from humm.graph.tabular import NETWORKS, NoisyOracle, compare, describe_variables, pc, profile, tabular_to_hcg, to_dag
from humm.paths import RESULTS
from humm.validate import sample_bnlearn

NAMES = ["asia", "sachs", "insurance", "alarm"]
data = {n: sample_bnlearn(n, 5000, seed=0) for n in NAMES}""",
"""md:## 1. The PC algorithm step by step (Asia)

PC removes edges with conditional-independence tests to build a skeleton, orients some edges using v-structures (`a → c ← b`), and propagates orientations with the Meek rules. Edges that stay undirected cannot be distinguished from data alone (Markov equivalence).""",
"""true, asia = data["asia"]
print(describe_variables(profile(asia), NETWORKS["asia"]["descriptions"]))
g = pc(asia, alpha=0.01)
print("\\nDirected edges:", sorted(g.directed))
print("Undirected:", sorted(tuple(sorted(e)) for e in g.undirected))
print("Orientation source:", g.oriented_by)
print("Metrics:", graph_report(true, to_dag(g)))""",
"""md:## 2. Prior-quality experiment (simulated LLM)

`NoisyOracle` is a simulated LLM that knows the true graph but errs on purpose. The error rate e is the fraction of variables that swap places in the causal order, and the probability of choosing the wrong direction in a pairwise query. e = 0 is a perfect prior.""",
"""rows = []
for n in NAMES:
    true, df = data[n]
    provs = {"none": None} | {f"e={e}": NoisyOracle(true, e, seed=1) for e in (0.0, 0.1, 0.2, 0.4)}
    rows.append(compare(df, true, provs, NETWORKS[n]["domain"], NETWORKS[n]["descriptions"], modes=("soft", "hard"), priors=("order", "pairwise")).assign(network=n))
oracle_res = pd.concat(rows, ignore_index=True)
oracle_res.pivot_table(index=["prior", "strategy", "mode"], columns="network", values="shd", sort=False)""",
"""oracle_res.pivot_table(index=["prior", "strategy", "mode"], columns="network", values="sid", sort=False)""",
"""fig, axes = plt.subplots(1, 4, figsize=(15, 3.2))
for ax, n in zip(axes, NAMES):
    d = oracle_res[oracle_res.network == n]
    base = d[d.prior == "none"].sid.iloc[0]
    for (strategy, mode), (color, ls) in {("order", "soft"): ("#2e86de", "--"), ("order", "hard"): ("#c0392b", "--"),
                                         ("pairwise", "soft"): ("#2e86de", "-"), ("pairwise", "hard"): ("#c0392b", "-")}.items():
        m = d[(d.strategy == strategy) & (d["mode"] == mode)]
        ax.plot([float(p[2:]) for p in m.prior], m.sid, "o", color=color, ls=ls, label=f"{strategy}·{mode}")
    ax.axhline(base, color="#7f8c8d", ls="--", label="statistics only")
    ax.set_title(n)
    ax.set_xlabel("prior error rate e")
axes[0].set_ylabel("SID (lower is better)")
axes[0].legend()
plt.tight_layout()""",
"""md:How to read the plot:

- **hard**: best when the prior is accurate (e = 0). With an error rate of only 10-20%, it is worse than statistics alone on the larger graphs (insurance, alarm).
- **soft**: the gain from an accurate prior is small, but a wrong prior does not make it much worse than statistics alone.
- **strategy**: with errors, pairwise (solid) degrades less than order (dashed). The simulated LLM errs differently in each case (swapping positions in the order vs flipping a pairwise direction), so the comparison is only indicative.
- **takeaway**: use soft for LLM priors of unknown quality, and hard for an ordering that an expert has checked. Section 4 asks where real LLMs fall on this error scale.""",
"""md:## 3. Reading the pipeline output (sachs, simulated LLM e = 0.1)

Along with the graph, `tabular_to_hcg` returns per-edge evidence, a validation report, and a **conflict list** (edges where the direction set by the data disagrees with the LLM ordering). An edge on which the data and the LLM ordering agree has evidence from two sources and can rise to verified. Conflicting edges are candidates for L3 (knowledge) review.""",
"""true, sachs = data["sachs"]
r = tabular_to_hcg(sachs, "sachs", NETWORKS["sachs"]["domain"], NETWORKS["sachs"]["descriptions"], NoisyOracle(true, 0.1, seed=1), mode="soft", prior="order", n_boot=20)
print(graph_report(true, r.dag))
print(r.report.summary()["verdict"], "| strong violations", r.report.summary()["strong_violations"], "| conflicts", r.conflicts)
chk = r.changes.assign(true_edge=[tuple(e.split("→")) in true.edges for e in r.changes.edge],
                      conflict=[tuple(e.split("→")) in {(u, v) for u, v, _ in r.conflicts} for e in r.changes.edge])
chk""",
"""md:Grouping by status and conflict shows the accuracy gap between edges where the two sources (data and LLM ordering) agree and edges where they disagree. The conflict list can serve as the list of edges for a human or L3 validation to look at first.""",
"""chk.groupby(["after", "conflict"]).true_edge.agg(n="size", true_rate="mean")""",
"""md:## 4. (GPU) Real LLM priors — Qwen3.5

We measure both strategies with two models. First, how well the prior matches the true graph: for order, the ordering accuracy; for pairwise, the accuracy of confident answers on true-edge pairs. Then we place the results combined with PC on top of the simulated-LLM curves from section 2. asia and sachs get variable descriptions; insurance and alarm get variable names only.

> Observed results: pairwise accuracy with the 9B model is 0.88 on sachs, 0.86 on insurance and 0.67 on alarm. Pairwise + hard improves SID substantially on sachs (49 → 16) and insurance (456 → 352) but worsens it on alarm (55 → 219). The order strategy gets truncated even at 4096 tokens once there are more than 20 variables. LLM responses are recorded to `results/llm_cache/`.
>
> In an earlier run, the order strategy failed with the 4B model, which was truncated mid-reasoning and produced no JSON (sachs, alarm), while the 9B model put all variables into one tier in alphabetical order (alarm, insurance). The fix raised the token limit to 4096, instructed a minimum tier count and a maximum tier size, and added the pairwise strategy.""",
"""import gc

import torch

from humm.providers.cache import CachedProvider
from humm.providers.hf import HFProvider

MODELS = {
    "Qwen/Qwen3.5-4B": "cuda:0",
    "Qwen/Qwen3.5-9B": "cuda:1",
}
llm_rows, orders = [], {}
for model_id, device in MODELS.items():
    lm = CachedProvider(HFProvider(model_id, device=device, batch_size=16), RESULTS / "llm_cache" / f"{model_id.split('/')[-1]}.json")
    for n in NAMES:
        true, df = data[n]
        res = compare(df, true, {model_id: lm}, NETWORKS[n]["domain"], NETWORKS[n]["descriptions"], modes=("soft", "hard"), priors=("order", "pairwise"))
        o = tabular_to_hcg(df, n, NETWORKS[n]["domain"], NETWORKS[n]["descriptions"], lm, prior="order", n_boot=0, validate_kw={"n_perm": 0})
        orders[f"{model_id}|{n}"] = {"tiers": o.knowledge.tiers, "raw": o.prior_raw}
        llm_rows.append(res.assign(network=n))
    lm.save()  # record responses so ReplayProvider can replay them without a GPU
    del lm
    gc.collect()
    torch.cuda.empty_cache()

llm_res = pd.concat(llm_rows, ignore_index=True)
llm_res.to_csv(RESULTS / "tabular_llm_prior.csv", index=False)
(RESULTS / "tabular_llm_orders.json").write_text(json.dumps(orders, ensure_ascii=False, indent=2))
soft = llm_res[llm_res["mode"] == "soft"].set_index(["prior", "network"])
order_q = soft[soft.strategy == "order"][["order_accuracy", "order_tiers"]]
pair_q = soft[soft.strategy == "pairwise"][["pairs_on_true_edges", "correct", "wrong", "unsure", "pair_accuracy"]]
order_q.join(pair_q).round(2)""",
"""summary = pd.concat([oracle_res[oracle_res.prior == "none"], llm_res], ignore_index=True)
summary.pivot_table(index=["prior", "strategy", "mode"], columns="network", values=["shd", "sid"], sort=False)""",
"""md:## 5. (GPU) HCG built with a real LLM (sachs)

Build the HCG on sachs from the pairwise prior (the hybrid-mode default) of the model with the higher pairwise accuracy, then save the validation report and the HCG.""",
"""best = llm_res[(llm_res.network == "sachs") & (llm_res.strategy == "pairwise")].sort_values("pair_accuracy").prior.iloc[-1]
lm = CachedProvider(HFProvider(best, device=MODELS[best], batch_size=16), RESULTS / "llm_cache" / f"{best.split('/')[-1]}.json")
true, sachs = data["sachs"]
r = tabular_to_hcg(sachs, "sachs", NETWORKS["sachs"]["domain"], NETWORKS["sachs"]["descriptions"], lm, mode="hybrid", prior="pairwise", n_boot=20)
r.report.save(RESULTS / "validation" / "sachs-llm.json")
r.hcg.save(RESULTS / "validation" / "sachs-llm.hcg.json")
print(best, graph_report(true, r.dag), "| conflicts", r.conflicts)
lm.save()
r.changes.assign(true_edge=[tuple(e.split("→")) in true.edges for e in r.changes.edge])""",
"""md:## 6. (CPU) Hybrid mode — replaying recorded responses

Replay the LLM responses recorded in section 4 (`results/llm_cache/`) to experiment without a GPU. First check whether the LLM's confidence matches its actual accuracy (calibration), then compare, per network, whether the data's direction or the LLM's direction is right more often.""",
"""import numpy as np

from humm.graph.tabular import DEV, HELDOUT, llm_pairwise
from humm.providers.cache import ReplayProvider

replay = {m: ReplayProvider(RESULTS / "llm_cache" / f"Qwen3.5-{m}.json", name=m) for m in ("4B", "9B")}
rows = []
for n in DEV:
    true, df = data[n]
    g0 = pc(df, 0.01)
    adj = sorted({tuple(sorted(e)) for e in g0.undirected} | {tuple(sorted(e)) for e in g0.directed})
    pws = {m: llm_pairwise(p, adj, NETWORKS[n]["domain"], NETWORKS[n]["descriptions"]).set_index(["a", "b"]).p_ab for m, p in replay.items()}
    for a, b in adj:
        if true.has_edge(a, b) or true.has_edge(b, a):
            rows.append({"network": n, "truth": int(true.has_edge(a, b)), "4B": pws["4B"][(a, b)], "9B": pws["9B"][(a, b)],
                         "data_dir": 1 if (a, b) in g0.directed else (0 if (b, a) in g0.directed else np.nan)})
cal = pd.DataFrame(rows)
conf, ok = np.maximum(cal["9B"], 1 - cal["9B"]), (cal["9B"] >= 0.5) == (cal.truth == 1)
pd.DataFrame({"n": ok.groupby(pd.cut(conf, [0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0001], right=False), observed=False).size(),
              "9B accuracy": ok.groupby(pd.cut(conf, [0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0001], right=False), observed=False).mean().round(2)}).T""",
"""d = cal.dropna(subset=["data_dir"])
d.groupby("network").apply(lambda x: pd.Series({"true edges oriented by data": len(x), "data accuracy": ((x.data_dir == 1) == (x.truth == 1)).mean(),
                                                "9B accuracy": ((x["9B"] >= 0.5) == (x.truth == 1)).mean()}), include_groups=False).round(2)""",
"""md:Which is right, the data or the LLM, differs by network (data on alarm, LLM on sachs). That is why we use **hybrid mode**: only pairs on which all models agree on a direction with confidence of at least τ take priority over the data, and the rest are handled as in soft mode. We compare SID while varying τ and the model combination.""",
"""providers = {"none": None, "9B": replay["9B"], "4B+9B": [replay["4B"], replay["9B"]]}
grid = pd.concat([compare(data[n][1], data[n][0], providers, NETWORKS[n]["domain"], NETWORKS[n]["descriptions"],
                          modes=("soft", "hard"), taus=(0.8, 0.9, 0.95, 0.99)).assign(network=n) for n in DEV], ignore_index=True)
grid.to_csv(RESULTS / "tabular_hybrid_grid.csv", index=False)
t = grid.pivot_table(index=["prior", "mode"], columns="network", values="sid", sort=False)
t.assign(mean_relative_SID=(t / t.loc[("none", "-")]).mean(axis=1).round(3))""",
"""md:**Default**: hybrid mode, with τ = 0.99 for a single provider and τ = 0.95 for two or more. This setting was no worse than statistics alone on all four development networks. Because τ was chosen on those same networks, section 7 rechecks it on held-out networks.""",
"""md:## 7. (GPU) Held-out network validation

Check whether the default beats statistics alone on four networks not used to choose τ (child, hepar2, win95pts, hailfinder). No variable descriptions are given, only names. First record the pairwise responses with the two models, then run the comparison by replay.""",
"""held = {n: sample_bnlearn(n, 5000, seed=0) for n in HELDOUT}
for model_id, device in MODELS.items():
    lm = CachedProvider(HFProvider(model_id, device=device, batch_size=16), RESULTS / "llm_cache" / f"{model_id.split('/')[-1]}.json")
    for n in HELDOUT:
        tabular_to_hcg(*held[n][::-1], NETWORKS[n]["domain"], provider=lm, mode="soft", n_boot=0, validate_kw={"n_perm": 0}) if False else \
            tabular_to_hcg(held[n][1], n, NETWORKS[n]["domain"], provider=lm, mode="soft", n_boot=0, validate_kw={"n_perm": 0})
    lm.save()
    del lm
    gc.collect()
    torch.cuda.empty_cache()

replay = {m: ReplayProvider(RESULTS / "llm_cache" / f"Qwen3.5-{m}.json", name=m) for m in ("4B", "9B")}
providers = {"none": None, "9B": replay["9B"], "4B+9B": [replay["4B"], replay["9B"]]}
heldout = pd.concat([compare(held[n][1], held[n][0], providers, NETWORKS[n]["domain"], modes=("soft", "hard", "hybrid")).assign(network=n)
                     for n in HELDOUT], ignore_index=True)
heldout.to_csv(RESULTS / "tabular_heldout.csv", index=False)
t = heldout.pivot_table(index=["prior", "mode"], columns="network", values="sid", sort=False)
t.assign(mean_relative_SID=(t / t.loc[("none", "-")]).mean(axis=1).round(3))""",
"""heldout[heldout["mode"] == "soft"].set_index(["prior", "network"])[["pairs_on_true_edges", "correct", "wrong", "unsure", "pair_accuracy"]].round(2)""",
"""md:## Result files

Files written by this notebook.

| File | Required | Notes |
|---|---|---|
| `results/tabular_llm_prior.csv` | yes | Section 4: prior accuracy and SHD/SID per model, network and strategy |
| `results/tabular_llm_orders.json` | yes | Section 4: causal orderings answered by the LLM, with raw responses |
| `results/validation/sachs-llm.json` | yes | Section 5: validation report |
| `results/validation/sachs-llm.hcg.json` | yes | Section 5: HCG built from the real LLM prior |
| `results/llm_cache/Qwen3.5-4B.json`, `Qwen3.5-9B.json` | yes | Recorded LLM responses (development and held-out networks); replayable without a GPU |
| `results/tabular_hybrid_grid.csv` | yes | Section 6: hybrid-mode τ search |
| `results/tabular_heldout.csv` | yes | Section 7: held-out network validation |""",
]

n = nbf.v4.new_notebook()
n.cells = [nbf.v4.new_markdown_cell(c[3:]) if c.startswith("md:") else nbf.v4.new_code_cell(c) for c in cells]
n.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(n, sys.argv[1])
