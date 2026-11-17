import sys
import nbformat as nbf

cells = [
"""md:# 05. DAG Validation — Falsifying a Graph with Data

In principle, a causal graph cannot be "proven". We try to falsify it against data, knowledge and experiments, and only graphs that survive can be used. `humm.validate` implements three of the six validation layers, the ones that can be done with data alone.

| Layer | Content | Status |
|---|---|---|
| L1 Structure | Cycles, self-loops, isolated nodes, temporal order violations | Implemented |
| L2 Data falsification | Local Markov condition (LMC) tests, permutation falsification test, edge dependence test, mechanism fit | Implemented |
| L3 Knowledge | Text evidence, LLM counterfactual checks, expert review | Planned |
| L4 Experiments | Compare interventional data with predicted effects | Planned |
| L5 Conclusion robustness | Effect range over alternative graphs, refutation tests | Partly in `humm.engine.refute` |
| L6 Benchmark | SHD and SID against the ground-truth graph | Implemented (discovery algorithms are planned) |

As seen in notebook 03, a graph that omits a confounder passes all the estimate refutation tests. Errors in the graph itself must be caught by L2 in this notebook.""",
"""import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from humm.engine import BinarySCM, LinearSCM
from humm.graph import HCG, Evidence, EvidenceKind
from humm.paths import RESULTS
from humm.validate import benchmark, fisher_z, g_test, sample_bnlearn, validate, validate_hcg""",
"""md:## 1. Conditional independence tests

The basic unit of L2 is the conditional independence test. Discrete data uses the G-test (stratified likelihood ratio) and continuous data uses Fisher-z (partial correlation). In the chain `a → b → c`, a and c are dependent, but conditioning on b makes them independent. A large p-value means "not inconsistent with independence".""",
"""CHAIN = nx.DiGraph([("a", "b"), ("b", "c")])
cont = LinearSCM(CHAIN, {("a", "b"): 1.0, ("b", "c"): 1.0}, {v: 1.0 for v in CHAIN}).sample(3000, seed=0)
disc = BinarySCM(CHAIN, {"a": np.array([0.5]), "b": np.array([0.2, 0.8]), "c": np.array([0.2, 0.8])}).sample(5000, seed=0)
pd.DataFrame({"Fisher-z (continuous)": [fisher_z(cont, "a", "c"), fisher_z(cont, "a", "c", ("b",))],
              "G-test (discrete)": [g_test(disc, "a", "c"), g_test(disc, "a", "c", ("b",))]},
             index=["a ⫫ c", "a ⫫ c | b"]).map(lambda p: f"{p:.2e}")""",
"""md:## 2. The true graph and corrupted graphs

We sample 5,000 observations from the bnlearn Asia network and validate the true graph and several deliberately corrupted graphs.

- **LMC violation rate**: the fraction of the conditional independences implied by the graph that the data reject. For the true graph it should be near the significance level (5%).
- **Permutation p-value**: the fraction of 100 graphs with randomly permuted node names that have as few violations as the given graph. The smaller it is, the more informative the graph is about the data.
- **Verdict**: pass requires that the number of violations is at chance level (binomial test p ≥ 0.05), that no strong violations remain after multiple-testing correction, and that the permutation p < 0.05. For small graphs with fewer than 5 LMC tests, the permutation test has no power, so that condition is dropped from the verdict.""",
"""true, asia = sample_bnlearn("asia", 5000, seed=0)


def edit(g, remove=(), add=()):
    h = g.copy()
    h.remove_edges_from(remove)
    h.add_edges_from(add)
    return h


cases = {
    "true graph": true,
    "either→dysp missing": edit(true, remove=[("either", "dysp")]),
    "smoke→bronc missing": edit(true, remove=[("smoke", "bronc")]),
    "lung→either reversed": edit(true, remove=[("lung", "either")], add=[("either", "lung")]),
    "smoke→lung reversed": edit(true, remove=[("smoke", "lung")], add=[("lung", "smoke")]),
    "asia→xray added": edit(true, add=[("asia", "xray")]),
}
reports = {k: validate(g, asia, n_perm=100) for k, g in cases.items()}
pd.DataFrame([{"graph": k, **{c: r.summary()[c] for c in ["verdict", "lmc_tests", "lmc_violation_rate", "excess_violation_p", "strong_violations", "permutation_p", "edges_unsupported"]},
               "top missing-edge candidate": r.missing_edge_candidates(1)[0][:2] if r.missing_edge_candidates(1) else "-"}
              for k, r in reports.items()]).round(3)""",
"""md:How to read the table:

- **Missing edges**: removing an edge raises the violation rate sharply, and the top missing-edge candidate points to the pair that was actually removed.
- **lung→either reversal**: this changes the v-structure (lung → either ← tub), so the data catch it.
- **smoke→lung reversal**: this is a change within the same Markov equivalence class, so observational data cannot distinguish it (pass). The direction must be settled by L3 knowledge or L4 experiments.
- **asia→xray added**: the overall graph verdict is pass, but the edge dependence test flags this edge as "not supported by the data".""",
"""fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True)
for ax, k in zip(axes, ["true graph", "either→dysp missing"]):
    perm = reports[k].permutation
    ax.hist(perm["perm_rates"], bins=20, color="#9aa5b1")
    ax.axvline(perm["violation_rate"], color="#c0392b", lw=2, label=f"given graph {perm['violation_rate']:.3f}")
    ax.set_title(f"{k} (permutation p = {perm['p_value']:.2f})")
    ax.set_xlabel("LMC violation rate")
    ax.legend()
axes[0].set_ylabel("random permuted graphs")
plt.tight_layout()""",
"""md:## 3. The HCG validation flow

In a real pipeline, an HCG built from text, an LLM or experts is validated against data. `validate_hcg` works as follows.

1. **L1 cleanup**: edges that violate the temporal order are marked refuted, and any remaining cycle is resolved by marking its weakest edge refuted.
2. **L2 tests**: LMC, permutation and edge tests are run on the remaining DAG.
3. **Update**: each edge gets data evidence (score = 1 − p) and its status is updated. To become verified, an edge needs at least two different kinds of evidence, data support, a directional basis (text, LLM, expert or experiment), and a passing graph verdict.

In the marketing example, suppose an LLM proposes the reversed edge "higher sales lead to more advertising" with low confidence. It contradicts the temporal order (season → ad → visits → sales), so L1 refutes it.""",
"""SALES = nx.DiGraph([("season", "ad"), ("ad", "visits"), ("visits", "sales"), ("season", "sales")])
scm = BinarySCM(SALES, {"season": np.array([0.4]), "ad": np.array([0.2, 0.7]), "visits": np.array([0.2, 0.75]),
                        "sales": np.array([0.15, 0.5, 0.45, 0.85])})
data = scm.sample(6000, seed=3)

h = HCG(graph_id="sales-demo")
for nid, label in [("season", "계절"), ("ad", "광고비"), ("visits", "방문"), ("sales", "매출")]:
    h.add_node(nid, label=label)
h.add_edge("season", "ad", evidence=[Evidence(kind=EvidenceKind.EXPERT, score=0.9, ref="마케팅팀 인터뷰")])
h.add_edge("ad", "visits", evidence=[Evidence(kind=EvidenceKind.TEXT, score=0.8, ref="report_07.pdf#p3")])
h.add_edge("visits", "sales", evidence=[Evidence(kind=EvidenceKind.TEXT, score=0.7), Evidence(kind=EvidenceKind.LLM_PRIOR, score=0.9)])
h.add_edge("season", "sales")
h.add_edge("sales", "ad", evidence=[Evidence(kind=EvidenceKind.LLM_PRIOR, score=0.3)])

report, changes = validate_hcg(h, data, time={"season": 0, "ad": 1, "visits": 2, "sales": 3}, n_perm=100)
print(report.summary()["verdict"], "|", report.notes)
changes""",
"""md:This graph has only 2 LMC tests, so the permutation test was not used in the verdict (insufficient power). Edges that have temporal order, data support and a directional basis become verified. season→sales has only data evidence and no directional basis, so it stays a hypothesis: data only say that "the two variables are related", not which way the effect runs. The validation result is saved as `validation.json`.""",
"""path = report.save(RESULTS / "validation" / "sales-demo.json")
h.save(RESULTS / "validation" / "sales-demo.hcg.json")
print(path)
report.mechanisms.round(3)""",
"""md:## 4. L6 benchmark

This is a regression test that measures SHD and SID on the same bnlearn networks whenever the pipeline changes. Causal discovery algorithms are not part of this notebook, so we only check the range of the metrics with the two extremes (empty graph and true graph).""",
"""names = ("asia", "sachs", "insurance", "alarm")
truth = {n: sample_bnlearn(n, 100)[0] for n in names}
empty = benchmark(lambda df: nx.DiGraph(), names=names, n=2000).assign(method="empty graph")
oracle = pd.concat([benchmark(lambda df, g=truth[n]: g.copy(), names=(n,), n=2000) for n in names]).assign(method="true graph")
pd.concat([empty, oracle])[["method", "network", "nodes", "edges", "shd", "sid", "f1_directed"]]""",
]

n = nbf.v4.new_notebook()
n.cells = [nbf.v4.new_markdown_cell(c[3:]) if c.startswith("md:") else nbf.v4.new_code_cell(c) for c in cells]
n.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(n, sys.argv[1])
