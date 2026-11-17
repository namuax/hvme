import sys
import nbformat as nbf

cells = [
"""md:# 03. Symbolic Causal Engine — Identification, Estimation, Refutation, Counterfactuals

`hvme.engine` is the **ground-truth reference** for all of Hvme. The answers to Hvme-Gym problems, the rewards for Hvme-LM reinforcement learning, and the verdicts of DAG validation all rely on its computations. For that reason it runs on graph theory and probability calculations alone, with no LLM involved.

| Module | What it does |
|---|---|
| `dsep` | d-separation tests, local Markov conditions |
| `identify` | Identification via backdoor (generalized adjustment criterion, O-set), then front-door, then instrumental variable |
| `scm` | Binary and linear structural causal models. Computes observational, interventional and counterfactual values exactly |
| `estimate` | Computes the identified estimand from data (stratification for binary, linear regression for continuous), with bootstrap confidence intervals |
| `refute` | Placebo treatment, random common cause and subset refutation tests |

This notebook runs without a GPU.""",
"""import logging

import networkx as nx
import numpy as np
import pandas as pd

from hvme.engine import (BinarySCM, LinearSCM, adjustment_sets, d_separated, estimate, identify, is_valid_adjustment,
                         optimal_adjustment, refute)

logging.disable(logging.WARNING)""",
"""md:## 1. d-separation: the three basic structures

Every conditional independence in a causal graph is determined by combinations of three structures: chain, fork and collider. Conditioning on the middle variable blocks a chain or a fork, whereas conditioning on a collider opens it.""",
"""structures = {
    "chain a→b→c": nx.DiGraph([("a", "b"), ("b", "c")]),
    "fork a←b→c": nx.DiGraph([("b", "a"), ("b", "c")]),
    "collider a→b←c": nx.DiGraph([("a", "b"), ("c", "b")]),
}
pd.DataFrame({name: {"a ⫫ c": d_separated(g, "a", "c"), "a ⫫ c | b": d_separated(g, "a", "c", "b")}
              for name, g in structures.items()}).T""",
"""md:## 2. Identification: backdoor, front-door, instrumental variable

`identify` decides whether the interventional effect P(y | do(x)) can be computed from observed variables alone and, if so, returns the estimand. Nodes passed as `latent` are treated as unobserved.

- **Confounding**: Z is a common cause of X and Y, so use backdoor adjustment.
- **Smoking-tar-cancer**: smoking (S) and lung cancer (C) share an unobserved common cause U, but S affects C only through tar (T), so use front-door adjustment.
- **Instrumental variable**: Z affects only X, so use IV.
- **Not identifiable**: there is only unobserved confounding and no other route.""",
"""CONFOUNDED = nx.DiGraph([("Z", "X"), ("Z", "Y"), ("X", "Y"), ("W", "Y")])
SMOKING = nx.DiGraph([("U", "S"), ("U", "C"), ("S", "T"), ("T", "C")])
IV = nx.DiGraph([("Z", "X"), ("U", "X"), ("U", "Y"), ("X", "Y")])
HIDDEN = nx.DiGraph([("U", "X"), ("U", "Y"), ("X", "Y")])

cases = [("confounded", CONFOUNDED, "X", "Y", set()), ("smoking-tar-cancer", SMOKING, "S", "C", {"U"}),
         ("instrument", IV, "X", "Y", {"U"}), ("not identifiable", HIDDEN, "X", "Y", {"U"}), ("reverse query", CONFOUNDED, "Y", "X", set())]
pd.DataFrame([{"case": n, "strategy": (e := identify(g, x, y, lat)).strategy, "variables": sorted(e.variables), "estimand": e.expression}
              for n, g, x, y, lat in cases])""",
"""md:### Choosing an adjustment set

There can be several valid adjustment sets. `adjustment_sets` finds minimal sets, smallest first, and `optimal_adjustment` returns the **O-set** (Henckel et al., 2022), which has the smallest estimation variance. In the confounding example, W is unrelated to X but is a cause of Y, so adding it to the adjustment set introduces no bias and only reduces variance.

The M-bias graph is a classic case where the intuition "control for every variable that looks relevant" fails. M is a collider, so controlling for M alone opens a path that was blocked.""",
"""M_BIAS = nx.DiGraph([("A", "X"), ("A", "M"), ("B", "M"), ("B", "Y"), ("X", "Y")])
print("Confounding — minimal sets:", adjustment_sets(CONFOUNDED, "X", "Y"), "| O-set:", optimal_adjustment(CONFOUNDED, "X", "Y"))
pd.Series({str(sorted(z)): is_valid_adjustment(M_BIAS, "X", "Y", z) for z in [set(), {"M"}, {"M", "A"}, {"M", "B"}]}, name="valid under M-bias")""",
"""md:## 3. Computing ground truth with an SCM

`BinarySCM` defines each variable as `v = 1[u_v < p_v(parents)]`. It compares "worlds" that share the same exogenous noise u and differ only in the intervention, so it can compute counterfactual values too. If the number of cells obtained by partitioning the noise space at the thresholds is at most 200,000, it returns the **exact value**; otherwise it uses a Monte Carlo approximation.

In the smoking example, we compare the naive contrast (difference in lung cancer rate between smokers and non-smokers) with the true causal effect (ATE).""",
"""scm = BinarySCM.random(SMOKING, seed=1)
obs = scm.prob("C", given={"S": 1}) - scm.prob("C", given={"S": 0})
print(f"Exact computation: {scm.exact}")
print(f"Naive contrast P(C|S=1) - P(C|S=0) = {obs:+.4f}")
print(f"Causal effect P(C|do(S=1)) - P(C|do(S=0)) = {scm.ate('S', 'C'):+.4f}")""",
"""md:### Counterfactual queries (Rung 3)

This computes the counterfactual query types used in CLadder. In the mediation graph `X → M → Y`, `X → Y` (Z is a confounder), it separates direct and indirect effects and computes the probabilities of necessity and sufficiency (PN, PS). PN and PS are meaningful when the treatment raises the outcome, so we picked a model with a positive ATE.""",
"""MED = nx.DiGraph([("Z", "X"), ("X", "M"), ("M", "Y"), ("X", "Y"), ("Z", "Y")])
med = BinarySCM.random(MED, seed=18)
pd.Series({
    "ATE": med.ate("X", "Y"),
    "CDE (M=1)": med.cde("X", "Y", {"M": 1}),
    "ETT": med.ett("X", "Y"),
    "NDE": med.nde("X", "Y", ["M"]),
    "NIE": med.nie("X", "Y", ["M"]),
    "PN": med.pn("X", "Y"),
    "PS": med.ps("X", "Y"),
}, name="value").round(4)""",
"""md:## 4. Estimating from data and comparing with the truth

We sample from the SCM, drop the unobserved variables, and estimate the effect from the graph alone. If the 95% confidence interval contains the SCM's true value, the engine's identification and estimation are correct. Binary data uses stratification and continuous data uses linear regression.""",
"""rows = []
for name, g, x, y, lat, seed in [("confounded", CONFOUNDED, "X", "Y", set(), 3), ("smoking-tar-cancer", SMOKING, "S", "C", {"U"}, 1)]:
    s = BinarySCM.random(g, seed=seed)
    est = estimate(g, s.sample(100_000, seed=seed + 1).drop(columns=list(lat)), x, y, latent=lat)
    rows.append({"model": "binary", "case": name, "method": est.method, "truth": s.ate(x, y), "estimate": est.value, "CI low": est.ci[0], "CI high": est.ci[1]})
for name, g, x, y, lat, seed in [("confounded", CONFOUNDED, "X", "Y", set(), 9), ("smoking-tar-cancer", SMOKING, "S", "C", {"U"}, 7), ("instrument", IV, "X", "Y", {"U"}, 5)]:
    l = LinearSCM.random(g, seed=seed)
    est = estimate(g, l.sample(100_000, seed=seed + 1).drop(columns=list(lat)), x, y, latent=lat)
    rows.append({"model": "linear", "case": name, "method": est.method, "truth": l.total_effect(x, y), "estimate": est.value, "CI low": est.ci[0], "CI high": est.ci[1]})
pd.DataFrame(rows).round(4)""",
"""md:## 5. Refutation tests

If the estimate comes from a correct model, it should pass all three tests. A test passes if the deviation from the expected value is within 2 bootstrap standard errors.""",
"""s = BinarySCM.random(CONFOUNDED, seed=3)
df = s.sample(50_000, seed=4)
refute(CONFOUNDED, df, "X", "Y").round(4)""",
"""md:Estimating with a wrong graph that omits the confounder Z produces bias, and the refutation tests do not catch this kind of error. They only check the estimation procedure on the given graph; errors in the graph itself must be caught by DAG validation (L2 data-based falsification, see notebook 05).""",
"""WRONG = nx.DiGraph([("X", "Y"), ("W", "Y")])
WRONG.add_node("Z")
print(f"True {s.ate('X', 'Y'):+.4f} | correct graph {estimate(CONFOUNDED, df, 'X', 'Y', n_boot=0).value:+.4f} | graph missing Z {estimate(WRONG, df, 'X', 'Y', n_boot=0).value:+.4f}")
refute(WRONG, df, "X", "Y").round(4)""",
]

n = nbf.v4.new_notebook()
n.cells = [nbf.v4.new_markdown_cell(c[3:]) if c.startswith("md:") else nbf.v4.new_code_cell(c) for c in cells]
n.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(n, sys.argv[1])
