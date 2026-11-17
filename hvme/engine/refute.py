"""Estimate refutation tests (same idea as DoWhy refuters, running on the Hvme estimator).

- placebo: shuffling the treatment at random should drive the effect to 0
- random_common_cause: adding an independent random variable as a common cause should leave the effect unchanged
- data_subset: the effect should hold on an 80% subsample
Decision rule: |new value - expected value| < 2 × bootstrap standard error
"""
import networkx as nx
import numpy as np
import pandas as pd

from .estimate import estimate


def refute(g: nx.DiGraph, df: pd.DataFrame, x: str, y: str, latent=(), n_subsets: int = 10, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = estimate(g, df, x, y, latent, n_boot=100, seed=seed)
    tol = 2 * base.se

    placebo = estimate(g, df.assign(**{x: rng.permutation(df[x].to_numpy())}), x, y, latent, n_boot=0, estimand=base.estimand).value

    g_w = g.copy()
    g_w.add_edges_from([("_w", x), ("_w", y)])
    w = rng.integers(0, 2, len(df)) if base.method.endswith(("stratified", "discrete")) else rng.normal(size=len(df))
    rcc = estimate(g_w, df.assign(_w=w), x, y, latent, n_boot=0).value

    subs = [estimate(g, df.sample(frac=0.8, random_state=int(s)), x, y, latent, n_boot=0, estimand=base.estimand).value
            for s in rng.integers(0, 2**31, n_subsets)]

    rows = [("placebo_treatment", 0.0, placebo), ("random_common_cause", base.value, rcc), ("data_subset", base.value, float(np.mean(subs)))]
    out = pd.DataFrame(rows, columns=["test", "expected", "new_value"])
    return out.assign(estimate=base.value, tolerance=tol, passed=(out.new_value - out.expected).abs() < tol)
