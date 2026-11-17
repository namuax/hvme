"""L2 data falsification: test the conditional independences the graph implies against data."""
import networkx as nx
import numpy as np
import pandas as pd

from ..engine.dsep import local_markov
from .ci import CITester


def lmc_tests(g: nx.DiGraph, ci: CITester, alpha: float = 0.05) -> pd.DataFrame:
    """Local Markov condition: for each node v and non-parent non-descendant w, test v ⫫ w | pa(v). Tests involving unobserved variables are skipped."""
    cols = set(ci.df.columns)
    rows = [(v, w, sorted(pa), ci(v, w, pa)) for v, rest, pa in local_markov(g) for w in sorted(rest) if {v, w} | pa <= cols]
    df = pd.DataFrame(rows, columns=["node", "other", "given", "p"])
    return df.assign(violated=df.p < alpha)


def permutation_test(g: nx.DiGraph, ci: CITester, n_perm: int = 100, alpha: float = 0.05, seed: int = 0) -> dict:
    """Compare the LMC violation rate with graphs whose node names are randomly permuted (Eulig et al., 2023).

    p_value = fraction of random graphs whose violation rate is at most the given graph's. The smaller (e.g. < 0.05), the more informative the graph is about the data.
    """
    rng = np.random.default_rng(seed)
    rate = lambda h: float(lmc_tests(h, ci, alpha).violated.mean()) if h.number_of_nodes() else 0.0
    observed = rate(g)
    nodes = list(g.nodes)
    perm = [rate(nx.relabel_nodes(g, dict(zip(nodes, rng.permutation(nodes))))) for _ in range(n_perm)]
    perm = np.nan_to_num(perm, nan=0.0)
    return {"violation_rate": float(observed), "perm_rates": [float(x) for x in perm], "p_value": float(np.mean(perm <= observed)) if n_perm else 1.0, "n_perm": n_perm}


def edge_tests(g: nx.DiGraph, ci: CITester, alpha: float = 0.05) -> pd.DataFrame:
    """Whether edge u → v is supported by data: u and v must remain dependent given the other parents of v."""
    cols = set(ci.df.columns)
    rows = [(u, v, sorted(set(g.predecessors(v)) - {u}), ci(u, v, set(g.predecessors(v)) - {u}))
            for u, v in g.edges if {u, v} | set(g.predecessors(v)) <= cols]
    df = pd.DataFrame(rows, columns=["src", "dst", "given", "p"])
    return df.assign(supported=df.p < alpha)


def _entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def mechanism_fit(g: nx.DiGraph, df: pd.DataFrame, discrete: bool) -> pd.DataFrame:
    """How well each node's parents explain it. Discrete: information gain (bits) = H(v) - H(v|parents); continuous: linear regression R²."""
    rows = []
    for v in [n for n in nx.topological_sort(g) if n in df]:
        pa = [p for p in g.predecessors(v) if p in df]
        if discrete:
            h = _entropy(df[v].value_counts(normalize=True).to_numpy())
            h_cond = sum(len(d) / len(df) * _entropy(d[v].value_counts(normalize=True).to_numpy()) for _, d in df.groupby(pa, observed=True)) if pa else h
            rows.append((v, pa, h, h - h_cond))
        else:
            a = np.column_stack([np.ones(len(df))] + [df[p].to_numpy(float) for p in pa])
            resid = df[v].to_numpy(float) - a @ np.linalg.lstsq(a, df[v].to_numpy(float), rcond=None)[0]
            rows.append((v, pa, float(df[v].var()), 1 - resid.var() / df[v].var()))
    return pd.DataFrame(rows, columns=["node", "parents", "baseline", "explained"]).assign(metric="info_gain_bits" if discrete else "r2")
