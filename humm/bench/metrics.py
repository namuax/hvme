"""Graph metrics (SHD, SID, F1) and probability metrics (accuracy, Brier, ECE)."""
import networkx as nx
import numpy as np

from ..engine.identify import is_valid_adjustment


# ---------- Graph structure metrics ----------
def _pairs(g):
    return {frozenset(e) for e in g.edges}


def shd(true: nx.DiGraph, est: nx.DiGraph) -> int:
    """Structural Hamming Distance: each missing edge, extra edge and reversal counts as 1."""
    t, e = set(true.edges), set(est.edges)
    reversed_ = {(u, v) for u, v in e - t if (v, u) in t}
    return len(_pairs(true) ^ _pairs(est)) + len(reversed_)


def _prf(t: set, e: set) -> dict:
    tp = len(t & e)
    p = tp / max(len(e), 1)
    r = tp / max(len(t), 1)
    return {"precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-12)}


def edge_f1(true: nx.DiGraph, est: nx.DiGraph) -> dict:
    """Returns both the direction-aware edge score and the direction-agnostic skeleton score."""
    return {"directed": _prf(set(true.edges), set(est.edges)), "skeleton": _prf(_pairs(true), _pairs(est))}


def sid(true: nx.DiGraph, est: nx.DiGraph) -> int:
    """Structural Intervention Distance (Peters & Bühlmann, 2015), DAG vs. DAG.

    Number of (i, j) pairs whose p(x_j | do(x_i)) is wrong when computed with i's parent set in the estimated graph.
    - If j is a parent of i in the estimated graph, it means 'no effect', so this is wrong when j is a descendant of i in the true graph.
    - Otherwise it is wrong if the estimated graph's parent set is not a valid adjustment set in the true graph.
    """
    wrong = 0
    desc = {i: nx.descendants(true, i) for i in true.nodes}
    for i in true.nodes:
        pa = set(est.predecessors(i))
        for j in true.nodes - {i}:
            wrong += (j in desc[i]) if j in pa else not is_valid_adjustment(true, i, j, pa)
    return int(wrong)


def graph_report(true: nx.DiGraph, est: nx.DiGraph) -> dict:
    f1 = edge_f1(true, est)
    return {"shd": shd(true, est), "sid": sid(true, est), "f1_directed": f1["directed"]["f1"], "f1_skeleton": f1["skeleton"]["f1"],
            "n_true": true.number_of_edges(), "n_est": est.number_of_edges()}


# ---------- Probability calibration metrics ----------
def brier(p, y) -> float:
    p, y = np.asarray(p, float), np.asarray(y, float)
    return float(np.mean((p - y) ** 2))


def ece(p, y, bins: int = 10) -> float:
    """Expected Calibration Error (binary: p = probability of true). Bins by the confidence of the predicted class."""
    p, y = np.asarray(p, float), np.asarray(y, int)
    conf = np.maximum(p, 1 - p)
    correct = ((p >= 0.5).astype(int) == y).astype(float)
    idx = np.minimum((conf - 0.5) * 2 * bins, bins - 1).astype(int)
    return float(sum(abs(correct[idx == b].mean() - conf[idx == b].mean()) * (idx == b).mean() for b in np.unique(idx)))
