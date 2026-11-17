"""그래프 지표(SHD·SID·F1)와 확률 지표(정확도·Brier·ECE)."""
import networkx as nx
import numpy as np

from ..engine.identify import is_valid_adjustment


# ---------- 그래프 구조 지표 ----------
def _pairs(g):
    return {frozenset(e) for e in g.edges}


def shd(true: nx.DiGraph, est: nx.DiGraph) -> int:
    """Structural Hamming Distance: 누락·추가·방향 반전을 각각 1로 센다."""
    t, e = set(true.edges), set(est.edges)
    reversed_ = {(u, v) for u, v in e - t if (v, u) in t}
    return len(_pairs(true) ^ _pairs(est)) + len(reversed_)


def _prf(t: set, e: set) -> dict:
    tp = len(t & e)
    p = tp / max(len(e), 1)
    r = tp / max(len(t), 1)
    return {"precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-12)}


def edge_f1(true: nx.DiGraph, est: nx.DiGraph) -> dict:
    """방향까지 맞춘 엣지 기준과 방향을 무시한 골격(skeleton) 기준을 함께 반환한다."""
    return {"directed": _prf(set(true.edges), set(est.edges)), "skeleton": _prf(_pairs(true), _pairs(est))}


def sid(true: nx.DiGraph, est: nx.DiGraph) -> int:
    """Structural Intervention Distance (Peters & Bühlmann, 2015), DAG 대 DAG.

    추정 그래프에서 i의 부모 집합으로 p(x_j | do(x_i))를 계산했을 때 틀리는 (i, j) 쌍의 수.
    - j가 추정 그래프에서 i의 부모이면 '효과 없음'을 뜻하므로, 참 그래프에서 j가 i의 자손일 때 틀린다.
    - 그 밖에는 추정 그래프의 부모 집합이 참 그래프에서 유효한 조정 집합이 아니면 틀린다.
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


# ---------- 확률 보정 지표 ----------
def brier(p, y) -> float:
    p, y = np.asarray(p, float), np.asarray(y, float)
    return float(np.mean((p - y) ** 2))


def ece(p, y, bins: int = 10) -> float:
    """Expected Calibration Error (이진: p = 참일 확률). 예측 클래스의 신뢰도 기준으로 구간을 나눈다."""
    p, y = np.asarray(p, float), np.asarray(y, int)
    conf = np.maximum(p, 1 - p)
    correct = ((p >= 0.5).astype(int) == y).astype(float)
    idx = np.minimum((conf - 0.5) * 2 * bins, bins - 1).astype(int)
    return float(sum(abs(correct[idx == b].mean() - conf[idx == b].mean()) * (idx == b).mean() for b in np.unique(idx)))
