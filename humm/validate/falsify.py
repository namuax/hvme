"""L2 데이터 반증: 그래프가 함의하는 조건부 독립을 데이터로 검정한다."""
import networkx as nx
import numpy as np
import pandas as pd

from ..engine.dsep import local_markov
from .ci import CITester


def lmc_tests(g: nx.DiGraph, ci: CITester, alpha: float = 0.05) -> pd.DataFrame:
    """국소 마르코프 조건: 각 노드 v와 부모가 아닌 비자손 w에 대해 v ⫫ w | pa(v) 를 검정한다. 관측되지 않은 변수가 낀 검정은 건너뛴다."""
    cols = set(ci.df.columns)
    rows = [(v, w, sorted(pa), ci(v, w, pa)) for v, rest, pa in local_markov(g) for w in sorted(rest) if {v, w} | pa <= cols]
    df = pd.DataFrame(rows, columns=["node", "other", "given", "p"])
    return df.assign(violated=df.p < alpha)


def permutation_test(g: nx.DiGraph, ci: CITester, n_perm: int = 100, alpha: float = 0.05, seed: int = 0) -> dict:
    """노드 이름을 무작위로 바꾼 그래프들과 LMC 위반 비율을 비교한다(Eulig et al., 2023).

    p_value = 위반 비율이 주어진 그래프 이하인 무작위 그래프의 비율. 작을수록(예: < 0.05) 그래프가 데이터에 대해 정보를 가진다.
    """
    rng = np.random.default_rng(seed)
    rate = lambda h: float(lmc_tests(h, ci, alpha).violated.mean()) if h.number_of_nodes() else 0.0
    observed = rate(g)
    nodes = list(g.nodes)
    perm = [rate(nx.relabel_nodes(g, dict(zip(nodes, rng.permutation(nodes))))) for _ in range(n_perm)]
    perm = np.nan_to_num(perm, nan=0.0)
    return {"violation_rate": float(observed), "perm_rates": [float(x) for x in perm], "p_value": float(np.mean(perm <= observed)) if n_perm else 1.0, "n_perm": n_perm}


def edge_tests(g: nx.DiGraph, ci: CITester, alpha: float = 0.05) -> pd.DataFrame:
    """엣지 u → v 가 데이터로 뒷받침되는가: v의 다른 부모를 조건으로 u와 v가 여전히 의존적이어야 한다."""
    cols = set(ci.df.columns)
    rows = [(u, v, sorted(set(g.predecessors(v)) - {u}), ci(u, v, set(g.predecessors(v)) - {u}))
            for u, v in g.edges if {u, v} | set(g.predecessors(v)) <= cols]
    df = pd.DataFrame(rows, columns=["src", "dst", "given", "p"])
    return df.assign(supported=df.p < alpha)


def _entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def mechanism_fit(g: nx.DiGraph, df: pd.DataFrame, discrete: bool) -> pd.DataFrame:
    """노드별로 부모가 그 노드를 얼마나 설명하는가. 이산: 정보 이득(비트) = H(v) - H(v|부모), 연속: 선형 회귀 R²."""
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
