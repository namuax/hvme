"""그래프 생성: 랜덤 DAG, 잠재 변수 지정, 작은 그래프의 마르코프 동치류(MEC) 전수 열거."""
from functools import lru_cache
from itertools import combinations, product

import networkx as nx
import numpy as np

from ..engine.dsep import d_separated


def random_dag(n: int, rng: np.random.Generator, p: float = 0.5, max_parents: int = 2, connected: bool = True) -> nx.DiGraph:
    """무작위 위상 순서를 정하고, 앞선 노드에서 뒤 노드로 확률 p의 엣지를 둔다. 부모 수는 max_parents로 제한한다."""
    while True:
        order = [f"V{i}" for i in rng.permutation(n)]
        g = nx.DiGraph()
        g.add_nodes_from(order)
        for j, v in enumerate(order):
            cand = [u for u in order[:j] if rng.random() < p]
            g.add_edges_from((u, v) for u in rng.permutation(cand)[:max_parents])
        if not connected or nx.is_weakly_connected(g):
            return nx.relabel_nodes(g, {v: f"V{i}" for i, v in enumerate(nx.topological_sort(g))})


# ---------- 마르코프 동치류 (Corr2Cause식 과제) ----------
def ci_signature(g: nx.DiGraph) -> frozenset:
    """그래프가 함의하는 모든 (a, b, S) 조건부 독립의 집합."""
    nodes = sorted(g.nodes)
    out = set()
    for a, b in combinations(nodes, 2):
        rest = [v for v in nodes if v not in (a, b)]
        for k in range(len(rest) + 1):
            for s in combinations(rest, k):
                if d_separated(g, a, b, s):
                    out.add((a, b, s))
    return frozenset(out)


@lru_cache(maxsize=None)
def all_dags(n: int) -> tuple[nx.DiGraph, ...]:
    """노드 n개(A, B, C, ...)의 모든 DAG. n=3 → 25개, n=4 → 543개."""
    nodes = [chr(65 + i) for i in range(n)]
    pairs = list(combinations(nodes, 2))
    dags = []
    for choice in product((0, 1, 2), repeat=len(pairs)):
        g = nx.DiGraph()
        g.add_nodes_from(nodes)
        g.add_edges_from((a, b) if c == 1 else (b, a) for (a, b), c in zip(pairs, choice) if c)
        if nx.is_directed_acyclic_graph(g):
            dags.append(g)
    return tuple(dags)


@lru_cache(maxsize=None)
def mec_index(n: int) -> dict[frozenset, list[nx.DiGraph]]:
    """CI 서명 → 같은 서명을 가진 DAG 목록(= 마르코프 동치류)."""
    index = {}
    for g in all_dags(n):
        index.setdefault(ci_signature(g), []).append(g)
    return index


def mec(g: nx.DiGraph) -> list[nx.DiGraph]:
    return mec_index(g.number_of_nodes())[ci_signature(g)]
