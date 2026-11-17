"""PC 알고리즘(PC-stable) + 배경지식. 결과는 CPDAG(방향 엣지 + 방향 미정 엣지)이다.

배경지식(Knowledge)
- tiers: 인과 순서 계층. 앞 계층 → 뒤 계층 방향만 허용한다.
- forbidden: 금지된 방향 (u, v). required: 반드시 있어야 하는 방향 (u, v).
- mode="hard": 배경지식을 골격 탐색 뒤 가장 먼저 적용한다(데이터 방향보다 우선).
  mode="soft": 데이터가 정한 방향(v-구조·Meek 규칙)을 먼저 정하고, 남은 방향 미정 엣지에만 배경지식을 쓴다.
  혼합: strong 인자로 받은 확실한 배경지식만 hard처럼, 나머지는 soft처럼 쓴다.
"""
from dataclasses import dataclass, field
from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd

from ...validate.ci import CITester


@dataclass
class Knowledge:
    tiers: list[list[str]] = field(default_factory=list)
    forbidden: set = field(default_factory=set)
    required: set = field(default_factory=set)

    @property
    def tier_of(self) -> dict:
        return {v: i for i, t in enumerate(self.tiers) for v in t}

    def allows(self, u, v) -> bool:
        t = self.tier_of
        return (u, v) not in self.forbidden and not (u in t and v in t and t[u] > t[v])

    def prefers(self, u, v) -> bool:
        """u → v 를 배경지식이 지지하는가(필수 엣지이거나, 계층이 앞서거나, 반대 방향이 금지됨)."""
        t = self.tier_of
        return (u, v) in self.required or (u in t and v in t and t[u] < t[v]) or ((v, u) in self.forbidden and (u, v) not in self.forbidden)


@dataclass
class CPDAG:
    nodes: list
    directed: set = field(default_factory=set)
    undirected: set = field(default_factory=set)  # frozenset({u, v})
    sepsets: dict = field(default_factory=dict)
    oriented_by: dict = field(default_factory=dict)  # (u, v) -> "vstruct" | "meek" | "knowledge" | "extension"

    def adjacent(self, u, v) -> bool:
        return frozenset((u, v)) in self.undirected or (u, v) in self.directed or (v, u) in self.directed

    def neighbors(self, u) -> set:
        return {v for v in self.nodes if v != u and self.adjacent(u, v)}

    def orient(self, u, v, how: str) -> bool:
        """u - v 를 u → v 로 바꾼다. 이미 방향이 있으면 바꾸지 않는다."""
        e = frozenset((u, v))
        if e not in self.undirected:
            return False
        self.undirected.discard(e)
        self.directed.add((u, v))
        self.oriented_by[(u, v)] = how
        return True


def skeleton(ci: CITester, alpha: float, knowledge: Knowledge, max_cond: int = 3) -> CPDAG:
    nodes = list(ci.df.columns)
    adj = {v: set(nodes) - {v} for v in nodes}
    keep = {frozenset(e) for e in knowledge.required}
    sepsets = {}
    for level in range(max_cond + 1):
        frozen = {v: set(a) for v, a in adj.items()}  # PC-stable: 단계 시작 시점의 인접 관계로 조건 집합을 고른다
        for x in nodes:
            for y in sorted(frozen[x]):
                if y not in adj[x] or frozenset((x, y)) in keep or len(frozen[x] - {y}) < level:
                    continue
                for s in combinations(sorted(frozen[x] - {y}), level):
                    if ci(x, y, s) > alpha:
                        adj[x].discard(y)
                        adj[y].discard(x)
                        sepsets[frozenset((x, y))] = set(s)
                        break
        if all(len(a) - 1 < level for a in adj.values()):
            break
    und = {frozenset((u, v)) for u in nodes for v in adj[u]}
    return CPDAG(nodes, set(), und, sepsets)


def orient_vstructures(g: CPDAG, knowledge: Knowledge) -> None:
    for z in g.nodes:
        for x, y in combinations(sorted(g.neighbors(z)), 2):
            if not g.adjacent(x, y) and z not in g.sepsets.get(frozenset((x, y)), {z}):
                for a in (x, y):
                    if knowledge.allows(a, z) and (z, a) not in g.directed:
                        g.orient(a, z, "vstruct")


def meek(g: CPDAG) -> None:
    """Meek 규칙 R1~R3을 바뀌는 것이 없을 때까지 적용한다."""
    changed = True
    while changed:
        changed = False
        for e in sorted(g.undirected, key=sorted):
            u, v = sorted(e)
            for a, b in ((u, v), (v, u)):
                parents_a = sorted(p for p, c in g.directed if c == a)
                r1 = any(not g.adjacent(p, b) for p in parents_a)
                r2 = any((a, w) in g.directed and (w, b) in g.directed for w in g.nodes)
                spouses = [w for w in g.nodes if (w, b) in g.directed and frozenset((a, w)) in g.undirected]
                r3 = any(not g.adjacent(c, d) for c, d in combinations(spouses, 2))
                if (r1 or r2 or r3) and g.orient(a, b, "meek"):
                    changed = True
                    break


def apply_knowledge(g: CPDAG, knowledge: Knowledge) -> None:
    for e in sorted(g.undirected, key=sorted):
        u, v = sorted(e)
        for a, b in ((u, v), (v, u)):
            if knowledge.prefers(a, b) and g.orient(a, b, "knowledge"):
                break


def to_dag(g: CPDAG, knowledge: Knowledge | None = None) -> nx.DiGraph:
    """방향 미정 엣지를 계층·위상 순서에 맞춰 정해 DAG로 만든다. 방향 엣지끼리 사이클이 있으면 해당 엣지를 방향 미정으로 되돌린다."""
    d = nx.DiGraph(sorted(g.directed))
    d.add_nodes_from(g.nodes)
    while not nx.is_directed_acyclic_graph(d):
        u, v = nx.find_cycle(d)[0]
        d.remove_edge(u, v)
        g.directed.discard((u, v))
        g.oriented_by.pop((u, v), None)
        g.undirected.add(frozenset((u, v)))
    tier = (knowledge or Knowledge()).tier_of
    rank = {v: i for i, v in enumerate(nx.lexicographical_topological_sort(d, key=lambda n: (tier.get(n, len(tier)), n)))}
    for e in sorted(g.undirected, key=sorted):
        u, v = sorted(e, key=rank.get)
        d.add_edge(u, v)
        g.oriented_by.setdefault((u, v), "extension")
    return d


def pc(df: pd.DataFrame, alpha: float = 0.01, knowledge: Knowledge | None = None, mode: str = "soft", max_cond: int = 3,
       method: str = "auto", strong: Knowledge | None = None) -> CPDAG:
    """strong: 데이터보다 먼저 적용할 배경지식(혼합 모드). mode="hard"이면 knowledge 전체가 strong이 된다.
    순서: strong 적용 → v-구조(strong에 어긋나지 않는 것만) → Meek → knowledge(soft) 적용 → Meek.
    """
    knowledge = knowledge or Knowledge()
    strong = knowledge if mode == "hard" else (strong or Knowledge())
    g = skeleton(CITester(df[sorted(df.columns)], method), alpha, knowledge, max_cond)  # 열 순서와 무관하게 같은 결과를 낸다
    g.directed |= {(u, v) for u, v in knowledge.required if frozenset((u, v)) in g.undirected}
    g.undirected -= {frozenset(e) for e in knowledge.required}
    apply_knowledge(g, strong)
    orient_vstructures(g, strong)
    meek(g)
    apply_knowledge(g, knowledge)
    meek(g)
    return g


def edge_frequency(df: pd.DataFrame, n_boot: int = 20, seed: int = 0, **kw) -> pd.DataFrame:
    """부트스트랩 재표본마다 PC를 다시 돌려, 인접(adjacency)과 방향(directed) 출현 빈도를 센다."""
    rng = np.random.default_rng(seed)
    adj, dire = {}, {}
    for _ in range(n_boot):
        g = pc(df.iloc[rng.integers(0, len(df), len(df))].reset_index(drop=True), **kw)
        for u, v in g.directed:
            dire[(u, v)] = dire.get((u, v), 0) + 1
        for e in {frozenset(x) for x in g.directed} | g.undirected:
            adj[e] = adj.get(e, 0) + 1
    rows = [{"src": u, "dst": v, "adjacency": adj.get(frozenset((u, v)), 0) / n_boot, "directed": dire.get((u, v), 0) / n_boot}
            for u, v in {tuple(sorted(e)) for e in adj} | set(dire)]
    return pd.DataFrame(rows).sort_values("adjacency", ascending=False).reset_index(drop=True)
