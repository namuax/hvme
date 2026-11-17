"""PC algorithm (PC-stable) + background knowledge. The result is a CPDAG (directed edges + undirected edges).

Background knowledge (Knowledge)
- tiers: causal order tiers. Only earlier tier -> later tier directions are allowed.
- forbidden: forbidden directions (u, v). required: directions that must be present (u, v).
- mode="hard": apply the knowledge first, right after skeleton search (takes priority over data-derived directions).
  mode="soft": first fix directions decided by the data (v-structures, Meek rules), then use the knowledge only for the remaining undirected edges.
  hybrid: only the certain knowledge passed as `strong` is applied like hard; the rest like soft.
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
        """Does the knowledge support u -> v (required edge, earlier tier, or opposite direction forbidden)?"""
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
        """Orient u - v as u -> v. Does nothing if it is already directed."""
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
        frozen = {v: set(a) for v, a in adj.items()}  # PC-stable: choose conditioning sets from the adjacency at the start of the step
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
    """Apply Meek rules R1-R3 until nothing changes."""
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
    """Orient undirected edges according to tier/topological order to make a DAG. If directed edges form a cycle, those edges are reverted to undirected."""
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
    """strong: background knowledge applied before the data (hybrid mode). With mode="hard", all of knowledge becomes strong.
    Order: apply strong -> v-structures (only those consistent with strong) -> Meek -> apply knowledge (soft) -> Meek.
    """
    knowledge = knowledge or Knowledge()
    strong = knowledge if mode == "hard" else (strong or Knowledge())
    g = skeleton(CITester(df[sorted(df.columns)], method), alpha, knowledge, max_cond)  # gives the same result regardless of column order
    g.directed |= {(u, v) for u, v in knowledge.required if frozenset((u, v)) in g.undirected}
    g.undirected -= {frozenset(e) for e in knowledge.required}
    apply_knowledge(g, strong)
    orient_vstructures(g, strong)
    meek(g)
    apply_knowledge(g, knowledge)
    meek(g)
    return g


def edge_frequency(df: pd.DataFrame, n_boot: int = 20, seed: int = 0, **kw) -> pd.DataFrame:
    """Rerun PC on each bootstrap resample and count how often each adjacency and each direction appears."""
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
