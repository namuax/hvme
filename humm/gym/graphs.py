"""Graph generation: random DAGs, latent-variable selection, and exhaustive enumeration of Markov equivalence classes (MEC) for small graphs."""
from functools import lru_cache
from itertools import combinations, product

import networkx as nx
import numpy as np

from ..engine.dsep import d_separated


def random_dag(n: int, rng: np.random.Generator, p: float = 0.5, max_parents: int = 2, connected: bool = True) -> nx.DiGraph:
    """Draw a random topological order and add an edge from each earlier node to each later node with probability p. Parent count is capped by max_parents."""
    while True:
        order = [f"V{i}" for i in rng.permutation(n)]
        g = nx.DiGraph()
        g.add_nodes_from(order)
        for j, v in enumerate(order):
            cand = [u for u in order[:j] if rng.random() < p]
            g.add_edges_from((u, v) for u in rng.permutation(cand)[:max_parents])
        if not connected or nx.is_weakly_connected(g):
            return nx.relabel_nodes(g, {v: f"V{i}" for i, v in enumerate(nx.topological_sort(g))})


# ---------- Markov equivalence classes (Corr2Cause-style task) ----------
def ci_signature(g: nx.DiGraph) -> frozenset:
    """The set of all conditional independences (a, b, S) implied by the graph."""
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
    """All DAGs on n nodes (A, B, C, ...). n=3 gives 25, n=4 gives 543."""
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
    """CI signature -> list of DAGs sharing it (= Markov equivalence class)."""
    index = {}
    for g in all_dags(n):
        index.setdefault(ci_signature(g), []).append(g)
    return index


def mec(g: nx.DiGraph) -> list[nx.DiGraph]:
    return mec_index(g.number_of_nodes())[ci_signature(g)]
