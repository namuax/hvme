"""Statistical relations → Markov equivalence class → causal hypothesis decision (Corr2Cause-style structural queries).

The premises are assumed to give all correlation and conditional-independence relations of the system (faithfulness).
- Skeleton: only pairs that are never independent under any conditioning set are adjacent.
- V-structure: for non-adjacent a, b with common neighbor c, if c is in none of the sets separating a and b, then a → c ← b.
- Equivalence class: orientations of the skeleton that are acyclic and have the same v-structures. Exhaustive enumeration suffices up to 6 variables (at most 15 edges).
  humm.gym.graphs.mec finds classes via the CI signatures of all DAGs, so it only works up to 4 variables.

Hypothesis definitions follow the Corr2Cause (Jin et al., 2024) labels. 'via' means there is a path through an intermediate variable, and is true even if a direct edge also exists
(checked against 1,162 test items).
"""
from itertools import combinations, product

import networkx as nx

HYPOTHESES = ("direct", "via", "cause", "collider", "confounder")


def holds(h: str, g: nx.DiGraph, a, b) -> bool:
    """Whether hypothesis h(a, b) holds in graph g."""
    return {"direct": lambda: g.has_edge(a, b),
            "via": lambda: any(len(p) > 2 for p in nx.all_simple_paths(g, a, b)),
            "cause": lambda: b in nx.descendants(g, a),
            "collider": lambda: any(g.has_edge(a, c) and g.has_edge(b, c) for c in g),
            "confounder": lambda: bool((nx.ancestors(g, a) & nx.ancestors(g, b)) - {a, b})}[h]()


def vstructures(g: nx.DiGraph) -> set[tuple]:
    """(a, c, b), a < b: a → c ← b and a, b are not adjacent."""
    return {(a, c, b) for c in g for a, b in combinations(sorted(g.predecessors(c)), 2) if not g.has_edge(a, b) and not g.has_edge(b, a)}


def from_relations(nodes, sepsets: dict[frozenset, list[frozenset]]) -> tuple[set[frozenset], set[tuple]]:
    """sepsets: independent pair → list of conditioning sets separating it (the empty set means marginal independence). Returns the skeleton and v-structures."""
    nodes = sorted(nodes)
    adj = {frozenset(p) for p in combinations(nodes, 2) if frozenset(p) not in sepsets}
    vs = {(a, c, b) for a, b in combinations(nodes, 2) if frozenset((a, b)) in sepsets for c in nodes
          if {frozenset((a, c)), frozenset((b, c))} <= adj and not any(c in s for s in sepsets[frozenset((a, b))])}
    return adj, vs


def equivalence_class(nodes, adj: set[frozenset], vs: set[tuple]) -> list[nx.DiGraph]:
    """All DAGs with skeleton adj and v-structures vs."""
    edges = sorted(tuple(sorted(e)) for e in adj)
    out = []
    for o in product((0, 1), repeat=len(edges)):
        g = nx.DiGraph()
        g.add_nodes_from(sorted(nodes))
        g.add_edges_from(e if f else e[::-1] for e, f in zip(edges, o))
        if nx.is_directed_acyclic_graph(g) and vstructures(g) == vs:
            out.append(g)
    return out


def mec_of(g: nx.DiGraph) -> list[nx.DiGraph]:
    """The Markov equivalence class of graph g."""
    return equivalence_class(g.nodes, {frozenset(e) for e in g.edges}, vstructures(g))
