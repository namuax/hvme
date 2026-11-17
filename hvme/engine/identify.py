"""Causal effect identification: backdoor (generalized adjustment criterion), frontdoor, instrumental variable (IV).

The latent argument names unobserved nodes. Adjustment, mediator and instrument sets use observed nodes only.
"""
from dataclasses import dataclass, field
from itertools import combinations

import networkx as nx

from .dsep import as_set


# ---------- Backdoor: generalized adjustment criterion (Shpitser et al. 2010; Perković et al. 2018) ----------
def causal_nodes(g: nx.DiGraph, x, y) -> set:
    """Nodes on causal paths from x to y (excluding x, including y). Empty if there is no causal path."""
    de = nx.descendants(g, x)
    return (de & nx.ancestors(g, y)) | {y} if y in de else set()


def forbidden(g: nx.DiGraph, x, y) -> set:
    """Nodes that must not be in an adjustment set: x and the descendants of nodes on causal paths."""
    return {x}.union(*({w} | nx.descendants(g, w) for w in causal_nodes(g, x, y)))


def proper_backdoor_graph(g: nx.DiGraph, x, y) -> nx.DiGraph:
    """The graph with the first edge of each causal path out of x removed."""
    h = g.copy()
    h.remove_edges_from([(x, w) for w in causal_nodes(g, x, y) if g.has_edge(x, w)])
    return h


def is_valid_adjustment(g: nx.DiGraph, x, y, z) -> bool:
    """Whether z is a valid adjustment set for the x → y effect."""
    z = as_set(z)
    return not (z & forbidden(g, x, y)) and nx.is_d_separator(proper_backdoor_graph(g, x, y), {x}, {y}, z)


def optimal_adjustment(g: nx.DiGraph, x, y) -> set:
    """O-set (Henckel, Perković & Maathuis 2022): the parents of causal-path nodes minus forbidden nodes. This is the adjustment set with the smallest variance."""
    pa = set().union(*(set(g.predecessors(w)) for w in causal_nodes(g, x, y)))
    return pa - forbidden(g, x, y)


def adjustment_sets(g: nx.DiGraph, x, y, latent=(), max_size: int = 4) -> list[set]:
    """List of minimal valid adjustment sets built from observed nodes, searched from smallest."""
    cand = sorted(set(g.nodes) - forbidden(g, x, y) - {y} - set(latent))
    found = []
    for k in range(max_size + 1):
        for z in map(set, combinations(cand, k)):
            if not any(f <= z for f in found) and is_valid_adjustment(g, x, y, z):
                found.append(z)
    return found


# ---------- Frontdoor (Pearl 1995) ----------
def is_frontdoor(g: nx.DiGraph, x, y, m) -> bool:
    """Whether m satisfies the frontdoor criterion.
    (i) m intercepts all causal paths x → y, (ii) there is no open backdoor path from x to m, and (iii) every backdoor path from m to y is blocked by x.
    """
    m = as_set(m)
    if m & {x, y}:
        return False
    cut = g.copy()
    cut.remove_nodes_from(m)
    no_x_out = g.copy()
    no_x_out.remove_edges_from(list(g.out_edges(x)))
    no_m_out = g.copy()
    no_m_out.remove_edges_from(list(g.out_edges(m)))
    return (y not in nx.descendants(cut, x)
            and nx.is_d_separator(no_x_out, {x}, m, set())
            and nx.is_d_separator(no_m_out, m, {y}, {x}))


def frontdoor_sets(g: nx.DiGraph, x, y, latent=(), max_size: int = 3) -> list[set]:
    cand = sorted(causal_nodes(g, x, y) - {y} - set(latent))
    found = []
    for k in range(1, max_size + 1):
        for m in map(set, combinations(cand, k)):
            if not any(f <= m for f in found) and is_frontdoor(g, x, y, m):
                found.append(m)
    return found


# ---------- Instrumental variable ----------
def is_instrument(g: nx.DiGraph, x, y, z) -> bool:
    """Unconditional instrument: z is connected to x (relevance), and z ⫫ y in the graph with x's outgoing edges removed (exclusion and exogeneity)."""
    if z in (x, y):
        return False
    no_x_out = g.copy()
    no_x_out.remove_edges_from(list(g.out_edges(x)))
    return not nx.is_d_separator(g, {z}, {x}, set()) and nx.is_d_separator(no_x_out, {z}, {y}, set())


def instruments(g: nx.DiGraph, x, y, latent=()) -> list[str]:
    cand = set(g.nodes) - {x, y} - set(latent) - nx.descendants(g, x)
    return sorted(z for z in cand if is_instrument(g, x, y, z))


# ---------- Combined identification ----------
@dataclass
class Estimand:
    strategy: str  # backdoor | frontdoor | iv | no_effect | not_identified
    treatment: str
    outcome: str
    variables: set = field(default_factory=set)
    expression: str = ""

    def __str__(self):
        return f"[{self.strategy}] {self.expression}"


def _fmt(s: set) -> str:
    return ",".join(sorted(s))


def identify(g: nx.DiGraph, x, y, latent=()) -> Estimand:
    """Try backdoor, then frontdoor, then IV."""
    latent = set(latent)
    if y not in nx.descendants(g, x):
        return Estimand("no_effect", x, y, expression=f"P({y}|do({x})) = P({y})")
    o_set = optimal_adjustment(g, x, y)
    backdoor = [o_set] if not (o_set & latent) else adjustment_sets(g, x, y, latent)
    if backdoor:
        z = backdoor[0]
        expr = f"Σ_{{{_fmt(z)}}} P({y}|{x},{_fmt(z)}) P({_fmt(z)})" if z else f"P({y}|{x})"
        return Estimand("backdoor", x, y, z, expr)
    front = frontdoor_sets(g, x, y, latent)
    if front:
        m = _fmt(front[0])
        return Estimand("frontdoor", x, y, front[0], f"Σ_{{{m}}} P({m}|{x}) Σ_{{{x}'}} P({y}|{x}',{m}) P({x}')")
    ivs = instruments(g, x, y, latent)
    if ivs:
        return Estimand("iv", x, y, {ivs[0]}, f"Cov({ivs[0]},{y}) / Cov({ivs[0]},{x})  (assumes a linear model)")
    return Estimand("not_identified", x, y, expression="not identifiable from the observed variables")


# ---------- Complete identification with unobserved confounding (Shpitser & Pearl 2006, ID algorithm decision only) ----------
def _c_components(nodes: set, bi: set) -> list[set]:
    h = nx.Graph()
    h.add_nodes_from(nodes)
    h.add_edges_from(tuple(e) for e in bi if set(e) <= nodes)
    return [set(c) for c in nx.connected_components(h)]


def id_identifiable(g: nx.DiGraph, bi, x, y) -> bool:
    """P(y | do(x))is identified. g is the directed graph over observed variables, bi the bidirected pairs representing unobserved confounding {frozenset({u, v}), ...}."""
    bi = {frozenset(e) for e in bi}

    def rec(ys: set, xs: set, nodes: set) -> bool:
        if not xs:
            return True
        sub = g.subgraph(nodes)
        an = set(ys).union(*(nx.ancestors(sub, v) for v in ys))
        if an != nodes:
            return rec(ys, xs & an, an)
        cut = sub.copy()
        cut.remove_edges_from(list(sub.in_edges(xs)))
        w = nodes - xs - set(ys).union(*(nx.ancestors(cut, v) for v in ys))
        if w:
            return rec(ys, xs | w, nodes)
        cs = _c_components(nodes - xs, bi)
        if len(cs) > 1:
            return all(rec(s, nodes - s, nodes) for s in cs)
        s, whole = cs[0], _c_components(nodes, bi)
        if whole == [nodes]:
            return False  # hedge
        if s in whole:
            return True
        big = next(c for c in whole if s < c)
        return rec(ys, xs & big, big)

    return rec({y}, {x}, set(g.nodes))
