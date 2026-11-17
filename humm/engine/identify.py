"""인과 효과 식별: 백도어(일반화 조정 기준), 프론트도어, 도구변수(IV).

latent 인자로 관측되지 않는 노드를 지정한다. 조정·매개·도구 집합에는 관측 노드만 쓴다.
"""
from dataclasses import dataclass, field
from itertools import combinations

import networkx as nx

from .dsep import as_set


# ---------- 백도어: 일반화 조정 기준 (Shpitser et al. 2010; Perković et al. 2018) ----------
def causal_nodes(g: nx.DiGraph, x, y) -> set:
    """x에서 y로 가는 인과 경로 위의 노드(x 제외, y 포함). 인과 경로가 없으면 빈 집합."""
    de = nx.descendants(g, x)
    return (de & nx.ancestors(g, y)) | {y} if y in de else set()


def forbidden(g: nx.DiGraph, x, y) -> set:
    """조정 집합에 들어가면 안 되는 노드: x와, 인과 경로 위 노드들의 자손."""
    return {x}.union(*({w} | nx.descendants(g, w) for w in causal_nodes(g, x, y)))


def proper_backdoor_graph(g: nx.DiGraph, x, y) -> nx.DiGraph:
    """x에서 인과 경로의 첫 엣지를 지운 그래프."""
    h = g.copy()
    h.remove_edges_from([(x, w) for w in causal_nodes(g, x, y) if g.has_edge(x, w)])
    return h


def is_valid_adjustment(g: nx.DiGraph, x, y, z) -> bool:
    """z가 x → y 효과의 유효한 조정 집합인가."""
    z = as_set(z)
    return not (z & forbidden(g, x, y)) and nx.is_d_separator(proper_backdoor_graph(g, x, y), {x}, {y}, z)


def optimal_adjustment(g: nx.DiGraph, x, y) -> set:
    """O-set (Henckel, Perković & Maathuis 2022): 인과 경로 노드들의 부모 중 금지 노드를 뺀 집합. 분산이 가장 작은 조정 집합이다."""
    pa = set().union(*(set(g.predecessors(w)) for w in causal_nodes(g, x, y)))
    return pa - forbidden(g, x, y)


def adjustment_sets(g: nx.DiGraph, x, y, latent=(), max_size: int = 4) -> list[set]:
    """관측 노드로 만든 최소(minimal) 유효 조정 집합 목록. 작은 집합부터 찾는다."""
    cand = sorted(set(g.nodes) - forbidden(g, x, y) - {y} - set(latent))
    found = []
    for k in range(max_size + 1):
        for z in map(set, combinations(cand, k)):
            if not any(f <= z for f in found) and is_valid_adjustment(g, x, y, z):
                found.append(z)
    return found


# ---------- 프론트도어 (Pearl 1995) ----------
def is_frontdoor(g: nx.DiGraph, x, y, m) -> bool:
    """m이 프론트도어 기준을 만족하는가.
    (i) m이 x → y 인과 경로를 모두 가로막고, (ii) x에서 m으로 열린 백도어 경로가 없고, (iii) m에서 y로 가는 백도어 경로가 x로 막힌다.
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


# ---------- 도구변수 ----------
def is_instrument(g: nx.DiGraph, x, y, z) -> bool:
    """무조건부 도구변수: z가 x와 연결되고(관련성), x의 출력 엣지를 지운 그래프에서 z ⫫ y (배제·외생성)."""
    if z in (x, y):
        return False
    no_x_out = g.copy()
    no_x_out.remove_edges_from(list(g.out_edges(x)))
    return not nx.is_d_separator(g, {z}, {x}, set()) and nx.is_d_separator(no_x_out, {z}, {y}, set())


def instruments(g: nx.DiGraph, x, y, latent=()) -> list[str]:
    cand = set(g.nodes) - {x, y} - set(latent) - nx.descendants(g, x)
    return sorted(z for z in cand if is_instrument(g, x, y, z))


# ---------- 통합 식별 ----------
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
    """백도어 → 프론트도어 → IV 순으로 식별을 시도한다."""
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
        return Estimand("iv", x, y, {ivs[0]}, f"Cov({ivs[0]},{y}) / Cov({ivs[0]},{x})  (선형 모형 가정)")
    return Estimand("not_identified", x, y, expression="관측 변수만으로 식별할 수 없다")


# ---------- 관측 안 된 교란이 있는 그래프의 완전한 식별 (Shpitser & Pearl 2006, ID 알고리즘의 판정만) ----------
def _c_components(nodes: set, bi: set) -> list[set]:
    h = nx.Graph()
    h.add_nodes_from(nodes)
    h.add_edges_from(tuple(e) for e in bi if set(e) <= nodes)
    return [set(c) for c in nx.connected_components(h)]


def id_identifiable(g: nx.DiGraph, bi, x, y) -> bool:
    """P(y | do(x))가 식별되는가. g는 관측 변수의 방향 그래프, bi는 관측 안 된 교란을 나타내는 양방향 쌍 {frozenset({u, v}), ...}."""
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
            return False  # 헤지
        if s in whole:
            return True
        big = next(c for c in whole if s < c)
        return rec(ys, xs & big, big)

    return rec({y}, {x}, set(g.nodes))
