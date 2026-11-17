"""통계적 관계 → 마르코프 동치류 → 인과 가설 판정 (Corr2Cause식 구조 질의, Step 9 SFT v2.2).

전제가 계의 상관·조건부 독립 관계를 모두 준다고 본다(충실성).
- 골격: 어떤 조건에서도 독립이 되지 않는 쌍만 인접한다.
- v-구조: 인접하지 않은 a, b의 공통 이웃 c가 a, b를 떼어 놓는 어느 조건 집합에도 없으면 a → c ← b이다.
- 동치류: 골격의 방향 조합 가운데 비순환이고 v-구조가 같은 것. 변수 6개(엣지 15개 이하)까지는 전수 열거로 충분하다.
  humm.gym.graphs.mec는 모든 DAG의 CI 서명으로 찾으므로 변수 4개까지만 쓸 수 있다.

가설 정의는 Corr2Cause(Jin et al., 2024) 라벨에 맞췄다. 'via'는 중간 변수를 거치는 경로가 있다는 뜻이며 직접 엣지가 함께 있어도 참이다
(테스트 1,162문항 대조, reports/2026-10-31_corr2cause_analysis.md 3절).
"""
from itertools import combinations, product

import networkx as nx

HYPOTHESES = ("direct", "via", "cause", "collider", "confounder")


def holds(h: str, g: nx.DiGraph, a, b) -> bool:
    """그래프 g에서 가설 h(a, b)가 성립하는가."""
    return {"direct": lambda: g.has_edge(a, b),
            "via": lambda: any(len(p) > 2 for p in nx.all_simple_paths(g, a, b)),
            "cause": lambda: b in nx.descendants(g, a),
            "collider": lambda: any(g.has_edge(a, c) and g.has_edge(b, c) for c in g),
            "confounder": lambda: bool((nx.ancestors(g, a) & nx.ancestors(g, b)) - {a, b})}[h]()


def vstructures(g: nx.DiGraph) -> set[tuple]:
    """(a, c, b), a < b: a → c ← b이고 a, b가 인접하지 않는다."""
    return {(a, c, b) for c in g for a, b in combinations(sorted(g.predecessors(c)), 2) if not g.has_edge(a, b) and not g.has_edge(b, a)}


def from_relations(nodes, sepsets: dict[frozenset, list[frozenset]]) -> tuple[set[frozenset], set[tuple]]:
    """sepsets: 독립인 쌍 → 그 쌍을 떼어 놓는 조건 집합 목록(빈 집합은 주변 독립). 골격과 v-구조를 돌려준다."""
    nodes = sorted(nodes)
    adj = {frozenset(p) for p in combinations(nodes, 2) if frozenset(p) not in sepsets}
    vs = {(a, c, b) for a, b in combinations(nodes, 2) if frozenset((a, b)) in sepsets for c in nodes
          if {frozenset((a, c)), frozenset((b, c))} <= adj and not any(c in s for s in sepsets[frozenset((a, b))])}
    return adj, vs


def equivalence_class(nodes, adj: set[frozenset], vs: set[tuple]) -> list[nx.DiGraph]:
    """골격 adj와 v-구조 vs를 갖는 모든 DAG."""
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
    """그래프 g의 마르코프 동치류."""
    return equivalence_class(g.nodes, {frozenset(e) for e in g.edges}, vstructures(g))
