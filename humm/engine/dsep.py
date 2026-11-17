"""d-분리와 그래프가 함의하는 조건부 독립."""
import networkx as nx


def as_set(a) -> set:
    return {a} if isinstance(a, str) else set(a)


def d_separated(g: nx.DiGraph, x, y, z=()) -> bool:
    """x ⫫ y | z 가 그래프 g에서 성립하는가 (x, y, z는 노드 하나 또는 노드 집합)."""
    return nx.is_d_separator(g, as_set(x), as_set(y), as_set(z))


def blocked_by(g: nx.DiGraph, path, z: set):
    """경로를 막는 노드와 이유. 막히지 않으면 None. 비충돌 노드가 z에 있거나, 충돌 노드와 그 자손이 모두 z 밖에 있으면 막힌다."""
    for a, c, b in zip(path, path[1:], path[2:]):
        collider = g.has_edge(a, c) and g.has_edge(b, c)
        if collider and not ({c} | nx.descendants(g, c)) & z:
            return c, "collider"
        if not collider and c in z:
            return c, "noncollider"
    return None


def paths_between(g: nx.DiGraph, a, b, cutoff: int = 7):
    return sorted(nx.all_simple_paths(g.to_undirected(as_view=True), a, b, cutoff=cutoff), key=lambda p: (len(p), p))


def local_markov(g: nx.DiGraph) -> list[tuple[str, set, set]]:
    """국소 마르코프 조건: (v, 부모가 아닌 비자손, 부모) 목록. 각 항목은 v ⫫ 비자손 | 부모 를 뜻한다."""
    out = []
    for v in nx.topological_sort(g):
        pa = set(g.predecessors(v))
        rest = set(g.nodes) - nx.descendants(g, v) - pa - {v}
        out.append((v, rest, pa))
    return [t for t in out if t[1]]
