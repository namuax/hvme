"""d-separation and the conditional independences a graph implies."""
import networkx as nx


def as_set(a) -> set:
    return {a} if isinstance(a, str) else set(a)


def d_separated(g: nx.DiGraph, x, y, z=()) -> bool:
    """x ⫫ y | z holds in graph g (x, y, z are each a node or a set of nodes)."""
    return nx.is_d_separator(g, as_set(x), as_set(y), as_set(z))


def blocked_by(g: nx.DiGraph, path, z: set):
    """The node blocking the path and why; None if not blocked. A path is blocked if a non-collider is in z, or a collider and all its descendants are outside z."""
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
    """Local Markov condition: list of (v, non-parent non-descendants, parents). Each item means v ⫫ non-descendants | parents."""
    out = []
    for v in nx.topological_sort(g):
        pa = set(g.predecessors(v))
        rest = set(g.nodes) - nx.descendants(g, v) - pa - {v}
        out.append((v, rest, pa))
    return [t for t in out if t[1]]
