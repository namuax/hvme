"""L1 structure checks: DAG-ness, self loops, isolated nodes, time-order violations."""
import networkx as nx


def check_structure(g: nx.DiGraph, time: dict | None = None) -> dict:
    """time: node → occurrence time (numeric). An edge whose cause happens after its effect is a time-order violation."""
    time = time or {}
    return {
        "is_dag": nx.is_directed_acyclic_graph(g),
        "cycles": [list(c) for c in list(nx.simple_cycles(g))[:20]],
        "self_loops": [list(e) for e in nx.selfloop_edges(g)],
        "isolated": sorted(nx.isolates(g)),
        "temporal_violations": [[u, v] for u, v in g.edges if u in time and v in time and time[u] > time[v]],
    }
