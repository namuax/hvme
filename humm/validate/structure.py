"""L1 구조 검증: DAG 여부, 자기 루프, 고립 노드, 시간 순서 위반."""
import networkx as nx


def check_structure(g: nx.DiGraph, time: dict | None = None) -> dict:
    """time: 노드 → 발생 시점(숫자). 원인이 결과보다 늦게 일어나는 엣지는 시간 순서 위반이다."""
    time = time or {}
    return {
        "is_dag": nx.is_directed_acyclic_graph(g),
        "cycles": [list(c) for c in list(nx.simple_cycles(g))[:20]],
        "self_loops": [list(e) for e in nx.selfloop_edges(g)],
        "isolated": sorted(nx.isolates(g)),
        "temporal_violations": [[u, v] for u, v in g.edges if u in time and v in time and time[u] > time[v]],
    }
