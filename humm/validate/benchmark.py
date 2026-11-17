"""L6 파이프라인 벤치마크: 참 그래프를 아는 bnlearn 네트워크에서 인과 탐색 결과를 SHD·SID·F1로 잰다."""
import time

import networkx as nx
import pandas as pd

from ..bench.metrics import graph_report


def sample_bnlearn(name: str, n: int, seed: int = 0) -> tuple[nx.DiGraph, pd.DataFrame]:
    """bnlearn 네트워크의 참 그래프와 표본(범주를 정수 코드로 바꾼 표)."""
    from pgmpy.example_models import load_model

    model = load_model(f"bnlearn/{name}")
    df = model.simulate(n_samples=n, seed=seed, show_progress=False)
    return nx.DiGraph(sorted(model.edges())), df[sorted(df.columns)].apply(lambda c: c.astype("category").cat.codes)


def benchmark(discover, names=("asia", "sachs", "insurance", "alarm"), n: int = 5000, seed: int = 0) -> pd.DataFrame:
    """discover(df) -> nx.DiGraph 를 각 네트워크에 돌려 지표를 모은다. 파이프라인을 바꿀 때마다 회귀 테스트로 쓴다."""
    rows = []
    for name in names:
        true, df = sample_bnlearn(name, n, seed)
        t = time.time()
        est = discover(df)
        est.add_nodes_from(true.nodes)
        rows.append({"network": name, "nodes": true.number_of_nodes(), "edges": true.number_of_edges(), **graph_report(true, est), "seconds": round(time.time() - t, 2)})
    return pd.DataFrame(rows)
