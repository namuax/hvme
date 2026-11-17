"""L6 pipeline benchmark: score causal discovery results with SHD, SID and F1 on bnlearn networks whose true graph is known."""
import time

import networkx as nx
import pandas as pd

from ..bench.metrics import graph_report


def sample_bnlearn(name: str, n: int, seed: int = 0) -> tuple[nx.DiGraph, pd.DataFrame]:
    """True graph and sample (categories integer-coded) of a bnlearn network."""
    from pgmpy.example_models import load_model

    model = load_model(f"bnlearn/{name}")
    df = model.simulate(n_samples=n, seed=seed, show_progress=False)
    return nx.DiGraph(sorted(model.edges())), df[sorted(df.columns)].apply(lambda c: c.astype("category").cat.codes)


def benchmark(discover, names=("asia", "sachs", "insurance", "alarm"), n: int = 5000, seed: int = 0) -> pd.DataFrame:
    """Run discover(df) -> nx.DiGraph on each network and collect metrics. Use as a regression test whenever the pipeline changes."""
    rows = []
    for name in names:
        true, df = sample_bnlearn(name, n, seed)
        t = time.time()
        est = discover(df)
        est.add_nodes_from(true.nodes)
        rows.append({"network": name, "nodes": true.number_of_nodes(), "edges": true.number_of_edges(), **graph_report(true, est), "seconds": round(time.time() - t, 2)})
    return pd.DataFrame(rows)
