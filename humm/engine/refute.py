"""추정치 반박 테스트 (DoWhy refuter와 같은 발상, Humm 추정기 위에서 동작).

- placebo: 처치를 무작위로 섞으면 효과가 0이 되어야 한다
- random_common_cause: 독립 난수 변수를 공통 원인으로 추가해도 효과가 그대로여야 한다
- data_subset: 80% 부분표본에서도 효과가 그대로여야 한다
판정 기준: |새 값 - 기대값| < 2 × 부트스트랩 표준오차
"""
import networkx as nx
import numpy as np
import pandas as pd

from .estimate import estimate


def refute(g: nx.DiGraph, df: pd.DataFrame, x: str, y: str, latent=(), n_subsets: int = 10, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = estimate(g, df, x, y, latent, n_boot=100, seed=seed)
    tol = 2 * base.se

    placebo = estimate(g, df.assign(**{x: rng.permutation(df[x].to_numpy())}), x, y, latent, n_boot=0, estimand=base.estimand).value

    g_w = g.copy()
    g_w.add_edges_from([("_w", x), ("_w", y)])
    w = rng.integers(0, 2, len(df)) if base.method.endswith(("stratified", "discrete")) else rng.normal(size=len(df))
    rcc = estimate(g_w, df.assign(_w=w), x, y, latent, n_boot=0).value

    subs = [estimate(g, df.sample(frac=0.8, random_state=int(s)), x, y, latent, n_boot=0, estimand=base.estimand).value
            for s in rng.integers(0, 2**31, n_subsets)]

    rows = [("placebo_treatment", 0.0, placebo), ("random_common_cause", base.value, rcc), ("data_subset", base.value, float(np.mean(subs)))]
    out = pd.DataFrame(rows, columns=["test", "expected", "new_value"])
    return out.assign(estimate=base.value, tolerance=tol, passed=(out.new_value - out.expected).abs() < tol)
