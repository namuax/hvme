"""식별된 추정식(Estimand)을 데이터로 계산한다. 이진 데이터는 층화(비모수), 연속 데이터는 선형 회귀를 쓴다."""
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd

from .identify import Estimand, identify


@dataclass
class Estimate:
    estimand: Estimand
    value: float
    se: float
    ci: tuple[float, float]
    method: str

    def __str__(self):
        return f"{self.estimand.strategy}/{self.method}: {self.value:.4f}  95% CI [{self.ci[0]:.4f}, {self.ci[1]:.4f}]"


# ---------- 이진 데이터 ----------
def backdoor_stratified(df: pd.DataFrame, x: str, y: str, z: list[str]) -> float:
    """Σ_z [P(y|x=1,z) - P(y|x=0,z)] P(z). 빈 층이 있으면(양수성 위반) NaN을 낸다."""
    d = df.assign(_all=0)
    keys = z or ["_all"]
    cell = d.groupby(keys + [x])[y].mean().unstack(x)
    pz = d.groupby(keys).size() / len(d)
    return float(((cell[1] - cell[0]) * pz).sum(min_count=len(pz)))


def frontdoor_discrete(df: pd.DataFrame, x: str, y: str, m: list[str]) -> float:
    """E[y|do(x=a)] = Σ_m P(m|a) Σ_x' P(y|x',m) P(x'), a=1 과 a=0 의 차."""
    pm_x = df.groupby(x)[m].value_counts(normalize=True).unstack(x).fillna(0)
    py_xm = df.groupby(m + [x])[y].mean().unstack(x)
    px = df[x].value_counts(normalize=True)
    inner = (py_xm * px).sum(axis=1)
    return float(((pm_x[1] - pm_x[0]) * inner.reindex(pm_x.index)).sum())


# ---------- 연속 데이터 (선형) ----------
def _ols(df: pd.DataFrame, y: str, xs: list[str]) -> np.ndarray:
    a = np.column_stack([np.ones(len(df))] + [df[c].to_numpy(float) for c in xs])
    return np.linalg.lstsq(a, df[y].to_numpy(float), rcond=None)[0][1:]


def backdoor_ols(df: pd.DataFrame, x: str, y: str, z: list[str]) -> float:
    return float(_ols(df, y, [x] + z)[0])


def frontdoor_linear(df: pd.DataFrame, x: str, y: str, m: list[str]) -> float:
    """Σ_m (x → m 계수) × (x를 통제한 m → y 계수). 매개변수끼리 엣지가 없는 경우에 맞다."""
    a = np.array([_ols(df, mi, [x])[0] for mi in m])
    b = _ols(df, y, m + [x])[: len(m)]
    return float(a @ b)


def iv_wald(df: pd.DataFrame, x: str, y: str, z: list[str]) -> float:
    c = df[[z[0], x, y]].cov()
    return float(c.loc[z[0], y] / c.loc[z[0], x])


METHODS = {
    ("backdoor", True): backdoor_stratified,
    ("backdoor", False): backdoor_ols,
    ("frontdoor", True): frontdoor_discrete,
    ("frontdoor", False): frontdoor_linear,
    ("iv", True): iv_wald,
    ("iv", False): iv_wald,
    ("no_effect", True): lambda *a: 0.0,
    ("no_effect", False): lambda *a: 0.0,
    ("not_identified", True): lambda *a: float("nan"),
    ("not_identified", False): lambda *a: float("nan"),
}


def is_binary(df: pd.DataFrame, cols) -> bool:
    return all(set(np.unique(df[c])) <= {0, 1} for c in cols)


def estimate(g: nx.DiGraph, df: pd.DataFrame, x: str, y: str, latent=(), n_boot: int = 200, seed: int = 0,
             estimand: Estimand | None = None) -> Estimate:
    """identify → 데이터 유형에 맞는 추정기 → 부트스트랩 표준오차·95% 신뢰구간."""
    e = estimand or identify(g, x, y, latent)
    v = sorted(e.variables)
    binary = is_binary(df, [x, y] + v)
    fn = METHODS[(e.strategy, binary)]
    rng = np.random.default_rng(seed)
    boots = np.array([fn(df.iloc[rng.integers(0, len(df), len(df))], x, y, v) for _ in range(n_boot)])
    se = float(np.nanstd(boots)) if n_boot else float("nan")
    ci = tuple(np.nanquantile(boots, [0.025, 0.975])) if n_boot else (float("nan"),) * 2
    return Estimate(e, fn(df, x, y, v), se, ci, fn.__name__)
