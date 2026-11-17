"""구조적 인과 모형(SCM). Humm-Gym 정답 계산과 엔진 검증에 쓴다.

- BinarySCM: 이진 변수, 문턱값 메커니즘 v = 1[u_v < p_v(pa)], u_v ~ U(0,1) 독립.
  관측·개입·반사실 질의를 모두 같은 '외생 잡음 세계' 위에서 계산한다.
  잡음 공간을 문턱값으로 나눈 칸(cell)이 max_cells 이하이면 정확 계산, 넘으면 몬테카를로로 근사한다.
- LinearSCM: 선형 가우스 모형. 총효과는 (I - W)^-1 로 정확히 계산한다.
"""
from dataclasses import dataclass, field
from itertools import product

import networkx as nx
import numpy as np
import pandas as pd


@dataclass
class BinarySCM:
    graph: nx.DiGraph
    cpt: dict[str, np.ndarray]  # cpt[v][k] = P(v=1 | 부모 비트 k), 부모 i번째가 비트 i
    max_cells: int = 200_000
    n_mc: int = 400_000
    seed: int = 0
    _u: tuple = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.order = list(nx.topological_sort(self.graph))
        self.parents = {v: sorted(self.graph.predecessors(v)) for v in self.order}
        self.noise()  # 정확 계산 여부(self.exact)를 생성 시점에 정한다

    @classmethod
    def random(cls, g: nx.DiGraph, seed: int = 0, low: float = 0.1, high: float = 0.9, **kw) -> "BinarySCM":
        rng = np.random.default_rng(seed)
        cpt = {v: rng.uniform(low, high, 2 ** g.in_degree(v)).round(2) for v in g.nodes}
        return cls(g, cpt, seed=seed, **kw)

    @classmethod
    def fit(cls, g: nx.DiGraph, df: pd.DataFrame, alpha: float = 1.0, **kw) -> "BinarySCM":
        """데이터로 확률표를 추정한다(라플라스 평활 alpha). 같은 그래프·확률표를 갖는 문턱값 SCM을 만들어 반사실을 계산한다.
        관찰 분포와 개입 분포는 그래프만 맞으면 식별되지만, 반사실(PN·PS 등 세계 간 결합)은 문턱값 메커니즘(단조성) 가정에 기댄다."""
        cpt = {}
        for v in g.nodes:
            pa = sorted(g.predecessors(v))
            bits = sum(df[p].to_numpy(int) << i for i, p in enumerate(pa)) if pa else np.zeros(len(df), int)
            n, k = np.bincount(bits, minlength=2 ** len(pa)), np.bincount(bits, weights=df[v].to_numpy(int), minlength=2 ** len(pa))
            cpt[v] = (k + alpha) / (n + 2 * alpha)
        return cls(g, cpt, **kw)

    # ---------- 외생 잡음 세계 ----------
    def noise(self) -> tuple[dict, np.ndarray]:
        """(U, w): 노드별 잡음 표본과 가중치. 정확 모드에서는 칸 중심점과 칸 넓이, MC 모드에서는 균등 가중치."""
        if self._u is None:
            edges = {v: np.unique(np.r_[0.0, self.cpt[v], 1.0]) for v in self.order}
            n_cells = int(np.prod([len(e) - 1 for e in edges.values()]))
            if n_cells <= self.max_cells:
                mids = [(e[1:] + e[:-1]) / 2 for e in edges.values()]
                lens = [np.diff(e) for e in edges.values()]
                grid = np.array(list(product(*mids))).T
                w = np.prod(np.array(list(product(*lens))), axis=1)
                self._u = (dict(zip(edges, grid)), w)
                self.exact = True
            else:
                rng = np.random.default_rng(self.seed)
                self._u = ({v: rng.uniform(size=self.n_mc) for v in self.order}, np.full(self.n_mc, 1 / self.n_mc))
                self.exact = False
        return self._u

    def world(self, do: dict | None = None, u: dict | None = None) -> dict[str, np.ndarray]:
        """잡음 u에서 개입 do를 적용한 세계의 모든 변수 값. do 값은 상수 또는 (다른 세계에서 온) 배열이다."""
        do = do or {}
        u = u or self.noise()[0]
        n = len(next(iter(u.values())))
        vals = {}
        for v in self.order:
            bits = sum(vals[p] << i for i, p in enumerate(self.parents[v])) if self.parents[v] else np.zeros(n, int)
            vals[v] = np.broadcast_to(do[v], n).astype(int) if v in do else (u[v] < self.cpt[v][bits]).astype(int)
        return vals

    def _mean(self, values, mask=None) -> float:
        w = self.noise()[1] if mask is None else self.noise()[1] * mask
        return float((w * values).sum() / w.sum())

    def sample(self, n: int, do: dict | None = None, seed: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        return pd.DataFrame(self.world(do, {v: rng.uniform(size=n) for v in self.order}))

    # ---------- Rung 1·2 ----------
    def prob(self, y: str, do: dict | None = None, given: dict | None = None) -> float:
        """P(y=1 | do(...), given...)."""
        w = self.world(do)
        mask = np.prod([w[k] == v for k, v in (given or {}).items()], axis=0) if given else None
        return self._mean(w[y], mask)

    def ate(self, x: str, y: str) -> float:
        return self.prob(y, {x: 1}) - self.prob(y, {x: 0})

    def cde(self, x: str, y: str, mediators: dict) -> float:
        """통제 직접효과: 매개변수를 고정값으로 묶은 상태에서의 x 효과."""
        return self.prob(y, {x: 1, **mediators}) - self.prob(y, {x: 0, **mediators})

    # ---------- Rung 3: 반사실 (같은 잡음을 공유하는 세계들) ----------
    def ett(self, x: str, y: str) -> float:
        """처치군 효과 E[Y_{x=1} - Y_{x=0} | X=1]."""
        f = self.world()
        return self._mean(self.world({x: 1})[y] - self.world({x: 0})[y], f[x] == 1)

    def nde(self, x: str, y: str, mediators: list[str]) -> float:
        """자연 직접효과 E[Y(x=1, M(x=0))] - E[Y(x=0)]."""
        m0 = self.world({x: 0})
        return self._mean(self.world({x: 1, **{m: m0[m] for m in mediators}})[y] - m0[y])

    def nie(self, x: str, y: str, mediators: list[str]) -> float:
        """자연 간접효과 E[Y(x=0, M(x=1))] - E[Y(x=0)]."""
        m0, m1 = self.world({x: 0}), self.world({x: 1})
        return self._mean(self.world({x: 0, **{m: m1[m] for m in mediators}})[y] - m0[y])

    def pn(self, x: str, y: str) -> float:
        """필요 확률 P(Y_{x=0}=0 | X=1, Y=1)."""
        f = self.world()
        return self._mean(self.world({x: 0})[y] == 0, (f[x] == 1) & (f[y] == 1))

    def ps(self, x: str, y: str) -> float:
        """충분 확률 P(Y_{x=1}=1 | X=0, Y=0)."""
        f = self.world()
        return self._mean(self.world({x: 1})[y] == 1, (f[x] == 0) & (f[y] == 0))


@dataclass
class LinearSCM:
    graph: nx.DiGraph
    weights: dict[tuple[str, str], float]
    noise_std: dict[str, float]

    def __post_init__(self):
        self.order = list(nx.topological_sort(self.graph))
        self.index = {v: i for i, v in enumerate(self.order)}

    @classmethod
    def random(cls, g: nx.DiGraph, seed: int = 0, w_range=(0.5, 2.0)) -> "LinearSCM":
        rng = np.random.default_rng(seed)
        weights = {e: float(rng.uniform(*w_range) * rng.choice([-1, 1])) for e in g.edges}
        return cls(g, weights, {v: float(rng.uniform(0.5, 1.5)) for v in g.nodes})

    def matrix(self) -> np.ndarray:
        w = np.zeros((len(self.order),) * 2)
        for (a, b), c in self.weights.items():
            w[self.index[a], self.index[b]] = c
        return w

    def total_effect(self, x: str, y: str) -> float:
        """y의 x에 대한 총효과 = 모든 인과 경로 계수 곱의 합 = (I - W)^-1 [x, y]."""
        t = np.linalg.inv(np.eye(len(self.order)) - self.matrix())
        return float(t[self.index[x], self.index[y]])

    def sample(self, n: int, do: dict | None = None, seed: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        do = do or {}
        vals = {}
        for v in self.order:
            pa = sum(self.weights[(p, v)] * vals[p] for p in self.graph.predecessors(v))
            vals[v] = np.full(n, float(do[v])) if v in do else pa + rng.normal(0, self.noise_std[v], n)
        return pd.DataFrame(vals)
