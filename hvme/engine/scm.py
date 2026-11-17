"""Structural causal model (SCM), used to compute Hvme-Gym answers and validate the engine.

- BinarySCM: binary variables, threshold mechanism v = 1[u_v < p_v(pa)], with independent u_v ~ U(0,1).
  Observational, interventional and counterfactual queries are all computed over the same 'exogenous noise world'.
  If the number of cells obtained by partitioning the noise space at the thresholds is at most max_cells, computation is exact; otherwise it is approximated by Monte Carlo.
- LinearSCM: linear Gaussian model. Total effects are computed exactly via (I - W)^-1.
"""
from dataclasses import dataclass, field
from itertools import product

import networkx as nx
import numpy as np
import pandas as pd


@dataclass
class BinarySCM:
    graph: nx.DiGraph
    cpt: dict[str, np.ndarray]  # cpt[v][k] = P(v=1 | parent bits k), the i-th parent is bit i
    max_cells: int = 200_000
    n_mc: int = 400_000
    seed: int = 0
    _u: tuple = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.order = list(nx.topological_sort(self.graph))
        self.parents = {v: sorted(self.graph.predecessors(v)) for v in self.order}
        self.noise()  # decide exact vs. Monte Carlo (self.exact) at construction

    @classmethod
    def random(cls, g: nx.DiGraph, seed: int = 0, low: float = 0.1, high: float = 0.9, **kw) -> "BinarySCM":
        rng = np.random.default_rng(seed)
        cpt = {v: rng.uniform(low, high, 2 ** g.in_degree(v)).round(2) for v in g.nodes}
        return cls(g, cpt, seed=seed, **kw)

    @classmethod
    def fit(cls, g: nx.DiGraph, df: pd.DataFrame, alpha: float = 1.0, **kw) -> "BinarySCM":
        """Estimate the probability tables from data (Laplace smoothing alpha). Builds a threshold SCM with the same graph and tables to compute counterfactuals.
        Observational and interventional distributions are identified if the graph is right, but counterfactuals (PN, PS and other cross-world quantities) rely on the threshold-mechanism (monotonicity) assumption."""
        cpt = {}
        for v in g.nodes:
            pa = sorted(g.predecessors(v))
            bits = sum(df[p].to_numpy(int) << i for i, p in enumerate(pa)) if pa else np.zeros(len(df), int)
            n, k = np.bincount(bits, minlength=2 ** len(pa)), np.bincount(bits, weights=df[v].to_numpy(int), minlength=2 ** len(pa))
            cpt[v] = (k + alpha) / (n + 2 * alpha)
        return cls(g, cpt, **kw)

    # ---------- Exogenous noise world ----------
    def noise(self) -> tuple[dict, np.ndarray]:
        """(U, w): per-node noise samples and weights. Exact mode uses cell centers and cell widths; MC mode uses uniform weights."""
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
        """All variable values in the world with intervention do applied to noise u. do values are constants or arrays (from another world)."""
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
        """Controlled direct effect: the effect of x with the mediators held at fixed values."""
        return self.prob(y, {x: 1, **mediators}) - self.prob(y, {x: 0, **mediators})

    # ---------- Rung 3: counterfactuals (worlds sharing the same noise) ----------
    def ett(self, x: str, y: str) -> float:
        """Effect on the treated E[Y_{x=1} - Y_{x=0} | X=1]."""
        f = self.world()
        return self._mean(self.world({x: 1})[y] - self.world({x: 0})[y], f[x] == 1)

    def nde(self, x: str, y: str, mediators: list[str]) -> float:
        """Natural direct effect E[Y(x=1, M(x=0))] - E[Y(x=0)]."""
        m0 = self.world({x: 0})
        return self._mean(self.world({x: 1, **{m: m0[m] for m in mediators}})[y] - m0[y])

    def nie(self, x: str, y: str, mediators: list[str]) -> float:
        """Natural indirect effect E[Y(x=0, M(x=1))] - E[Y(x=0)]."""
        m0, m1 = self.world({x: 0}), self.world({x: 1})
        return self._mean(self.world({x: 0, **{m: m1[m] for m in mediators}})[y] - m0[y])

    def pn(self, x: str, y: str) -> float:
        """Probability of necessity P(Y_{x=0}=0 | X=1, Y=1)."""
        f = self.world()
        return self._mean(self.world({x: 0})[y] == 0, (f[x] == 1) & (f[y] == 1))

    def ps(self, x: str, y: str) -> float:
        """Probability of sufficiency P(Y_{x=1}=1 | X=0, Y=0)."""
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
        """Total effect of x on y = sum over all causal paths of the product of coefficients = (I - W)^-1 [x, y]."""
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
