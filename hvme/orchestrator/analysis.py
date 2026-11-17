"""Analysis of a single query: identification -> estimation -> (counterfactual) -> refutation -> evidence paths -> graph sensitivity (L5) -> assumption list."""
import re
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import pandas as pd

from ..engine import BinarySCM, Estimand, estimate, identify, refute
from ..engine.identify import causal_nodes
from ..graph.schema import HCG, EdgeStatus, EvidenceKind
from .query import Query

IDENTIFIED = ("backdoor", "frontdoor", "iv")
MAIN = {"ate": "ate", "association": "ate", "mediation": "nie", "ett": "ett"}  # headline number per kind


def latent_of(g: nx.DiGraph, df: pd.DataFrame) -> list[str]:
    """Nodes that are in the graph but have no data column (e.g. variables that appear only in text) are treated as unobserved."""
    return sorted(set(g) - set(df.columns))


def mediators_of(q: Query, g: nx.DiGraph, df: pd.DataFrame) -> list[str]:
    return q.mediators or sorted((causal_nodes(g, q.treatment, q.outcome) - {q.outcome}) & set(df.columns))


def fit_scm(g: nx.DiGraph, df: pd.DataFrame, nodes) -> BinarySCM:
    """Fit a binary SCM using only the observed ancestors of the nodes of interest."""
    keep = set().union(*(nx.ancestors(g, v) | {v} for v in nodes)) & set(df.columns)
    return BinarySCM.fit(nx.DiGraph(g.subgraph(sorted(keep))), df)


# Key number per kind. ate/association come from the identified estimand; mediation/ett are computed on the fitted SCM.
def _scm_values(q: Query, g: nx.DiGraph, df: pd.DataFrame) -> dict:
    x, y, m = q.treatment, q.outcome, mediators_of(q, g, df)
    scm = fit_scm(g, df, [x, y, *m])
    out = {"ate_scm": scm.ate(x, y)}
    out |= {"nde": scm.nde(x, y, m), "nie": scm.nie(x, y, m)} if q.kind == "mediation" else {}
    out |= {"ett": scm.ett(x, y)} if q.kind == "ett" else {}
    return out


def headline(q: Query, g: nx.DiGraph, df: pd.DataFrame) -> float:
    """Headline numbers used in the sensitivity analysis."""
    lat = latent_of(g, df)
    return {"ate": lambda: estimate(g, df, q.treatment, q.outcome, lat, n_boot=0).value,
            "association": lambda: estimate(g, df, q.treatment, q.outcome, lat, n_boot=0).value,
            "mediation": lambda: _scm_values(q, g, df)["nie"],
            "ett": lambda: _scm_values(q, g, df)["ett"]}[q.kind]()


def association(df: pd.DataFrame, x: str, y: str) -> float:
    return float(df.loc[df[x] == 1, y].mean() - df.loc[df[x] == 0, y].mean())


def bootstrap(fn, df: pd.DataFrame, n_boot: int, seed: int = 0) -> dict:
    """fn(df) -> {name: value}. Resamples the data and gives a 95% interval per name."""
    rng = np.random.default_rng(seed)
    point = fn(df)
    boots = pd.DataFrame([fn(df.iloc[rng.integers(0, len(df), len(df))]) for _ in range(n_boot)])
    return {k: {"value": float(v), "ci": tuple(float(c) for c in np.nanquantile(boots[k], [0.025, 0.975])) if n_boot else (np.nan, np.nan)}
            for k, v in point.items()}


def display_names(hcg: HCG, text: str = "") -> dict:
    """id -> display name: expression used in the question > shortest Korean alias > id."""
    hangul = lambda w: bool(re.search("[가-힣]", w))
    out = {}
    for n in hcg.nodes:
        names = sorted({n.label, *n.aliases} - {""}, key=lambda w: (len(w), w))
        said = [w for w in names if w in text]
        out[n.id] = (said and max(said, key=len)) or next((w for w in names if hangul(w)), n.id)
    return out


# ---------- evidence paths ----------
def edge_info(e) -> dict:
    quote = next((x.span for x in e.evidence if x.kind == EvidenceKind.TEXT), "")
    return {"edge": f"{e.src}→{e.dst}", "status": e.status.value, "confidence": round(e.confidence, 3), "sign": e.sign,
            "evidence": sorted({x.kind.value for x in e.evidence}), "quote": quote}


def evidence_paths(hcg: HCG, x: str, y: str, k: int = 5) -> list[list[dict]]:
    """Causal paths from x to y (the k shortest) with the status, confidence and evidence of the edges on them."""
    g = hcg.to_networkx()
    paths = sorted(nx.all_simple_paths(g, x, y, cutoff=6), key=lambda p: (len(p), p))[:k]
    return [[edge_info(hcg.edge(u, v)) for u, v in zip(p, p[1:])] for p in paths]


def backdoor_paths(g: nx.DiGraph, x: str, y: str, k: int = 3) -> list[list[str]]:
    """x-y paths that start with an arrow into x (candidate confounding paths)."""
    u = g.to_undirected(as_view=True)
    paths = [p for p in nx.all_simple_paths(u, x, y, cutoff=6) if g.has_edge(p[1], x)]
    return sorted(paths, key=lambda p: (len(p), p))[:k]


# ---------- L5: graph sensitivity ----------
def alternatives(hcg: HCG, x: str, y: str, max_edges: int = 8):
    """Alternative graphs that show how much the conclusion depends on each edge. Verified edges can also be wrong, so status is ignored.
    - active edges touching ancestors of x and y (max_edges, lowest confidence first): graph with the edge deleted, graph with it flipped (if still a DAG)
    - refuted edges with no opposite-direction edge: graph with the edge restored (if a DAG)
    - graph with an unobserved common cause added between x and y"""
    g = hcg.to_networkx()
    rel = nx.ancestors(g, x) | nx.ancestors(g, y) | {x, y}
    edges = sorted((e for e in hcg.active_edges() if {e.src, e.dst} & rel), key=lambda e: (e.confidence, e.key))
    gone = sorted((e for e in hcg.edges if e.status == EdgeStatus.REFUTED and {e.src, e.dst} & rel and not g.has_edge(e.dst, e.src)), key=lambda e: e.key)
    for e in edges[:max_edges]:
        h = g.copy()
        h.remove_edge(e.src, e.dst)
        yield f"삭제 {e.src}→{e.dst} ({e.status.value})", h
        r = h.copy()
        r.add_edge(e.dst, e.src)
        yield from [(f"반전 {e.dst}→{e.src} ({e.status.value})", r)] * nx.is_directed_acyclic_graph(r)
    for e in gone[:max_edges]:
        h = g.copy()
        h.add_edge(e.src, e.dst)
        yield from [(f"복원 {e.src}→{e.dst} (refuted)", h)] * nx.is_directed_acyclic_graph(h)
    h = g.copy()
    h.add_edges_from([("_U", x), ("_U", y)])
    yield f"숨은 교란 _U→{x},{y}", h


def sensitivity(q: Query, hcg: HCG, df: pd.DataFrame, base: float, base_ci: tuple) -> tuple[pd.DataFrame, dict]:
    rows = []
    for name, h in alternatives(hcg, q.treatment, q.outcome):
        e = identify(h, q.treatment, q.outcome, latent_of(h, df))
        ok = e.strategy in IDENTIFIED + ("no_effect",) and (q.kind in ("ate", "association") or e.strategy != "frontdoor" and "_U" not in h)
        rows.append({"scenario": name, "strategy": e.strategy if ok else "not_identified", "variables": ",".join(sorted(e.variables)),
                     "value": headline(q, h, df) if ok else np.nan})
    tab = pd.DataFrame(rows, columns=["scenario", "strategy", "variables", "value"])
    v = tab.value.dropna()
    summary = {"scenarios": len(tab), "identified": int(len(v)), "min": float(v.min()) if len(v) else np.nan, "max": float(v.max()) if len(v) else np.nan,
               "flipped": int((np.sign(v) == -np.sign(base)).sum()) if base else int((v != 0).sum()), "vanished": int((v == 0).sum()) if base else 0,
               "within_ci": int(v.between(*base_ci).sum()),
               "not_identified": tab.scenario[tab.value.isna()].tolist()}
    return tab, summary


# ---------- assumptions ----------
def assumptions(q: Query, e: Estimand, hcg: HCG, df: pd.DataFrame, g: nx.DiGraph) -> list[str]:
    x, y = q.treatment, q.outcome
    out = [f"그래프 '{hcg.graph_id}'가 참 인과 구조를 담고 있다(활성 엣지 {len(hcg.active_edges())}개)."]
    adjust = [f"조정 집합 {{{', '.join(sorted(e.variables))}}}이 {x}–{y} 사이 교란 경로를 모두 막는다(그 밖의 숨은 교란이 없다).",
              "양수성: 조정 변수의 모든 층에 처치·비처치 단위가 모두 있다."] if e.variables else [f"{x}–{y} 사이에 교란 경로가 없어 조정이 필요 없다(숨은 교란이 없다고 본다)."]
    out += {"backdoor": adjust,
            "frontdoor": [f"매개 집합 {{{', '.join(sorted(e.variables))}}}이 {x}의 효과를 모두 전달하고, 매개 변수와 {y} 사이에 숨은 교란이 없다."],
            "iv": [f"도구 변수 {', '.join(sorted(e.variables))}는 {x}를 통해서만 {y}에 영향을 주고 {y}와 공통 원인이 없다. 효과는 선형(동질)이라고 본다."],
            "no_effect": [f"그래프에 {x}에서 {y}로 가는 인과 경로가 없다. 효과가 0이라는 결론은 이 그래프 구조에 전적으로 기댄다."],
            "not_identified": []}[e.strategy]
    out += ["반사실 계산은 데이터로 적합한 이진 SCM(문턱값 메커니즘, 노드별 독립 잡음)에 기댄다."] * (q.kind in ("mediation", "ett"))
    out += [f"자연 직접·간접 효과 분해는 매개 변수 {', '.join(mediators_of(q, g, df)) or '(없음)'}와 {y} 사이에 처치의 영향을 받는 교란이 없다고 가정한다."] * (q.kind == "mediation")
    lat = [v for v in latent_of(g, df) if v in nx.ancestors(g, x) | nx.ancestors(g, y)]
    out += [f"데이터에 없는 변수 {', '.join(lat)}는 관측되지 않은 것으로 다뤘다."] * bool(lat)
    on_path = {k for p in nx.all_simple_paths(g, x, y, cutoff=6) for k in zip(p, p[1:])} | {(u, v) for v in e.variables for u in g.predecessors(v)}
    weak = sorted(f"{a}→{b}" for a, b in on_path if g.has_edge(a, b) and hcg.edge(a, b).status != EdgeStatus.VERIFIED)
    out += [f"검증(verified)되지 않은 엣지에 기댄다: {', '.join(weak)}."] * bool(weak)
    out += ["SUTVA: 한 단위의 처치가 다른 단위의 결과에 영향을 주지 않는다."]
    return out


@dataclass
class Analysis:
    query: Query
    estimand: Estimand
    estimate: dict  # {name: {"value", "ci"}} — ate (estimand), association, nde, nie, ett ...
    refutation: pd.DataFrame
    sensitivity: pd.DataFrame
    sensitivity_summary: dict
    paths: list
    backdoor: list
    assumptions: list
    names: dict = field(default_factory=dict)  # id -> display name
    experiment: dict = field(default_factory=dict)

    @property
    def main(self) -> str:
        return MAIN[self.query.kind]


def analyze(q: Query, hcg: HCG, df: pd.DataFrame, n_boot: int = 200, n_boot_scm: int = 30, seed: int = 0) -> Analysis:
    g = hcg.to_networkx()
    x, y = q.treatment, q.outcome
    lat = latent_of(g, df)
    e = identify(g, x, y, lat)
    est = estimate(g, df, x, y, lat, n_boot=n_boot, seed=seed, estimand=e)
    values = {"ate": {"value": est.value, "ci": tuple(map(float, est.ci))}}
    values |= {"association": bootstrap(lambda d: {"association": association(d, x, y)}, df, n_boot, seed)["association"]} if q.kind == "association" else {}
    values |= bootstrap(lambda d: _scm_values(q, g, d), df, n_boot_scm, seed) if q.kind in ("mediation", "ett") else {}
    main = MAIN[q.kind]
    ref = refute(g, df, x, y, lat, seed=seed) if e.strategy in IDENTIFIED else pd.DataFrame(columns=["test", "expected", "new_value", "passed"])
    sens, summ = sensitivity(q, hcg, df, values[main]["value"], values[main]["ci"])
    return Analysis(q, e, values, ref, sens, summ, evidence_paths(hcg, x, y), backdoor_paths(g, x, y) if e.strategy != "no_effect" else [],
                    assumptions(q, e, hcg, df, g), display_names(hcg, q.text))
