"""질의 하나의 분석: 식별 → 추정 → (반사실) → 반박 → 근거 경로 → 그래프 민감도(L5) → 가정 목록."""
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
MAIN = {"ate": "ate", "association": "ate", "mediation": "nie", "ett": "ett"}  # 유형별 대표 수치


def latent_of(g: nx.DiGraph, df: pd.DataFrame) -> list[str]:
    """그래프에는 있지만 데이터 열이 없는 노드(텍스트에서만 나온 변수 등)는 관측되지 않은 변수로 다룬다."""
    return sorted(set(g) - set(df.columns))


def mediators_of(q: Query, g: nx.DiGraph, df: pd.DataFrame) -> list[str]:
    return q.mediators or sorted((causal_nodes(g, q.treatment, q.outcome) - {q.outcome}) & set(df.columns))


def fit_scm(g: nx.DiGraph, df: pd.DataFrame, nodes) -> BinarySCM:
    """관심 노드들의 관측된 조상만으로 이진 SCM을 적합한다."""
    keep = set().union(*(nx.ancestors(g, v) | {v} for v in nodes)) & set(df.columns)
    return BinarySCM.fit(nx.DiGraph(g.subgraph(sorted(keep))), df)


# 유형별 핵심 수치. ate·association은 식별된 추정식, mediation·ett는 적합한 SCM에서 계산한다.
def _scm_values(q: Query, g: nx.DiGraph, df: pd.DataFrame) -> dict:
    x, y, m = q.treatment, q.outcome, mediators_of(q, g, df)
    scm = fit_scm(g, df, [x, y, *m])
    out = {"ate_scm": scm.ate(x, y)}
    out |= {"nde": scm.nde(x, y, m), "nie": scm.nie(x, y, m)} if q.kind == "mediation" else {}
    out |= {"ett": scm.ett(x, y)} if q.kind == "ett" else {}
    return out


def headline(q: Query, g: nx.DiGraph, df: pd.DataFrame) -> float:
    """민감도 분석에 쓰는 대표 수치."""
    lat = latent_of(g, df)
    return {"ate": lambda: estimate(g, df, q.treatment, q.outcome, lat, n_boot=0).value,
            "association": lambda: estimate(g, df, q.treatment, q.outcome, lat, n_boot=0).value,
            "mediation": lambda: _scm_values(q, g, df)["nie"],
            "ett": lambda: _scm_values(q, g, df)["ett"]}[q.kind]()


def association(df: pd.DataFrame, x: str, y: str) -> float:
    return float(df.loc[df[x] == 1, y].mean() - df.loc[df[x] == 0, y].mean())


def bootstrap(fn, df: pd.DataFrame, n_boot: int, seed: int = 0) -> dict:
    """fn(df) → {이름: 값}. 표본을 다시 뽑아 이름별 95% 구간을 낸다."""
    rng = np.random.default_rng(seed)
    point = fn(df)
    boots = pd.DataFrame([fn(df.iloc[rng.integers(0, len(df), len(df))]) for _ in range(n_boot)])
    return {k: {"value": float(v), "ci": tuple(float(c) for c in np.nanquantile(boots[k], [0.025, 0.975])) if n_boot else (np.nan, np.nan)}
            for k, v in point.items()}


def display_names(hcg: HCG, text: str = "") -> dict:
    """id → 설명에 쓸 이름: 질문에 나온 표현 > 가장 짧은 한국어 별칭 > id."""
    hangul = lambda w: bool(re.search("[가-힣]", w))
    out = {}
    for n in hcg.nodes:
        names = sorted({n.label, *n.aliases} - {""}, key=lambda w: (len(w), w))
        said = [w for w in names if w in text]
        out[n.id] = (said and max(said, key=len)) or next((w for w in names if hangul(w)), n.id)
    return out


# ---------- 근거 경로 ----------
def edge_info(e) -> dict:
    quote = next((x.span for x in e.evidence if x.kind == EvidenceKind.TEXT), "")
    return {"edge": f"{e.src}→{e.dst}", "status": e.status.value, "confidence": round(e.confidence, 3), "sign": e.sign,
            "evidence": sorted({x.kind.value for x in e.evidence}), "quote": quote}


def evidence_paths(hcg: HCG, x: str, y: str, k: int = 5) -> list[list[dict]]:
    """x에서 y로 가는 인과 경로(짧은 것부터 k개)와 경로 위 엣지의 상태·신뢰도·증거."""
    g = hcg.to_networkx()
    paths = sorted(nx.all_simple_paths(g, x, y, cutoff=6), key=lambda p: (len(p), p))[:k]
    return [[edge_info(hcg.edge(u, v)) for u, v in zip(p, p[1:])] for p in paths]


def backdoor_paths(g: nx.DiGraph, x: str, y: str, k: int = 3) -> list[list[str]]:
    """x로 들어오는 화살표로 시작하는 x–y 경로(교란 경로 후보)."""
    u = g.to_undirected(as_view=True)
    paths = [p for p in nx.all_simple_paths(u, x, y, cutoff=6) if g.has_edge(p[1], x)]
    return sorted(paths, key=lambda p: (len(p), p))[:k]


# ---------- L5: 그래프 민감도 ----------
def alternatives(hcg: HCG, x: str, y: str, max_edges: int = 8):
    """결론이 각 엣지에 얼마나 기대는지 보는 대안 그래프. verified 엣지도 틀릴 수 있으므로 상태와 관계없이 본다.
    - x·y의 조상에 닿는 활성 엣지(신뢰도 낮은 순 max_edges개): 지운 그래프, 뒤집은 그래프(DAG이면)
    - 반증된 엣지 중 반대 방향 엣지가 없는 것: 되살린 그래프(DAG이면)
    - x와 y 사이에 관측되지 않은 공통 원인을 넣은 그래프"""
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


# ---------- 가정 ----------
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
    estimate: dict  # {이름: {"value", "ci"}} — ate(추정식), association, nde, nie, ett ...
    refutation: pd.DataFrame
    sensitivity: pd.DataFrame
    sensitivity_summary: dict
    paths: list
    backdoor: list
    assumptions: list
    names: dict = field(default_factory=dict)  # id → 설명용 이름
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
