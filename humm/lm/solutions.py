"""Solution-template trajectories: write the correct solution of a Humm-Gym item as sentences that follow the engine's computation (SFT teacher data).

Follows the steps of the Gym CoT instruction (1 variables/graph -> 2 query type -> 3 estimand -> 4 computation) and ends with 'Final answer: ...'.
Every number is computed by the engine from the item's probability table. For structure-reasoning types where 9B CoT is weak (corr2cause, identify, backadj),
correct trajectories are hard to collect by rejection sampling, so this template is the main teacher.
"""
import re
from itertools import product

import networkx as nx
import numpy as np

from ..engine import BinarySCM, adjustment_sets, frontdoor_sets, identify, instruments, optimal_adjustment
from ..engine.discovery import holds, mec_of
from ..engine.dsep import blocked_by, paths_between
from ..engine.identify import causal_nodes, forbidden, proper_backdoor_graph
from ..gym.check import _scm
from ..gym.formats import LETTERS, target_text
from ..gym.graphs import mec
from ..gym.queries import HYP, SCORE_EDGES, _rel
from ..gym.verbalize import Lexicon, josa

KIND_NAME = {
    "marginal": ("주변 확률(Rung 1)", "marginal probability (Rung 1)"),
    "correlation": ("관측 조건부 확률의 차이, 즉 상관(Rung 1)", "difference of observational conditional probabilities, i.e. correlation (Rung 1)"),
    "exp_away": ("공통 결과를 조건으로 둔 상관, 즉 설명 효과(Rung 1)", "correlation given a common effect, i.e. explaining away (Rung 1)"),
    "ate": ("개입 효과 ATE(Rung 2)", "average treatment effect, ATE (Rung 2)"),
    "cde": ("통제 직접효과 CDE(Rung 2)", "controlled direct effect, CDE (Rung 2)"),
    "ate_score": ("개입 효과 ATE의 크기 구간(Rung 2)", "size bucket of the ATE (Rung 2)"),
    "ett": ("처치군 효과 ETT(Rung 3)", "effect of treatment on the treated, ETT (Rung 3)"),
    "nde": ("자연 직접효과 NDE(Rung 3)", "natural direct effect, NDE (Rung 3)"),
    "nie": ("자연 간접효과 NIE(Rung 3)", "natural indirect effect, NIE (Rung 3)"),
    "pn": ("필요 확률 PN(Rung 3)", "probability of necessity, PN (Rung 3)"),
    "ps": ("충분 확률 PS(Rung 3)", "probability of sufficiency, PS (Rung 3)"),
    "dsep": ("조건부 독립, 즉 d-분리 판정(Rung 1)", "conditional independence, i.e. d-separation (Rung 1)"),
    "backadj": ("조정 집합의 유효성(백도어·일반화 조정 기준, Rung 2)", "validity of an adjustment set (back-door / generalized adjustment criterion, Rung 2)"),
    "identify": ("인과 효과의 식별 전략(Rung 2)", "identification strategy for a causal effect (Rung 2)"),
    "corr2cause": ("통계적 관계로부터의 인과 추론(마르코프 동치류, Rung 1)", "causal inference from statistical relations (Markov equivalence class, Rung 1)"),
    "det_cf": ("결정론적 반사실(Rung 3)", "deterministic counterfactual (Rung 3)"),
}


def _t(lang, ko, en):
    return ko if lang == "ko" else en


def jo(word: str, kind: str) -> str:
    """Particle only: jo('계절', '이') -> '이'."""
    return josa(word, kind)[len(word):]


def _p(x: float) -> str:
    return f"{x:.4f}"


class Writer:
    """Helper for writing the solution of one item: names, probabilities, path notation."""

    def __init__(self, d: dict):
        self.d, self.lang, self.meta = d, d["lang"], d["meta"]
        self.g = nx.DiGraph([tuple(e) for e in d["graph"]])
        self.g.add_nodes_from(self.meta.get("cpt", {}))
        self.g.add_nodes_from(d.get("latent", []))
        self.lex = Lexicon({k: tuple(v) for k, v in self.meta["vocab"].items()}, self.lang) if "vocab" in self.meta else None
        self.scm: BinarySCM | None = _scm(d) if "cpt" in self.meta else None

    def t(self, ko, en):
        return _t(self.lang, ko, en)

    def n(self, v) -> str:
        return self.lex.name(v)

    def ns(self, vs) -> str:
        return self.lex.names(sorted(vs)) if vs else self.t("없음", "none")

    def set_(self, vs) -> str:
        """Set notation: {A, B}, or the empty set."""
        return f"{{{self.lex.names(sorted(vs))}}}" if vs else "∅"

    def eq(self, assign: dict) -> str:
        """Format like 'ad=on, season=off'."""
        return ", ".join(f"{self.n(v)}={self.lex.state(v, x)}" for v, x in assign.items())

    def edges(self, g=None) -> str:
        g = self.g if g is None else g
        es = sorted(g.edges, key=lambda e: (self.n(e[0]), self.n(e[1])))
        return ", ".join(f"{self.n(u)} → {self.n(v)}" for u, v in es) or self.t("(엣지 없음)", "(no edges)")

    def path(self, p) -> str:
        """A path with arrow directions: A → B ← C."""
        out = self.n(p[0])
        for u, v in zip(p, p[1:]):
            out += f" → {self.n(v)}" if self.g.has_edge(u, v) else f" ← {self.n(v)}"
        return out

    def joint(self, assign: dict) -> float:
        w, vals = self.scm.noise()[1], self.scm.world()
        mask = np.prod([vals[k] == x for k, x in assign.items()], axis=0) if assign else np.ones_like(w)
        return float((w * mask).sum() / w.sum())

    def cond(self, y: str, given: dict, do: dict | None = None) -> float:
        return self.scm.prob(y, do=do, given=given or None)

    def head(self, extra: str = "") -> list[str]:
        g, latent = self.g, self.d.get("latent", [])
        lines = [self.t(f"1) 변수와 그래프: 변수는 {self.ns(g.nodes)}이다. 직접 인과 엣지는 {self.edges()}이다.",
                        f"1) Variables and graph: the variables are {self.ns(g.nodes)}. The direct causal edges are {self.edges()}.")]
        lines += [self.t(f"   관측되지 않는 변수: {self.ns(latent)}.", f"   Unobserved: {self.ns(latent)}.")] if latent else []
        lines += self.target()
        k = KIND_NAME[self.d["query_type"]]
        lines.append(self.t(f"2) 질의 유형: {k[0]}이다.{extra}", f"2) Query type: {k[1]}.{extra}"))
        return lines

    def target(self) -> list[str]:
        """Variables the question refers to: written right after the graph to reduce errors of misreading treatment and outcome (identify)."""
        m, qt = self.meta, self.d["query_type"]
        if qt in ("identify", "backadj"):
            return [self.t(f"   질문의 처치는 {self.n(m['x'])}, 결과는 {self.n(m['y'])}이다.", f"   The question is about the treatment {self.n(m['x'])} and the outcome {self.n(m['y'])}.")]
        if qt == "dsep":
            z = self.t(f", 조건은 {self.set_(m['z'])}", f" given {self.set_(m['z'])}")
            return [self.t(f"   질문의 두 변수는 {self.n(m['a'])}{jo(self.n(m['a']), '과')} {self.n(m['b'])}{z}이다.",
                           f"   The question is about {self.n(m['a'])} and {self.n(m['b'])}{z}.")]
        return []

    def final(self, lines: list[str]) -> str:
        return "\n".join(lines + [f"Final answer: {target_text(self.d)}"])


# ---------- Path analysis (d-separation) ----------
def describe_paths(w: Writer, g: nx.DiGraph, paths, z: set, limit: int = 6) -> tuple[list[str], bool]:
    """Write open/blocked status per path, showing open paths first. (list of sentences, whether an open path exists)."""
    status = [(p, blocked_by(g, p, z)) for p in paths]
    status.sort(key=lambda s: s[1] is not None)
    lines = []
    for p, b in status[:limit]:
        why = (w.t("열려 있다", "is open") if b is None else
               w.t(f"{w.n(b[0])}에서 막힌다({'충돌 노드이고 그 자신과 자손이 조건에 없다' if b[1] == 'collider' else '조건에 포함된 비충돌 노드이다'})",
                   f"is blocked at {w.n(b[0])} ({'a collider that is not conditioned on, nor any of its descendants' if b[1] == 'collider' else 'a conditioned non-collider'})"))
        lines.append(f"   - {w.path(p)}: {why}")
    rest = len(status) - limit
    lines += [w.t(f"   - 나머지 경로 {rest}개도 모두 막힌다.", f"   - The remaining {rest} paths are all blocked.")] if rest > 0 and status[limit][1] is not None else []
    return lines, any(b is None for _, b in status)


# ---------- Numeric queries ----------
def _adjust_terms(w: Writer, x, y, z: list, xval: int) -> tuple[float, list[str]]:
    """P(Y=1|do(X=x)) = Σ_z P(Y=1|X=x,z) P(z). (value, term description)."""
    total, parts = 0.0, []
    for bits in product((0, 1), repeat=len(z)):
        a = dict(zip(z, bits))
        py, pz = w.cond(y, {x: xval, **a}), w.joint(a)
        total += py * pz
        parts.append(f"{_p(py)}×{_p(pz)}")
    return total, parts


def sol_effect(w: Writer, kind: str) -> str:
    d, m = w.d, w.meta
    x, y = m["x"], m["y"]
    z = sorted(w.g.predecessors(x))
    zt = w.ns(z)
    y1 = w.lex.is_(y, 1) if w.lang == "en" else f"{w.n(y)}={w.lex.state(y, 1)}"
    lines = w.head()
    if z:
        lines.append(w.t(f"3) 추정식: {w.n(x)}의 부모 {{{zt}}}{jo(zt, '이')} 모든 백도어 경로를 막으므로 P({y1} | do({w.n(x)}=v)) = Σ_z P({y1} | {w.n(x)}=v, z) P(z)이다.",
                         f"3) Estimand: the parents of {w.n(x)}, {{{zt}}}, block every back-door path, so P({y1} | do({w.n(x)}=v)) = Σ_z P({y1} | {w.n(x)}=v, z) P(z)."))
    else:
        lines.append(w.t(f"3) 추정식: {w.n(x)}에 부모가 없어 교란 경로가 없으므로 P({y1} | do({w.n(x)}=v)) = P({y1} | {w.n(x)}=v)이다.",
                         f"3) Estimand: {w.n(x)} has no parents, so there is no confounding path and P({y1} | do({w.n(x)}=v)) = P({y1} | {w.n(x)}=v)."))
    p1, t1 = _adjust_terms(w, x, y, z, 1)
    p0, t0 = _adjust_terms(w, x, y, z, 0)
    s1, s0 = w.lex.state(x, 1), w.lex.state(x, 0)
    lines.append(w.t(f"4) 계산: do({w.n(x)}={s1}): {' + '.join(t1)} = {_p(p1)}. do({w.n(x)}={s0}): {' + '.join(t0)} = {_p(p0)}. ATE = {_p(p1)} − {_p(p0)} = {p1 - p0:+.4f}.",
                     f"4) Computation: do({w.n(x)}={s1}): {' + '.join(t1)} = {_p(p1)}. do({w.n(x)}={s0}): {' + '.join(t0)} = {_p(p0)}. ATE = {_p(p1)} − {_p(p0)} = {p1 - p0:+.4f}."))
    assert abs((p1 - p0) - w.scm.ate(x, y)) < 1e-6
    if kind == "ate_score":
        lvl = int(np.digitize(p1 - p0, SCORE_EDGES)) + 1
        lines.append(w.t(f"   {p1 - p0:+.4f}은 {lvl}단계 '{d['options'][lvl - 1]}' 구간에 든다.", f"   {p1 - p0:+.4f} falls in level {lvl}, '{d['options'][lvl - 1]}'."))
    else:
        lines.append(_conclude(w, p1 - p0, 0.0))
    return w.final(lines)


def _conclude(w: Writer, val: float, threshold: float) -> str:
    """Decide yes/no from the value and the question direction (increase / decrease)."""
    up = w.meta["direction"] == "up"
    holds = (val > threshold) == up
    rel = w.t("크다" if val > threshold else "작다", "greater" if val > threshold else "smaller")
    return w.t(f"   값 {val:+.4f}이 기준 {threshold:g}보다 {rel}. 질문은 '{'높아지는가' if up else '낮아지는가'}'를 물으므로 답은 '{'예' if holds else '아니오'}'이다.",
               f"   The value {val:+.4f} is {rel} than {threshold:g}. The question asks whether it {'increases' if up else 'decreases'}, so the answer is {'yes' if holds else 'no'}.")


def sol_marginal(w: Writer) -> str:
    y = w.meta["y"]
    pa = sorted(w.g.predecessors(y))
    lines = w.head()
    lines.append(w.t(f"3) 추정식: P({w.n(y)}={w.lex.state(y, 1)}) = Σ_pa P({w.n(y)}={w.lex.state(y, 1)} | pa) P(pa), pa = {{{w.ns(pa)}}}.",
                     f"3) Estimand: P({w.n(y)}={w.lex.state(y, 1)}) = Σ_pa P({w.n(y)}={w.lex.state(y, 1)} | pa) P(pa), pa = {{{w.ns(pa)}}}."))
    total, parts = 0.0, []
    for bits in product((0, 1), repeat=len(pa)):
        a = dict(zip(pa, bits))
        py, pp = w.cond(y, a), w.joint(a)
        total += py * pp
        parts.append(f"{_p(py)}×{_p(pp)}" if pa else _p(py))
    lines.append(w.t(f"4) 계산: {' + '.join(parts)} = {_p(total)}. 따라서 P({w.n(y)}={w.lex.state(y, 0)}) = {_p(1 - total)}.",
                     f"4) Computation: {' + '.join(parts)} = {_p(total)}, so P({w.n(y)}={w.lex.state(y, 0)}) = {_p(1 - total)}."))
    assert abs(total - w.scm.prob(y)) < 1e-6
    lines.append(_conclude(w, total, 0.5))
    return w.final(lines)


def sol_correlation(w: Writer) -> str:
    x, y = w.meta["x"], w.meta["y"]
    p1, p0 = w.cond(y, {x: 1}), w.cond(y, {x: 0})
    lines = w.head(w.t(" 개입이 아니라 관측이므로 교란 경로를 통한 연관도 포함된다.", " This is observation, not intervention, so association through confounding paths is included."))
    lines.append(w.t(f"3) 추정식: P({w.n(y)}={w.lex.state(y, 1)} | {w.n(x)}={w.lex.state(x, 1)}) − P({w.n(y)}={w.lex.state(y, 1)} | {w.n(x)}={w.lex.state(x, 0)}). "
                     "각 조건부 확률은 결합분포 P(x, y) / P(x)로 구한다.",
                     f"3) Estimand: P({w.n(y)}={w.lex.state(y, 1)} | {w.n(x)}={w.lex.state(x, 1)}) − P({w.n(y)}={w.lex.state(y, 1)} | {w.n(x)}={w.lex.state(x, 0)}), "
                     "each computed as P(x, y) / P(x) from the joint distribution."))
    j11, j01, px1 = w.joint({x: 1, y: 1}), w.joint({x: 0, y: 1}), w.joint({x: 1})
    lines.append(w.t(f"4) 계산: {_p(j11)} / {_p(px1)} = {_p(p1)}, {_p(j01)} / {_p(1 - px1)} = {_p(p0)}. 차이 = {p1 - p0:+.4f}.",
                     f"4) Computation: {_p(j11)} / {_p(px1)} = {_p(p1)}, {_p(j01)} / {_p(1 - px1)} = {_p(p0)}. Difference = {p1 - p0:+.4f}."))
    lines.append(_conclude(w, p1 - p0, 0.0))
    return w.final(lines)


def sol_cde(w: Writer) -> str:
    x, y, fix = w.meta["x"], w.meta["y"], w.meta["fix"]
    p1, p0 = w.cond(y, {}, {x: 1, **fix}), w.cond(y, {}, {x: 0, **fix})
    lines = w.head()
    lines.append(w.t(f"3) 추정식: 매개 변수를 {w.eq(fix)}로 고정한 두 개입의 차이 P(y | do({w.n(x)}=1, M=m)) − P(y | do({w.n(x)}=0, M=m)). "
                     "개입한 변수의 확률표만 지우는 절단 인수분해(truncated factorization)로 계산한다.",
                     f"3) Estimand: the difference of two interventions with the mediators fixed at {w.eq(fix)}, P(y | do({w.n(x)}=1, M=m)) − P(y | do({w.n(x)}=0, M=m)), "
                     "computed by the truncated factorization (drop the factors of the intervened variables)."))
    lines.append(w.t(f"4) 계산: {_p(p1)} − {_p(p0)} = {p1 - p0:+.4f}.", f"4) Computation: {_p(p1)} − {_p(p0)} = {p1 - p0:+.4f}."))
    lines.append(_conclude(w, p1 - p0, 0.0))
    return w.final(lines)


def sol_ett(w: Writer) -> str:
    x, y = w.meta["x"], w.meta["y"]
    z = sorted(w.g.predecessors(x))
    lines = w.head()
    lines.append(w.t(f"3) 추정식: ETT = E[Y | {w.n(x)}=1] − E[Y_{{x=0}} | {w.n(x)}=1]. 부모 {{{w.ns(z)}}}{jo(w.ns(z), '이')} 백도어 경로를 막으므로 "
                     f"E[Y_{{x=0}} | {w.n(x)}=1] = Σ_z P(y | {w.n(x)}=0, z) P(z | {w.n(x)}=1)이다.",
                     f"3) Estimand: ETT = E[Y | {w.n(x)}=1] − E[Y_{{x=0}} | {w.n(x)}=1]. The parents {{{w.ns(z)}}} block the back-door paths, so "
                     f"E[Y_{{x=0}} | {w.n(x)}=1] = Σ_z P(y | {w.n(x)}=0, z) P(z | {w.n(x)}=1)."))
    obs = w.cond(y, {x: 1})
    total, parts = 0.0, []
    for bits in product((0, 1), repeat=len(z)):
        a = dict(zip(z, bits))
        py, pz = w.cond(y, {x: 0, **a}), (w.joint({x: 1, **a}) / w.joint({x: 1}))
        total += py * pz
        parts.append(f"{_p(py)}×{_p(pz)}")
    val = obs - total
    lines.append(w.t(f"4) 계산: E[Y | x=1] = {_p(obs)}, 반사실 항 = {' + '.join(parts)} = {_p(total)}. ETT = {val:+.4f}.",
                     f"4) Computation: E[Y | x=1] = {_p(obs)}, counterfactual term = {' + '.join(parts)} = {_p(total)}. ETT = {val:+.4f}."))
    assert abs(val - w.scm.ett(x, y)) < 1e-6
    up = w.meta["direction"] == "up"
    holds = (val > 0) == up
    lines.append(w.t(f"   ETT가 {'양수' if val > 0 else '음수'}이므로 {w.n(x)}{jo(w.n(x), '이')} 0이었다면 확률은 {'더 낮았다' if val > 0 else '더 높았다'}. 답은 '{'예' if holds else '아니오'}'이다.",
                     f"   ETT is {'positive' if val > 0 else 'negative'}, so had {w.n(x)} been 0 the probability would have been {'lower' if val > 0 else 'higher'}. "
                     f"The answer is {'yes' if holds else 'no'}."))
    return w.final(lines)


def sol_mediation(w: Writer, kind: str) -> str:
    x, y, med = w.meta["x"], w.meta["y"], w.meta["med"]
    s = w.scm
    m0, m1 = s.world({x: 0}), s.world({x: 1})
    y0 = s._mean(m0[y])
    cross = s._mean(s.world({x: 1, **{m: m0[m] for m in med}})[y]) if kind == "nde" else s._mean(s.world({x: 0, **{m: m1[m] for m in med}})[y])
    val = cross - y0
    lines = w.head()
    expr = "E[Y(1, M(0))] − E[Y(0)]" if kind == "nde" else "E[Y(0, M(1))] − E[Y(0)]"
    lines.append(w.t(f"3) 추정식: 매개 변수 M = {{{w.ns(med)}}}. {kind.upper()} = {expr}. 외생 잡음을 공유하는 두 세계(x=0, x=1)를 만들고, "
                     f"{'결과는 x=1 세계에서, 매개 변수는 x=0 세계의 값으로' if kind == 'nde' else '결과는 x=0 세계에서, 매개 변수만 x=1 세계의 값으로'} 계산한다(매개 공식과 같다).",
                     f"3) Estimand: mediators M = {{{w.ns(med)}}}. {kind.upper()} = {expr}. Build two worlds (x=0, x=1) sharing the exogenous noise and evaluate "
                     f"{'the outcome with x=1 but the mediators at their x=0 values' if kind == 'nde' else 'the outcome with x=0 but the mediators at their x=1 values'} (equivalent to the mediation formula)."))
    lines.append(w.t(f"4) 계산: 교차 세계 항 = {_p(cross)}, E[Y(0)] = {_p(y0)}. {kind.upper()} = {val:+.4f}.",
                     f"4) Computation: cross-world term = {_p(cross)}, E[Y(0)] = {_p(y0)}. {kind.upper()} = {val:+.4f}."))
    assert abs(val - (s.nde(x, y, med) if kind == "nde" else s.nie(x, y, med))) < 1e-6
    lines.append(_conclude(w, val, 0.0))
    return w.final(lines)


# ---------- Structure queries ----------
def sol_dsep(w: Writer) -> str:
    a, b, z = w.meta["a"], w.meta["b"], set(w.meta["z"])
    lines = w.head()
    lines.append(w.t(f"3) 판정 기준: {w.n(a)}{jo(w.n(a), '과')} {w.n(b)} 사이의 모든 경로가 조건 {w.set_(z)}에 의해 막히면 d-분리, 즉 독립이다.",
                     f"3) Criterion: {w.n(a)} and {w.n(b)} are independent (d-separated) iff every path between them is blocked given {w.set_(z)}."))
    paths = paths_between(w.g, a, b)
    body, open_ = describe_paths(w, w.g, paths, z)
    lines.append(w.t(f"4) 경로 점검 ({len(paths)}개):", f"4) Checking the {len(paths)} paths:"))
    lines += body or [w.t("   - 두 변수를 잇는 경로가 없다.", "   - No path connects the two variables.")]
    lines.append(w.t(f"   {'열린 경로가 있으므로 독립이 아니다' if open_ else '모든 경로가 막히므로 독립이다'}.",
                     f"   {'There is an open path, so they are not independent' if open_ else 'All paths are blocked, so they are independent'}."))
    if w.meta.get("asks") == "dependent":
        lines.append(w.t(f"   질문은 의존 여부를 묻으므로 답은 '{'예' if open_ else '아니오'}'이다.", f"   The question asks whether they are dependent, so the answer is {'yes' if open_ else 'no'}."))
    assert (not open_) == (bool(w.d["label"]) != (w.meta.get("asks") == "dependent"))
    return w.final(lines)


def sol_backadj(w: Writer) -> str:
    if "methods" in w.meta:
        return sol_methods(w)
    x, y, z = w.meta["x"], w.meta["y"], set(w.meta["z"])
    lines = w.head()
    lines.append(w.t(f"3) 판정 기준(일반화 조정 기준): (i) Z = {w.set_(z)}에 {w.n(x)} → {w.n(y)} 인과 경로 위의 노드나 그 자손이 없고, "
                     f"(ii) {w.n(x)}에서 나가는 인과 경로의 첫 엣지를 지운 그래프에서 Z가 {w.n(x)}{jo(w.n(x), '과')} {w.n(y)} 사이의 모든 경로를 막아야 한다.",
                     f"3) Criterion (generalized adjustment criterion): (i) Z = {w.set_(z)} contains no node on a causal path {w.n(x)} → {w.n(y)} nor a descendant of one, and "
                     f"(ii) in the graph without the first edges of the causal paths out of {w.n(x)}, Z blocks every path between {w.n(x)} and {w.n(y)}."))
    bad = sorted(z & forbidden(w.g, x, y))
    lines.append(w.t(f"4) (i) 금지 노드(인과 경로 위 노드와 그 자손, {w.n(x)} 포함): {{{w.ns(forbidden(w.g, x, y))}}}. "
                     f"Z와 겹치는 노드: {w.ns(bad) if bad else '없음'}.",
                     f"4) (i) Forbidden nodes (on causal paths or their descendants, plus {w.n(x)}): {{{w.ns(forbidden(w.g, x, y))}}}. "
                     f"Overlap with Z: {w.ns(bad) if bad else 'none'}."))
    pb = proper_backdoor_graph(w.g, x, y)
    body, open_ = describe_paths(w, pb, paths_between(pb, x, y), z)
    removed = sorted((x, v) for v in causal_nodes(w.g, x, y) if w.g.has_edge(x, v))
    cut = ', '.join(f'{w.n(u)} → {w.n(v)}' for u, v in removed)
    lines.append(w.t(f"   (ii) {cut}{jo(w.n(removed[-1][1]), '을')} 지운 그래프의 경로:",
                     f"   (ii) Paths after removing {', '.join(f'{w.n(u)} → {w.n(v)}' for u, v in removed)}:"))
    lines += body or [w.t("   - 남은 경로가 없다.", "   - No path remains.")]
    ok = not bad and not open_
    lines.append(w.t(f"   {'두 조건을 모두 만족하므로 충분하다' if ok else ('(i)을 어기므로' if bad else '열린 경로가 남으므로') + ' 충분하지 않다'}.",
                     f"   {'Both conditions hold, so it is sufficient' if ok else ('Condition (i) fails' if bad else 'An open path remains') + ', so it is not sufficient'}."))
    assert ok == bool(w.d["label"])
    return w.final(lines)


def adjustment_check(w: Writer, z: set) -> tuple[list[str], bool]:
    """Validity-check sentences and verdict for adjustment set z."""
    x, y = w.meta["x"], w.meta["y"]
    bad = sorted(z & forbidden(w.g, x, y))
    lines = [w.t(f"   (i) 금지 노드와 겹치는 노드: {w.ns(bad) if bad else '없음'}.", f"   (i) Overlap with the forbidden nodes: {w.ns(bad) if bad else 'none'}.")]
    pb = proper_backdoor_graph(w.g, x, y)
    body, open_ = describe_paths(w, pb, paths_between(pb, x, y), z, limit=4)
    lines.append(w.t("   (ii) 인과 경로의 첫 엣지를 지운 그래프의 경로:", "   (ii) Paths after removing the first edges of the causal paths:"))
    lines += body or [w.t("   - 남은 경로가 없다.", "   - No path remains.")]
    return lines, not bad and not open_


def sol_methods(w: Writer) -> str:
    """CLadder-style comparison of two methods: yes when method 1's adjustment set is valid and method 2's is not."""
    x, y, (m1, m2) = w.meta["x"], w.meta["y"], w.meta["methods"]
    lines = w.head(w.t(" 방법마다 조정 집합(나누어 보는 변수)이 유효한지 본다.", " Check whether each method's adjustment set (the variables it stratifies by) is valid."))
    lines.append(w.t(f"3) 판정 기준(일반화 조정 기준): 금지 노드(인과 경로 위 노드와 그 자손, {w.n(x)} 포함) = {{{w.ns(forbidden(w.g, x, y))}}}. "
                     "조정 집합은 금지 노드를 포함하지 않고, 인과 경로의 첫 엣지를 지운 그래프에서 모든 경로를 막아야 한다.",
                     f"3) Criterion (generalized adjustment): forbidden nodes (on causal paths or their descendants, plus {w.n(x)}) = {{{w.ns(forbidden(w.g, x, y))}}}. "
                     "A valid set contains no forbidden node and blocks every path once the first edges of the causal paths are removed."))
    ok = []
    for i, m in enumerate((m1, m2), 1):
        lines.append(w.t(f"4-{i}) 방법 {i}: 조정 집합 {w.set_(m)}.", f"4-{i}) Method {i}: adjustment set {w.set_(m)}."))
        body, valid = adjustment_check(w, set(m))
        lines += body + [w.t(f"   방법 {i}의 조정 집합은 {'유효하다' if valid else '유효하지 않다'}.", f"   Method {i} is {'valid' if valid else 'not valid'}.")]
        ok.append(valid)
    lines.append(w.t(f"   방법 1이 {'유효하고 방법 2는 유효하지 않으므로 방법 1이 더 올바르다' if ok[0] else '유효하지 않으므로 방법 1이 더 올바르다고 할 수 없다'}.",
                     f"   {'Method 1 is valid and Method 2 is not, so Method 1 is more correct' if ok[0] else 'Method 1 is not valid, so it is not more correct'}."))
    assert ok[0] == bool(w.d["label"]) and ok[0] != ok[1]
    return w.final(lines)


def sol_det_cf(w: Writer) -> str:
    from ..gym.v3 import det_values
    m = w.meta
    rules = {v: (op, [tuple(l) for l in lits]) for v, (op, lits) in m["rules"].items()}
    order = list(nx.topological_sort(w.g))
    x, y, xf = m["x"], m["y"], m["fact"][m["x"]]
    cf = det_values(order, rules, m["exo"], {x: 1 - xf})
    rule = lambda v: (" ∧ " if rules[v][0] == "and" else " ∨ ").join(f"{w.n(p)}={w.lex.state(p, a)}" for p, a in rules[v][1])
    lines = w.head(w.t(" 확률이 없고 규칙이 결정론적이므로, 외생 변수를 관측값에 고정하고 개입한 뒤 규칙을 차례로 적용하면 된다.",
                       " There are no probabilities and the rules are deterministic, so fix the exogenous variables at their observed values, intervene, and apply the rules in order."))
    lines.append(w.t("3) 규칙(조건이 참이면 첫 번째 상태):", "3) Rules (the first state when the condition holds):"))
    lines += [f"   - {w.n(v)} = {w.lex.state(v, 1)} ⟺ {rule(v)}" for v in order if v in rules]
    exo = {v: a for v, a in m["exo"].items() if v != x}
    lines.append(w.t(f"4) 외생 변수 고정: {w.eq(exo) or '없음'}. 개입: do({w.n(x)}={w.lex.state(x, 1 - xf)}) (실제로는 {w.lex.state(x, xf)}).",
                     f"4) Fix the exogenous variables: {w.eq(exo) or 'none'}. Intervene: do({w.n(x)}={w.lex.state(x, 1 - xf)}) (actually {w.lex.state(x, xf)})."))
    for v in order:
        if v in rules and v != x and v in nx.ancestors(w.g, y) | {y}:
            lines.append(w.t(f"   - {w.n(v)}: {rule(v)}{'이 참' if cf[v] else '이 거짓'} → {w.n(v)}={w.lex.state(v, cf[v])}",
                             f"   - {w.n(v)}: {rule(v)} is {'true' if cf[v] else 'false'} → {w.n(v)}={w.lex.state(v, cf[v])}"))
    holds = cf[y] == m["ask"]
    lines.append(w.t(f"   반사실 세계에서 {w.n(y)}={w.lex.state(y, cf[y])}이다. 질문은 {w.n(y)}={w.lex.state(y, m['ask'])}인지 묻으므로 답은 '{'예' if holds else '아니오'}'이다.",
                     f"   In the counterfactual world {w.n(y)}={w.lex.state(y, cf[y])}. The question asks whether {w.n(y)}={w.lex.state(y, m['ask'])}, so the answer is {'yes' if holds else 'no'}."))
    assert holds == bool(w.d["label"])
    return w.final(lines)


def sol_identify(w: Writer) -> str:
    x, y, latent = w.meta["x"], w.meta["y"], set(w.d.get("latent", []))
    g, lines = w.g, w.head()
    lines.append(w.t("3) 순서: 인과 경로 존재 → 백도어 조정 → 프론트도어 조정 → 도구변수 순으로 점검하고, 처음 성립하는 방법을 고른다.",
                     "3) Order: check that a causal path exists, then back-door adjustment, front-door adjustment and an instrumental variable; pick the first that applies."))
    e = identify(g, x, y, latent)
    step = []
    if y not in nx.descendants(g, x):
        step.append(w.t(f"4) {w.n(x)}에서 {w.n(y)}로 가는 방향 경로가 없다. 따라서 인과 효과는 0이다.", f"4) There is no directed path from {w.n(x)} to {w.n(y)}, so the causal effect is zero."))
    else:
        step.append(w.t(f"4) 인과 경로가 있다({w.path(nx.shortest_path(g, x, y))}).", f"4) A causal path exists ({w.path(nx.shortest_path(g, x, y))})."))
        o = optimal_adjustment(g, x, y)
        if e.strategy == "backdoor":
            step.append(w.t(f"   백도어: 관측 변수 집합 {{{w.ns(e.variables)}}}로 조정하면 기준을 만족한다.", f"   Back-door: adjusting for the observed set {{{w.ns(e.variables)}}} satisfies the criterion."))
        else:
            hidden = sorted(o & latent)
            step.append(w.t(f"   백도어: 최적 조정 집합 {{{w.ns(o)}}}에 관측되지 않는 {w.ns(hidden)}{jo(w.ns(hidden), '이')} 들어 있고, 관측 변수로 만든 어떤 집합도 기준을 만족하지 않는다." if hidden else
                            "   백도어: 관측 변수로 만든 어떤 집합도 기준을 만족하지 않는다.",
                            f"   Back-door: the optimal adjustment set {{{w.ns(o)}}} contains the unobserved {w.ns(hidden)}, and no set of observed variables satisfies the criterion." if hidden else
                            "   Back-door: no set of observed variables satisfies the criterion."))
            fd = frontdoor_sets(g, x, y, latent)
            if e.strategy == "frontdoor":
                step.append(w.t(f"   프론트도어: {{{w.ns(fd[0])}}}{jo(w.ns(fd[0]), '이')} {w.n(x)}의 효과를 모두 전달하고, {w.n(x)}–매개 사이에 열린 백도어 경로가 없으며, 매개–{w.n(y)} 사이 백도어 경로는 {w.n(x)}{jo(w.n(x), '로')} 막힌다.",
                                f"   Front-door: {{{w.ns(fd[0])}}} intercepts every causal path, there is no open back-door path from {w.n(x)} to it, and its back-door paths to {w.n(y)} are blocked by {w.n(x)}."))
            else:
                step.append(w.t("   프론트도어: 세 조건을 모두 만족하는 매개 집합이 없다.", "   Front-door: no mediator set satisfies the three conditions."))
                iv = instruments(g, x, y, latent)
                step.append(w.t(f"   도구변수: {w.n(iv[0])}{jo(w.n(iv[0]), '이')} {w.n(x)}{jo(w.n(x), '과')} 연결되어 있고, {w.n(x)}의 출력 엣지를 지우면 {w.n(y)}{jo(w.n(y), '과')} d-분리된다." if iv else
                                "   도구변수: 조건을 만족하는 변수가 없다. 따라서 관측 변수만으로는 식별할 수 없다.",
                                f"   Instrument: {w.n(iv[0])} is connected to {w.n(x)} and is d-separated from {w.n(y)} once the edges out of {w.n(x)} are removed." if iv else
                                "   Instrument: no variable qualifies, so the effect is not identifiable from the observed variables."))
    lines += step
    k = w.d["label"]
    lines.append(w.t(f"   해당하는 보기: {LETTERS[k]}. {w.d['options'][k]}.", f"   Matching option: {LETTERS[k]}. {w.d['options'][k]}."))
    assert w.meta["option_order"][k] == e.strategy
    return w.final(lines)


def sol_corr2cause(w: Writer) -> str:
    m, lang = w.meta, w.lang
    names = m["names"]
    g = w.g
    g.add_nodes_from(names)
    nm = lambda v: names[v]
    es = lambda h: ", ".join(f"{nm(u)} → {nm(v)}" for u, v in sorted(h.edges, key=lambda e: (nm(e[0]), nm(e[1])))) or _t(lang, "(엣지 없음)", "(no edges)")
    skel = sorted({tuple(sorted((nm(u), nm(v)))) for u, v in g.edges})
    vs = sorted({(nm(a), nm(c), nm(b)) for c in g for a in g.predecessors(c) for b in g.predecessors(c) if nm(a) < nm(b) and not g.has_edge(a, b) and not g.has_edge(b, a)})
    v33 = "call" in m  # tool-free CoT variant: the original benchmark's 'via' definition, exhaustive enumeration up to 6 variables
    members = mec_of(g) if v33 else mec(g)
    hyp = (re.search(r"(?:Hypothesis|가설): (.+?\.) ", w.d["question"])[1] if v33 else
           HYP[lang][m["hypothesis"]].format(a=nm(m["a"]), b=nm(m["b"]), i=josa(nm(m["a"]), "이")[1:], n=josa(nm(m["a"]), "은")[1:], w=josa(nm(m["a"]), "과")[1:]))
    truth = [(holds if v33 else _rel)(m["hypothesis"], h, m["a"], m["b"]) for h in members]
    lines = [_t(lang, f"1) 변수: {', '.join(sorted(nm(v) for v in g))}. 그래프는 주어지지 않고 통계적 관계만 주어진다.",
                f"1) Variables: {', '.join(sorted(nm(v) for v in g))}. Only statistical relations are given, not the graph."),
             _t(lang, f"2) 질의 유형: {KIND_NAME['corr2cause'][0]}이다. 가설이 관계와 맞는 모든 그래프에서 참일 때만 '반드시 참'이다.",
                f"2) Query type: {KIND_NAME['corr2cause'][1]}. The hypothesis is implied only if it holds in every graph consistent with the relations."),
             _t(lang, f"3) 관계로부터 골격을 세운다: 어떤 조건에서도 독립이 되지 않는 쌍만 인접한다 → {', '.join(f'{a}–{b}' for a, b in skel) or '(없음)'}. "
                      f"v-구조(서로 인접하지 않은 두 변수가 공통 이웃을 조건으로 할 때만 의존): {', '.join(f'{a} → {c} ← {b}' for a, c, b in vs) or '없음'}.",
                f"3) Build the skeleton: only pairs that never become independent are adjacent → {', '.join(f'{a}–{b}' for a, b in skel) or '(none)'}. "
                f"V-structures (non-adjacent pairs that become dependent given their common neighbour): {', '.join(f'{a} → {c} ← {b}' for a, c, b in vs) or 'none'}."),
             _t(lang, f"4) 이 골격과 v-구조를 갖는 DAG(마르코프 동치류)는 {len(members)}개이다. 가설 '{hyp}'를 각각 확인한다:",
                f"4) There are {len(members)} DAGs with this skeleton and these v-structures (the Markov equivalence class). Check the hypothesis '{hyp}' in each:")]
    lines += [f"   - {es(h)}: {_t(lang, '참' if t else '거짓', 'true' if t else 'false')}" for h, t in zip(members, truth)]
    lines.append(_t(lang, f"   {'모든 그래프에서 참이므로 반드시 참이다' if all(truth) else '거짓인 그래프가 있으므로 반드시 참이라고 할 수 없다'}.",
                    f"   {'It holds in every graph, so it is implied' if all(truth) else 'It fails in at least one graph, so it is not implied'}."))
    assert all(truth) == bool(w.d["label"])
    return w.final(lines)


SOLVERS = {
    "marginal": sol_marginal, "correlation": sol_correlation, "ate": lambda w: sol_effect(w, "ate"), "ate_score": lambda w: sol_effect(w, "ate_score"),
    "cde": sol_cde, "ett": sol_ett, "nde": lambda w: sol_mediation(w, "nde"), "nie": lambda w: sol_mediation(w, "nie"),
    "dsep": sol_dsep, "backadj": sol_backadj, "identify": sol_identify, "corr2cause": sol_corr2cause, "det_cf": sol_det_cf,
}


def solve(d: dict) -> str:
    """Solution trajectory for one item. The item must have meta['vocab'] (for corr2cause, meta['names'])."""
    return SOLVERS[d["query_type"]](Writer(d))
