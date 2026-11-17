"""Hume-Gym v3 생성기 (Step 9 SFT v2): 문형 다양화·처치/결과 전환·CLadder식 확률 문맥·결정론적 반사실.

SFT 1차는 유형마다 문형이 하나뿐이라 모델이 문형으로 질의 유형을 외웠다(reports/2026-10-28_hume_lm_sft.md 3장). v3는
- 질의 유형마다 한국어·영어 문형을 여러 개 두고, 처치 상태(t)와 결과 상태(o)를 무작위로 뒤집는다.
- 수치 질의 문맥의 절반은 전체 확률표 대신 질의에 필요한 관측 확률(주변·결합·조건부)만 준다. 이때 정답 계산에 쓸
  causal_engine 호출 인자(meta.call)를 문맥과 같은 숫자로 만들고, 도구 값이 참값과 맞는 경우에만 이 문맥을 쓴다.
- 결정론적 반사실(det_cf)을 추가한다.

수치 질의의 값은 모두 '처치 세계 − 기준 세계'(humm.lm.tools.evaluate)이고, meta.q에 엔진 질의를 비트로 남겨 check가 재계산한다.

v3.1(SFT v2.1, GENERATORS_31): SFT v2는 CLadder 문장을 도구 인자로 옮기지 못했다(reports/2026-10-28_hume_lm_sft_v2.md 3.2).
영어 수치 질의와 det_cf 일부를 CLadder식 표면(style "bare")으로 쓴다: 무의미 변수명, 상태 이름 없는 'kwox / not kwox',
'The probability of not kwox and kwoz is 5%.' 결합 문형, 'For those who are …' 조건부 문형, 'We know that … causes …' 규칙.
도구 인자에서는 kwox=yes / kwox=no로 쓴다. CLadder 문항 자체는 쓰지 않는다.

v3.2(GENERATORS_32): SFT v2.1은 'would having a sister positively affect heart condition?'처럼 상태 없이 변수 이름만 쓴 질문에서
결과 상태를 잘못 골랐다(reports/2026-10-29_hume_lm_sft_v21.md 4절). ate·nde·nie 일반 문항 일부를 이 문형(form "named")으로 쓴다.
확률 문장이 늘 상태 1을 다루므로 상태 없는 이름은 상태 1을 뜻하고, 이때 처치·결과는 t = o = 1이다.

v3.3(GENERATORS_33): v2.1은 Corr2Cause 원 벤치(변수 5–6개가 89%)에서 동치류를 손으로 나열하다 틀렸다(reports/2026-10-31_corr2cause_analysis.md).
corr2cause를 변수 3–6개(5–6개 60%)로 늘리고, 원 벤치처럼 모든 상관·조건부 독립 관계를 나열하며, 가설 'via'를 원 벤치 정의(중간 변수를 거치는 경로)로 쓴다.
정답은 humm.engine.discovery로 계산하고, 풀이는 causal_discovery 도구 호출(meta.call)이다.

v3.5(GENERATORS_35): v2.6은 CLadder의 상태 없는 nie·nde 질문에서 처치를 거꾸로 골랐다(reports/2026-11-02_hume_lm_sft_v26.md).
v3.2 named는 처치가 늘 '높음·있음' 같은 두드러진 상태라 모델이 상태 이름으로 처치를 골랐다. CLadder의 처치(X=1)는 조건부 확률 문장에서
기준 다음에 나오는 상태이고 Gym도 같은 순서로 쓴다. named 문항의 절반은 처치 변수의 두 상태 이름을 맞바꿔(flip) 순서만 단서로 남긴다.
있음/없음류 상태(yes, on, taken, done)는 CLadder에서도 늘 있음 쪽이 X=1이므로 맞바꾸지 않는다.

v3.6(GENERATORS_36): v2.8은 CLadder exp_away(설명 효과)를 0.25만 맞혔다(reports/2026-11-05_hume_lm_sft_v211.md). 엔진·Gym에 이 유형이 없어
공통 결과로 거르지 않은 correlation이나 ate를 불렀다(값 +0.0000). exp_away는 X → C ← Y에서 C=c인 경우만 볼 때 X에 따라 Y가 달라지는지 묻는다.
엔진 질의는 correlation에 관측 조건 given = {C: c}를 붙인 것이다. CLadder처럼 P(X)와 P(Y | X, C) 넷을 준다.
"""
from itertools import combinations

import networkx as nx
import numpy as np

from ..engine import BinarySCM, d_separated, is_valid_adjustment
from ..engine.discovery import holds, mec_of
from ..engine.identify import causal_nodes
from ..lm.tools import ToolError, evaluate, run, run_discovery
from . import queries as Q
from .graphs import random_dag
from .queries import MARGIN, Item, Lexicon, _cpt, _effect_setup, _lex, _pick, _q, _seed, _was
from .verbalize import PREAMBLE, Lexicon, describe_graph, josa


def _pct(p):
    return f"{round(p * 100)}%"


def statement(lex, target: dict, given: dict, p: float, style: str) -> str:
    """확률 문장 하나. style: gym(기존 문체) | cl(CLadder식 문체)."""
    if lex.lang == "ko":
        tgt = "이고 ".join(lex.is_(v, x) for v, x in target.items())
        if given:
            return f"{lex.conds(given)}일 때, {tgt}일 확률은 {_pct(p)}이다." if style == "gym" else f"{lex.conds(given)}인 경우 중 {tgt}인 비율은 {_pct(p)}이다."
        return f"{tgt}일 확률은 {_pct(p)}이다." if style == "gym" else f"전체적으로 {tgt}일 확률은 {_pct(p)}이다."
    tgt = " and ".join(lex.is_(v, x) for v, x in target.items())
    if given:
        return f"When {lex.conds(given)}, the probability that {tgt} is {_pct(p)}." if style == "gym" else f"For cases where {lex.conds(given)}, the probability that {tgt} is {_pct(p)}."
    return f"The probability that {tgt} is {_pct(p)}." if style == "gym" else f"The overall probability that {tgt} is {_pct(p)}."


def prob_str(lex, target: dict, given: dict, p: float) -> str:
    """도구 인자용: 'P(매출=높음 | 광고 집행=집행) = 0.30'."""
    eq = lambda a: ", ".join(f"{lex.name(v)}={lex.state(v, x)}" for v, x in a.items())
    return f"P({eq(target)}{' | ' + eq(given) if given else ''}) = {round(p * 100) / 100:.2f}"


def observed(scm: BinarySCM, target: dict, given: dict) -> float:
    f = scm.world()
    m = np.prod([f[k] == v for k, v in given.items()], axis=0) if given else None
    hit = np.prod([f[k] == v for k, v in target.items()], axis=0)
    return scm._mean(hit, m)


def cpt_statements(scm: BinarySCM) -> list[tuple[dict, dict]]:
    return [({v: 1}, {u: (k >> i) & 1 for i, u in enumerate(scm.parents[v])}) for v in scm.order for k in range(len(scm.cpt[v]))]


def min_statements(qtype: str, g: nx.DiGraph, c: dict):
    """질의에 필요한 관측 확률만(CLadder식). 해당 구조가 아니면 None."""
    x, y = c.get("x"), c["y"]
    yx = lambda z: [({y: 1}, {x: a, **dict(zip(z, bits))}) for a in (0, 1) for bits in np.ndindex(*(2,) * len(z))]
    if qtype == "marginal":
        pa = sorted(g.predecessors(y))
        return [({y: 1}, {})] if not pa else [({pa[0]: 1}, {}), ({y: 1}, {pa[0]: 0}), ({y: 1}, {pa[0]: 1})] if len(pa) == 1 else None
    if qtype == "correlation":
        return [({x: 1}, {}), ({x: 0, y: 1}, {}), ({x: 1, y: 1}, {})]
    if qtype == "exp_away":
        return [({x: 1}, {})] + [({y: 1}, {x: a, c["col"]: b}) for a in (0, 1) for b in (0, 1)]
    z = sorted(g.predecessors(x))
    if qtype == "ate" and len(z) <= 1:
        return [({v: 1}, {}) for v in z] + yx(z)
    if qtype == "ett" and len(z) <= 1:
        return yx(z) + [({z[0]: 1}, {x: a}) for a in (0, 1)] if z else yx(z)
    if qtype in ("nde", "nie") and not z and len(c["med"]) == 1:
        m = c["med"][0]
        return [({y: 1}, {x: a, m: b}) for a in (0, 1) for b in (0, 1)] + [({m: 1}, {x: a}) for a in (0, 1)]
    return None


def make_call(lex, g, scm, stmts, q) -> dict:
    """문맥 확률 문장과 같은 숫자로 causal_engine 호출 인자를 만든다."""
    st = lambda d: {lex.name(v): lex.state(v, x) for v, x in d.items()}
    call = {"variables": {lex.name(v): [lex.state(v, 1), lex.state(v, 0)] for v in sorted(g.nodes)},
            "graph": [f"{lex.name(u)} -> {lex.name(v)}" for u, v in sorted(g.edges)],
            "probabilities": [prob_str(lex, t, gv, observed(scm, t, gv)) for t, gv in stmts],
            "query": q["query"]}
    call |= {k: st(q[k]) for k in ("treatment", "baseline") if k in q} | {"outcome": st(q["outcome"])}
    call |= {"mediators": [lex.name(m) for m in q["mediators"]]} if q.get("mediators") else {}
    call |= {"fixed": st(q["fixed"])} if q.get("fixed") else {}
    call |= {"given": st(q["given"])} if q.get("given") else {}
    return call


# ---------- 문형 ----------
def _ctx(lex, c, t, o):
    x, y = c.get("x"), c["y"]
    d = {"Y": lex.name(y), "yo": lex.is_(y, o), "yn": lex.is_(y, 1 - o)}
    if x is not None:
        d |= {"X": lex.name(x), "t": lex.state(x, t), "b": lex.state(x, 1 - t), "xt": lex.is_(x, t), "xb": lex.is_(x, 1 - t)}
    if c.get("med"):
        d["M"] = lex.names(c["med"])
    if "col" in c:
        d["cv"] = lex.is_(c["col"], c["cv"])
    return d


def _ko(d):
    """한국어 조사가 붙은 조각."""
    return d | {k + "_을": josa(d[k], "을") for k in ("X", "M") if k in d} | {k + "_이": josa(d[k], "이") for k in ("X", "M") if k in d} | \
        ({"t_로": _q(d["t"]), "b_로": _q(d["b"]), "t_였": _was(d["t"]), "b_였": _was(d["b"])} if "t" in d else {})


PHRASES = {
    "marginal": {
        "ko": [lambda d, up: f"전체적으로 {d['yo']}일 확률이 {d['yn']}일 확률보다 {'높은가' if up else '낮은가'}?",
               lambda d, up: f"전체적으로 보면 {d['yo']}일 확률은 50%보다 {'큰가' if up else '작은가'}?",
               lambda d, up: f"모든 경우를 합쳐 볼 때, {d['yo']}인 경우가 {d['yn']}인 경우보다 {'흔한가' if up else '드문가'}?"],
        "en": [lambda d, up: f"Overall, is it {'more' if up else 'less'} likely that {d['yo']} than that {d['yn']}?",
               lambda d, up: f"Overall, is the probability that {d['yo']} {'above' if up else 'below'} 50%?",
               lambda d, up: f"Considering all cases, is the chance that {d['yo']} {'greater' if up else 'smaller'} than the chance that {d['yn']}?"],
    },
    "correlation": {
        "ko": [lambda d, up: f"{d['xt']}임을 관측하면, {d['xb']}임을 관측할 때보다 {d['yo']}일 확률이 {'높은가' if up else '낮은가'}?",
               lambda d, up: f"{d['xt']}인 것을 관측했을 때 {d['yo']}일 가능성이 {'더 큰가' if up else '더 작은가'}?",
               lambda d, up: f"관측 자료에서 {d['xt']}인 집단은 {d['xb']}인 집단보다 {d['yo']}인 비율이 {'높은가' if up else '낮은가'}?"],
        "en": [lambda d, up: f"Is the chance that {d['yo']} {'larger' if up else 'smaller'} when observing that {d['xt']}?",
               lambda d, up: f"Is the probability that {d['yo']} {'higher' if up else 'lower'} when we observe that {d['xt']} than when we observe that {d['xb']}?",
               lambda d, up: f"Among cases where {d['xt']}, is it {'more' if up else 'less'} common that {d['yo']} than among cases where {d['xb']}?"],
    },
    "ate": {
        "ko": [lambda d, up: f"개입을 통해 {d['X_을']} 강제로 {d['t_로']} 정하면, {d['b_로']} 정할 때보다 {d['yo']}일 확률이 {'높아지는가' if up else '낮아지는가'}?",
               lambda d, up: f"{d['X_을']} {d['t_로']} 바꾸는 조치는 {d['b_로']} 두는 것에 비해 {d['yo']}일 가능성을 {'높이는가' if up else '낮추는가'}?",
               lambda d, up: f"모든 대상의 {d['X_을']} {d['t_로']} 만드는 정책은 모두를 {d['b_로']} 만드는 정책보다 {d['yo']}일 확률을 {'높이는가' if up else '낮추는가'}?"],
        "en": [lambda d, up: f"If we intervene to set {d['X']} to {d['t']} instead of {d['b']}, does the probability that {d['yo']} {'increase' if up else 'decrease'}?",
               lambda d, up: f"Will setting {d['X']} to {d['t']} {'increase' if up else 'decrease'} the chance that {d['yo']}, compared with setting it to {d['b']}?",
               lambda d, up: f"Would a policy that makes {d['X']} {d['t']} for everyone, rather than {d['b']} for everyone, {'raise' if up else 'lower'} the chance that {d['yo']}?"],
    },
    "ett": {
        "ko": [lambda d, up: f"실제로 {d['xb']}인 경우들만 볼 때, 만약 {d['X_이']} {d['t_였']}다면 {d['yo']}일 확률은 {'더 높았겠는가' if up else '더 낮았겠는가'}?",
               lambda d, up: f"{d['xb']}인 대상들에게 {d['X_이']} {d['t_였']}다면, {d['yo']}일 가능성이 {'더 컸겠는가' if up else '더 작았겠는가'}?",
               lambda d, up: f"{d['xb']}였던 사례들을 생각하자. 그 사례들에서 {d['X_이']} {d['t_였']}더라면 {d['yo']}일 확률이 {'올라갔겠는가' if up else '내려갔겠는가'}?"],
        "en": [lambda d, up: f"For cases where {d['xb']}, would it be {'more' if up else 'less'} likely that {d['yo']} if {d['X']} had been {d['t']}?",
               lambda d, up: f"Among cases where {d['xb']}, would the probability that {d['yo']} have been {'higher' if up else 'lower'} if {d['X']} had been {d['t']}?",
               lambda d, up: f"Consider only the cases in which {d['xb']}. Had {d['X']} instead been {d['t']}, would it have become {'more' if up else 'less'} likely that {d['yo']}?"],
    },
    "nde": {
        "ko": [lambda d, up: f"{d['M_을']} 거치는 간접 경로를 제외하고(매개 변수는 {d['xb']}일 때의 값으로 둔다), {d['X_이']} '{d['b']}'에서 {d['t_로']} 바뀌면 {d['yo']}일 확률이 {'높아지는가' if up else '낮아지는가'}?",
               lambda d, up: f"{d['M_을']} 통한 매개 효과를 무시한다면, {d['xt']}인 것은 {d['xb']}인 것에 비해 {d['yo']}일 가능성을 {'높이는가' if up else '낮추는가'}?",
               lambda d, up: f"{d['M_이']} {d['xb']}일 때의 값에 머문다고 할 때, 직접 경로만으로 {d['xt']}이면 {d['yo']}일 확률이 {'올라가는가' if up else '내려가는가'}?"],
        "en": [lambda d, up: f"If we disregard the mediation effect through {d['M']}, would changing {d['X']} from {d['b']} to {d['t']} {'increase' if up else 'decrease'} the chance that {d['yo']}?",
               lambda d, up: f"Keeping {d['M']} at the values they would take when {d['xb']}, does {d['X']} being {d['t']} rather than {d['b']} {'raise' if up else 'lower'} the probability that {d['yo']}?",
               lambda d, up: f"Through the direct path alone (not through {d['M']}), would {d['xt']} make it {'more' if up else 'less'} likely that {d['yo']} than {d['xb']}?"],
    },
    "nie": {
        "ko": [lambda d, up: f"{d['xb']}인 상태를 유지하면서 {d['M']}만 {d['xt']}일 때의 값으로 바뀐다면, {d['yo']}일 확률이 {'높아지는가' if up else '낮아지는가'}?",
               lambda d, up: f"{d['xt']}인 것은 {d['M_을']} 거치는 경로를 통해서 {d['yo']}일 가능성을 {'높이는가' if up else '낮추는가'}? (기준은 {d['xb']})",
               lambda d, up: f"{d['M_을']} 통한 간접 효과만 볼 때, {d['X_이']} {d['b']} 대신 {d['t']}이면 {d['yo']}일 확률이 {'올라가는가' if up else '내려가는가'}?"],
        "en": [lambda d, up: f"Suppose {d['xb']}, but {d['M']} take the values they would have if {d['xt']}. Does the probability that {d['yo']} {'increase' if up else 'decrease'}?",
               lambda d, up: f"Does {d['X']} being {d['t']} rather than {d['b']} {'increase' if up else 'decrease'} the chance that {d['yo']} through {d['M']}?",
               lambda d, up: f"Is the indirect effect of {d['xt']} (versus {d['xb']}) on the chance that {d['yo']}, through {d['M']} only, {'positive' if up else 'negative'}?"],
    },
}
PHRASES["exp_away"] = {
    "ko": [lambda d, up: f"{d['cv']}인 경우만 볼 때, {d['xt']}이면 {d['xb']}일 때보다 {d['yo']}일 확률이 {'높은가' if up else '낮은가'}?",
           lambda d, up: f"{d['cv']}인 대상들 가운데 {d['xt']}인 집단은 {d['xb']}인 집단보다 {d['yo']}인 비율이 {'높은가' if up else '낮은가'}?",
           lambda d, up: f"{d['cv']}임을 알고 있을 때, {d['xt']}임을 관측하면 {d['yo']}일 가능성이 {'커지는가' if up else '작아지는가'}?"],
    "en": [lambda d, up: f"Among cases where {d['cv']}, is the chance that {d['yo']} {'higher' if up else 'lower'} when {d['xt']} than when {d['xb']}?",
           lambda d, up: f"If we look only at cases where {d['cv']}, does the probability that {d['yo']} {'increase' if up else 'decrease'} when {d['xt']}?",
           lambda d, up: f"Restricting attention to cases where {d['cv']}, is it {'more' if up else 'less'} likely that {d['yo']} when we observe that {d['xt']} rather than {d['xb']}?"],
}
PHRASES["cde"] = {
    "ko": [lambda d, up, f=f: f"{d['fix']}인 상태로 고정하고, " + f(d, up) for f in PHRASES["ate"]["ko"][:2]],
    "en": [lambda d, up, f=f: f"Holding fixed that {d['fix']}: " + f(d, up)[0].lower() + f(d, up)[1:] for f in PHRASES["ate"]["en"][:2]],
}
SETUPS = {
    "marginal": lambda rng: (lambda g: {"g": g, "y": sorted(g.nodes)[-1]})(random_dag(int(rng.integers(3, 7)), rng, p=0.5, max_parents=2)),
    "correlation": _effect_setup, "ate": _effect_setup, "ett": _effect_setup,
    "cde": lambda rng: (lambda c: c and c | {"fix": {m: int(rng.integers(2)) for m in c["med"]}})(_effect_setup(rng, mediated=True)),
    "nde": lambda rng: _effect_setup(rng, mediated=True), "nie": lambda rng: _effect_setup(rng, mediated=True),
    "exp_away": lambda rng: (lambda x, y, col: {"g": nx.DiGraph([(x, col), (y, col)]), "x": x, "y": y, "col": col, "cv": int(rng.integers(2))})(*(f"V{i}" for i in rng.permutation(3))),
}
RUNG = {"marginal": 1, "correlation": 1, "ate": 2, "cde": 2, "ett": 3, "nde": 3, "nie": 3}


# ---------- CLadder식 표면 (v3.1) ----------
class Bare(Lexicon):
    """무의미 변수: 문장에서는 'kwox' / 'not kwox', 도구 인자 상태는 yes/no."""

    def is_(self, v, val):
        return self.name(v) if val else f"not {self.name(v)}"


def bare_lex(g, rng) -> Bare:
    names = set()
    while len(names) < g.number_of_nodes():
        names.add("".join(rng.choice(list(a)) for a in ("bcdfghjklmnprstvxyz", "wwrl", "aeiouy", "bcdfghjklmnpqrstvxz")))
    return Bare({v: (n, "예", "아니오", n, "yes", "no") for v, n in zip(sorted(g.nodes), sorted(sorted(names), key=lambda _: rng.random()))}, "en")  # 집합 순서(해시)에 기대지 않게 먼저 정렬


def _graph_text(g, lex) -> list[str]:
    """CLadder처럼 문장 첫 글자는 대문자(Kwox has a direct effect on swoq.)."""
    return [t[0].upper() + t[1:] for t in describe_graph(g, lex)] if isinstance(lex, Bare) else describe_graph(g, lex)


def bare_statement(lex, target: dict, given: dict, p: float) -> str:
    tgt = " and ".join(lex.is_(v, x) for v, x in target.items())
    if given:
        return f"For those who are {' and are '.join(lex.is_(v, x) for v, x in given.items())}, the probability of {tgt} is {_pct(p)}."
    return f"The {'overall ' if len(target) == 1 else ''}probability of {tgt} is {_pct(p)}."


BARE_PHRASES = {
    "marginal": lambda d, up: f"Is {d['yo']} {'more' if up else 'less'} likely than {d['yn']} overall?",
    "correlation": lambda d, up: f"Is the chance of {d['yo']} {'larger' if up else 'smaller'} when observing {d['xt']}?",
    "ate": lambda d, up: f"Will {d['xt']} {'increase' if up else 'decrease'} the chance of {d['yo']}?",
    "ett": lambda d, up: f"For those who are {d['xb']}, would it be {'more' if up else 'less'} likely to see {d['yo']} if the individual was {d['xt']}?",
    "nde": lambda d, up: f"If we disregard the mediation effect through {d['M']}, would {d['xt']} {'positively' if up else 'negatively'} affect {d['yo']}?",
    "nie": lambda d, up: f"Does {d['xt']} {'positively' if up else 'negatively'} affect {d['yo']} through {d['M']}?",
}


NAMED_PHRASES = {  # 처치·결과를 상태 없이 이름만 쓴다(CLadder 'affect heart condition')
    "ate": {"ko": lambda d, up: f"{d['X_이']} {d['Y']}에 {'양(+)' if up else '음(−)'}의 인과 효과를 주는가?",
            "en": lambda d, up: f"Does {d['X']} have a {'positive' if up else 'negative'} causal effect on {d['Y']}?"},
    "nde": {"ko": lambda d, up: f"{d['M_을']} 통한 매개 효과를 무시하면, {d['X_이']} {d['Y']}에 {'양(+)' if up else '음(−)'}의 영향을 주는가?",
            "en": lambda d, up: f"If we disregard the mediation effect through {d['M']}, would {d['X']} {'positively' if up else 'negatively'} affect {d['Y']}?"},
    "nie": {"ko": lambda d, up: f"{d['X_이']} {d['M_을']} 거쳐서 {d['Y']}에 {'양(+)' if up else '음(−)'}의 영향을 주는가?",
            "en": lambda d, up: f"Does {d['X']} {'positively' if up else 'negatively'} affect {d['Y']} through {d['M']}?"},
}


def numeric(qtype, rng, lang, domain, bare: float = 0.0, named: float = 0.0, digits: float = 0.0, flip: float = 0.0):
    c = SETUPS[qtype](rng)
    if c is None:
        return None
    t = 1 if qtype in ("nde", "nie") and rng.random() < 0.5 else int(rng.integers(2))
    o = int(rng.integers(2))
    named_ = bool(named) and qtype in NAMED_PHRASES and rng.random() < named  # named = 0이면 v3.1과 같은 난수 흐름
    t, o = (1, 1) if named_ else (t, o)
    q = {"query": "correlation" if qtype == "exp_away" else qtype, "outcome": {c["y"]: o}}
    q |= {"treatment": {c["x"]: t}, "baseline": {c["x"]: 1 - t}} if qtype != "marginal" else {}
    q |= {"mediators": c["med"]} if qtype in ("nde", "nie") else {"fixed": c["fix"]} if qtype == "cde" else {"given": {c["col"]: c["cv"]}} if qtype == "exp_away" else {}
    for _ in range(30):
        scm = BinarySCM.random(c["g"], seed=_seed(rng))
        val = evaluate(scm, q)[0]
        if abs(val) >= MARGIN:
            break
    else:
        return None
    bare_ = bool(bare) and lang == "en" and qtype in BARE_PHRASES and rng.random() < bare
    lex = bare_lex(c["g"], rng) if bare_ else _lex(c["g"], rng, lang, domain)
    if digits and not bare_ and rng.random() < digits:  # v3.4: 학습 도메인 이름에 상태 1/0(테스트 무의미 이름의 상태 표기)
        lex = Lexicon({v: (w[0], "1", "0", w[3], "1", "0") for v, w in lex.vocab.items()}, lang)
    if flip and named_ and not bare_ and lex.vocab[c["x"]][4] not in PRESENT and rng.random() < flip:  # v3.5: 처치 변수의 상태 1이 두드러지지 않은 쪽(낮음·느림)이 되게 맞바꾼다
        w = lex.vocab[c["x"]]
        lex = Lexicon(lex.vocab | {c["x"]: (w[0], w[2], w[1], w[3], w[5], w[4])}, lang)
    stmts, style = min_statements(qtype, c["g"], c) if rng.random() < 0.5 or bare_ else None, "cl" if rng.random() < 0.5 else "gym"
    style = "bare" if bare_ else style
    named_ = named_ and not bare_  # bare 이름은 원래 상태 1이다
    call = None
    if stmts:
        call = make_call(lex, c["g"], scm, stmts, q)
        try:
            tool_val = run(call)[0]
        except ToolError:
            tool_val = None
        if tool_val is None or abs(tool_val - val) > 0.01 or (tool_val > 0) != (val > 0):
            stmts = None
    if not stmts:
        stmts = cpt_statements(scm)
        call = make_call(lex, c["g"], scm, stmts, q)
    want = bool(rng.random() < 0.5)
    up = (val > 0) == want
    d = _ctx(lex, c, t, o) | ({"fix": lex.conds(c["fix"])} if qtype == "cde" else {})
    forms = PHRASES[qtype][lang]
    k = int(rng.integers(len(forms)))
    dl = _ko(d) if lang == "ko" else d
    question = BARE_PHRASES[qtype](d, up) if bare_ else NAMED_PHRASES[qtype][lang](dl, up) if named_ else forms[k](dl, up)
    k = "bare" if bare_ else "named" if named_ else k
    sents = [bare_statement(lex, tt, gv, observed(scm, tt, gv)) if bare_ else statement(lex, tt, gv, observed(scm, tt, gv), style) for tt, gv in stmts]
    text = PREAMBLE[lang] + " " + " ".join(_graph_text(c["g"], lex) + sents)
    meta = {k2: v for k2, v in c.items() if k2 != "g"} | {"direction": "up" if up else "down", "cpt": _cpt(scm), "vocab": lex.vocab, "q": q,
                                                          "call": call, "style": style, "minimal": stmts is not None and len(stmts) < len(cpt_statements(scm)),
                                                          "form": k} | ({"sents": sents} if bare_ else {})
    return Item(qtype, RUNG.get(qtype, 1), "noul", lang, domain, text, question, int(want), value=round(val, 6), graph=list(c["g"].edges), meta=meta)  # exp_away(v3.6)는 RUNG에 없고 Rung 1이다


# ---------- 구조 질의 문형 ----------
def gen_dsep(rng, lang, domain):
    it = Q.gen_dsep(rng, lang, domain)
    if it is None or rng.random() < 0.5:
        return it
    lex = _lex_of(it)
    a, b, z = it.meta["a"], it.meta["b"], it.meta["z"]
    if lang == "ko":
        cond = f"{lex.names(z)}의 값을 알고 있을 때, " if z else ""
        q = f"{cond}{josa(lex.name(a), '을')} 알면 {lex.name(b)}에 대한 예측이 달라지는가? 즉 두 변수는 서로 의존하는가?"
    else:
        cond = f"Once we know {lex.names(z)}, " if z else ""
        q = f"{cond}does learning {lex.name(a)} change what we expect about {lex.name(b)}? In other words, are they dependent{' given ' + lex.names(z) if z else ''}?"
    it.question, it.label, it.meta["asks"] = q, 1 - it.label, "dependent"
    return it


def _lex_of(it):
    return Q.Lexicon({k: tuple(v) for k, v in it.meta["vocab"].items()}, it.lang)


def gen_backadj(rng, lang, domain):
    """절반은 기존 문형, 절반은 CLadder식 두 방법 비교(유효한 조정 집합 하나와 유효하지 않은 집합 하나)."""
    if rng.random() < 0.5:
        return Q.gen_backadj(rng, lang, domain)
    for _ in range(40):
        g = random_dag(int(rng.integers(3, 7)), rng, p=0.45, max_parents=3)
        pairs = [(x, y) for x in g for y in sorted(nx.descendants(g, x))]
        x, y = pairs[rng.integers(len(pairs))]
        rest = sorted(set(g.nodes) - {x, y})
        sets = [sorted(_pick(rng, rest, int(rng.integers(0, min(2, len(rest)) + 1)))) for _ in range(6)]
        good = [s for s in sets if is_valid_adjustment(g, x, y, s)]
        bad = [s for s in sets if not is_valid_adjustment(g, x, y, s)]
        if good and bad:
            break
    else:
        return None
    want = bool(rng.random() < 0.5)
    m1, m2 = (good[0], bad[0]) if want else (bad[0], good[0])
    lex = _lex(g, rng, lang, domain)
    if lang == "ko":
        desc = lambda s: f"{lex.names(s)}의 값별로 나누어 {josa(lex.name(x), '과')} {lex.name(y)}의 상관을 본다." if s else f"{josa(lex.name(x), '과')} {lex.name(y)}의 상관을 전체에서 바로 본다."
        q = f"방법 1: {desc(m1)} 방법 2: {desc(m2)} {josa(lex.name(x), '이')} {lex.name(y)}에 미치는 영향을 알아내려면 방법 1이 방법 2보다 더 올바른가?"
    else:
        desc = lambda s: f"We look at how {lex.name(x)} correlates with {lex.name(y)} case by case according to {lex.names(s)}." if s else f"We look directly at how {lex.name(x)} correlates with {lex.name(y)} in general."
        q = f"Method 1: {desc(m1)} Method 2: {desc(m2)} To understand how {lex.name(x)} affects {lex.name(y)}, is it more correct to use Method 1 than Method 2?"
    return Item("backadj", 2, "noul", lang, domain, PREAMBLE[lang] + " " + " ".join(describe_graph(g, lex)), q, int(want), graph=list(g.edges),
                meta={"x": x, "y": y, "z": m1, "methods": [m1, m2], "vocab": lex.vocab})


# ---------- 결정론적 반사실 ----------
def det_rule_text(lex, v, rule) -> str:
    """rule = (op, [(parent, value), ...]): v=1 iff 리터럴들이 op(and/or)로 성립."""
    op, lits = rule
    if lex.lang == "ko":
        cond = ("이고 " if op == "and" else "이거나 ").join(lex.is_(p, a) for p, a in lits)
        return f"{cond}이면 {lex.is_(v, 1)}이 되고, 그렇지 않으면 {lex.is_(v, 0)}이 된다."
    cond = (" and " if op == "and" else " or ").join(lex.is_(p, a) for p, a in lits)
    return f"If {cond}, then {lex.is_(v, 1)}; otherwise {lex.is_(v, 0)}."


def det_values(order, rules, exo: dict, do: dict | None = None) -> dict:
    do, vals = do or {}, {}
    for v in order:
        if v in do:
            vals[v] = do[v]
        elif v in rules:
            op, lits = rules[v]
            hits = [vals[p] == a for p, a in lits]
            vals[v] = int(all(hits) if op == "and" else any(hits))
        else:
            vals[v] = exo[v]
    return vals


def bare_rule_text(lex, v, rule) -> str:
    """CLadder식 규칙: 'gyzp and not hwax causes jyka', 부정 리터럴 하나는 'tijv causes not gyzp'."""
    op, lits = rule
    if len(lits) == 1 and not lits[0][1]:
        return f"{lex.name(lits[0][0])} causes not {lex.name(v)}"
    return f"{f' {op} '.join(lex.is_(p, a) for p, a in lits)} causes {lex.name(v)}"


def gen_det_cf(rng, lang, domain, bare: float = 0.0):
    for _ in range(40):
        g = random_dag(int(rng.integers(3, 6)), rng, p=0.55, max_parents=2)
        order = list(nx.topological_sort(g))
        rules = {v: ("and" if rng.random() < 0.5 else "or", [(p, int(rng.integers(2))) for p in sorted(g.predecessors(v))]) for v in order if g.in_degree(v)}
        exo = {v: int(rng.integers(2)) for v in order if v not in rules}
        pairs = [(x, y) for x in order for y in sorted(nx.descendants(g, x))]
        x, y = pairs[rng.integers(len(pairs))]
        fact = det_values(order, rules, exo)
        cf = det_values(order, rules, exo, {x: 1 - fact[x]})
        if cf[y] != fact[y] or rng.random() < 0.4:  # 반사실 값이 사실과 같은 문항도 일부 남긴다
            break
    else:
        return None
    want = bool(rng.random() < 0.5)
    ask = cf[y] if want else 1 - cf[y]
    bare_ = bool(bare) and lang == "en" and rng.random() < bare
    lex = bare_lex(g, rng) if bare_ else _lex(g, rng, lang, domain)
    obs = {v: a for v, a in exo.items() if v != x}
    rules_txt = [det_rule_text(lex, v, rules[v]) for v in order if v in rules]
    if bare_:
        know = "We know that " + ", and we know that ".join(bare_rule_text(lex, v, rules[v]) for v in order if v in rules) + "."
        seen = f" We observed {' and '.join(f'an individual is {lex.is_(v, a)}' for v, a in obs.items())}." if obs else ""
        q = f"Would an individual is {lex.is_(y, ask)} if {lex.is_(x, 1 - fact[x])} instead of {lex.is_(x, fact[x])}?"
    elif lang == "ko":
        know = "변수 사이의 규칙은 다음과 같이 결정론적이다: " + " ".join(rules_txt)
        seen = f" 관측 결과 {lex.conds(obs)}이다." if obs else ""
        q = f"실제로는 {lex.is_(x, fact[x])}이다. 만약 {josa(lex.name(x), '이')} {_was(lex.state(x, 1 - fact[x]))}다면, {lex.is_(y, ask)}이었겠는가?"
    else:
        know = "The variables follow these deterministic rules: " + " ".join(rules_txt)
        seen = f" We observed that {lex.conds(obs)}." if obs else ""
        q = f"In fact, {lex.is_(x, fact[x])}. If {lex.name(x)} had been {lex.state(x, 1 - fact[x])} instead, would it be the case that {lex.is_(y, ask)}?"
    text = PREAMBLE[lang] + " " + " ".join(_graph_text(g, lex)) + " " + know + seen
    return Item("det_cf", 3, "noul", lang, domain, text, q, int(want), graph=list(g.edges),
                meta={"x": x, "y": y, "ask": ask, "rules": {v: [op, [list(l) for l in lits]] for v, (op, lits) in rules.items()}, "exo": exo,
                      "fact": fact, "vocab": lex.vocab, "style": "bare" if bare_ else "plain"})


# ---------- v3.3 corr2cause: 원 벤치 표면, 변수 3–6개, causal_discovery 도구 ----------
C2C_N, C2C_P, C2C_MAX_MEC, C2C_MAX_REL = (3, 4, 5, 6), (0.15, 0.25, 0.3, 0.3), 32, 60  # 원 벤치는 동치류 최대 32개, 관계 최대 52개다
HYP33 = {
    "ko": {"direct": ["{a}{i} {b}의 직접 원인이다."], "via": ["{a}{n} 다른 변수를 거쳐 {b}의 원인이 된다.", "{a}에서 다른 변수를 거쳐 {b}{r} 가는 인과 경로가 있다."],
           "cause": ["{a}{n} {b}의 (직접 또는 간접) 원인이다."], "collider": ["{a}{w} {b}의 공통 결과(충돌 변수)가 적어도 하나 있다."],
           "confounder": ["{a}{w} {b}의 공통 원인(교란 변수)이 적어도 하나 있다."]},
    "en": {"direct": ["{a} directly causes {b}."], "via": ["{a} causes something else which causes {b}.", "{a} is a cause for {b}, but not a direct one."],
           "cause": ["{a} is a (direct or indirect) cause of {b}."], "collider": ["There exists at least one collider (i.e., common effect) of {a} and {b}."],
           "confounder": ["There exists at least one confounder (i.e., common cause) of {a} and {b}."]},
}


def _and(xs: list[str]) -> str:
    return xs[0] if len(xs) == 1 else f"{', '.join(xs[:-1])} and {xs[-1]}"


def c2c_relations(g: nx.DiGraph, lang: str) -> tuple[list[str], list[str]]:
    """원 벤치처럼 쌍마다 주변 관계와, 그 쌍을 떼어 놓는 모든 조건 집합을 쓴다. (상관 문장, 독립 문장)."""
    nodes, dep, ind = sorted(g), [], []
    for u, v in combinations(nodes, 2):
        pair = f"{josa(u, '과')} {josa(v, '은')}"
        if d_separated(g, u, v):
            ind.append(f"{pair} 서로 독립이다." if lang == "ko" else f"{u} is independent of {v}.")
        else:
            dep.append(f"{pair} 상관관계가 있다." if lang == "ko" else f"{u} correlates with {v}.")
        rest = [w for w in nodes if w not in (u, v)]
        for s in (list(s) for k in range(1, len(rest) + 1) for s in combinations(rest, k)):
            if d_separated(g, u, v, s):
                ind.append(f"{josa(', '.join(s), '이')} 주어지면 {pair} 독립이다." if lang == "ko" else f"{u} and {v} are independent given {_and(s)}.")
    return dep, ind


def gen_corr2cause(rng, lang, domain):
    """정답 '참'과 '거짓' 가설을 동치류에서 전수 계산한 뒤 원하는 쪽에서 고른다(50:50)."""
    want = bool(rng.random() < 0.5)
    pool = "PQRSTUVWXYZ" if domain in Q.TEST_DOMAINS else "ABCDEFGHJKLMN"
    for _ in range(40):
        n = int(rng.choice(C2C_N, p=C2C_P))
        g = random_dag(n, rng, p=float(rng.uniform(0.3, 0.8)), max_parents=3, connected=bool(rng.random() < 0.8))
        g = nx.relabel_nodes(g, dict(zip(sorted(g), _pick(rng, list(pool), n))))  # 이름 순서가 위상 순서를 드러내지 않게 무작위로 붙인다
        members = mec_of(g)
        cands = [(h, a, b) for h in HYP33["en"] for a in g for b in g if a != b and all(holds(h, m, a, b) for m in members) == want]
        dep, ind = c2c_relations(g, lang)
        if cands and len(members) <= C2C_MAX_MEC and len(dep + ind) <= C2C_MAX_REL:
            h, a, b = cands[rng.integers(len(cands))]
            break
    else:
        return None
    names = sorted(g)
    if lang == "ko":
        rel = " ".join(dep + [("한편, " if dep and i == 0 else "") + x for i, x in enumerate(ind)])
        ctx = f"변수 {', '.join(names)} {n}개로 이루어진 닫힌 계를 생각하라. 이 변수들 사이의 통계적 관계는 다음이 전부이다: {rel}"
    else:
        rel = " ".join(dep + [("However, " if dep and i == 0 else "") + x for i, x in enumerate(ind)])
        ctx = f"Suppose there is a closed system of {n} variables, {_and(names)}. All the statistical relations among these {n} variables are as follows: {rel}"
    phr = HYP33[lang][h]
    hyp = phr[rng.integers(len(phr))].format(a=a, b=b, i=josa(a, "이")[1:], n=josa(a, "은")[1:], w=josa(a, "과")[1:], r=josa(b, "로")[len(b):])
    q = f"가설: {hyp} 위의 통계적 관계로부터 이 가설이 반드시 참이라고 할 수 있는가?" if lang == "ko" else f"Hypothesis: {hyp} Does the premise necessarily imply the hypothesis?"
    call = {"variables": names, "relations": dep + ind, "hypothesis": h, "a": a, "b": b}
    assert run_discovery(call)[0] == want
    return Item("corr2cause", 1, "noul", lang, domain, ctx, q, int(want), graph=sorted(g.edges),
                meta={"hypothesis": h, "a": a, "b": b, "mec_size": len(members), "names": {v: v for v in names}, "call": call})


GENERATORS = {**{k: (lambda k: lambda rng, lang, domain: numeric(k, rng, lang, domain))(k) for k in RUNG},
              "ate_score": Q.gen_ate_score, "dsep": gen_dsep, "backadj": gen_backadj, "identify": Q.gen_identify, "corr2cause": Q.gen_corr2cause,
              "det_cf": gen_det_cf}
MIX = {"ate": 0.11, "nie": 0.09, "nde": 0.08, "ett": 0.11, "cde": 0.05, "correlation": 0.08, "marginal": 0.06, "ate_score": 0.03,
       "backadj": 0.10, "dsep": 0.06, "identify": 0.07, "corr2cause": 0.06, "det_cf": 0.10}

BARE = 0.5  # v3.1: 영어 수치 질의(cde·ate_score 제외)와 det_cf 중 CLadder식 표면 비율
GENERATORS_31 = GENERATORS | {k: (lambda k: lambda rng, lang, domain: numeric(k, rng, lang, domain, bare=BARE))(k) for k in BARE_PHRASES} | \
    {"det_cf": lambda rng, lang, domain: gen_det_cf(rng, lang, domain, bare=BARE)}
NAMED = 0.25  # v3.2: ate·nde·nie 일반 문항 중 상태 없는 이름 문형 비율
GENERATORS_32 = GENERATORS_31 | {k: (lambda k: lambda rng, lang, domain: numeric(k, rng, lang, domain, bare=BARE, named=NAMED))(k) for k in NAMED_PHRASES}
GENERATORS_33 = GENERATORS_32 | {"corr2cause": gen_corr2cause}
DIGITS = 1.0  # v3.4: 수치 질의 전부를 상태 1/0으로(학습 보충용, 다른 유형은 v3.3과 같다)
GENERATORS_34 = GENERATORS_33 | {k: (lambda k: lambda rng, lang, domain: numeric(k, rng, lang, domain, bare=BARE, named=NAMED, digits=DIGITS))(k) for k in RUNG}
PRESENT = ("yes", "on", "taken", "done", "1")  # v3.5: 맞바꾸지 않는 있음/없음류·1/0 상태 1
FLIP = 0.5  # v3.5: named 문항(ate·nde·nie만, 학습 보충용) 중 처치 상태 이름을 맞바꾼 비율
GENERATORS_35 = GENERATORS_33 | {k: (lambda k: lambda rng, lang, domain: numeric(k, rng, lang, domain, named=1.0, flip=FLIP))(k) for k in NAMED_PHRASES}
# v3.6 exp_away는 이전 버전 생성기(BARE_PHRASES로 만든 GENERATORS_31~35)에 들어가지 않게 여기서 더한다. RUNG(수치 유형 목록으로도 쓴다)에는 넣지 않는다
BARE_PHRASES["exp_away"] = lambda d, up: f"If we look at those who are {d['cv']}, does the chance of {d['yo']} {'increase' if up else 'decrease'} when {d['xt']}?"
EXP_AWAY = 0.5  # v3.6: exp_away 영어 문항 중 CLadder식 표면 비율(다른 유형은 v3.5와 같다)
GENERATORS_36 = GENERATORS_35 | {"exp_away": lambda rng, lang, domain: numeric("exp_away", rng, lang, domain, bare=EXP_AWAY)}
