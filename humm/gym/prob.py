"""Hume-Gym v3.7 (SFT v2.12): 값을 JSON으로 묻는 수치 질의(kind "prob").

4B 평가표(reports/2026-11-06_hume_lm_eval_table.md)에서 CaLM·CauGym의 수치 과제는 예/아니오와 함께 값 자체(PROB)를 채점했다.
v2.11은 거기서 셋을 잃었다.
- ETT 값의 정의가 교과서(실제 − 반사실)라 엔진 value(반사실 − 실제)와 부호가 반대다. 엔진 결과의 'ETT =' 줄을 옮겨 쓴다.
- PN·PS(Tian–Pearl 상·하한)가 엔진에 없었다. 엔진 질의 pn·ps와 bound를 쓴다.
- 처치에서 결과로 가는 경로가 없거나 직접 엣지뿐인 문항은 확률 없이 값이 0이다. 확률 목록을 비우고 매개 집합도 비워 엔진이 경로에서 찾게 한다.

문맥은 그 벤치들처럼 엣지마다 한 문장, 확률은 소수 넷째 자리다. 질문 문형은 유형마다 둘이고, 하나는 벤치의 문형을, 하나는 다르게 쓴 문형이다.
벤치 문항 자체는 쓰지 않는다. 정답 값은 문맥과 같은 숫자로 만든 엔진 호출(meta.call)의 값이고, 참 확률표의 값과 0.01 안일 때만 문항으로 쓴다.
"""
import json

import networkx as nx
import numpy as np

from ..engine import BinarySCM
from ..lm.tools import ToolError, evaluate, run
from .graphs import random_dag
from .queries import MARGIN, Item, _cpt, _lex, _seed
from .v3 import GENERATORS_36, Bare, bare_lex, cpt_statements, observed
from .verbalize import PREAMBLE, Lexicon

TYPES = ("ate", "cde", "ett", "nde", "nie", "pn", "ps")
ZERO = 0.3  # ate·ett·nde·nie 중 경로가 없어 값이 0인 문항(확률 없음) 비율
FULL = {"ate": "average treatment effect", "cde": "controlled direct effect", "ett": "effect of treatment on the treated", "nde": "natural direct effect",
        "nie": "natural indirect effect", "pn": "probability of necessity", "ps": "probability of sufficiency"}
RUNG = {"ate": 2, "cde": 2, "ett": 3, "nde": 3, "nie": 3, "pn": 3, "ps": 3}

QUESTIONS = {
    "ate": [lambda d: f"If {d['X']} is changed to be {d['t']}, will {d['Y']} be more likely to be {d['o']}?",
            lambda d: f"Would setting {d['X']} to {d['t']} by intervention, rather than to {d['b']}, make {d['Y']} more likely to be {d['o']}?"],
    "cde": [lambda d: f"Conditioned on {d['fix']}, if the {d['X']} had been {d['t']}, would the {d['Y']} have been more likely to be {d['o']}?",
            lambda d: f"Holding {d['fix']} fixed, would changing {d['X']} from {d['b']} to {d['t']} make {d['Y']} more likely to be {d['o']}?"],
    "ett": [lambda d: f"For those with {d['X']} being {d['b']}, if their {d['X']} had been {d['t']}, would the {d['Y']} have been more likely to be {d['o']}?",
            lambda d: f"Among the units whose {d['X']} is {d['b']}, would {d['Y']} have been more likely to be {d['o']} had their {d['X']} been {d['t']}?"],
    "nde": [lambda d: f"Suppose the mediator keeps constant when {d['X']} is changed to be {d['t']}, would the {d['Y']} have been more likely to be {d['o']}?",
            lambda d: f"If {d['X']} changed from {d['b']} to {d['t']} while every mediator kept the value it has under {d['X']} being {d['b']}, would {d['Y']} be more likely to be {d['o']}?"],
    "nie": [lambda d: f"Suppose {d['X']} is held constant and the mediator changes to whatever value it would have attained under {d['X']} changing to be {d['t']}, "
                      f"would {d['Y']} have been more likely to be {d['o']}?",
            lambda d: f"If {d['X']} stayed {d['b']} but the mediators took the values they would have under {d['X']} being {d['t']}, would {d['Y']} be more likely to be {d['o']}?"],
    "pn": [lambda d: f"Given that {d['X']} was {d['t']} and {d['Y']} was {d['o']}, what is the {d['bound']} bound of the probability of the {d['Y']} would have been {d['n']} "
                     f"if the {d['X']} had been {d['b']}?",
           lambda d: f"{d['X'][0].upper() + d['X'][1:]} was {d['t']} and {d['Y']} was {d['o']}. What is the {d['bound']} bound of the probability that {d['Y']} would have been "
                     f"{d['n']} had {d['X']} been {d['b']}?"],
    "ps": [lambda d: f"Given that {d['X']} was {d['b']} and {d['Y']} was {d['n']}, what is the {d['bound']} bound of the probability that {d['Y']} would have been {d['o']} "
                     f"if the {d['X']} had been {d['t']}?",
           lambda d: f"{d['X'][0].upper() + d['X'][1:]} was {d['b']} and {d['Y']} was {d['n']}. What is the {d['bound']} bound of the probability that {d['Y']} would have been "
                     f"{d['o']} had {d['X']} been {d['t']}?"],
}
FORMATS = {True: ['Provide the calculation result to four decimal places and a final "yes" or "no" answer in JSON format, like {"ANSWER": "Yes", "PROB": "0.1234"}:',
                  'Give the value to four decimal places and a yes or no answer as JSON, e.g. {"ANSWER": "No", "PROB": "-0.0312"}.'],
           False: ['Provide the calculation result to four decimal places in JSON format, like {"PROB": "0.1234"}:',
                   'Give the value to four decimal places as JSON, e.g. {"PROB": "0.4321"}.']}


def _zero_ok(qtype, g, x, y) -> bool:
    """경로 구조만으로 값이 0인가: ate·ett는 x → y 경로 없음, nde는 직접 엣지 없음, nie는 매개 경로 없음."""
    med = nx.descendants(g, x) & nx.ancestors(g, y)
    return {"ate": y not in nx.descendants(g, x), "ett": y not in nx.descendants(g, x), "nde": not g.has_edge(x, y), "nie": not med}[qtype]


def _setup(qtype, rng, zero: bool):
    for _ in range(50):
        g = random_dag(int(rng.integers(3, 7)), rng, p=0.5, max_parents=2)
        med = lambda x, y: nx.descendants(g, x) & nx.ancestors(g, y)
        pairs = [(x, y) for x in g for y in g if x != y and (_zero_ok(qtype, g, x, y) if zero else
                                                              y in nx.descendants(g, x) and (qtype not in ("nde", "nie", "cde") or med(x, y)) and
                                                              (qtype != "nde" or g.has_edge(x, y)))]
        if pairs:
            x, y = pairs[rng.integers(len(pairs))]
            return g, x, y, sorted(med(x, y))
    return None


def _statements(qtype, g, x, y, med, t, o, bound):
    """문맥 확률(대상, 조건). 처치 부모가 하나 이하이면 필요한 것만(백도어는 부모), 아니면 전체 확률표."""
    z = sorted(g.predecessors(x))
    yx = [({y: o}, {x: a, **dict(zip(z, bits))}) for a in (1, 0) for bits in np.ndindex(*(2,) * len(z))]
    if qtype in ("ate", "pn", "ps") and len(z) <= 1:
        base = [({v: 1}, {}) for v in z] + yx
    elif qtype == "ett" and len(z) <= 1:
        base = yx + [({z[0]: 1}, {x: 1 - t})] if z else yx
    else:
        base = None
    return base + _joint(x, y, t, o) if qtype in ("pn", "ps") and base is not None else base


def _joint(x, y, t, o):
    """PN·PS 상·하한에 드는 관측 확률 P(y), P(x, y), P(x', y')."""
    return [({y: o}, {}), ({x: t, y: o}, {}), ({x: 1 - t, y: 1 - o}, {})]


def prob_str(lex, target: dict, given: dict, p: float) -> str:
    eq = lambda a: ", ".join(f"{lex.name(v)}={lex.state(v, s)}" for v, s in a.items())
    return f"P({eq(target)}{' | ' + eq(given) if given else ''}) = {p:.4f}"


def sentence(lex, target: dict, given: dict, p: float) -> str:
    be = lambda a: " and ".join(f"{lex.name(v)} being {lex.state(v, s)}" for v, s in a.items())
    return f"For those with {be(given)}, the probability of {be(target)} is {p:.4f}." if given else f"The probability of {be(target)} is {p:.4f}."


def answer_json(qtype: str, val: float) -> dict:
    """정답 JSON. ETT의 PROB는 실제 − 반사실(엔진 value의 반대 부호), 예/아니오는 value > 0이다."""
    prob = -val if qtype == "ett" else val
    out = {"PROB": f"{round(prob, 4) + 0.0:.4f}"}
    return out if qtype in ("pn", "ps") else {"ANSWER": "Yes" if round(val, 4) > 0 else "No"} | out


def gen_prob(qtype, rng, lang, domain):
    zero = qtype in ("ate", "ett", "nde", "nie") and rng.random() < ZERO
    c = _setup(qtype, rng, zero)
    if c is None:
        return None
    g, x, y, med = c
    t, o = int(rng.integers(2)), int(rng.integers(2))
    bound = str(rng.choice(["lower", "upper"])) if qtype in ("pn", "ps") else None
    fix = {m: int(rng.integers(2)) for m in med} if qtype == "cde" else {}
    q = {"query": qtype, "treatment": {x: t}, "baseline": {x: 1 - t}, "outcome": {y: o}, "mediators": med, "fixed": fix, "bound": bound}
    for _ in range(30):
        scm = BinarySCM.random(g, seed=_seed(rng))
        val = evaluate(scm, q)[0]
        joint_ok = min(observed(scm, {x: t, y: o}, {}), observed(scm, {x: 1 - t, y: 1 - o}, {})) >= 0.05
        if zero or (abs(val) >= MARGIN if qtype not in ("pn", "ps") else joint_ok and 0.02 < val < 0.98):
            break
    else:
        return None
    lex = bare_lex(g, rng) if rng.random() < 0.5 else _lex(g, rng, lang, domain)
    if isinstance(lex, Bare):  # 무의미 이름의 상태는 high/low(벤치 표기)
        lex = Lexicon({v: w[:4] + ("high", "low") for v, w in lex.vocab.items()}, "en")
    stmts = [] if zero else _statements(qtype, g, x, y, med, t, o, bound) or cpt_statements(scm) + (_joint(x, y, t, o) if qtype in ("pn", "ps") else [])
    st = lambda a: {lex.name(v): lex.state(v, s) for v, s in a.items()}
    call = {"variables": {lex.name(v): [lex.state(v, 1), lex.state(v, 0)] for v in sorted(g.nodes)},
            "graph": [f"{lex.name(u)} -> {lex.name(v)}" for u, v in sorted(g.edges)],
            "probabilities": [prob_str(lex, tg, gv, round(observed(scm, tg, gv), 4)) for tg, gv in stmts],
            "query": qtype, "treatment": st(q["treatment"]), "baseline": st(q["baseline"]), "outcome": st(q["outcome"])}
    call |= {"mediators": []} if qtype in ("nde", "nie") else {"fixed": st(fix)} if fix else {"bound": bound} if bound else {}
    try:
        tool_val = run(call)[0]
    except ToolError:
        return None
    if abs(tool_val - val) > 0.01 or (qtype not in ("pn", "ps") and not zero and (tool_val > 0) != (val > 0)):
        return None
    d = {"X": lex.name(x), "Y": lex.name(y), "t": lex.state(x, t), "b": lex.state(x, 1 - t), "o": lex.state(y, o), "n": lex.state(y, 1 - o), "bound": bound,
         "fix": " and ".join(f"{lex.name(m)} being {lex.state(m, s)}" for m, s in fix.items())}
    k, order = int(rng.integers(2)), list(nx.topological_sort(g))
    graph = [f"{lex.name(u)[0].upper() + lex.name(u)[1:]} has a direct effect on {lex.name(v)}." for u, v in sorted(g.edges, key=lambda e: (order.index(e[0]), order.index(e[1])))]
    sents = [sentence(lex, tg, gv, round(observed(scm, tg, gv), 4)) for tg, gv in stmts]
    context = f"Input Info: {PREAMBLE['en']} {' '.join(graph)}\n{' '.join(sents)}".rstrip()
    yesno = qtype not in ("pn", "ps")
    question = f"Instruction: Consider the {FULL[qtype]} ({qtype.upper()}) of {d['X']} on {d['Y']}.\nQuestion: {QUESTIONS[qtype][k](d)}\n{FORMATS[yesno][k]}"
    ans = answer_json(qtype, tool_val)
    meta = {"x": x, "y": y, "med": med, "fix": fix, "bound": bound, "zero": zero, "form": k, "cpt": _cpt(scm), "vocab": lex.vocab, "call": call, "answer": ans,
            "style": "prob"}
    label = int(ans["ANSWER"] == "Yes") if yesno else -1
    return Item(qtype, RUNG[qtype], "prob", "en", domain, context, question, label, value=float(ans["PROB"]), graph=list(g.edges), meta=meta)


def recompute(d: dict) -> int:
    """저장한 호출을 다시 돌려 정답을 맞춘다. 값이 저장한 PROB와 다르면 -9."""
    ans = answer_json(d["query_type"], run(d["meta"]["call"])[0])
    return (int(ans["ANSWER"] == "Yes") if "ANSWER" in ans else -1) if ans == d["meta"]["answer"] else -9


GENERATORS_37 = GENERATORS_36 | {f"prob_{k}": (lambda k: lambda rng, lang, domain: gen_prob(k, rng, "en", domain))(k) for k in TYPES}
