"""도구 사용 풀이 궤적 (Step 9 SFT v2). 대화 메시지 목록으로 만들어 채팅 템플릿(tools=TOOLS)으로 학습한다.

- 수치 질의: ① 변수·상태·그래프 → ② 질의 유형과 비교 대상 → ③ 추정식 → causal_engine 호출 → (도구 결과) → ④ 값의 부호로 결론.
  숫자는 모델이 계산하지 않는다. 확률 문장은 문맥의 숫자를 그대로 옮긴다(meta.call).
- 구조 질의(dsep·backadj·identify·corr2cause)와 det_cf: 도구 없이 humm.lm.solutions의 풀이 한 턴.

v2.1: CLadder식 문항(meta.style "bare")은 'X / not X' 표기 규약과 문장 → 확률 인자 대응을 적는다.
오류 복구 궤적(repair)은 첫 호출에 흔한 실수(상태 선언 불일치, 문장식 표기, 엣지 표기)를 넣고 오류 메시지를 받아 고쳐 다시 부른다.

v2.2: Gym v3.3 corr2cause(meta.call)는 causal_discovery 호출 궤적이다. 가설 문장을 도구의 가설 종류로 옮기는 것까지 모델이 하고,
골격·v-구조·동치류와 판정은 도구가 한다. 오류 복구는 관계 하나를 빠뜨린 첫 호출이다.

v2.3: 도구가 둘이 되자 수치 질문에 causal_discovery를 부르는 오류가 생겼다(named). 수치 질의의 오류 복구에 '도구 선택'을 더한다:
확률 문장을 causal_discovery에 넘겨 오류를 받고 causal_engine으로 바꾼다. 도구 없는 CoT판 corr2cause(cot_examples)는 손풀이를 보존한다.

v2.4: v2.3은 오류를 맞게 진단하고도 같은 인자로 다시 불렀다(상태 1/0 문항). 도구 선택 복구는 named 문항에만 두고, 상태 1/0 문항에서
상태를 yes/no로 선언한 실수(digits)와, 진단만 하고 인자를 그대로 다시 넣어 같은 오류를 두 번 받은 뒤 실제로 고치는 궤적(repeat)을 더한다.

v2.5: 그래프만 주어지는 구조 질의(dsep·backadj·identify)는 causal_graph 호출 궤적이다. 모델은 그래프·질의·대상 변수를 옮기고,
경로 점검과 판정은 도구가 한다. 손풀이는 도구 없는 CoT판(cot_examples)으로 보존한다. ate_score 결론에는 값과 구간 경계의 비교를 적는다.
v2.7: named 문항의 처치는 '조건부 확률 문장에서 기준 다음에 나오는 상태'라고 적는다(Gym v3.5, 처치 상태 이름을 맞바꾼 문항과 함께).
v2.8: 처치 변수 상태를 선언과 다른 yes/no로 쓰고, 선언 상태를 알려 주는 엔진 오류를 받아 고치는 복구 궤적(undeclared)을 더한다.
v2.9: 복구 실수 넷을 더한다. 처치 변수를 yes/no로 선언하고 확률에는 본문 상태를 씀(declared), nde·nie 매개 집합 누락(mediators),
결과를 그래프 노드와 다른 이름으로 따로 선언(alias), 조건 하나의 처치 상태를 잘못 옮겨 같은 조건이 두 번 나옴(duplicate).
det_cf는 causal_graph를 잘못 부른 뒤(손실 제외) 오류를 받고 손풀이로 돌아가는 궤적을 더한다(detcf_messages).
v2.13: 부모·조상은 causal_graph path(ancestry_messages). v2.14: 앞문·도구변수·교란 쌍 식별은 frontdoor·instrument·confounded(graphid_messages).
"""
import copy
import re

import numpy as np

from ..gym.check import _scm
from ..gym.formats import render, target_text
from ..gym.queries import SCORE_EDGES
from ..gym.verbalize import Lexicon
from ..gym.formats import LETTERS
from .solutions import KIND_NAME, Writer, jo, solve
from .tools import STMT, format_call, parse_statement, respond, run, run_discovery, run_graph

NUMERIC = ("marginal", "correlation", "ate", "cde", "ett", "nde", "nie", "ate_score", "exp_away")
GRAPH = ("dsep", "backadj", "identify")


def _call_for_score(d: dict) -> dict:
    """ate_score(v2 생성기)는 meta.call이 없어 전체 확률표로 호출 인자를 만든다."""
    from ..gym.v3 import cpt_statements, make_call
    m = d["meta"]
    lex = Lexicon({k: tuple(v) for k, v in m["vocab"].items()}, d["lang"])
    scm = _scm(d)
    q = {"query": "ate", "treatment": {m["x"]: 1}, "baseline": {m["x"]: 0}, "outcome": {m["y"]: 1}}
    return make_call(lex, scm.graph, scm, cpt_statements(scm), q)


def _estimand(qt: str, call: dict, ko: bool) -> str:
    (x, t), = call.get("treatment", {"": ""}).items()
    b = call.get("baseline", {x: ""})[x]
    (y, o), = call["outcome"].items()
    med = ", ".join(call.get("mediators", []))
    fix = ", ".join(f"{k}={v}" for k, v in call.get("fixed", {}).items())
    gv = ", ".join(f"{k}={v}" for k, v in call.get("given", {}).items())
    f = {"marginal": f"P({y}={o}) − 0.5",
         "correlation": f"P({y}={o} | {x}={t}) − P({y}={o} | {x}={b})",
         "exp_away": f"P({y}={o} | {x}={t}, {gv}) − P({y}={o} | {x}={b}, {gv})",
         "ate": f"P({y}={o} | do({x}={t})) − P({y}={o} | do({x}={b}))",
         "cde": f"P({y}={o} | do({x}={t}, {fix})) − P({y}={o} | do({x}={b}, {fix}))",
         "ett": f"P({y}_{{{x}={t}}}={o} | {x}={b}) − P({y}={o} | {x}={b})",
         "nde": f"P({y}_{{{x}={t}, M_{{{x}={b}}}}}={o}) − P({y}_{{{x}={b}}}={o}), M = {{{med}}}",
         "nie": f"P({y}_{{{x}={b}, M_{{{x}={t}}}}}={o}) − P({y}_{{{x}={b}}}={o}), M = {{{med}}}"}[qt]
    note = {"marginal": ("결과 상태의 전체 확률이 절반보다 큰지 본다", "whether the overall probability of the outcome state exceeds one half"),
            "correlation": ("관측(조건부) 확률의 차이이며, 교란 경로를 통한 연관도 포함된다", "a difference of observational conditional probabilities, including association through confounding paths"),
            "exp_away": (f"공통 결과 {gv}인 경우만 보는 관측 확률의 차이이다. 공통 결과로 거르면 서로 독립인 원인끼리도 연관이 생기므로(설명 효과) given에 {gv}를 넣는다",
                         f"a difference of observational probabilities among the units with the common effect {gv}; conditioning on a common effect makes its independent causes associated (explaining away), so {gv} goes into given"),
            "ate": ("개입 분포는 백도어 조정(부모로 조정) 또는 절단 인수분해로 구한다", "the interventional distributions follow from back-door adjustment (on the parents) or the truncated factorization"),
            "cde": ("매개 변수를 개입으로 고정한 두 세계를 비교한다", "compare two worlds with the mediators held fixed by intervention"),
            "ett": (f"실제로 {x}={b}인 경우에 대한 반사실 비교이다", f"a counterfactual comparison among the units that actually have {x}={b}"),
            "nde": ("결과는 처치 세계에서, 매개 변수는 기준 세계의 값으로 둔다", "the outcome under the treatment with the mediators at their baseline values"),
            "nie": ("처치는 기준에 두고 매개 변수만 처치 세계의 값으로 바꾼다", "the treatment stays at baseline while only the mediators take their treatment values")}[qt]
    return f"{f} ({note[0 if ko else 1]})"


REPAIRS = ("states", "phrase", "edge", "tool", "digits")


def _corrupt(call: dict, kind: str, ko: bool) -> tuple[dict, str]:
    """흔한 첫 호출 실수 하나를 넣은 인자와, 오류를 보고 고칠 때 쓸 설명."""
    bad, t = copy.deepcopy(call), (lambda a, b: a if ko else b)
    if kind == "digits":  # 상태 1/0 문항에서 상태를 yes/no로 선언하고 처치·결과도 yes/no로 씀(확률 문장은 1/0 그대로)
        yn = {"1": "yes", "0": "no"}
        bad["variables"] = {v: ["yes", "no"] for v in bad["variables"]}
        bad |= {k: {v: yn[s] for v, s in bad[k].items()} for k in ("treatment", "baseline", "outcome", "fixed") if k in bad}
        return bad, t("확률 문장은 상태를 1/0으로 쓰는데 상태 선언은 [yes, no]로 했다. 모든 변수의 상태 선언을 [1, 0]으로 바꾸고 처치·결과도 1/0으로 쓴다. 확률 문장은 그대로 둔다.",
                      "The probabilities write the states as 1/0, but I declared them as [yes, no]. I declare the states of every variable as [1, 0] "
                      "and write the treatment and outcome with 1/0, keeping the probabilities unchanged.")
    if kind == "states":  # CLadder식으로 상태를 'kwox / not kwox'로 선언하고 확률은 yes/no로 씀
        v = next(iter(bad["outcome"]))
        a, b = bad["variables"][v]
        bad["variables"][v] = [v, f"not {v}"]
        return bad, t(f"{v}의 상태를 [{v}, not {v}]로 선언했지만 확률 문장에서는 {v}={a}처럼 썼다. 상태 선언을 [{a}, {b}]로 맞춘다.",
                      f"I declared the states of {v} as [{v}, not {v}] but wrote {v}={a} in the probabilities. Declare its states as [{a}, {b}] to match.")
    if kind == "undeclared":  # v2.8: 처치 변수 상태를 선언과 다른 yes/no로 씀(CLadder 'smoking=yes', 선언은 smoker / non-smoker)
        (x, _), = bad["treatment"].items()
        a, b = bad["variables"][x]
        yn = {a: "yes", b: "no"}
        bad["probabilities"] = [re.sub(rf"(?<=[(|,])(\s*){re.escape(x)}=({re.escape(a)}|{re.escape(b)})(?=\s*[,|)])", lambda g: f"{g[1]}{x}={yn[g[2]]}", p)
                                for p in bad["probabilities"]]
        bad |= {k: {x: yn[bad[k][x]]} for k in ("treatment", "baseline")}
        return bad, t(f"{x}의 상태는 [{a}, {b}]로 선언했는데 확률 문장과 처치에 yes/no를 썼다. 오류가 알려 준 선언 상태대로 {x}={a}, {x}={b}로 쓴다.",
                      f"I declared the states of {x} as [{a}, {b}] but wrote yes/no in the probabilities and the treatment. As the error says, I use the declared states {x}={a} and {x}={b}.")
    if kind == "declared":  # v2.9: undeclared의 반대. 처치 변수를 yes/no로 선언하고 확률·처치는 본문 상태(CLadder 'manager=sign', 선언은 yes / no)
        (x, _), = bad["treatment"].items()
        a, b = bad["variables"][x]
        bad["variables"][x] = ["yes", "no"]
        return bad, t(f"{x}의 상태를 [yes, no]로 선언했는데 확률 문장과 처치는 {x}={a}, {x}={b}로 썼다. 선언을 문제의 상태 [{a}, {b}]로 바꾼다.",
                      f"I declared the states of {x} as [yes, no] but wrote {x}={a} and {x}={b} in the probabilities and the treatment. I declare its states as the problem's [{a}, {b}].")
    if kind == "alias":  # v2.9: 결과를 그래프 노드와 다른 이름으로 따로 선언(CLadder 'recovering from the disease' / 'disease')
        (y, o), = bad["outcome"].items()
        al = t(f"{y} 결과", f"{y} outcome")
        bad["variables"][al] = bad["variables"][y]
        bad["probabilities"] = [re.sub(rf"(?<=[(|,])(\s*){re.escape(y)}=", lambda g: f"{g[1]}{al}=", p) for p in bad["probabilities"]]
        bad["outcome"] = {al: o}
        return bad, t(f"결과를 그래프 노드 {y}와 다른 이름 '{al}'로 따로 선언해 그래프에 없는 변수가 생겼다. 확률 문장과 결과도 그래프 노드 이름 {y}로 쓰고 '{al}'은 지운다.",
                      f"I declared the outcome under a name, '{al}', that differs from the graph node {y}, so it is in no edge. I write the probabilities and the outcome with the node name {y} and drop '{al}'.")
    if kind == "duplicate":  # v2.9: 조건부 문장 하나의 처치 상태를 잘못 옮겨 같은 조건이 두 번 나옴
        i, j, x, a, b = _duplicate(call)
        bad["probabilities"][j] = re.sub(rf"(?<=[|,])(\s*){re.escape(x)}={re.escape(b)}(?=\s*[,)])", lambda g: f"{g[1]}{x}={a}", call["probabilities"][j])
        return bad, t(f"'{call['probabilities'][i]}'와 '{bad['probabilities'][j]}'는 같은 조건이다. 문제는 조건마다 확률을 한 번만 주므로 두 번째 문장의 상태를 잘못 옮겼다. 문맥을 다시 보면 그 문장의 조건은 {x}={b}이다.",
                      f"'{call['probabilities'][i]}' and '{bad['probabilities'][j]}' have the same condition. The problem gives each condition once, so I copied a state wrongly in the second; rereading the context, its condition is {x}={b}.")
    if kind == "typo":  # v2.10: 무의미 변수명을 옮겨 적다 철자를 틀림(Gym 오답 fjfq → fjq). 선언·확률 문장·엣지 중 한 곳
        names = [v for v in bad["variables"] if any(v in e.split(" -> ") for e in bad["graph"])]
        v = names[sum(map(ord, "".join(names))) % len(names)]
        w, where = _typo(v), sum(map(ord, v)) % 3
        name = lambda x: re.sub(rf"(?<![\w]){re.escape(v)}(?=\s*=)", w, x)
        if where == 0:
            bad["variables"] = {w if k == v else k: s for k, s in bad["variables"].items()}
            bad["probabilities"] = [name(p) for p in bad["probabilities"]]
            bad |= {k: {w if a == v else a: b for a, b in bad[k].items()} for k in ("treatment", "baseline", "outcome", "fixed") if k in bad}
            bad |= {"mediators": [w if m == v else m for m in bad["mediators"]]} if "mediators" in bad else {}
            return bad, t(f"변수 {v}를 {w}로 잘못 옮겨 적어 선언했다. 문제에 나온 대로 {v}로 선언하고 확률 문장에도 {v}로 쓴다.",
                          f"I copied the variable {v} wrongly and declared it as {w}. I declare it as {v}, as in the question, and write {v} in the probabilities too.")
        if where == 1 and any(name(p) != p for p in bad["probabilities"]):
            bad["probabilities"] = [name(p) for p in bad["probabilities"]]
            return bad, t(f"확률 문장의 {w}는 문제의 변수 {v}를 잘못 옮겨 적은 것이다. 그 문장들에 {v}로 쓴다.",
                          f"{w} in the probabilities is the question's variable {v} copied wrongly. I write {v} in those statements.")
        i = next(i for i, e in enumerate(bad["graph"]) if v in e.split(" -> "))
        bad["graph"][i] = " -> ".join(w if x == v else x for x in bad["graph"][i].split(" -> "))
        return bad, t(f"엣지 '{bad['graph'][i]}'에 문제의 변수 {v}를 {w}로 잘못 옮겨 적었다. '{call['graph'][i]}'로 쓴다.",
                      f"In the edge '{bad['graph'][i]}' I copied the question's variable {v} wrongly as {w}. I write '{call['graph'][i]}'.")
    if kind == "phrase":  # name=state 대신 문장처럼 씀
        i = max(range(len(bad["probabilities"])), key=lambda j: len(bad["probabilities"][j]))
        body, num = STMT.match(bad["probabilities"][i]).groups()[:2]
        wrong = re.sub(r"([^,|=]+?)=([^,|]+)", lambda g: f"{g[1]} is {g[2]}", body)
        bad["probabilities"][i] = f"P({wrong}) = {num}"
        return bad, t(f"확률 문장 '{bad['probabilities'][i]}' 대신 name=state 형식인 '{call['probabilities'][i]}'를 쓴다.",
                      f"'{bad['probabilities'][i]}' is not in name=state form. Rewrite it as {call['probabilities'][i]}.")
    bad["graph"][0] = bad["graph"][0].replace("->", "→")
    return bad, t(f"엣지 표기 '{bad['graph'][0]}' 대신 '{call['graph'][0]}'처럼 '->'를 쓴다.",
                  f"Edges must use '->', so '{bad['graph'][0]}' becomes '{call['graph'][0]}'.")


def _typo(v: str) -> str:
    """철자 실수 하나: 글자 빠짐, 글자 끼어듦, 이웃 글자 바꿈, 공백(Gym 오답의 fjq·bczcz·hvjx·cqh x)."""
    k = len(v) // 2
    cands = [v[:k] + v[k + 1:], v[:k] + v[k - 1:], v[:k - 1] + v[k] + v[k - 1] + v[k + 1:], v[:k] + " " + v[k:]]
    i = sum(map(ord, v)) % 4
    return next(w for w in cands[i:] + cands[:i] if w != v)


def _duplicate(call: dict) -> tuple | None:
    """처치 상태만 다르고 값이 다른 두 조건부 문장 (i, j, 처치 변수, i의 상태, j의 상태). 없으면 None."""
    if not call.get("treatment"):
        return None
    (x, _), = call["treatment"].items()
    st = [parse_statement(p, call["variables"]) for p in call["probabilities"]]
    for i, (ti, gi, pi) in enumerate(st):
        for j, (tj, gj, pj) in enumerate(st):
            if i != j and pi != pj and ti == tj and x in gi and x in gj and gi[x] != gj[x] and {k: v for k, v in gi.items() if k != x} == {k: v for k, v in gj.items() if k != x}:
                return i, j, x, *(call["variables"][x][1 - g[x]] for g in (gi, gj))
    return None


def _given(call: dict, sents: list[str]) -> list[str]:
    """CLadder식 문장과 도구 인자의 대응(결합·조건부·주변)."""
    kind = lambda p: "conditional" if "|" in p else "joint" if "," in p.split(")")[0] else "marginal"
    return ["   Given probabilities:"] + [f"   - \"{s}\" → {p} ({kind(p)})" for s, p in zip(sents, call["probabilities"])]


def tool_messages(d: dict, repair: str | None = None, repeat: bool = False) -> list[dict]:
    """수치 질의 하나의 도구 사용 대화: user → assistant(호출) → tool → assistant(결론).
    repair: 첫 호출에 실수를 넣어 user → assistant(잘못된 호출) → tool(오류) → assistant(고친 호출) → tool → assistant(결론).
    repeat: 오류 뒤 진단만 하고 같은 인자로 다시 부른 턴(손실 제외)과 두 번째 오류를 넣고, 그다음 턴에서 실제로 고친다."""
    m, ko = d["meta"], d["lang"] == "ko"
    qt = "ate" if d["query_type"] == "ate_score" else d["query_type"]
    call = m.get("call") or _call_for_score(d)
    t = lambda a, b: a if ko else b
    var = ", ".join(f"{k}({'/'.join(v)})" for k, v in call["variables"].items())
    edges = ", ".join(e.replace(" -> ", " → ") for e in call["graph"])
    (y, o), = call["outcome"].items()
    up = m.get("direction", "up") == "up"
    if qt == "marginal":
        cmp_ = t(f"결과 {y}={o}의 전체 확률", f"the overall probability of the outcome {y}={o}")
    else:
        (x, tv), = call["treatment"].items()
        cmp_ = t(f"처치 {x}={tv}, 기준 {x}={call['baseline'][x]}, 결과 {y}={o}", f"treatment {x}={tv}, baseline {x}={call['baseline'][x]}, outcome {y}={o}")
    k = KIND_NAME[d["query_type"]]
    asks = t(f"질문은 값이 {'양수(높아짐)' if up else '음수(낮아짐)'}인지 묻는다", f"the question asks whether the value is {'positive (an increase)' if up else 'negative (a decrease)'}")
    if d["query_type"] == "ate_score":
        asks = t("질문은 값이 어느 구간에 드는지 묻는다", "the question asks which level the value falls in")
    bare = m.get("style") == "bare"
    note = [f"   Notation: a bare name such as '{y}' means {y}=yes, and 'not {y}' means {y}=no."] if bare else []
    if m.get("form") == "named":  # v3.5: 처치는 조건부 확률 문장에서 기준 다음에 나오는 상태, 결과는 확률 문장이 다루는 상태다
        bv = call["baseline"][x]
        note = [t(f"   질문은 변수 {x}, {y}의 상태를 밝히지 않는다. 조건부 확률 문장은 {x}의 기준 상태를 먼저, 처치 상태를 나중에 쓰므로 처치는 {x}={tv}(기준 {x}={bv})이다. 결과는 확률 문장이 다루는 상태인 {y}={o}이다.",
                  f"   The question names {x} and {y} without states. The conditional probabilities list the baseline state of {x} first and the treatment state second, so the treatment is {x}={tv} (baseline {x}={bv}). The outcome is the state the probabilities are stated for, {y}={o}.")]
    first = "\n".join([
        t(f"1) 변수(상태): {var}. 직접 인과 엣지: {edges}.", f"1) Variables (states): {var}. Direct causal edges: {edges}."), *note,
        t(f"2) 질의 유형: {k[0]}이다. 비교 대상: {cmp_}. {asks}.", f"2) Query type: {k[1]}. Compared: {cmp_}; {asks}."),
        t(f"3) 추정식: {_estimand(qt, call, ko)}. 확률 값이 주어진 수치 질의이므로 문맥의 확률 {len(call['probabilities'])}개를 그대로 넘겨 causal_engine으로 계산한다.",
          f"3) Estimand: {_estimand(qt, call, ko)}. Probabilities are given, so compute it with causal_engine, passing the {len(call['probabilities'])} given probabilities unchanged."),
        *(_given(call, m["sents"]) if bare else []),
    ])
    val, result = run(call)
    if d["query_type"] == "ate_score":
        lvl = int(np.digitize(val, SCORE_EDGES)) + 1
        pp, lo, hi = val * 100, ([None] + [e * 100 for e in SCORE_EDGES])[lvl - 1], ([e * 100 for e in SCORE_EDGES] + [None])[lvl - 1]
        cmp_ko = (f"{lo:+.0f}%p 이상" if hi is None else f"{hi:+.0f}%p 미만" if lo is None else f"{lo:+.0f}%p 이상 {hi:+.0f}%p 미만").replace("-", "−")
        cmp_en = (f"at least {lo:+.0f} pp" if hi is None else f"below {hi:+.0f} pp" if lo is None else f"between {lo:+.0f} and {hi:+.0f} pp").replace("-", "−")
        last = t(f"4) 엔진 값은 {val:+.4f}, 곧 {pp:+.1f}%p이다. 구간 경계는 −30, −10, +10, +30%p이고 {pp:+.1f}%p는 {cmp_ko}이므로 {lvl}단계 '{d['options'][lvl - 1]}'이다.",
                 f"4) The engine value is {val:+.4f}, i.e. {pp:+.1f} pp. The level boundaries are −30, −10, +10 and +30 pp, and {pp:+.1f} pp is {cmp_en}, so it is level {lvl}, '{d['options'][lvl - 1]}'.")
    else:
        holds = (val > 0) == up
        last = t(f"4) 엔진 값은 {val:+.4f}로 {'양수' if val > 0 else '음수'}이다. 질문은 {'높아지는가(양수)' if up else '낮아지는가(음수)'}를 물으므로 답은 '{'예' if holds else '아니오'}'이다.",
                 f"4) The engine value is {val:+.4f}, which is {'positive' if val > 0 else 'negative'}. The question asks about {'an increase (positive)' if up else 'a decrease (negative)'}, so the answer is {'yes' if holds else 'no'}.")
    tc = lambda a, f="causal_engine": [{"type": "function", "function": {"name": f, "arguments": a}}]
    msgs = [{"role": "user", "content": render(d, "tool")}, {"role": "assistant", "content": first, "tool_calls": tc(call)}]
    if repair == "tool":  # 도구 선택 실수: 확률 문장을 causal_discovery의 관계로 넘긴다
        bad = {"variables": list(call["variables"]), "relations": call["probabilities"], "hypothesis": "cause", "a": next(iter(call.get("treatment", call["outcome"]))), "b": y}
        err = respond(format_call(bad, "causal_discovery"))
        why = t("causal_discovery는 상관·독립 관계만 읽는 도구이다. 이 질문은 확률 값이 주어진 수치 질의이므로 causal_engine을 쓴다. "
                "확률 문장은 probabilities 인자로 옮기고 변수 상태, 그래프, 질의 유형, 처치, 결과를 함께 채운다.",
                "causal_discovery only reads correlation and independence relations. This question gives probabilities, so I use causal_engine: "
                "the probability statements go into probabilities, together with the variable states, the graph, the query type, the treatment and the outcome.")
        msgs[1] |= {"tool_calls": tc(bad, "causal_discovery"), "train": False}  # 잘못 고르는 호출 자체는 배우지 않는다
    elif repair:
        bad, why = _corrupt(call, repair, ko)
        err = respond(format_call(bad), repair=False)  # v2.15 엔진 보정(상태 재선언·엣지 보강) 전의 오류로 복구를 배운다
        msgs[1]["tool_calls"] = tc(bad)
    if repair:
        assert err.startswith("error:"), err
        fix = t(f"4) 도구가 오류를 돌려주었다. {why}" + ("" if repair == "tool" else " 인자를 고쳐 다시 호출한다."),
                f"4) The tool returned an error. {why}" + ("" if repair == "tool" else " I call it again with the corrected arguments."))
        msgs += [{"role": "tool", "content": err}]
        if repeat:  # 진단은 맞게 하고 인자는 그대로 둔 재호출(v2.3의 실패 모양, 배우지 않는다) → 같은 오류 → 실제로 고친다
            msgs += [{"role": "assistant", "content": fix, "tool_calls": msgs[1]["tool_calls"], "train": False}, {"role": "tool", "content": err}]
            fix = t(f"5) 같은 오류가 다시 났다. 앞 호출의 인자가 첫 호출과 똑같아 고친 것이 없었다. {why} 이번에는 그 부분을 실제로 바꾼 인자로 호출한다.",
                    f"5) The same error came back: the previous call repeated the first call's arguments unchanged. {why} This time I actually change that part of the arguments.")
        msgs += [{"role": "assistant", "content": fix, "tool_calls": tc(call)}]
        last = ("6" if repeat else "5") + last[1:]
    return msgs + [{"role": "tool", "content": result}, {"role": "assistant", "content": f"{last}\nFinal answer: {target_text(d)}"}]


HYP_MEANING = {
    "direct": ("{a}에서 {b}{r} 가는 직접 엣지가 있다", "a direct edge from {a} to {b}"),
    "via": ("{a}에서 다른 변수를 거쳐 {b}{r} 가는 경로가 있다(직접 엣지가 함께 있어도 된다)", "a directed path from {a} to {b} through another variable (a direct edge may also exist)"),
    "cause": ("{a}에서 {b}{r} 가는 경로가 있다(직접 또는 간접)", "a directed path from {a} to {b}, direct or indirect"),
    "collider": ("{a}{w} {b}{i} 함께 가리키는 변수가 있다", "some variable that both {a} and {b} point into"),
    "confounder": ("{a}{w} {b}의 공통 조상인 다른 변수가 있다", "another variable that is an ancestor of both {a} and {b}"),
}


def discovery_messages(d: dict, repair: bool = False) -> list[dict]:
    """corr2cause(v3.3) 하나의 도구 사용 대화. repair: 첫 호출에서 주변 관계 하나를 빠뜨리고 오류를 받아 고친다."""
    call, ko = d["meta"]["call"], d["lang"] == "ko"
    t = lambda a, b: a if ko else b
    h, a, b = call["hypothesis"], call["a"], call["b"]
    mean = HYP_MEANING[h][0 if ko else 1].format(a=a, b=b, w=jo(a, "과"), r=jo(b, "로"), i=jo(b, "이"))
    first = "\n".join([
        t(f"1) 변수: {', '.join(call['variables'])}. 그래프는 주어지지 않고 통계적 관계만 주어진다.",
          f"1) Variables: {', '.join(call['variables'])}. Only statistical relations are given, not the graph."),
        t(f"2) 질의 유형: {KIND_NAME['corr2cause'][0]}이다. 가설은 '{mean}'를 뜻하므로 causal_discovery의 가설 종류는 {h}({a}, {b})이다. "
          "관계와 맞는 모든 그래프에서 참일 때만 '반드시 참'이다.",
          f"2) Query type: {KIND_NAME['corr2cause'][1]}. The hypothesis means {mean}, so the causal_discovery hypothesis is {h}({a}, {b}). "
          "It is implied only if it holds in every graph consistent with the relations."),
        t(f"3) 전제의 관계 {len(call['relations'])}개를 그대로 causal_discovery에 넘겨 골격, v-구조, 마르코프 동치류를 구하고 가설을 모든 그래프에서 확인한다.",
          f"3) Pass the {len(call['relations'])} relations of the premise unchanged to causal_discovery, which builds the skeleton, the v-structures "
          "and the Markov equivalence class and checks the hypothesis in every graph."),
    ])
    implied, result = run_discovery(call)
    k, n = map(int, re.search(r"holds in (\d+) of (\d+)", result).groups())
    last = t(f"4) 가설은 동치류의 그래프 {n}개 중 {k}개에서 성립한다. {'모든 그래프에서 참이므로 반드시 참이다' if implied else '거짓인 그래프가 있으므로 반드시 참이라고 할 수 없다'}.",
             f"4) The hypothesis holds in {k} of the {n} graphs in the class. {'It holds in every graph, so it is implied' if implied else 'It fails in at least one graph, so it is not implied'}.")
    tc = lambda c: [{"type": "function", "function": {"name": "causal_discovery", "arguments": c}}]
    msgs = [{"role": "user", "content": render(d, "tool")}, {"role": "assistant", "content": first, "tool_calls": tc(call)}]
    if repair:
        marg = [r for r in call["relations"] if "given" not in r and "주어지면" not in r]
        drop = marg[sum(map(ord, d["id"])) % len(marg)]  # 빠뜨리는 위치는 문항마다 다르게(늘 첫 관계면 그 자리를 외운다)
        err = respond(format_call(call | {"relations": [r for r in call["relations"] if r != drop]}, "causal_discovery"))
        assert err.startswith("error:"), err
        msgs[1]["tool_calls"] = tc(call | {"relations": [r for r in call["relations"] if r != drop]})
        msgs += [{"role": "tool", "content": err},
                 {"role": "assistant", "content": t(f"4) 관계 '{drop}'를 빠뜨렸다. 전제의 관계를 모두 넣어 다시 호출한다.",
                                                    f"4) I left out the relation '{drop}'. I call it again with every relation of the premise."), "tool_calls": tc(call)}]
        last = "5" + last[1:]
    return msgs + [{"role": "tool", "content": result}, {"role": "assistant", "content": f"{last}\nFinal answer: {target_text(d)}"}]


def graph_call(d: dict) -> dict:
    """구조 질의의 causal_graph 인자(변수 이름은 문맥의 이름)."""
    w, m, qt = Writer(d), d["meta"], d["query_type"]
    call = {"variables": [w.n(v) for v in sorted(w.g.nodes)], "graph": [f"{w.n(u)} -> {w.n(v)}" for u, v in sorted(w.g.edges)]}
    call |= {"unobserved": [w.n(v) for v in sorted(d["latent"])]} if d.get("latent") else {}
    if qt == "dsep":
        return call | {"query": "dsep", "a": w.n(m["a"]), "b": w.n(m["b"]), "given": [w.n(v) for v in sorted(m["z"])]}
    call |= {"query": "adjustment" if qt == "backadj" else "identify", "a": w.n(m["x"]), "b": w.n(m["y"])}
    return call | ({"sets": [[w.n(v) for v in sorted(z)] for z in (m["methods"] if "methods" in m else [m["z"]])]} if qt == "backadj" else {})


def graph_messages(d: dict, repair: bool = False) -> list[dict]:
    """구조 질의 하나의 causal_graph 사용 대화. repair: 첫 호출의 엣지 표기를 '→'로 써서 오류를 받고 고친다."""
    w, m, qt, ko = Writer(d), d["meta"], d["query_type"], d["lang"] == "ko"
    t = lambda a, b: a if ko else b
    call = graph_call(d)
    what = {"dsep": ("d-분리(조건부 독립) 여부를", "whether they are d-separated (conditionally independent)"),
            "backadj": ("조정 집합의 유효성을", "whether each adjustment set is valid"),
            "identify": ("식별 전략(백도어 → 프론트도어 → 도구변수 순서)을", "the identification strategy (back-door, then front-door, then instrument)")}[qt]
    first = "\n".join(w.head() + [t(f"3) 확률 없이 그래프만 주어진 구조 질의이므로 그래프를 그대로 causal_graph에 넘겨 {what[0]} 판정한다.",
                                   f"3) Only the graph is given, with no probabilities, so I pass it to causal_graph to decide {what[1]}.")])
    verdict, result = run_graph(call)
    if qt == "dsep":
        sep = verdict == "yes"
        last = t(f"4) 도구는 d-분리 여부를 '{verdict}'로 돌려주었다. 따라서 {'독립이다' if sep else '독립이 아니다'}.",
                 f"4) The tool says d-separated: {verdict}. So they are {'independent' if sep else 'not independent'}.")
        if m.get("asks") == "dependent":
            last += t(f" 질문은 의존 여부를 물으므로 답은 '{'아니오' if sep else '예'}'이다.", f" The question asks whether they are dependent, so the answer is {'no' if sep else 'yes'}.")
    elif qt == "backadj" and "methods" in m:
        v1, v2 = verdict.split(", ")
        last = t(f"4) 도구 판정은 방법 1 '{v1}', 방법 2 '{v2}'이다. " + ("방법 1이 유효하고 방법 2는 유효하지 않으므로 방법 1이 더 올바르다." if v1 == "valid" else "방법 1이 유효하지 않으므로 방법 1이 더 올바르다고 할 수 없다."),
                 f"4) The tool says Method 1 is {v1} and Method 2 is {v2}. " + ("Method 1 is valid and Method 2 is not, so Method 1 is more correct." if v1 == "valid" else "Method 1 is not valid, so it is not more correct."))
    elif qt == "backadj":
        last = t(f"4) 도구 판정은 조정 집합 {w.set_(m['z'])}: '{verdict}'이다. 따라서 {'충분하다' if verdict == 'valid' else '충분하지 않다'}.",
                 f"4) The tool says the adjustment set {w.set_(m['z'])} is {verdict}. So it is {'sufficient' if verdict == 'valid' else 'not sufficient'}.")
    else:
        k = d["label"]
        assert m["option_order"][k] == verdict
        last = t(f"4) 도구 판정은 '{result.splitlines()[-1].split(': ')[1]}'이다. 해당하는 보기: {LETTERS[k]}. {d['options'][k]}.",
                 f"4) The tool says identified by: {result.splitlines()[-1].split(': ')[1]}. Matching option: {LETTERS[k]}. {d['options'][k]}.")
    tc = lambda c: [{"type": "function", "function": {"name": "causal_graph", "arguments": c}}]
    msgs = [{"role": "user", "content": render(d, "tool")}, {"role": "assistant", "content": first, "tool_calls": tc(call)}]
    if repair:
        bad = call | {"graph": [e.replace("->", "→") for e in call["graph"]]}
        err = respond(format_call(bad, "causal_graph"))
        assert err.startswith("error:") and call["graph"], err
        msgs[1]["tool_calls"] = tc(bad)
        msgs += [{"role": "tool", "content": err},
                 {"role": "assistant", "content": t(f"4) 엣지 표기 '{bad['graph'][0]}' 대신 '{call['graph'][0]}'처럼 '->'를 써서 다시 호출한다.",
                                                    f"4) Edges must use '->', so '{bad['graph'][0]}' becomes '{call['graph'][0]}'. I call it again."), "tool_calls": tc(call)}]
        last = "5" + last[1:]
    return msgs + [{"role": "tool", "content": result}, {"role": "assistant", "content": f"{last}\nFinal answer: {target_text(d)}"}]


def detcf_messages(d: dict) -> list[dict]:
    """v2.9: det_cf를 causal_graph로 부른 실수(손실 제외) → 오류 → 도구 없이 손풀이."""
    w, m, ko = Writer(d), d["meta"], d["lang"] == "ko"
    t = lambda a, b: a if ko else b
    bad = {"variables": [w.n(v) for v in sorted(w.g.nodes)], "graph": [f"{w.n(u)} -> {w.n(v)}" for u, v in sorted(w.g.edges)],
           "query": "counterfactual", "a": w.n(m["x"]), "b": w.n(m["y"])}
    err = respond(format_call(bad, "causal_graph"))
    assert err.startswith("error:"), err
    first = "\n".join(w.head() + [t("3) 그래프가 주어졌으므로 causal_graph로 반사실을 판정한다.", "3) The graph is given, so I pass it to causal_graph to decide the counterfactual.")])
    fix = t("도구가 오류를 돌려주었다. causal_graph는 dsep, adjustment, identify만 판정한다. 결정론적 반사실은 규칙이 주어졌으므로 도구 없이 규칙을 차례로 적용해 푼다.\n\n",
            "The tool returned an error: causal_graph only decides dsep, adjustment and identify. A deterministic counterfactual comes with its rules, so I solve it without a tool by applying the rules in order.\n\n")
    return [{"role": "user", "content": render(d, "tool")},
            {"role": "assistant", "content": first, "tool_calls": [{"type": "function", "function": {"name": "causal_graph", "arguments": bad}}], "train": False},
            {"role": "tool", "content": err}, {"role": "assistant", "content": fix + solve(d)}]


PROB_ESTIMAND = {
    "ate": "P({y}={o} | do({x}={t})) − P({y}={o} | do({x}={b}))",
    "cde": "P({y}={o} | do({x}={t}, {fix})) − P({y}={o} | do({x}={b}, {fix}))",
    "ett": "E[{y}={o} actual − {y}={o} had {x}={t} | actually {x}={b}] (actual minus counterfactual)",
    "nde": "P({y}_{{{x}={t}, M_{{{x}={b}}}}}={o}) − P({y}_{{{x}={b}}}={o})",
    "nie": "P({y}_{{{x}={b}, M_{{{x}={t}}}}}={o}) − P({y}_{{{x}={b}}}={o})",
    "pn": "the {bound} bound of P({y} would not be {o} had {x}={b} | {x}={t}, {y}={o})",
    "ps": "the {bound} bound of P({y}={o} had {x}={t} | {x}={b}, {y} not {o})",
}


def prob_messages(d: dict) -> list[dict]:
    """v2.12: 값을 JSON으로 묻는 문항(Gym v3.7)의 도구 궤적. ETT의 PROB는 엔진의 'ETT =' 줄(실제 − 반사실), PN·PS는 bound,
    확률이 없는 문항은 빈 확률 목록, 매개 변수를 밝히지 않은 nde·nie는 빈 매개 집합으로 부른다."""
    m, qt, call = d["meta"], d["query_type"], d["meta"]["call"]
    var = ", ".join(f"{k}({'/'.join(v)})" for k, v in call["variables"].items())
    edges = ", ".join(e.replace(" -> ", " → ") for e in call["graph"])
    (x, tv), (y, o) = next(iter(call["treatment"].items())), next(iter(call["outcome"].items()))
    bv = call["baseline"][x]
    fix = ", ".join(f"{k}={v}" for k, v in call.get("fixed", {}).items())
    est = PROB_ESTIMAND[qt].format(x=x, t=tv, b=bv, y=y, o=o, fix=fix, bound=m["bound"])
    cmp_ = {"ett": f"actual {x}={bv}, counterfactual {x}={tv}, outcome {y}={o}", "pn": f"actual {x}={tv} and {y}={o}, counterfactual {x}={bv}",
            "ps": f"actual {x}={bv} and {y} not {o}, counterfactual {x}={tv}, outcome {y}={o}"}.get(qt, f"treatment {x}={tv}, baseline {x}={bv}, outcome {y}={o}")
    data = ("No probabilities are given, so the value follows from the graph alone: I pass an empty list of probabilities." if not call["probabilities"] else
            f"Probabilities are given, so I compute it with causal_engine, passing the {len(call['probabilities'])} given probabilities unchanged.")
    med = " The question does not name the mediators, so I leave mediators empty and the engine uses every variable on a directed path from the treatment to the outcome." \
        if qt in ("nde", "nie") else ""
    first = "\n".join([f"1) Variables (states): {var}. Direct causal edges: {edges or 'none'}.",
                        f"2) Query type: {KIND_NAME[qt][1]}. Compared: {cmp_}.",
                        f"3) Estimand: {est}. {data}{med}"])
    val, result = run(call)
    ans = m["answer"]
    if qt == "ett":
        last = (f"4) The question asks for the ETT value, defined as actual minus counterfactual: the engine's ETT line gives {ans['PROB']}. "
                f"The yes/no answer asks whether the counterfactual world is more likely: value = counterfactual − actual = {val:+.4f}, so the answer is {ans['ANSWER']}.")
    elif qt in ("pn", "ps"):
        last = f"4) The engine returns the {m['bound']} bound {val:.4f}, so PROB is {ans['PROB']}."
    else:
        last = f"4) The engine value is {val:+.4f}, so PROB is {ans['PROB']} and the outcome is {'more' if ans['ANSWER'] == 'Yes' else 'not more'} likely: the answer is {ans['ANSWER']}."
    return [{"role": "user", "content": render(d, "tool")},
            {"role": "assistant", "content": first, "tool_calls": [{"type": "function", "function": {"name": "causal_engine", "arguments": call}}]},
            {"role": "tool", "content": result}, {"role": "assistant", "content": f"{last}\nFinal answer: {target_text(d)}"}]


ASK = {"parent": "whether {a} is a parent (direct cause) of {b}", "ancestor": "whether {a} is an ancestor of {b} (a directed path from {a} to {b})",
       "cause": "whether {a} causes {b}, directly or through other variables, i.e. whether {a} is an ancestor of {b}"}


def ancestry_messages(d: dict) -> list[dict]:
    """v2.13: 부모·조상 질의(Gym v3.8)의 causal_graph path 궤적. d-분리(독립 여부)는 원인 여부가 아니므로 path를 부른다."""
    m, call = d["meta"], d["meta"]["call"]
    a, b = call["a"], call["b"]
    first = "\n".join([f"1) Variables: {', '.join(call['variables'])}. Direct causal edges: {', '.join(e.replace(' -> ', ' → ') for e in call['graph'])}.",
                        f"2) Query type: graph structure, {ASK[m['ask']].format(a=a, b=b)}.",
                        "3) Only the graph is given, so I pass it to causal_graph with query path, which follows the arrows. "
                        "Dependence (d-separation) is not the question here: two variables can be dependent without one causing the other."])
    verdict, result = run_graph(call)
    line = f"{a} is a parent (direct cause) of {b}: {'yes' if verdict == 'parent' else 'no'}" if m["ask"] == "parent" else \
        f"{a} is an ancestor (direct or indirect cause) of {b}: {'no' if verdict == 'none' else 'yes'}"
    last = f"4) The tool says {line}. So the answer is {'Yes' if d['label'] else 'No'}."
    return [{"role": "user", "content": render(d, "tool")},
            {"role": "assistant", "content": first, "tool_calls": [{"type": "function", "function": {"name": "causal_graph", "arguments": call}}]},
            {"role": "tool", "content": result}, {"role": "assistant", "content": f"{last}\nFinal answer: {target_text(d)}"}]


GRAPHID = {"frontdoor": ("which option set satisfies the front-door criterion for the effect of {a} on {b}",
                         "I pass it to causal_graph with query frontdoor and the option sets as `sets`; the engine checks that a set intercepts every directed path "
                         "from {a} to {b} and that no back-door path stays open. A valid adjustment (back-door) set is not the question here."),
           "instrument": ("which option set consists of instrumental variables for the effect of {a} on {b}",
                          "I pass it to causal_graph with query instrument and the option sets as `sets`; the engine checks that each variable is a direct cause of {a} "
                          "and is connected to {b} only through {a}. Identification of the effect is not the question here."),
           "colbias": ("whether {a} affects {b}, i.e. whether {a} is an ancestor of {b}; the correlation is measured only among units selected on a common effect",
                       "I pass it to causal_graph with query path, which follows the arrows. A correlation within a selected group (collider bias) is not evidence that one variable affects the other, so d-separation is not the question."),
           "cei": ("whether the causal effect of {a} on {b} is identified when some pairs share unobserved confounders",
                   "I pass it to causal_graph with query identify and each confounded pair as 'X <-> Y' in `confounded`; the engine runs the complete ID algorithm.")}


def graphid_messages(d: dict) -> list[dict]:
    """v2.14: 앞문·도구변수·교란 쌍 식별(Gym v3.9)의 causal_graph 궤적. 질의를 질문에 맞게 고르고 보기를 sets로, 교란 쌍을 confounded로 옮긴다."""
    m, call = d["meta"], d["meta"]["call"]
    a, b = call["a"], call["b"]
    what, how = (x.format(a=a, b=b) for x in GRAPHID[m["ask"]])
    head = [f"1) Variables: {', '.join(call['variables'])}. Direct causal edges: {', '.join(e.replace(' -> ', ' → ') for e in call['graph'])}."]
    head += [f"   Unobserved confounders: {', '.join(call['confounded'])}."] if call.get("confounded") else []
    head += [f"   Options: " + "; ".join(f"{LETTERS[i]}. {{{o}}}" for i, o in enumerate(d["options"]))] if d["options"] else []
    first = "\n".join(head + [f"2) Query type: graph structure, {what}.", f"3) Only the graph is given, so {how}"])
    verdict, result = run_graph(call)
    if m["ask"] == "colbias":
        anc = verdict != "none"
        last = (f"4) The tool says {a} is an ancestor (direct or indirect cause) of {b}: {'yes' if anc else 'no'}. So {a} {'affects' if anc else 'does not affect'} {b}"
                + ("" if anc else "; the correlation among the selected group comes from conditioning on the common effect") + f". The answer is {'Yes' if d['label'] else 'No'}.")
    elif m["ask"] == "cei":
        last = f"4) The tool says identified: {'yes' if d['label'] else 'no'}. So the answer is {'Yes' if d['label'] else 'No'}."
    else:
        vs, k = verdict.split(", "), d["label"]
        last = f"4) The tool says: " + "; ".join(f"{LETTERS[i]} {{{o}}} {v}" for i, (o, v) in enumerate(zip(d["options"], vs))) + \
               f". Only option {LETTERS[k]} is valid. Matching option: {LETTERS[k]}. {d['options'][k]}."
    return [{"role": "user", "content": render(d, "tool")},
            {"role": "assistant", "content": first, "tool_calls": [{"type": "function", "function": {"name": "causal_graph", "arguments": call}}]},
            {"role": "tool", "content": result}, {"role": "assistant", "content": f"{last}\nFinal answer: {target_text(d)}"}]


def messages(d: dict, repair: str | None = None, repeat: bool = False) -> list[dict]:
    if d["kind"] == "prob":
        return prob_messages(d)
    if d["query_type"] == "ancestry":
        return ancestry_messages(d)
    if d["query_type"] == "graphid":
        return graphid_messages(d)
    if d["query_type"] == "det_cf" and repair:
        return detcf_messages(d)
    if d["query_type"] in GRAPH:
        return graph_messages(d, bool(repair))
    if d["query_type"] == "corr2cause" and "call" in d["meta"]:
        return discovery_messages(d, bool(repair))
    if d["query_type"] in NUMERIC:
        return tool_messages(d, repair, repeat)
    return [{"role": "user", "content": render(d, "tool")}, {"role": "assistant", "content": solve(d)}]


def _repairs(m: dict, v29: bool = True) -> tuple:
    """문항 표면에 맞는 첫 호출 실수 후보. CLadder식은 상태 선언, 상태 1/0은 yes/no 선언, 도구 선택은 이름 문형에만(v2.4).
    v2.8: 처치 상태가 yes/no·1/0이 아니면 선언에 없는 yes/no를 쓰는 실수(undeclared)를 더한다.
    v2.9: 같은 조건에 yes/no 선언(declared), 같은 조건 두 번(duplicate), 결과 다른 이름(alias)을 더한다(v29=False면 v2.8 후보).
    v2.9의 nde·nie 매개 집합 누락(mediators)은 v2.12 엔진이 빈 매개 집합을 경로에서 찾아 오류가 아니게 되어 뺐다."""
    call = m.get("call", {})
    und = ("undeclared", "declared")[:1 + v29] if call.get("treatment") and not set(call["variables"][next(iter(call["treatment"]))]) & {"yes", "no", "1", "0"} else ()
    und += v29 * (("duplicate",) * bool(call and _duplicate(call)) + ("alias",) * bool(call.get("treatment")))  # 고친 엔진은 엣지 없는 변수를 불일치일 때만 짚는다
    if m.get("style") == "bare":
        return ("states", "states", "phrase", "edge", *und)
    if m.get("form") == "named":
        return ("phrase", "edge", "tool", *und)
    if {tuple(w[4:]) for w in m.get("vocab", {}).values()} == {("1", "0")}:
        return ("digits", "digits", "phrase", "edge")
    return ("phrase", "edge", *und)


def tool_examples(items: list[dict], repair: float = 0.0, seed: int = 0, repeat: float = 0.0, v29: bool = True) -> list[dict]:
    """repair: 수치 질의 중 오류 복구 궤적으로 만들 비율(실수 종류는 _repairs). 호출이 두 번 들어가므로 확률 문장이 8개 이하인 문항(최소 문맥)에만 넣는다.
    repeat: 복구 궤적 중 같은 오류를 두 번 받은 뒤 고치는 궤적의 비율(별도 난수라 실수 종류의 흐름은 그대로다).
    v29=False: v2.9 복구 후보와 det_cf 도구 오선택 복구를 빼 v2.8 예제를 그대로 만든다(학습 잡음 측정용)."""
    rng = np.random.default_rng(seed)
    kinds = [str(rng.choice(_repairs(d["meta"], v29)))
             if len(d["meta"].get("call", {}).get("probabilities", [])) <= 8 and rng.random() < repair else None for d in items]
    kinds = [k if d["query_type"] in NUMERIC else "missing" if k and "call" in d["meta"] else "edge" if k and d["query_type"] in GRAPH and d["graph"]
             else "detcf" if k and v29 and d["query_type"] == "det_cf" else None for d, k in zip(items, kinds)]  # v2.9: det_cf는 도구 오선택 복구(난수 흐름은 그대로 둔다)
    twice = np.random.default_rng([seed, 1]).random(len(items)) < repeat
    twice = [bool(r) and k not in (None, "tool", "missing", "detcf") and d["query_type"] in NUMERIC for d, k, r in zip(items, kinds, twice)]
    return [{"id": d["id"], "messages": messages(d, k, r), "tools": True, "task": d["query_type"], "source": "repeat" if r else "repair" if k else "template"}
            for d, k, r in zip(items, kinds, twice)]


def typo_examples(items: list[dict], seed: int = 3, repeat: float = 0.0) -> list[dict]:
    """v2.10: 무의미 변수명 철자 실수 복구 궤적만 만든다(v2.8 예제는 그대로 두고 따로 더한다). repeat: 같은 오류를 두 번 받은 뒤 고치는 궤적의 비율."""
    twice = np.random.default_rng(seed).random(len(items)) < repeat
    return [{"id": d["id"], "messages": tool_messages(d, "typo", bool(r)), "tools": True, "task": d["query_type"], "source": "repeat" if r else "repair"}
            for d, r in zip(items, twice)]


def error_prefix(d: dict, kind: str, repeat: bool = False) -> str:
    """GRPO 3차: 오류 복구 rollout의 앞부분. 실수를 넣은 첫 호출과 엔진 오류(repeat면 진단만 하고 같은 인자로 다시 부른 턴과 두 번째 오류까지)를
    generate_tools 출력 모양으로 만든다. 모델은 그 뒤에서 이어 쓴다."""
    from .tools import format_call
    msgs = tool_messages(d, kind, repeat)[1:5 if repeat else 3]
    return "".join(f"\n<tool_response>\n{m['content']}\n</tool_response>\n" if m["role"] == "tool" else
                   m["content"] + "\n\n" + format_call(m["tool_calls"][0]["function"]["arguments"], m["tool_calls"][0]["function"]["name"]) for m in msgs)


def cot_examples(items: list[dict]) -> list[dict]:
    """도구 없는 CoT 풀이(render 'cot', 도구 목록 없음). 도구 궤적으로 바꾼 유형의 손풀이를 보존한다."""
    return [{"id": d["id"], "messages": [{"role": "user", "content": render(d, "cot")}, {"role": "assistant", "content": solve(d)}], "tools": False,
             "task": d["query_type"], "source": "cot"} for d in items]


def transcript(msgs: list[dict]) -> str:
    """채점용: 어시스턴트 출력과 도구 결과를 이어 붙인 텍스트(마지막 'Final answer'를 읽는다)."""
    from .tools import format_call
    out = []
    for m in msgs[1:]:
        if m["role"] == "assistant":
            out.append(m["content"] + "".join("\n\n" + format_call(c["function"]["arguments"], c["function"]["name"]) for c in m.get("tool_calls", [])))
        else:
            out.append(f"<tool_response>\n{m['content']}\n</tool_response>")
    return "\n".join(out)
