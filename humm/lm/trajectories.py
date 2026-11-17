"""Tool-use solution trajectories. Built as lists of chat messages and trained with the chat template (tools=TOOLS).

- Numeric queries: 1 variables, states, graph -> 2 query type and comparison target -> 3 estimand -> causal_engine call -> (tool result) -> 4 conclusion from the sign of the value.
  The model does not compute numbers. Probability statements copy the numbers in the context (meta.call).
- CLadder-style items (meta.style "bare") state the 'X / not X' notation convention and the statement -> probability-argument mapping.
- Error-recovery trajectories (repair) put a common mistake in the first call (state declaration mismatch, sentence-style notation, edge notation), receive the error message, fix it and call again.
- Gym corr2cause (meta.call) is a causal_discovery call trajectory. The model translates the hypothesis sentence into the tool's hypothesis kind;
  skeleton, v-structures, equivalence class and verdict come from the tool. Error recovery is a first call that omits one relation.
- With two tools, numeric questions can wrongly call causal_discovery (named items). Recovery for numeric queries adds 'tool choice':
  the probability statements are passed to causal_discovery, the error is received, and the call is switched to causal_engine. The tool-free CoT corr2cause variant (cot_examples) keeps the hand solution.
- A tool-choice recovery that only diagnoses the error and re-sends the same arguments does not fix anything, so tool-choice recovery is limited to named items. For items with states 1/0, add the mistake of
  declaring states as yes/no (digits), and trajectories that diagnose but re-send the same arguments, receive the same error twice and only then really fix it (repeat).
- Structure queries given only a graph (dsep, backadj, identify) are causal_graph call trajectories. The model transcribes the graph, query and target variables;
  path checking and the verdict come from the tool. Hand solutions are kept as the tool-free CoT variant (cot_examples). The ate_score conclusion states the comparison of the value with the bucket boundaries.
- For named items the treatment is described as 'the state that follows the baseline in the conditional probability statements' (together with items whose treatment-state names are swapped).
- Recovery trajectories also cover treatment states written as yes/no that differ from the declaration, with an engine error that reveals the declared states (undeclared).
- More recovery mistakes: treatment declared as yes/no while probabilities use the text's states (declared), a missing mediator set for nde/nie (mediators),
  the outcome declared separately under a name different from the graph node (alias), and a treatment state mistranscribed in one condition so the same condition appears twice (duplicate).
  For det_cf, a trajectory where causal_graph is wrongly called (excluded from the loss), the error is received and the solution returns to the hand solution (detcf_messages).
- Parents/ancestors use causal_graph path (ancestry_messages). Front-door, instrument and confounded-pair identification use frontdoor, instrument and confounded (graphid_messages).
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
    """ate_score items have no meta.call, so the call arguments are built from the full probability table."""
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
    """Arguments with one common first-call mistake, and the explanation to use when fixing it after seeing the error."""
    bad, t = copy.deepcopy(call), (lambda a, b: a if ko else b)
    if kind == "digits":  # on 1/0-state items, declare states as yes/no and write treatment/outcome as yes/no (probability statements stay 1/0)
        yn = {"1": "yes", "0": "no"}
        bad["variables"] = {v: ["yes", "no"] for v in bad["variables"]}
        bad |= {k: {v: yn[s] for v, s in bad[k].items()} for k in ("treatment", "baseline", "outcome", "fixed") if k in bad}
        return bad, t("확률 문장은 상태를 1/0으로 쓰는데 상태 선언은 [yes, no]로 했다. 모든 변수의 상태 선언을 [1, 0]으로 바꾸고 처치·결과도 1/0으로 쓴다. 확률 문장은 그대로 둔다.",
                      "The probabilities write the states as 1/0, but I declared them as [yes, no]. I declare the states of every variable as [1, 0] "
                      "and write the treatment and outcome with 1/0, keeping the probabilities unchanged.")
    if kind == "states":  # declare states CLadder-style as 'kwox / not kwox' but write probabilities with yes/no
        v = next(iter(bad["outcome"]))
        a, b = bad["variables"][v]
        bad["variables"][v] = [v, f"not {v}"]
        return bad, t(f"{v}의 상태를 [{v}, not {v}]로 선언했지만 확률 문장에서는 {v}={a}처럼 썼다. 상태 선언을 [{a}, {b}]로 맞춘다.",
                      f"I declared the states of {v} as [{v}, not {v}] but wrote {v}={a} in the probabilities. Declare its states as [{a}, {b}] to match.")
    if kind == "undeclared":  # treatment state written as yes/no, different from the declaration (CLadder 'smoking=yes', declared smoker / non-smoker)
        (x, _), = bad["treatment"].items()
        a, b = bad["variables"][x]
        yn = {a: "yes", b: "no"}
        bad["probabilities"] = [re.sub(rf"(?<=[(|,])(\s*){re.escape(x)}=({re.escape(a)}|{re.escape(b)})(?=\s*[,|)])", lambda g: f"{g[1]}{x}={yn[g[2]]}", p)
                                for p in bad["probabilities"]]
        bad |= {k: {x: yn[bad[k][x]]} for k in ("treatment", "baseline")}
        return bad, t(f"{x}의 상태는 [{a}, {b}]로 선언했는데 확률 문장과 처치에 yes/no를 썼다. 오류가 알려 준 선언 상태대로 {x}={a}, {x}={b}로 쓴다.",
                      f"I declared the states of {x} as [{a}, {b}] but wrote yes/no in the probabilities and the treatment. As the error says, I use the declared states {x}={a} and {x}={b}.")
    if kind == "declared":  # opposite of undeclared: treatment declared as yes/no while probabilities and treatment use the text's states (CLadder 'manager=sign', declared yes / no)
        (x, _), = bad["treatment"].items()
        a, b = bad["variables"][x]
        bad["variables"][x] = ["yes", "no"]
        return bad, t(f"{x}의 상태를 [yes, no]로 선언했는데 확률 문장과 처치는 {x}={a}, {x}={b}로 썼다. 선언을 문제의 상태 [{a}, {b}]로 바꾼다.",
                      f"I declared the states of {x} as [yes, no] but wrote {x}={a} and {x}={b} in the probabilities and the treatment. I declare its states as the problem's [{a}, {b}].")
    if kind == "alias":  # outcome declared separately under a name different from the graph node (CLadder 'recovering from the disease' / 'disease')
        (y, o), = bad["outcome"].items()
        al = t(f"{y} 결과", f"{y} outcome")
        bad["variables"][al] = bad["variables"][y]
        bad["probabilities"] = [re.sub(rf"(?<=[(|,])(\s*){re.escape(y)}=", lambda g: f"{g[1]}{al}=", p) for p in bad["probabilities"]]
        bad["outcome"] = {al: o}
        return bad, t(f"결과를 그래프 노드 {y}와 다른 이름 '{al}'로 따로 선언해 그래프에 없는 변수가 생겼다. 확률 문장과 결과도 그래프 노드 이름 {y}로 쓰고 '{al}'은 지운다.",
                      f"I declared the outcome under a name, '{al}', that differs from the graph node {y}, so it is in no edge. I write the probabilities and the outcome with the node name {y} and drop '{al}'.")
    if kind == "duplicate":  # one conditional statement's treatment state is mistranscribed, so the same condition appears twice
        i, j, x, a, b = _duplicate(call)
        bad["probabilities"][j] = re.sub(rf"(?<=[|,])(\s*){re.escape(x)}={re.escape(b)}(?=\s*[,)])", lambda g: f"{g[1]}{x}={a}", call["probabilities"][j])
        return bad, t(f"'{call['probabilities'][i]}'와 '{bad['probabilities'][j]}'는 같은 조건이다. 문제는 조건마다 확률을 한 번만 주므로 두 번째 문장의 상태를 잘못 옮겼다. 문맥을 다시 보면 그 문장의 조건은 {x}={b}이다.",
                      f"'{call['probabilities'][i]}' and '{bad['probabilities'][j]}' have the same condition. The problem gives each condition once, so I copied a state wrongly in the second; rereading the context, its condition is {x}={b}.")
    if kind == "typo":  # a meaningless variable name is misspelled while transcribing (fjfq -> fjq); in the declaration, probability statements or one edge
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
    if kind == "phrase":  # written like a sentence instead of name=state
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
    """One spelling mistake: a dropped character, an inserted character, swapped neighbouring characters, or a space."""
    k = len(v) // 2
    cands = [v[:k] + v[k + 1:], v[:k] + v[k - 1:], v[:k - 1] + v[k] + v[k - 1] + v[k + 1:], v[:k] + " " + v[k:]]
    i = sum(map(ord, v)) % 4
    return next(w for w in cands[i:] + cands[:i] if w != v)


def _duplicate(call: dict) -> tuple | None:
    """Two conditional statements that differ only in the treatment state and have different values: (i, j, treatment variable, state of i, state of j). None if there are none."""
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
    """Mapping between CLadder-style statements and tool arguments (joint, conditional, marginal)."""
    kind = lambda p: "conditional" if "|" in p else "joint" if "," in p.split(")")[0] else "marginal"
    return ["   Given probabilities:"] + [f"   - \"{s}\" → {p} ({kind(p)})" for s, p in zip(sents, call["probabilities"])]


def tool_messages(d: dict, repair: str | None = None, repeat: bool = False) -> list[dict]:
    """Tool-use conversation for one numeric query: user -> assistant (call) -> tool -> assistant (conclusion).
    repair: put a mistake in the first call: user -> assistant (wrong call) -> tool (error) -> assistant (fixed call) -> tool -> assistant (conclusion).
    repeat: after the error, add a turn that only diagnoses and re-calls with the same arguments (excluded from the loss) plus a second error, then really fix it in the next turn."""
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
    if m.get("form") == "named":  # the treatment is the state that follows the baseline in conditional probability statements; the outcome is the state the probability statements cover
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
    if repair == "tool":  # tool-choice mistake: pass the probability statements as causal_discovery relations
        bad = {"variables": list(call["variables"]), "relations": call["probabilities"], "hypothesis": "cause", "a": next(iter(call.get("treatment", call["outcome"]))), "b": y}
        err = respond(format_call(bad, "causal_discovery"))
        why = t("causal_discovery는 상관·독립 관계만 읽는 도구이다. 이 질문은 확률 값이 주어진 수치 질의이므로 causal_engine을 쓴다. "
                "확률 문장은 probabilities 인자로 옮기고 변수 상태, 그래프, 질의 유형, 처치, 결과를 함께 채운다.",
                "causal_discovery only reads correlation and independence relations. This question gives probabilities, so I use causal_engine: "
                "the probability statements go into probabilities, together with the variable states, the graph, the query type, the treatment and the outcome.")
        msgs[1] |= {"tool_calls": tc(bad, "causal_discovery"), "train": False}  # the wrongly chosen call itself is not learned
    elif repair:
        bad, why = _corrupt(call, repair, ko)
        err = respond(format_call(bad), repair=False)  # learn recovery from the error as it was before the engine's automatic corrections (state redeclaration, edge completion)
        msgs[1]["tool_calls"] = tc(bad)
    if repair:
        assert err.startswith("error:"), err
        fix = t(f"4) 도구가 오류를 돌려주었다. {why}" + ("" if repair == "tool" else " 인자를 고쳐 다시 호출한다."),
                f"4) The tool returned an error. {why}" + ("" if repair == "tool" else " I call it again with the corrected arguments."))
        msgs += [{"role": "tool", "content": err}]
        if repeat:  # a re-call with a correct diagnosis but unchanged arguments (a failure pattern, not learned) -> same error -> really fix it
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
    """Tool-use conversation for one corr2cause item. repair: the first call omits one marginal relation, receives the error and fixes it."""
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
        drop = marg[sum(map(ord, d["id"])) % len(marg)]  # vary the omitted position per item (if it were always the first relation, the model would memorize that slot)
        err = respond(format_call(call | {"relations": [r for r in call["relations"] if r != drop]}, "causal_discovery"))
        assert err.startswith("error:"), err
        msgs[1]["tool_calls"] = tc(call | {"relations": [r for r in call["relations"] if r != drop]})
        msgs += [{"role": "tool", "content": err},
                 {"role": "assistant", "content": t(f"4) 관계 '{drop}'를 빠뜨렸다. 전제의 관계를 모두 넣어 다시 호출한다.",
                                                    f"4) I left out the relation '{drop}'. I call it again with every relation of the premise."), "tool_calls": tc(call)}]
        last = "5" + last[1:]
    return msgs + [{"role": "tool", "content": result}, {"role": "assistant", "content": f"{last}\nFinal answer: {target_text(d)}"}]


def graph_call(d: dict) -> dict:
    """causal_graph arguments for a structure query (variable names are those in the context)."""
    w, m, qt = Writer(d), d["meta"], d["query_type"]
    call = {"variables": [w.n(v) for v in sorted(w.g.nodes)], "graph": [f"{w.n(u)} -> {w.n(v)}" for u, v in sorted(w.g.edges)]}
    call |= {"unobserved": [w.n(v) for v in sorted(d["latent"])]} if d.get("latent") else {}
    if qt == "dsep":
        return call | {"query": "dsep", "a": w.n(m["a"]), "b": w.n(m["b"]), "given": [w.n(v) for v in sorted(m["z"])]}
    call |= {"query": "adjustment" if qt == "backadj" else "identify", "a": w.n(m["x"]), "b": w.n(m["y"])}
    return call | ({"sets": [[w.n(v) for v in sorted(z)] for z in (m["methods"] if "methods" in m else [m["z"]])]} if qt == "backadj" else {})


def graph_messages(d: dict, repair: bool = False) -> list[dict]:
    """causal_graph conversation for one structure query. repair: the first call writes edges with '→', receives the error and fixes it."""
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
    """Mistake of calling causal_graph for det_cf (excluded from the loss) -> error -> hand solution without tools."""
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
    """Tool trajectory for items that ask for the value as JSON. ETT's PROB is the engine's 'ETT =' line (actual - counterfactual), PN/PS are bounds,
    items without probabilities use an empty probability list, and nde/nie that do not name mediators are called with an empty mediator set."""
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
    """causal_graph path trajectory for parent/ancestor queries. d-separation (independence) is not the same as being a cause, so path is called."""
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
    """causal_graph trajectory for front-door, instrument and confounded-pair identification. The query is chosen to match the question; options go into sets and confounded pairs into confounded."""
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
    """Candidate first-call mistakes that fit the item surface. CLadder-style items get state declaration, 1/0-state items get yes/no declaration, tool choice applies only to named items.
    If the treatment state is not yes/no or 1/0, add the mistake of using an undeclared yes/no (undeclared).
    Also adds yes/no declaration (declared), the same condition twice (duplicate) and a different outcome name (alias); with v29=False only the earlier candidates are used.
    A missing mediator set for nde/nie (mediators) is left out because the engine now finds an empty mediator set from the paths, so it is no longer an error."""
    call = m.get("call", {})
    und = ("undeclared", "declared")[:1 + v29] if call.get("treatment") and not set(call["variables"][next(iter(call["treatment"]))]) & {"yes", "no", "1", "0"} else ()
    und += v29 * (("duplicate",) * bool(call and _duplicate(call)) + ("alias",) * bool(call.get("treatment")))  # the fixed engine flags edge-less variables only on a mismatch
    if m.get("style") == "bare":
        return ("states", "states", "phrase", "edge", *und)
    if m.get("form") == "named":
        return ("phrase", "edge", "tool", *und)
    if {tuple(w[4:]) for w in m.get("vocab", {}).values()} == {("1", "0")}:
        return ("digits", "digits", "phrase", "edge")
    return ("phrase", "edge", *und)


def tool_examples(items: list[dict], repair: float = 0.0, seed: int = 0, repeat: float = 0.0, v29: bool = True) -> list[dict]:
    """repair: fraction of numeric queries to turn into error-recovery trajectories (mistake kinds come from _repairs). The call appears twice, so only items with at most 8 probability statements (minimal context) are used.
    repeat: fraction of recovery trajectories that receive the same error twice before fixing it (uses a separate RNG, so the sequence of mistake kinds is unchanged).
    v29=False: leave out the later recovery candidates and the det_cf tool-misselection recovery to reproduce the earlier examples exactly (for measuring training noise)."""
    rng = np.random.default_rng(seed)
    kinds = [str(rng.choice(_repairs(d["meta"], v29)))
             if len(d["meta"].get("call", {}).get("probabilities", [])) <= 8 and rng.random() < repair else None for d in items]
    kinds = [k if d["query_type"] in NUMERIC else "missing" if k and "call" in d["meta"] else "edge" if k and d["query_type"] in GRAPH and d["graph"]
             else "detcf" if k and v29 and d["query_type"] == "det_cf" else None for d, k in zip(items, kinds)]  # det_cf gets tool-misselection recovery (the RNG sequence is unchanged)
    twice = np.random.default_rng([seed, 1]).random(len(items)) < repeat
    twice = [bool(r) and k not in (None, "tool", "missing", "detcf") and d["query_type"] in NUMERIC for d, k, r in zip(items, kinds, twice)]
    return [{"id": d["id"], "messages": messages(d, k, r), "tools": True, "task": d["query_type"], "source": "repeat" if r else "repair" if k else "template"}
            for d, k, r in zip(items, kinds, twice)]


def typo_examples(items: list[dict], seed: int = 3, repeat: float = 0.0) -> list[dict]:
    """Build only recovery trajectories for misspelled meaningless variable names (added separately, leaving the other examples unchanged). repeat: fraction of trajectories that receive the same error twice before fixing it."""
    twice = np.random.default_rng(seed).random(len(items)) < repeat
    return [{"id": d["id"], "messages": tool_messages(d, "typo", bool(r)), "tools": True, "task": d["query_type"], "source": "repeat" if r else "repair"}
            for d, r in zip(items, twice)]


def error_prefix(d: dict, kind: str, repeat: bool = False) -> str:
    """Prefix of an error-recovery rollout for GRPO: the first call containing a mistake and the engine error (with repeat, also the diagnose-only turn that re-calls with the same arguments and the second error),
    in the shape of generate_tools output. The model continues after it."""
    from .tools import format_call
    msgs = tool_messages(d, kind, repeat)[1:5 if repeat else 3]
    return "".join(f"\n<tool_response>\n{m['content']}\n</tool_response>\n" if m["role"] == "tool" else
                   m["content"] + "\n\n" + format_call(m["tool_calls"][0]["function"]["arguments"], m["tool_calls"][0]["function"]["name"]) for m in msgs)


def cot_examples(items: list[dict]) -> list[dict]:
    """Tool-free CoT solutions (render 'cot', no tool list). Preserves the hand solutions of types that were converted to tool trajectories."""
    return [{"id": d["id"], "messages": [{"role": "user", "content": render(d, "cot")}, {"role": "assistant", "content": solve(d)}], "tools": False,
             "task": d["query_type"], "source": "cot"} for d in items]


def transcript(msgs: list[dict]) -> str:
    """For scoring: assistant output and tool results concatenated (the last 'Final answer' is read)."""
    from .tools import format_call
    out = []
    for m in msgs[1:]:
        if m["role"] == "assistant":
            out.append(m["content"] + "".join("\n\n" + format_call(c["function"]["arguments"], c["function"]["name"]) for c in m.get("tool_calls", [])))
        else:
            out.append(f"<tool_response>\n{m['content']}\n</tool_response>")
    return "\n".join(out)
