"""Hume-Gym v3.9 (SFT v2.14): 그래프만 주는 식별 질의 셋. 앞문 기준 집합(frontdoor)·도구변수(instrument)·관측 안 된 교란이 있는 식별(cei).

9B v2.12 CaLM에서 FAS 7·IV 40·CEI 7–40이었다(reports/2026-11-08_hume_lm_9b_sft_v212.md). 모델이 adjustment·identify를 잘못 불렀고,
엔진에는 앞문 기준·도구변수 집합 판정과 교란 쌍을 받는 인자가 없었다. 엔진에 frontdoor·instrument 질의와 confounded 인자(ID 알고리즘)를 더하고,
이 문항으로 호출 궤적을 만든다.

- frontdoor·instrument: 보기 셋 중 하나만 맞는 객관식(kind "choice"). 오답 보기의 절반 이상은 헷갈리는 집합이다.
  앞문: 일부 경로만 가로막는 매개, 뒷문이 열린 매개, 처치의 부모, 결과의 자식. 도구변수: 결과와 다른 길로 이어진 처치의 부모, 처치의 자손, 부모가 아닌 조상, 처치 자신.
- 도구변수는 처치의 부모이면서 처치를 지우면 결과와 d-분리되는 변수다(DoWhy 정의, CaLM IV 정답과 같다).
- cei: 교란 쌍을 방향 엣지 수의 0.2–0.8배 넣고 P(b | do(a))가 식별되는지 묻는다(예/아니오 반반).
- colbias: CLadder식 충돌 편향. a → c ← b에서 c인 사람만 본 a·b 상관을 주고 'a가 b에 영향을 주는가/주지 않는가'를 묻는다(CaLM CB 63, 모델이 dsep을 불렀다).
  답은 path 질의(a가 b의 조상인가)로 정한다. 문형만으로 답이 정해지지 않게 30%는 a → b 엣지를 더한다.
문형은 유형마다 둘이고 하나는 CaLM 문형을 본떴다(벤치 문항은 쓰지 않았다).
"""
from itertools import combinations

import networkx as nx

from ..engine import id_identifiable, is_frontdoor, is_instrument
from ..lm.tools import run_graph
from .ancestry import GENERATORS_38, LETTERS
from .graphs import random_dag
from .queries import Item, _lex
from .v3 import Bare, _graph_text, bare_lex
from .verbalize import PREAMBLE, Lexicon

ASKS = ("frontdoor", "instrument", "cei", "colbias")
QUESTIONS = {
    "frontdoor": ["Which set of variables satisfies the front-door criterion relative to an ordered pair of variables ({a}, {b}) in the above causal graph?",
                  "Which of the following sets can serve as a front-door adjustment set for the effect of {a} on {b}?"],
    "instrument": ["Which set of variables is the instrumental variables relative to an ordered pair of variables ({a}, {b}) in the above causal graph?",
                   "Which of the following can be used as instrumental variables for estimating the effect of {a} on {b}?"],
    "cei": ["Whether the causal effect of {a} on {b} is identified or not?",
            "Can the causal effect of {a} on {b} be identified from the observed variables?"],
    "colbias": ["If we look at {c}, does it mean that {a} affects {b}?", "If we look at {c}, does it mean that {a} does not affect {b}?"],
}


def gen_colbias(rng, domain):
    """a → c ← b(30%는 a → b도). 질문은 c인 사람만 본 상관에서 a가 b에 영향을 주는지 묻는다."""
    g = nx.DiGraph([("a", "c"), ("b", "c")] + ([("a", "b")] if rng.random() < 0.3 else []))
    lex = bare_lex(g, rng) if rng.random() < 0.5 else _lex(g, rng, "en", domain)
    nm, bare = lex.name, isinstance(lex, Bare)
    who = f"those who are {nm('c')}" if bare else f"those for whom {lex.is_('c', 1)}"
    r = float(rng.choice([-1, 1]) * rng.uniform(0.01, 0.3))
    context = ("Imagine a self-contained, hypothetical world with only the following conditions, and without any unmentioned factors or causal relationships: "
               + " ".join(t[0].upper() + t[1:] for t in _graph_text(g, lex)) + f"\nFor {who}, the correlation between {nm('a')} and {nm('b')} is {r:.2f}.")
    form = int(rng.integers(2))
    question = QUESTIONS["colbias"][form].format(a=nm("a"), b=nm("b"), c=who)
    affects = g.has_edge("a", "b")
    label = int(affects if form == 0 else not affects)
    call = {"variables": [nm(v) for v in sorted(g.nodes, key=nm)], "graph": [f"{nm(u)} -> {nm(v)}" for u, v in sorted(g.edges)], "query": "path", "a": nm("a"), "b": nm("b")}
    assert (run_graph(call)[0] != "none") == affects
    meta = {"ask": "colbias", "a": "a", "b": "b", "names": sorted(g.nodes), "bi": [], "sets": [], "vocab": lex.vocab, "surface": "bare" if bare else "named",
            "form": form, "call": call, "style": "graphid"}
    return Item("graphid", 1, "noul", "en", domain, context, question, label, graph=list(g.edges), meta=meta)


def _options(g, ask, rng):
    """(처치, 결과, 맞는 집합, 헷갈리는 오답 집합들, 나머지 오답 집합들) 또는 None."""
    nodes = sorted(g.nodes)
    pairs = [(x, y) for x in nodes for y in nodes if x != y and nx.has_path(g, x, y)]
    rng.shuffle(pairs)
    for x, y in pairs[:20]:
        if ask == "frontdoor":
            mids = sorted(nx.descendants(g, x) & nx.ancestors(g, y))
            cands = [set(c) for k in (1, 2) for c in combinations(mids, k)]
            good = [m for m in cands if is_frontdoor(g, x, y, m) and not any(is_frontdoor(g, x, y, s) for s in cands if s < m)]
            hard = [m for m in cands if not is_frontdoor(g, x, y, m)] + [{p} for p in g.predecessors(x)] + [{c} for c in g.successors(y)]
            ok = lambda m: is_frontdoor(g, x, y, m)
        else:
            good = [{z} for z in sorted(g.predecessors(x)) if is_instrument(g, x, y, z)]
            good += [set(c) for c in combinations([next(iter(z)) for z in good], 2)] if len(good) > 1 and rng.random() < 0.3 else []
            hard = [{z} for z in g.predecessors(x) if not is_instrument(g, x, y, z)] + [{z} for z in sorted(nx.descendants(g, x) - {y})] + \
                   [{z} for z in sorted(nx.ancestors(g, x) - set(g.predecessors(x)))] + [{x}]  # 집합 순회는 해시 순서라 정렬한다(프로세스 간 재현)
            ok = lambda m: all(z in g.predecessors(x) and is_instrument(g, x, y, z) for z in m)
        rest = [{z} for z in nodes if z not in (x, y) and not ok({z})]
        if good and len({frozenset(m) for m in hard + rest}) >= 2:
            return x, y, good[rng.integers(len(good))], hard, rest
    return None


def gen_graphid(ask, rng, lang, domain):
    if ask == "colbias":
        return gen_colbias(rng, domain)
    n = int(rng.integers(5, 16))
    g = random_dag(n, rng, p=float(rng.uniform(0.25, 0.6)), max_parents=4)
    label, bi = None, []
    if ask == "cei":
        nodes = sorted(g.nodes)
        allp = [frozenset(p) for p in combinations(nodes, 2)]
        k = max(1, min(len(allp), round(float(rng.choice([0.2, 0.4, 0.6, 0.8])) * max(g.number_of_edges(), 2))))
        bi = [allp[i] for i in rng.choice(len(allp), k, replace=False)]
        want = int(rng.random() < 0.5)
        pairs = [(x, y) for x in nodes for y in nodes if x != y and nx.has_path(g, x, y)]
        rng.shuffle(pairs)
        hit = next(((x, y) for x, y in pairs[:30] if id_identifiable(g, bi, x, y) == bool(want)), None)
        if hit is None:
            return None
        (x, y), label = hit, want
    else:
        got = _options(g, ask, rng)
        if got is None:
            return None
        x, y, good, hard, rest = got
        wrong, seen = [], {frozenset(good)}
        for _ in range(2):
            pool = [m for m in (hard if hard and rng.random() < 0.6 else rest or hard) if frozenset(m) not in seen]
            pool = pool or [m for m in hard + rest if frozenset(m) not in seen]
            m = pool[rng.integers(len(pool))]
            wrong.append(m)
            seen.add(frozenset(m))
        label = int(rng.integers(3))
        sets = wrong[:label] + [good] + wrong[label:]
    surface = str(rng.choice(["letters", "bare", "named"] if n <= 12 else ["letters", "bare"]))  # 영역 어휘는 12개까지만 넉넉하다
    if surface == "letters":
        lex = Lexicon({v: (L, "예", "아니오", L, "yes", "no") for v, L in zip(sorted(g.nodes, key=lambda _: rng.random()), LETTERS + "OPQ")}, "en")
    else:
        lex = bare_lex(g, rng) if surface == "bare" else _lex(g, rng, "en", domain)
    nm = lex.name
    edges = sorted(g.edges, key=lambda e: (nm(e[0]), nm(e[1])))
    if rng.random() < 0.5:
        text = [f"{nm(u)} causes {nm(v)}" for u, v in edges]
        context = "You will be presented with a causal graph in the following form: " + (", ".join(text[:-1]) + ", and " + text[-1] if len(text) > 1 else text[0]) + "."
    else:
        context = f"{PREAMBLE['en']} " + " ".join(s[0].upper() + s[1:] for s in _graph_text(g, lex))
    conf = [" and ".join(sorted(nm(v) for v in p)) for p in bi]
    if bi:
        context += (" There exist unobserved confounders between: " + ", ".join(conf[:-1]) + (", and " if len(conf) > 1 else "") + conf[-1] + "."
                    if rng.random() < 0.5 else " " + " ".join(f"{c} share an unobserved common cause." for c in conf))
    form = int(rng.integers(2))
    question = QUESTIONS[ask][form].format(a=nm(x), b=nm(y))
    call = {"variables": [nm(v) for v in sorted(g.nodes, key=nm)], "graph": [f"{nm(u)} -> {nm(v)}" for u, v in edges]}
    call |= {"confounded": [" <-> ".join(sorted(nm(v) for v in p)) for p in bi]} if bi else {}
    call |= {"query": "identify" if ask == "cei" else ask, "a": nm(x), "b": nm(y)}
    options = []
    if ask != "cei":
        call["sets"] = [sorted(nm(v) for v in m) for m in sets]
        options = [", ".join(s) for s in call["sets"]]
    verdict = run_graph(call)[0]
    assert (verdict == "identified") == bool(label) if ask == "cei" else verdict.split(", ").index("valid") == label and verdict.count("not valid") == 2
    meta = {"ask": ask, "a": x, "b": y, "names": sorted(g.nodes), "bi": [sorted(p) for p in bi], "sets": [sorted(m) for m in sets] if ask != "cei" else [],
            "vocab": lex.vocab, "surface": surface, "form": form, "call": call, "style": "graphid"}
    return Item("graphid", 2, "noul" if ask == "cei" else "choice", "en", domain, context, question, label, options=options, graph=list(g.edges), meta=meta)


def recompute(d: dict) -> int:
    m = d["meta"]
    g = nx.DiGraph([tuple(e) for e in d["graph"]])
    g.add_nodes_from(m["names"])
    x, y = m["a"], m["b"]
    if m["ask"] == "colbias":
        return int(nx.has_path(g, x, y) if m["form"] == 0 else not nx.has_path(g, x, y))
    if m["ask"] == "cei":
        return int(id_identifiable(g, [frozenset(p) for p in m["bi"]], x, y))
    ok = (lambda s: is_frontdoor(g, x, y, set(s))) if m["ask"] == "frontdoor" else (lambda s: all(g.has_edge(z, x) and is_instrument(g, x, y, z) for z in s))
    valid = [i for i, s in enumerate(m["sets"]) if ok(s)]
    assert len(valid) == 1, valid
    return valid[0]


GENERATORS_39 = GENERATORS_38 | {f"graphid_{k}": (lambda k: lambda rng, lang, domain: gen_graphid(k, rng, "en", domain))(k) for k in ASKS}
