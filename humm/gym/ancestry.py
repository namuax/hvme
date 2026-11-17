"""Humm-Gym v3.8 (SFT v2.13): 그래프만 주고 부모·조상(원인)을 묻는 구조 질의(query_type "ancestry").

9B 평가표(reports/2026-11-08_humm_lm_9b_sft_v212.md 3.1절)에서 CaLM AR·CA FP·FA가 기반 80–87 → 57로 떨어졌다.
엔진에 부모·조상 질의가 없어 모델이 causal_graph의 dsep을 불러 'd-분리 아님'을 '원인이다·부모다'로 읽었다.
엔진에 path 질의를 더하고(humm.lm.tools.run_graph), 이 문항으로 path 호출 궤적을 만든다.

묻는 것(meta.ask): parent(직접 원인, 엣지), ancestor(방향 경로), cause(직접 또는 다른 변수를 거쳐 원인 = ancestor).
오답 문항의 절반 이상은 헷갈리기 쉬운 쌍이다: 부모 질문은 부모가 아닌 조상과 거꾸로 된 쌍, 조상·원인 질문은 거꾸로 된 쌍(b가 a의 조상)과 공통 원인을 가진 쌍.
표면: 이름은 글자(A, B, …)·무의미 이름·영역 어휘, 엣지는 'A causes B, …' 목록이나 'A has a direct effect on B.' 문장.
문형은 유형마다 둘 이상이고 하나는 CaLM 문형을 본떴다(벤치 문항은 쓰지 않았다).
"""
import networkx as nx

from ..lm.tools import run_graph
from .graphs import random_dag
from .queries import Item, _lex
from .v3 import GENERATORS_36, _graph_text, bare_lex
from .verbalize import PREAMBLE, Lexicon

ASKS = ("parent", "ancestor", "cause")
LETTERS = "ABCDEFGHIJKLMN"
QUESTIONS = {
    "parent": ["Does {a} serve as the parent node of {b}?", "Is {a} a direct cause of {b}?", "Is there a direct edge from {a} to {b}, making {a} a parent of {b}?"],
    "ancestor": ["Does {a} serve as the ancestor node of {b}?", "Is {a} an ancestor of {b}, that is, does a directed path lead from {a} to {b}?"],
    "cause": ["Does {a} cause {b}?", "Does {a} cause {b}, directly or through other variables?"],
}


def _pairs(g, ask):
    """(양성 쌍, 헷갈리는 음성 쌍, 나머지 음성 쌍)."""
    nodes = list(g)
    yes = [(a, b) for a in nodes for b in nodes if a != b and (g.has_edge(a, b) if ask == "parent" else nx.has_path(g, a, b))]
    rev = [(b, a) for a, b in yes]
    if ask == "parent":
        hard = [(a, b) for a in nodes for b in nodes if a != b and nx.has_path(g, a, b) and not g.has_edge(a, b)] + rev
    else:
        hard = rev + [(a, b) for a in nodes for b in nodes if a != b and (a, b) not in yes and (b, a) not in yes
                      and nx.ancestors(g, a) & nx.ancestors(g, b)]
    rest = [(a, b) for a in nodes for b in nodes if a != b and (a, b) not in yes and (a, b) not in hard]
    return yes, hard, rest


def gen_ancestry(ask, rng, lang, domain):
    g = random_dag(int(rng.integers(4, 13)), rng, p=float(rng.uniform(0.3, 0.6)), max_parents=4)
    yes, hard, rest = _pairs(g, ask)
    label = int(rng.random() < 0.5)
    pool = yes if label else hard if hard and (rng.random() < 0.6 or not rest) else rest
    if not pool:
        return None
    a, b = pool[rng.integers(len(pool))]
    surface = str(rng.choice(["letters", "bare", "named"]))
    if surface == "letters":
        lex = Lexicon({v: (L, "예", "아니오", L, "yes", "no") for v, L in zip(sorted(g.nodes, key=lambda _: rng.random()), LETTERS)}, "en")
    else:
        lex = bare_lex(g, rng) if surface == "bare" else _lex(g, rng, "en", domain)
    n = lex.name
    edges = sorted(g.edges, key=lambda e: (n(e[0]), n(e[1])))
    if rng.random() < 0.5:
        text = [f"{n(u)} causes {n(v)}" for u, v in edges]
        graph = (", ".join(text[:-1]) + ", and " + text[-1] if len(text) > 1 else text[0]) + "."
        context = f"Consider a causal graph in which {graph}"
    else:
        context = f"{PREAMBLE['en']} " + " ".join(s[0].upper() + s[1:] for s in _graph_text(g, lex))
    form = int(rng.integers(len(QUESTIONS[ask])))
    question = QUESTIONS[ask][form].format(a=n(a), b=n(b))
    call = {"variables": [n(v) for v in sorted(g.nodes, key=n)], "graph": [f"{n(u)} -> {n(v)}" for u, v in edges], "query": "path", "a": n(a), "b": n(b)}
    verdict = run_graph(call)[0]
    assert (verdict == "parent" if ask == "parent" else verdict != "none") == bool(label)
    meta = {"ask": ask, "a": a, "b": b, "names": sorted(g.nodes), "vocab": lex.vocab, "surface": surface, "form": form, "call": call, "style": "ancestry"}
    return Item("ancestry", 1, "noul", "en", domain, context, question, label, graph=list(g.edges), meta=meta)


def recompute(d: dict) -> int:
    m = d["meta"]
    g = nx.DiGraph([tuple(e) for e in d["graph"]])
    g.add_nodes_from(m["names"])
    return int(g.has_edge(m["a"], m["b"]) if m["ask"] == "parent" else nx.has_path(g, m["a"], m["b"]))


GENERATORS_38 = GENERATORS_36 | {f"ancestry_{k}": (lambda k: lambda rng, lang, domain: gen_ancestry(k, rng, "en", domain))(k) for k in ASKS}
