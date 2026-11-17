"""Hvme-Gym: structural queries that give only a graph and ask for parents or ancestors (causes) (query_type "ancestry").

Motivation: CaLM AR, CA FP and FA dropped from 80-87 (base) to 57 after earlier fine-tuning.
The engine had no parent/ancestor query, so the model called dsep on causal_graph and read 'not d-separated' as 'is a cause / is a parent'.
A path query is added to the engine (hvme.lm.tools.run_graph), and these items produce path-call trajectories.

Asked (meta.ask): parent (direct cause, an edge), ancestor (directed path), cause (direct or via other variables = ancestor).
At least half of the negative items use confusable pairs: for parent questions, non-parent ancestors and reversed pairs; for ancestor/cause questions, reversed pairs (b is an ancestor of a) and pairs with a common cause.
Surface: names are letters (A, B, ...), meaningless names or domain vocabulary; edges are an 'A causes B, ...' list or 'A has a direct effect on B.' sentences.
Each type has two or more templates, one modeled on the CaLM template (no benchmark items are used).
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
    """(positive pairs, confusable negative pairs, remaining negative pairs)."""
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
