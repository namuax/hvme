"""Hvme-Gym: identification queries that give only a graph: front-door criterion set (frontdoor), instrumental variable (instrument), and identification with unobserved confounding (cei).

Earlier CaLM results were FAS 7, IV 40, CEI 7-40. The model misused adjustment and identify, and
the engine had no front-door/instrument set check and no argument for confounded pairs. frontdoor and instrument queries and a confounded argument (ID algorithm) are added to the engine, and
these items produce call trajectories.

- frontdoor, instrument: multiple choice with exactly one correct option out of three (kind "choice"). At least half of the wrong options are confusable sets.
  Front-door: a mediator blocking only some paths, a mediator with an open back door, a parent of the treatment, a child of the outcome. Instrument: a treatment parent linked to the outcome by another path, a treatment descendant, a non-parent ancestor, the treatment itself.
- An instrument is a parent of the treatment that is d-separated from the outcome once the treatment is removed (DoWhy definition, same as the CaLM IV answer).
- cei: add confounded pairs numbering 0.2-0.8 times the directed edge count and ask whether P(b | do(a)) is identifiable (half yes, half no).
- colbias: CLadder-style collider bias. For a -> c <- b, give the a-b correlation among those with c and ask whether a affects b (the model had called dsep here).
  The answer comes from a path query (is a an ancestor of b). To keep the template from giving the answer away, 30% add an a -> b edge.
Each type has two templates, one modeled on the CaLM template (no benchmark items are used).
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
    """a -> c <- b (30% also a -> b). The question asks whether a affects b given the correlation among those with c."""
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
    """(treatment, outcome, correct set, confusable wrong sets, remaining wrong sets) or None."""
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
                   [{z} for z in sorted(nx.ancestors(g, x) - set(g.predecessors(x)))] + [{x}]  # set iteration follows hash order, so sort for reproducibility across processes
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
    surface = str(rng.choice(["letters", "bare", "named"] if n <= 12 else ["letters", "bare"]))  # domain vocabulary is only sufficient for up to 12 variables
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
