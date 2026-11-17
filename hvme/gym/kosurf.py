"""Gym v3.10: Korean CLadder-style surface (nonsense variable names).

KoCLadder is a machine translation of CLadder, so its sentences differ from the Gym Korean templates
('xevo 가 gyzp 에 직접적인 영향을 미칩니다', 'xevo 가 아니면서 gyzp 인 확률은 27% 입니다', '우리는 … 를 유발한다는 것을 안다').
Earlier rounds learned this surface from machine-translated Gym items, which carried translation noise
(swapped method order, mixed state names). Here an English CLadder-style (bare) item is generated first and
its structure (graph, probability statements, rules, query) is rendered into Korean directly from templates.
Labels and engine values are those of the English item, so no translation check is needed. Speech level
(한다 / 합니다), particle spacing and the preamble are sampled per item.

    from hvme.gym import kosurf
    en, ko = kosurf.generate_pairs(3000, seed=1100)
"""
import re

import networkx as nx
import numpy as np

from . import v3
from .dataset import generate
from .queries import Item
from .graphid import GENERATORS_39

PREAMBLES = [
    ("다음 조건들만 있는 자립적이고 가상의 세계를 상상해 보라. 언급되지 않은 어떤 요인이나 인과 관계는 없다.",
     "다음 조건들만 존재하는 단일한 가상의 세계를 상상해 보십시오. 언급되지 않은 어떤 요인이나 인과 관계도 없다고 가정합니다."),
    ("오직 다음 조건들만 존재하고 언급되지 않은 요인이나 인과 관계는 없는, 가상의 세계를 상상해 보자.",
     "단일하고 가상의 세계를 상상해 보며, 오직 다음 조건들만 존재하고 언급되지 않은 어떤 요인이나 인과 관계도 없다고 가정해 봅시다:"),
    ("언급되지 않은 요인이나 인과 관계 없이 다음 조건만 있는 폐쇄적이고 가상의 세계를 상상해 보라:",
     "언급되지 않은 요인이나 인과 관계가 없는, 다음 조건들만 있는 가상의 세계를 상상해 보세요."),
]
P_STMT = re.compile(r"P\((.+?)(?: \| (.+?))?\) = ([\d.]+)")


class Ko:
    """Korean style of one item: fixes the sentence ending (한다 / 합니다) and particle spacing."""

    def __init__(self, rng, names: dict):
        self.polite = bool(rng.random() < 0.5)
        self.sp = " " if rng.random() < 0.7 else ""  # KoCLadder translations usually put a space before particles ('xevo 가')
        self.n = names
        self.cap = bool(rng.random() < 0.5)

    def end(self, plain: str, polite: str) -> str:
        return polite if self.polite else plain

    def j(self, v, kind: str) -> str:  # nonsense names are read as ending in a vowel (가, 는, 를, 와, 로)
        return f"{self.n[v]}{self.sp}{dict(이='가', 은='는', 을='를', 과='와', 로='로', 의='의', 에='에')[kind]}"

    def head(self, v, kind: str) -> str:
        s = self.j(v, kind)
        return s[0].upper() + s[1:] if self.cap else s

    # literal: the name if val is 1, '<name> 가 아님' if val is 0
    def noun(self, v, val):
        return self.n[v] if val else f"{self.j(v, '이')} 아님"

    def adn(self, v, val):  # adnominal: '… 인', '… 가 아닌'
        return f"{self.n[v]}{self.sp}인" if val else f"{self.j(v, '이')} 아닌"

    def conj(self, v, val, last: bool, op="and"):
        if last:
            return self.adn(v, val)
        if op == "or":
            return f"{self.n[v]}{self.sp}이거나" if val else f"{self.j(v, '이')} 아니거나"
        return f"{self.n[v]}{self.sp}이면서" if val else f"{self.j(v, '이')} 아니면서"

    def cond(self, v, val):  # conditional: '… 이면', '… 가 아니면'
        return f"{self.n[v]}{self.sp}이면" if val else f"{self.j(v, '이')} 아니면"

    def obj(self, v, val):  # object: '… 를', '… 가 아님을'
        return f"{self.n[v]}{self.sp}를" if val else f"{self.j(v, '이')} 아님을"

    def subj(self, v, val):  # subject: '… 가', '… 가 아닌 것이'
        return self.j(v, "이") if val else f"{self.j(v, '이')} 아닌 것이"

    def fut(self, v, val):  # future adnominal: '… 일', '… 가 아닐'
        return f"{self.n[v]}{self.sp}일" if val else f"{self.j(v, '이')} 아닐"


def _assign(text: str, inv: dict) -> list[tuple[str, int]]:
    return [(inv[a.split("=")[0]], int(a.split("=")[1] == "yes")) for a in text.split(", ")]


def _graph(k: Ko, g: nx.DiGraph) -> list[str]:
    out = []
    for v in nx.topological_sort(g):
        kids = sorted(g.successors(v))
        if kids:
            objs = ", ".join(k.n[c] for c in kids[:-1])
            objs = f"{objs}{k.sp}와 {k.n[kids[-1]]}" if objs else k.n[kids[-1]]
            out.append(f"{k.head(v, '은' if len(out) % 2 else '이')} {objs}{k.sp}에 직접적인 영향을 {k.end('미친다', '미칩니다')}.")
    return out


def _stmt(k: Ko, target, given, p) -> str:
    pct = f"{round(p * 100)}%"
    be = k.end("이다", "입니다")
    if given:
        g = " ".join(k.conj(v, a, i == len(given) - 1) for i, (v, a) in enumerate(given))
        (v, a), = target
        tgt = f"{k.n[v]}{k.sp}의 확률" if a else f"{k.j(v, '이')} 아닐 확률"
        return f"{g} 사람들의 경우 {tgt}은 {pct}{be}."
    if len(target) == 1:
        (v, a), = target
        return f"{k.n[v]}{k.sp}의 전체 확률은 {pct}{be}." if a else f"전체적으로 {k.j(v, '이')} 아닐 확률은 {pct}{be}."
    t = " ".join(k.conj(v, a, i == len(target) - 1) for i, (v, a) in enumerate(target))
    return f"{t} 확률은 {pct}{be}."


def _question(k: Ko, it) -> str:
    m, q = it.meta, it.meta["q"]
    up = m["direction"] == "up"
    (y, o), = q["outcome"].items()
    if it.query_type == "marginal":
        return f"전체적으로 {k.fut(y, o)} 가능성이 {k.fut(y, 1 - o)} 가능성보다 더 {'높' if up else '낮'}{k.end('은가', '습니까')}?"
    (x, t), = q["treatment"].items()
    if it.query_type == "correlation":
        return f"{k.obj(x, t)} 관찰했을 때 {k.fut(y, o)} 확률이 더 {'큰' if up else '작은'}{k.end('가', '가요')}?"
    if it.query_type == "exp_away":
        (c, cv), = q["given"].items()
        return f"{k.adn(c, cv)} 경우만 볼 때, {k.cond(x, t)} {k.fut(y, o)} 확률이 {k.end('증가하는가' if up else '감소하는가', '증가합니까' if up else '감소합니까')}?"
    if it.query_type == "ate":
        verb = ("증가시키는가", "증가시킬까요") if up else ("감소시키는가", "감소시킬까요")
        return f"{k.subj(x, t)} {k.fut(y, o)} 확률을 {k.end(*verb)}?"
    if it.query_type == "ett":
        return f"{k.adn(x, 1 - t)} 사람들의 경우, 그 개인이 {k.adn(x, t)} 상태였다면 {k.fut(y, o)} 가능성이 더 {'높' if up else '낮'}았{k.end('겠는가', '을까요')}?"
    med = ", ".join(k.n[v] for v in m["med"])
    eff = "긍정적인" if up else "부정적인"
    if it.query_type == "nde":
        return f"{med}{k.sp}를 통한 매개 효과를 무시할 때, {k.subj(x, t)} {k.fut(y, o)} 확률에 {eff} 영향을 {k.end('미치는가', '미칠까요')}?"
    return f"{k.subj(x, t)} {med}{k.sp}를 통해 {k.fut(y, o)} 확률에 {eff} 영향을 {k.end('미치는가', '미칩니까')}?"


def _numeric(it, rng) -> Item:
    vocab = it.meta["vocab"]
    k = Ko(rng, {v: w[3] for v, w in vocab.items()})
    inv = {w[3]: v for v, w in vocab.items()}
    g = nx.DiGraph(it.graph)
    sents = [_stmt(k, _assign(t, inv), _assign(gv, inv) if gv else [], float(p)) for t, gv, p in P_STMT.findall(" ".join(it.meta["call"]["probabilities"]))]
    pre = PREAMBLES[int(rng.integers(len(PREAMBLES)))][int(k.polite)]
    return Item(it.query_type, it.rung, it.kind, "ko", it.domain, pre + " " + " ".join(_graph(k, g) + sents), _question(k, it), it.label, value=it.value,
                graph=it.graph, meta=it.meta | {"style": "kobare"})


def _det_cf(it, rng) -> Item:
    m = it.meta
    k = Ko(rng, {v: w[3] for v, w in m["vocab"].items()})
    g = nx.DiGraph(it.graph)
    g.add_nodes_from(m["fact"])
    order = list(nx.topological_sort(g))
    rules = []
    for v in order:
        if v not in m["rules"]:
            continue
        op, lits = m["rules"][v]
        if len(lits) == 1 and not lits[0][1]:
            rules.append(f"{k.j(lits[0][0], '은')} {k.j(v, '이')} 아님을 유발한다")
        else:
            cause = " ".join(k.conj(p, a, i == len(lits) - 1, op) for i, (p, a) in enumerate(lits))
            rules.append(f"{cause} 것이 {k.j(v, '을')} 유발한다")
    know = "우리는 " + ", 그리고 ".join(rules) + f"는 것을 {k.end('안다', '알고 있습니다')}."
    obs = {v: a for v, a in m["exo"].items() if v != m["x"]}
    seen = (f" 우리는 개인이 {' 그리고 '.join(k.noun(v, a) for v, a in obs.items())}임을 {k.end('관찰했다', '관찰했습니다')}.") if obs else ""
    x, y, fx = m["x"], m["y"], m["fact"][m["x"]]
    q = f"{k.adn(x, fx)} 대신 {k.adn(x, 1 - fx)} 경우라면, 그 개인은 {k.fut(y, m['ask'])}{k.end('까', '까요')}?"
    pre = PREAMBLES[int(rng.integers(len(PREAMBLES)))][int(k.polite)]
    return Item("det_cf", 3, it.kind, "ko", it.domain, pre + " " + " ".join(_graph(k, g)) + " " + know + seen, q, it.label, graph=it.graph,
                meta=m | {"style": "kobare"})


def _backadj(it, rng) -> Item | None:
    m = it.meta
    if "methods" not in m:
        return None
    g = nx.DiGraph(it.graph)
    g.add_nodes_from(m["vocab"])
    lex = v3.bare_lex(g, rng)
    k = Ko(rng, {v: w[3] for v, w in lex.vocab.items()})
    x, y = m["x"], m["y"]
    desc = lambda s: (f"{', '.join(k.n[v] for v in s)}{k.sp}에 따라 이 상관관계를 사례별로 살펴본다." if s
                      else f"{k.j(x, '과')} {k.n[y]} 간의 일반적인 상관관계를 직접 살펴본다.")
    q = (f"방법 1: {desc(m['methods'][0])} 방법 2: {desc(m['methods'][1])} {k.j(x, '이')} {k.n[y]}{k.sp}에 어떻게 영향을 미치는지 이해하려면 "
         f"방법 1{k.sp}을 방법 2{k.sp}보다 사용하는 것이 더 {k.end('정확한가', '정확합니까')}?")
    pre = PREAMBLES[int(rng.integers(len(PREAMBLES)))][int(k.polite)]
    return Item("backadj", 2, it.kind, "ko", it.domain, pre + " " + " ".join(_graph(k, g)), q, it.label, graph=it.graph,
                meta=m | {"vocab": lex.vocab, "style": "kobare"})


MIX_310 = {"det_cf": 0.24, "correlation": 0.14, "backadj": 0.14, "ate": 0.1, "marginal": 0.08, "ett": 0.08, "nie": 0.08, "nde": 0.08, "exp_away": 0.06}
GEN_EN = GENERATORS_39 | {k: (lambda k: lambda rng, lang, domain: v3.numeric(k, rng, "en", domain, bare=1.0))(k) for k in v3.BARE_PHRASES} | {
    "det_cf": lambda rng, lang, domain: v3.gen_det_cf(rng, "en", domain, bare=1.0),
    "backadj": lambda rng, lang, domain: (lambda it: it if it is not None and "methods" in it.meta else None)(v3.gen_backadj(rng, "en", domain))}


def to_korean(it: dict, rng) -> dict | None:
    """English CLadder-style item (dict) -> Korean item (dict)."""
    obj = Item(**{f: it[f] for f in ("query_type", "rung", "kind", "lang", "domain", "context", "question", "label")},
               value=it.get("value"), graph=[tuple(e) for e in it["graph"]], meta=it["meta"])
    ko = (_det_cf if it["query_type"] == "det_cf" else _backadj if it["query_type"] == "backadj" else _numeric)(obj, rng)
    return None if ko is None else ko.to_dict() | {"id": f"ko-{it['id']}_v310", "split": it["split"], "src_id": it["id"], "src_lang": "en"}


def generate_pairs(n: int, seed: int = 1100, split: str = "train") -> tuple[list[dict], list[dict]]:
    en = generate(n, split, seed=seed, mix=MIX_310, generators=GEN_EN, langs=("en",))
    rng = np.random.default_rng(seed + 1)
    ko = [to_korean(d, rng) for d in en]
    keep = [i for i, d in enumerate(ko) if d is not None]
    return [en[i] for i in keep], [ko[i] for i in keep]
