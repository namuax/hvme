"""Render graphs and probability tables as Korean or English sentences."""
from dataclasses import dataclass

import networkx as nx

from ..engine.scm import BinarySCM


def _batchim(word: str) -> int:
    """Final-syllable coda: 0 none, 1 present, 2 rieul coda. Non-Hangul is treated as 0."""
    ch = word[-1]
    code = ord(ch) - 0xAC00
    if not 0 <= code < 11172:  # Uppercase Latin letters follow their Korean pronunciation: L, R end in a rieul coda; M, N have a coda
        return {"L": 2, "R": 2, "M": 1, "N": 1}.get(ch, 0)
    jong = code % 28
    return 2 if jong == 8 else int(jong > 0)


def josa(word: str, kind: str) -> str:
    """kind: one of '은', '이', '을', '과', '로'. Attach the particle matching the final coda."""
    b = _batchim(word)
    table = {"은": ("는", "은", "은"), "이": ("가", "이", "이"), "을": ("를", "을", "을"), "과": ("와", "과", "과"), "로": ("로", "으로", "로")}
    return word + table[kind][b]


PREAMBLE = {
    "ko": "다음 조건만 있는 가상의 세계를 생각하라. 여기에 언급되지 않은 요인이나 인과 관계는 없다.",
    "en": "Imagine a self-contained, hypothetical world with only the following conditions, and without any unmentioned factors or causal relationships:",
}


@dataclass
class Lexicon:
    """Mapping from graph nodes to variable vocabulary, plus sentence fragments."""
    vocab: dict  # node -> (ko_name, ko_on, ko_off, en_name, en_on, en_off)
    lang: str

    def name(self, v):
        return self.vocab[v][0 if self.lang == "ko" else 3]

    def state(self, v, val):
        i = (1 if val else 2) if self.lang == "ko" else (4 if val else 5)
        return self.vocab[v][i]

    def is_(self, v, val):
        """'매출이 '높음'' / 'sales is high'."""
        return f"{josa(self.name(v), '이')} '{self.state(v, val)}'" if self.lang == "ko" else f"{self.name(v)} is {self.state(v, val)}"

    def names(self, vs):
        vs = [self.name(v) for v in vs]
        return ", ".join(vs) if self.lang == "ko" else (" and ".join(vs) if len(vs) < 3 else ", ".join(vs[:-1]) + ", and " + vs[-1])

    def conds(self, assign: dict):
        parts = [self.is_(v, x) for v, x in assign.items()]
        return ", ".join(parts) if self.lang == "ko" else " and ".join(parts)


def describe_graph(g: nx.DiGraph, lex: Lexicon, latent=()) -> list[str]:
    lines = []
    for v in nx.topological_sort(g):
        kids = sorted(g.successors(v))
        if kids:
            lines.append(f"{josa(lex.name(v), '은')} {lex.names(kids)}에 직접 영향을 준다." if lex.lang == "ko"
                         else f"{lex.name(v)} has a direct effect on {lex.names(kids)}.")
    lines += [f"{josa(lex.name(u), '은')} 관측되지 않는다." if lex.lang == "ko" else f"{lex.name(u)} is unobserved." for u in sorted(latent)]
    return lines


def describe_cpt(scm: BinarySCM, lex: Lexicon) -> list[str]:
    """Write each variable's conditional probability table as sentences, including everything needed to compute the answer."""
    lines = []
    for v in scm.order:
        pa = scm.parents[v]
        for k, p in enumerate(scm.cpt[v]):
            pct = f"{round(p * 100)}%"
            assign = {u: (k >> i) & 1 for i, u in enumerate(pa)}
            lines.append((f"{lex.conds(assign)}일 때, {lex.is_(v, 1)}일 확률은 {pct}이다." if pa else f"{lex.is_(v, 1)}일 확률은 {pct}이다.")
                         if lex.lang == "ko" else
                         (f"When {lex.conds(assign)}, the probability that {lex.is_(v, 1)} is {pct}." if pa
                          else f"The probability that {lex.is_(v, 1)} is {pct}."))
    return lines


def context(lex: Lexicon, *blocks: list[str]) -> str:
    return PREAMBLE[lex.lang] + " " + " ".join(s for b in blocks for s in b)
