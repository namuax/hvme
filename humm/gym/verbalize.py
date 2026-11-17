"""그래프와 확률표를 한국어·영어 문장으로 바꾼다."""
from dataclasses import dataclass

import networkx as nx

from ..engine.scm import BinarySCM


def _batchim(word: str) -> int:
    """마지막 글자의 받침: 0 없음, 1 있음, 2 ㄹ받침. 한글이 아니면 0으로 본다."""
    ch = word[-1]
    code = ord(ch) - 0xAC00
    if not 0 <= code < 11172:  # 영문 대문자는 읽는 소리 기준: L·R(엘·알)은 ㄹ받침, M·N(엠·엔)은 받침 있음
        return {"L": 2, "R": 2, "M": 1, "N": 1}.get(ch, 0)
    jong = code % 28
    return 2 if jong == 8 else int(jong > 0)


def josa(word: str, kind: str) -> str:
    """kind: '은', '이', '을', '과', '로' 중 하나. 받침에 맞는 조사를 붙인다."""
    b = _batchim(word)
    table = {"은": ("는", "은", "은"), "이": ("가", "이", "이"), "을": ("를", "을", "을"), "과": ("와", "과", "과"), "로": ("로", "으로", "로")}
    return word + table[kind][b]


PREAMBLE = {
    "ko": "다음 조건만 있는 가상의 세계를 생각하라. 여기에 언급되지 않은 요인이나 인과 관계는 없다.",
    "en": "Imagine a self-contained, hypothetical world with only the following conditions, and without any unmentioned factors or causal relationships:",
}


@dataclass
class Lexicon:
    """그래프 노드 → 변수 어휘 대응과 문장 조각."""
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
    """각 변수의 조건부 확률표를 문장으로 쓴다. 정답 계산에 필요한 정보가 모두 들어간다."""
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
