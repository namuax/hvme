"""Synthetic text corpus: generate documents from a ground-truth DAG. An evaluation set with ground truth for the text -> causal graph pipeline.

What is mixed into the documents
- 2-3 expressions (synonyms) per variable: tests canonicalization
- Causal sentence patterns (increase/decrease, active/passive, causal connectives) and hedged expressions ("~할 수도 있다")
- Distractor sentences: correlation statements (not causal), no-effect statements (no edge), irrelevant sentences
- Erroneous documents: flip a direction or claim a nonexistent edge (the aggregation stage must filter these out)
"""
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from ...gym.verbalize import josa

# (canonical id, Korean expressions, English expressions). Listed per domain in a plausible causal order (earlier -> later). Ground-truth edges follow this order.
TEXT_VOCAB = {
    "marketing": [
        ("season", ["계절 수요", "성수기 수요"], ["seasonal demand", "peak-season demand"]),
        ("ad_spend", ["광고비", "광고 예산", "마케팅 지출"], ["ad spending", "advertising budget", "marketing spend"]),
        ("discount", ["할인 행사", "프로모션 횟수"], ["discount events", "promotion frequency"]),
        ("web_traffic", ["웹 방문", "사이트 트래픽", "방문자 수"], ["web traffic", "site visits", "visitor count"]),
        ("conversion", ["구매 전환율", "전환율"], ["conversion rate", "purchase conversion"]),
        ("sales", ["매출", "매출액", "판매액"], ["sales", "revenue", "sales revenue"]),
        ("satisfaction", ["고객 만족도", "만족도"], ["customer satisfaction", "satisfaction score"]),
        ("churn", ["고객 이탈률", "이탈률"], ["customer churn", "churn rate"]),
    ],
    "manufacturing": [
        ("equip_age", ["설비 노후도", "장비 연식"], ["equipment age", "machine age"]),
        ("maintenance", ["예방 정비", "정비 빈도"], ["preventive maintenance", "maintenance frequency"]),
        ("temperature", ["공정 온도", "작업장 온도"], ["process temperature", "shop-floor temperature"]),
        ("vibration", ["설비 진동", "진동 수준"], ["machine vibration", "vibration level"]),
        ("defect_rate", ["불량률", "결함 비율"], ["defect rate", "defective ratio"]),
        ("downtime", ["설비 가동 중단", "다운타임"], ["machine downtime", "unplanned stoppage"]),
        ("output", ["생산량", "산출량"], ["production output", "throughput"]),
        ("delivery_delay", ["납기 지연", "출하 지연"], ["delivery delays", "shipment delays"]),
    ],
    "it_ops": [
        ("deploy", ["배포 빈도", "릴리스 횟수"], ["deployment frequency", "release count"]),
        ("config_change", ["설정 변경", "구성 변경"], ["configuration changes", "config changes"]),
        ("traffic", ["요청 트래픽", "유입 트래픽"], ["request traffic", "incoming load"]),
        ("cpu", ["CPU 사용률", "프로세서 부하"], ["CPU utilization", "processor load"]),
        ("memory_leak", ["메모리 누수", "메모리 누출"], ["memory leaks", "memory leakage"]),
        ("latency", ["응답 지연", "응답 시간"], ["response latency", "response time"]),
        ("error_rate", ["오류율", "에러 비율"], ["error rate", "failure ratio"]),
        ("outage", ["서비스 장애", "장애 건수"], ["service outages", "incident count"]),
    ],
}

FILLER = {
    "ko": ["이번 보고서는 지난 분기 자료를 바탕으로 작성되었다.", "자세한 수치는 부록에 정리했다.", "담당 부서는 다음 달에 후속 회의를 열 예정이다.",
           "자료 수집은 외부 업체와 함께 진행했다.", "일부 항목은 집계 기준이 바뀌어 직접 비교하기 어렵다."],
    "en": ["This report is based on data from the last quarter.", "Detailed figures are listed in the appendix.",
           "The team will hold a follow-up meeting next month.", "Data collection was carried out with an external vendor.",
           "Some items are hard to compare directly because the counting rules changed."],
}


def _ko(kind, a, b):
    """Korean sentence patterns. a = cause, b = effect."""
    return {
        "up": [f"{josa(a, '이')} 늘면 {b}도 증가한다.", f"{josa(a, '은')} {josa(b, '을')} 높인다.", f"{josa(b, '은')} {a} 때문에 올라간다.",
               f"{josa(a, '이')} 증가하면서 {josa(b, '이')} 커졌다.", f"{a} 증가가 {b} 상승으로 이어진다."],
        "down": [f"{josa(a, '은')} {josa(b, '을')} 낮춘다.", f"{josa(a, '이')} 늘수록 {josa(b, '은')} 줄어든다.", f"{a} 증가는 {b} 감소로 이어진다.",
                 f"{josa(b, '은')} {a} 때문에 떨어졌다."],
        "hedge": [f"{josa(a, '이')} {josa(b, '을')} 변화시킬 수도 있다.", f"{josa(a, '이')} {b}에 영향을 줄 가능성이 있다."],
        "corr": [f"{josa(a, '과')} {josa(b, '은')} 함께 움직이는 경향이 있다.", f"{josa(a, '과')} {b} 사이에는 상관관계가 관찰된다."],
        "none": [f"{josa(a, '은')} {b}에 영향을 주지 않는다.", f"분석 결과 {josa(a, '은')} {josa(b, '과')} 무관했다."],
    }[kind]


def _cap(s):
    return s[:1].upper() + s[1:]


def _en(kind, a, b):
    A, B = _cap(a), _cap(b)
    return {
        "up": [f"Higher {a} leads to higher {b}.", f"{A} increases {b}.", f"{B} rises because of {a}.",
               f"As {a} went up, {b} also increased.", f"{B} is driven by {a}."],
        "down": [f"{A} reduces {b}.", f"Higher {a} lowers {b}.", f"{B} falls as {a} increases.", f"{B} dropped because of {a}."],
        "hedge": [f"{A} may change {b}.", f"{A} could possibly affect {b}."],
        "corr": [f"{A} and {b} tend to move together.", f"{A} is correlated with {b}."],
        "none": [f"{A} has no effect on {b}.", f"The analysis found {a} to be unrelated to {b}."],
    }[kind]


@dataclass
class Corpus:
    domain: str
    lang: str
    graph: nx.DiGraph  # ground-truth DAG over canonical ids; edge attribute sign
    docs: list  # [{"id", "text", "noisy"}]
    aliases: dict  # expression -> canonical id
    annotations: dict = field(default_factory=dict)  # sentence -> ground-truth claim (for the simulated LLM and error analysis)

    @property
    def names(self) -> dict:
        """canonical id -> representative expression (first alias)."""
        out = {}
        for s, c in self.aliases.items():
            out.setdefault(c, s)
        return out


def make_corpus(domain: str, lang: str = "ko", n_docs: int = 12, n_edges: int = 9, mentions: int = 3, noisy_docs: float = 0.15,
                seed: int = 0, graph: nx.DiGraph | None = None) -> Corpus:
    """graph: pass the ground-truth DAG directly (edge attribute sign). If not given, n_edges random edges following the domain order are drawn."""
    rng = np.random.default_rng(seed)
    vocab = TEXT_VOCAB[domain]
    ids = [v[0] for v in vocab]
    alias = {v[0]: (v[1] if lang == "ko" else v[2]) for v in vocab}
    pairs = [(ids[i], ids[j]) for i in range(len(ids)) for j in range(i + 1, len(ids))]  # only pairs that follow the plausible order
    g = nx.DiGraph()
    g.add_nodes_from(ids)
    for k in (rng.permutation(len(pairs))[:n_edges] if graph is None else []):
        g.add_edge(*pairs[k], sign=str(rng.choice(["+", "-"])))
    g.add_edges_from(graph.edges(data=True) if graph is not None else [])
    non_edges = [(a, b) for a in ids for b in ids if a != b and not g.has_edge(a, b) and not g.has_edge(b, a)]
    templ = _ko if lang == "ko" else _en
    def say(kind, a, b, **note):
        """Build one sentence and record its ground-truth claim (including surface expressions)."""
        sa, sb = str(rng.choice(alias[a])), str(rng.choice(alias[b]))
        s = str(rng.choice(templ(kind, sa, sb)))
        notes[s] = {"cause": a, "effect": b, "cause_text": sa, "effect_text": sb, "hedged": False, "polarity": "?"} | note
        return s

    docs, notes = [[] for _ in range(n_docs)], {}
    g = nx.relabel_nodes(g, str)
    for a, b, d in g.edges(data=True):  # each true edge appears in `mentions` documents
        for doc in rng.choice(n_docs, mentions, replace=False):
            kind = "hedge" if rng.random() < 0.15 else ("up" if d["sign"] == "+" else "down")
            docs[doc].append(say(kind, a, b, relation="causal", polarity=d["sign"] if kind != "hedge" else "?", hedged=kind == "hedge"))
    for doc in range(n_docs):  # distractor sentences: correlation, no-effect, irrelevant
        a, b = list(g.edges)[rng.integers(g.number_of_edges())]
        docs[doc].append(say("corr", a, b, relation="correlational"))
        a, b = non_edges[rng.integers(len(non_edges))]
        docs[doc].append(say("none", a, b, relation="no_effect"))
        docs[doc] += list(rng.choice(FILLER[lang], 2, replace=False))
    noisy = set(rng.choice(n_docs, int(round(noisy_docs * n_docs)), replace=False).tolist())
    for doc in noisy:  # erroneous documents: claim one flipped edge and one nonexistent edge
        a, b = list(g.edges)[rng.integers(g.number_of_edges())]
        docs[doc].append(say("up", b, a, relation="causal", polarity="+", wrong=True))
        a, b = non_edges[rng.integers(len(non_edges))]
        docs[doc].append(say("up", a, b, relation="causal", polarity="+", wrong=True))
    out = [{"id": f"{domain}-{lang}-doc{i:02d}", "text": " ".join(rng.permutation(sents)), "noisy": i in noisy} for i, sents in enumerate(docs)]
    return Corpus(domain, lang, g, out, {s: c for c, ss in alias.items() for s in ss}, notes)
