"""합성 텍스트 코퍼스: 정답 DAG에서 문서를 만든다. 텍스트 → 인과 그래프 파이프라인의 정답 있는 평가셋이다.

문서에 섞이는 것
- 변수마다 2~3개의 표현(동의어): 정규화(canonicalization)를 시험한다
- 인과 문형(증가·감소, 능동·수동, 원인 접속 표현), 불확실 표현("~할 수도 있다")
- 방해 문장: 상관 진술(인과 아님), 무효과 진술(엣지 없음), 관계없는 문장
- 오류 문서: 방향을 뒤집거나 없는 엣지를 주장한다(집계 단계가 걸러야 한다)
"""
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from ...gym.verbalize import josa

# (정규 id, 한국어 표현들, 영어 표현들). 도메인마다 그럴듯한 인과 순서(앞 → 뒤)로 나열했다. 정답 엣지는 이 순서를 따른다.
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
    """한국어 문형. a=원인, b=결과."""
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
    graph: nx.DiGraph  # 정규 id 기준 정답 DAG, 엣지 속성 sign
    docs: list  # [{"id", "text", "noisy"}]
    aliases: dict  # 표현 → 정규 id
    annotations: dict = field(default_factory=dict)  # 문장 → 정답 주장(가상 LLM과 오류 분석용)

    @property
    def names(self) -> dict:
        """정규 id → 대표 표현(첫 번째 별칭)."""
        out = {}
        for s, c in self.aliases.items():
            out.setdefault(c, s)
        return out


def make_corpus(domain: str, lang: str = "ko", n_docs: int = 12, n_edges: int = 9, mentions: int = 3, noisy_docs: float = 0.15,
                seed: int = 0, graph: nx.DiGraph | None = None) -> Corpus:
    """graph: 정답 DAG를 직접 준다(엣지 속성 sign). 주지 않으면 도메인 순서를 따르는 엣지 n_edges개를 무작위로 뽑는다."""
    rng = np.random.default_rng(seed)
    vocab = TEXT_VOCAB[domain]
    ids = [v[0] for v in vocab]
    alias = {v[0]: (v[1] if lang == "ko" else v[2]) for v in vocab}
    pairs = [(ids[i], ids[j]) for i in range(len(ids)) for j in range(i + 1, len(ids))]  # 그럴듯한 순서를 따르는 쌍만
    g = nx.DiGraph()
    g.add_nodes_from(ids)
    for k in (rng.permutation(len(pairs))[:n_edges] if graph is None else []):
        g.add_edge(*pairs[k], sign=str(rng.choice(["+", "-"])))
    g.add_edges_from(graph.edges(data=True) if graph is not None else [])
    non_edges = [(a, b) for a in ids for b in ids if a != b and not g.has_edge(a, b) and not g.has_edge(b, a)]
    templ = _ko if lang == "ko" else _en
    def say(kind, a, b, **note):
        """문장 하나를 만들고 정답 주장(표면 표현 포함)을 기록한다."""
        sa, sb = str(rng.choice(alias[a])), str(rng.choice(alias[b]))
        s = str(rng.choice(templ(kind, sa, sb)))
        notes[s] = {"cause": a, "effect": b, "cause_text": sa, "effect_text": sb, "hedged": False, "polarity": "?"} | note
        return s

    docs, notes = [[] for _ in range(n_docs)], {}
    g = nx.relabel_nodes(g, str)
    for a, b, d in g.edges(data=True):  # 참 엣지마다 mentions개 문서에 등장
        for doc in rng.choice(n_docs, mentions, replace=False):
            kind = "hedge" if rng.random() < 0.15 else ("up" if d["sign"] == "+" else "down")
            docs[doc].append(say(kind, a, b, relation="causal", polarity=d["sign"] if kind != "hedge" else "?", hedged=kind == "hedge"))
    for doc in range(n_docs):  # 방해 문장: 상관·무효과·관계없는 문장
        a, b = list(g.edges)[rng.integers(g.number_of_edges())]
        docs[doc].append(say("corr", a, b, relation="correlational"))
        a, b = non_edges[rng.integers(len(non_edges))]
        docs[doc].append(say("none", a, b, relation="no_effect"))
        docs[doc] += list(rng.choice(FILLER[lang], 2, replace=False))
    noisy = set(rng.choice(n_docs, int(round(noisy_docs * n_docs)), replace=False).tolist())
    for doc in noisy:  # 오류 문서: 뒤집힌 엣지 하나, 없는 엣지 하나를 주장한다
        a, b = list(g.edges)[rng.integers(g.number_of_edges())]
        docs[doc].append(say("up", b, a, relation="causal", polarity="+", wrong=True))
        a, b = non_edges[rng.integers(len(non_edges))]
        docs[doc].append(say("up", a, b, relation="causal", polarity="+", wrong=True))
    out = [{"id": f"{domain}-{lang}-doc{i:02d}", "text": " ".join(rng.permutation(sents)), "noisy": i in noisy} for i, sents in enumerate(docs)]
    return Corpus(domain, lang, g, out, {s: c for c, ss in alias.items() for s in ss}, notes)
