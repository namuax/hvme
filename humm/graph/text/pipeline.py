"""텍스트 → HCG 파이프라인 (02_proposal.md 4.2, T1~T7).

T1 분할 → T2·T3 인과 주장 추출 → T5 정규화 → T6 문서 간 집계 → T7 반사실 검증 → HCG(텍스트·LLM 증거) → L1 정리
"""
from dataclasses import dataclass

import pandas as pd

from ..schema import HCG, EdgeStatus, Evidence, EvidenceKind, Source
from .aggregate import aggregate
from .canonical import canonicalize
from .extract import chunk, extract_claims
from .verify import verify_edges


@dataclass
class TextResult:
    hcg: HCG
    claims: pd.DataFrame
    mapping: dict
    candidates: pd.DataFrame  # 집계 결과(검증 전)
    edges: pd.DataFrame  # 최종 채택 엣지
    raw: dict


def text_to_hcg(docs: list[dict], graph_id: str, domain: str, provider, verify: bool = True, min_support: float = 1.0, min_score: float = 0.55,
                verify_threshold: float | None = None, max_chars: int = 1500) -> TextResult:
    """min_score: 문서 간 집계 점수 하한. 0.55면 서로 다른 문서 두 곳 이상(또는 단정 1 + 불확실 1)이 같은 방향을 말해야 한다.
    verify: 반사실 검증 확률을 LLM 증거로 붙인다. verify_threshold를 주면 p_fwd가 그보다 낮은 후보를 버린다.
            기본값(None)은 버리지 않는다. LLM 반사실 판단은 텍스트가 아니라 세상 상식을 반영하기 때문이다
            (reports/2026-10-25_text_llm.md)."""
    chunks = chunk(docs, max_chars)
    claims, raw_extract = extract_claims(provider, chunks)
    mapping, raw_canon = canonicalize(provider, list(claims.cause) + list(claims.effect), domain) if len(claims) else ({}, "")
    claims = claims.assign(cause_c=claims.cause.map(mapping), effect_c=claims.effect.map(mapping))
    cand = aggregate(claims, min_support, min_score)
    ver = verify_edges(provider, cand, domain) if verify else pd.DataFrame({"src": cand.src, "dst": cand.dst, "p_fwd": float("nan"), "p_rev": float("nan")})
    edges = cand.merge(ver, on=["src", "dst"])
    edges = edges[edges.p_fwd >= verify_threshold] if verify and verify_threshold is not None else edges

    hcg = HCG(graph_id=graph_id)
    members = {}
    for phrase, name in mapping.items():
        members.setdefault(name, []).append(phrase)
    for name, ms in sorted(members.items()):
        hcg.add_node(name, label=name, aliases=sorted(set(ms) - {name}), source=[Source(kind="text", ref=d) for d in sorted(set(claims[claims.cause_c.eq(name) | claims.effect_c.eq(name)].doc_id))][:5])
    for r in edges.itertuples():
        ev = [Evidence(kind=EvidenceKind.TEXT, score=round(r.support / len(r.docs) * r.score, 3), ref=d, span=q[:200], method="text:claim")
              for d, q in list(zip(r.docs, r.quotes))[:5]]
        ev += [Evidence(kind=EvidenceKind.LLM_PRIOR, score=round(r.p_fwd, 3), method="llm:counterfactual")] if verify and r.p_fwd == r.p_fwd else []  # NaN 제외
        hcg.add_edge(r.src, r.dst, sign=r.sign, evidence=ev)
    hcg.break_cycles()
    for e in hcg.edges:
        e.promote()
    return TextResult(hcg, claims, mapping, cand, edges.reset_index(drop=True), {"extract": raw_extract, "canonical": raw_canon})
