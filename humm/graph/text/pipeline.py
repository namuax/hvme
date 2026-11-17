"""Text -> HCG pipeline (T1-T7).

T1 split -> T2/T3 causal claim extraction -> T5 canonicalization -> T6 cross-document aggregation -> T7 counterfactual verification -> HCG (text/LLM evidence) -> L1 cleanup
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
    candidates: pd.DataFrame  # aggregation result (before verification)
    edges: pd.DataFrame  # finally accepted edges
    raw: dict


def text_to_hcg(docs: list[dict], graph_id: str, domain: str, provider, verify: bool = True, min_support: float = 1.0, min_score: float = 0.55,
                verify_threshold: float | None = None, max_chars: int = 1500) -> TextResult:
    """min_score: lower bound on the cross-document aggregation score. At 0.55, at least two different documents (or one assertive + one hedged) must state the same direction.
    verify: attach the counterfactual verification probability as LLM evidence. If verify_threshold is given, candidates with p_fwd below it are dropped.
            The default (None) drops nothing, because the LLM counterfactual judgment reflects world knowledge, not the text."""
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
        ev += [Evidence(kind=EvidenceKind.LLM_PRIOR, score=round(r.p_fwd, 3), method="llm:counterfactual")] if verify and r.p_fwd == r.p_fwd else []  # exclude NaN
        hcg.add_edge(r.src, r.dst, sign=r.sign, evidence=ev)
    hcg.break_cycles()
    for e in hcg.edges:
        e.promote()
    return TextResult(hcg, claims, mapping, cand, edges.reset_index(drop=True), {"extract": raw_extract, "canonical": raw_canon})
