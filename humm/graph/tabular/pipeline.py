"""Tabular data -> HCG pipeline (S1-S5).

S1 profile -> S2 LLM causal order -> S3 PC (background knowledge) -> S4 resolve remaining undirected edges with LLM pairwise queries -> S5 bootstrap frequencies -> HCG -> validation
"""
from dataclasses import dataclass

import networkx as nx
import pandas as pd

from ...validate import ValidationReport, validate_hcg
from ..schema import HCG, Evidence, EvidenceKind
from .pc import CPDAG, Knowledge, edge_frequency, meek, pc, to_dag
from .prior import average_pairwise, knowledge_from, llm_order, llm_pairwise, strong_knowledge
from .profile import describe_variables, profile


@dataclass
class TabularResult:
    hcg: HCG
    cpdag: CPDAG
    dag: nx.DiGraph
    knowledge: Knowledge
    frequency: pd.DataFrame
    pairwise: pd.DataFrame
    report: ValidationReport
    changes: pd.DataFrame
    prior_raw: str
    conflicts: list  # edges where the data-derived direction conflicts with the LLM order (for L3 review)


def orient_with_pairwise(g: CPDAG, pw: pd.DataFrame, threshold: float = 0.6) -> None:
    """Orient only pairs the LLM is at least threshold confident about, then rerun the Meek rules."""
    for r in pw.itertuples():
        if r.p_ab >= threshold:
            g.orient(r.a, r.b, "llm")
        elif r.p_ab <= 1 - threshold:
            g.orient(r.b, r.a, "llm")
    meek(g)


def tabular_to_hcg(df: pd.DataFrame, graph_id: str, domain: str = "", descriptions: dict | None = None, provider=None,
                   alpha: float = 0.01, mode: str = "hybrid", prior: str = "pairwise", n_boot: int = 20, time: dict | None = None,
                   max_cond: int = 3, validate_kw: dict | None = None, tau: float | None = None) -> TabularResult:
    """provider: one provider or a list of providers (pairwise probabilities are averaged).
    prior: 'order'    ask the LLM for the full causal order, then ask pairwise only for edges still undecided (first provider only).
           'pairwise' ask for the direction of every adjacent pair in the PC skeleton and use confident answers as forbidden-direction knowledge (default).
           'both'     use both.
    mode:  'soft' data first, 'hard' prior knowledge first,
           'hybrid' only pairs that all providers are at least tau confident about in the same direction override the data; the rest use soft (default).
    tau:   if unset, 0.99 for one provider and 0.95 for several.
    """
    prof = profile(df)
    empty = pd.DataFrame(columns=["a", "b", "p_ab"])
    providers = list(provider) if isinstance(provider, (list, tuple)) else [provider] * (provider is not None)
    use_order, use_pairs = bool(providers) and prior in ("order", "both"), bool(providers) and prior in ("pairwise", "both")
    tiers, raw = llm_order(providers[0], list(df.columns), domain, describe_variables(prof, descriptions)) if use_order else ([], "")
    g0 = pc(df, alpha, max_cond=max_cond)
    adjacent = sorted({tuple(sorted(e)) for e in g0.undirected} | {tuple(sorted(e)) for e in g0.directed})
    pws = [llm_pairwise(p, adjacent, domain, descriptions) for p in providers] if use_pairs and adjacent else []
    pw = average_pairwise(pws) if pws else empty
    knowledge = knowledge_from(tiers, pw)
    tau = tau or (0.99 if len(providers) == 1 else 0.95)
    strong = strong_knowledge(pws, tau) if mode == "hybrid" and pws else None
    g = pc(df, alpha, knowledge, "hard" if mode == "hard" else "soft", max_cond, strong=strong)
    undecided = sorted(tuple(sorted(e)) for e in g.undirected)
    extra = llm_pairwise(providers[0], undecided, domain, descriptions) if use_order and not use_pairs and undecided else empty
    orient_with_pairwise(g, extra)
    pw = pd.concat([x for x in (pw, extra) if len(x)]) if len(pw) or len(extra) else empty
    dag = to_dag(g, knowledge)
    freq = edge_frequency(df, n_boot, alpha=alpha, knowledge=knowledge, mode=mode, max_cond=max_cond) if n_boot else pd.DataFrame(columns=["src", "dst", "adjacency", "directed"])
    adj = {frozenset((r.src, r.dst)): r.adjacency for r in freq.itertuples()}
    p_llm = {(r.a, r.b): r.p_ab for r in pw.itertuples()} | {(r.b, r.a): 1 - r.p_ab for r in pw.itertuples()}

    hcg = HCG(graph_id=graph_id)
    for r in prof.itertuples():
        hcg.add_node(r.variable, dtype=r.kind, label=(descriptions or {}).get(r.variable, ""))
    for u, v in dag.edges:
        how = g.oriented_by.get((u, v), "data")
        ev = [Evidence(kind=EvidenceKind.DATA, score=round(adj.get(frozenset((u, v)), 0.5), 3), method=f"pc:{how}", ref=f"n={len(df)}")]
        llm_score = round(p_llm.get((u, v), 0.7), 3)
        ev += [Evidence(kind=EvidenceKind.LLM_PRIOR, score=llm_score, method="llm:prior")] if how in ("knowledge", "llm") else []
        agrees = how not in ("knowledge", "llm") and knowledge.prefers(u, v)
        ev += [Evidence(kind=EvidenceKind.LLM_PRIOR, score=llm_score, method="llm:agrees")] if agrees else []
        hcg.add_edge(u, v, evidence=ev)
    conflicts = [(u, v, g.oriented_by.get((u, v), "data")) for u, v in dag.edges if knowledge.prefers(v, u)]
    report, changes = validate_hcg(hcg, df, time=time, **(validate_kw or {}))
    return TabularResult(hcg, g, dag, knowledge, freq, pw, report, changes, raw, conflicts)


def compare(df: pd.DataFrame, true: nx.DiGraph, providers: dict, domain: str = "", descriptions: dict | None = None,
            alpha: float = 0.01, modes=("soft", "hard", "hybrid"), priors=("pairwise",), taus=()) -> pd.DataFrame:
    """Compare 'statistics only' with per-provider, per-strategy prior combinations on the same data. providers: {name: provider or None}."""
    from ...bench.metrics import graph_report
    from .prior import order_agreement, pairwise_agreement

    rows = []
    for name, prov in providers.items():
        runs = [(m, None) for m in modes] + [("hybrid", t) for t in taus]
        for strategy in (priors if prov else ("-",)):
            for mode, tau in (runs if prov else [("-", None)]):
                r = tabular_to_hcg(df, name, domain, descriptions, prov, alpha, mode if prov else "soft", strategy if prov else "pairwise",
                                   n_boot=0, validate_kw={"n_perm": 0}, tau=tau)
                rows.append({"prior": name, "strategy": strategy, "mode": mode if tau is None else f"hybrid τ={tau}", **graph_report(true, r.dag),
                             **{f"order_{k}": v for k, v in order_agreement(r.knowledge.tiers, true).items() if k in ("accuracy", "tiers")},
                             **pairwise_agreement(r.pairwise, true)})
    return pd.DataFrame(rows)
