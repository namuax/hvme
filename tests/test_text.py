import networkx as nx
import pandas as pd
import pytest

from humm.engine import BinarySCM
from humm.graph import EdgeStatus, Evidence, EvidenceKind, HCG
from humm.graph.text import (TEXT_VOCAB, TextOracle, aggregate, canonicalize, chunk, evaluate, link_nodes, make_corpus, merge_hcg, parse_claims,
                             text_to_hcg)


@pytest.fixture(scope="module")
def corpus():
    return make_corpus("marketing", "ko", seed=1)


def test_corpus_is_plausible_and_annotated(corpus):
    order = [v[0] for v in TEXT_VOCAB["marketing"]]
    assert all(order.index(a) < order.index(b) for a, b in corpus.graph.edges)  # true edges follow a plausible order
    kinds = {n["relation"] for n in corpus.annotations.values()}
    assert kinds == {"causal", "correlational", "no_effect"}
    assert any(n.get("wrong") for n in corpus.annotations.values()) and any(d["noisy"] for d in corpus.docs)
    assert all(n["cause_text"] in s for s, n in corpus.annotations.items())


def test_parse_claims_and_chunk():
    raw = '```json\n{"claims": [{"cause": "광고비", "effect": "매출", "relation": "causal", "polarity": "+", "hedged": false, "quote": "q"},' \
          '{"cause": "x", "effect": "y", "relation": "maybe"}]}\n```'
    assert [c["cause"] for c in parse_claims(raw)] == ["광고비"] and parse_claims("no json") == []
    chunks = chunk([{"id": "d", "text": "A causes B. " * 50}], max_chars=200)
    assert len(chunks) > 1 and all(len(c["text"]) <= 200 for c in chunks)


def test_aggregate_rules():
    rows = [("d1", "A", "B", "causal", "+", False), ("d1", "A", "B", "causal", "+", False), ("d2", "A", "B", "causal", "+", True),
            ("d3", "B", "A", "causal", "+", False), ("d4", "A", "B", "no_effect", "?", False), ("d5", "C", "D", "correlational", "?", False)]
    c = pd.DataFrame(rows, columns=["doc_id", "cause_c", "effect_c", "relation", "polarity", "hedged"]).assign(quote="q")
    e = aggregate(c, min_score=0.0)
    r = e.set_index(["src", "dst"]).loc[("A", "B")]
    assert r.support == 1.5 and r.contra == 2 and r.score == pytest.approx(1.5 / 4.5)  # duplicates within one document count once, hedged claims count 0.5
    assert ("C", "D") not in set(zip(e.src, e.dst))  # a correlation claim alone does not create an edge


def test_canonicalize_with_oracle(corpus):
    phrases = [n["cause_text"] for n in corpus.annotations.values()]
    mapping, _ = canonicalize(TextOracle(corpus), phrases, "marketing")
    by_id = {}
    for p, name in mapping.items():
        by_id.setdefault(corpus.aliases[p], set()).add(name)
    assert all(len(names) == 1 for names in by_id.values())


def test_pipeline_perfect_oracle_recovers_graph(corpus):
    r = text_to_hcg(corpus.docs, "m", "marketing", TextOracle(corpus, 0.0))
    m = evaluate(list(zip(r.edges.src, r.edges.dst)), r.mapping, corpus)
    assert m["precision"] == 1.0 and m["recall"] >= 0.85 and m["group_purity"] == 1.0  # one true edge contradicted by an erroneous document may hit the score floor
    loose = text_to_hcg(corpus.docs, "m", "marketing", TextOracle(corpus, 0.0), verify=False, min_score=0.4)
    assert evaluate(list(zip(loose.edges.src, loose.edges.dst)), loose.mapping, corpus)["precision"] < 1.0  # a single-document erroneous claim survives
    assert {"llm:counterfactual"} <= {x.method for e in r.hcg.edges for x in e.evidence}  # verification does not filter; it is attached as evidence


def test_clean_strips_change_words():
    from humm.graph.text.extract import clean
    assert [clean(x) for x in ["Higher visitor count", "성수기 수요 증가", "'매출'", "lower"]] == ["visitor count", "성수기 수요", "매출", "lower"]


def test_merge_resolves_direction_by_direction_evidence():
    table = HCG(graph_id="t")
    for v in ("x", "y"):
        table.add_node(v)
    table.add_edge("y", "x", evidence=[Evidence(kind=EvidenceKind.DATA, score=0.9, method="pc:extension"), Evidence(kind=EvidenceKind.DATA, score=0.9, method="ci:g_test")])
    text = HCG(graph_id="s")
    for v in ("X var", "Y var"):
        text.add_node(v)
    text.add_edge("X var", "Y var", evidence=[Evidence(kind=EvidenceKind.TEXT, score=0.6), Evidence(kind=EvidenceKind.LLM_PRIOR, score=0.8)])
    m = merge_hcg(text, table, {"X var": "x", "Y var": "y"})
    assert m.edge("y", "x").status == EdgeStatus.REFUTED and m.edge("x", "y").status == EdgeStatus.SUPPORTED
    assert {x.method for x in m.edge("x", "y").evidence} >= {"ci:g_test"}  # adjacency evidence is carried over to the surviving edge


def test_link_nodes_by_alias(corpus):
    df = BinarySCM.random(corpus.graph, seed=5).sample(500, seed=1)
    table = HCG.from_networkx(nx.DiGraph(), "t")
    for c in df.columns:
        table.add_node(c)
    text = text_to_hcg(corpus.docs, "m", "marketing", TextOracle(corpus, 0.0)).hcg
    link = link_nodes(text, table, {v[0]: v[1] + v[2] for v in TEXT_VOCAB["marketing"]})
    assert all(corpus.aliases.get(t) == c or c in t for t, c in link.items()) and len(link) == len(text.nodes)
