import networkx as nx
import pytest

from humm.graph import HCG, EdgeStatus, Evidence, EvidenceKind, to_compact_json, to_edge_text


def sales_graph():
    g = HCG(graph_id="sales")
    for nid, label in [("season", "계절"), ("ad", "광고비"), ("visits", "방문"), ("sales", "매출")]:
        g.add_node(nid, label=label)
    g.add_edge("season", "ad", sign="+")
    g.add_edge("ad", "visits", sign="+", evidence=[Evidence(kind=EvidenceKind.TEXT, score=0.9)])
    g.add_edge("visits", "sales", sign="+", mechanism="방문이 구매로 전환된다",
               evidence=[Evidence(kind=EvidenceKind.TEXT, score=0.8), Evidence(kind=EvidenceKind.DATA, score=0.9)])
    g.add_edge("season", "sales", sign="+")
    return g


def test_unknown_node_rejected():
    g = sales_graph()
    with pytest.raises(AssertionError):
        g.add_edge("ad", "profit")


def test_noisy_or_confidence():
    e = sales_graph().edge("visits", "sales")
    assert e.confidence == pytest.approx(1 - (1 - 0.6 * 0.8) * (1 - 0.8 * 0.9))


def test_promote_needs_two_kinds():
    g = sales_graph()
    assert g.edge("ad", "visits").promote() == EdgeStatus.HYPOTHESIS
    assert g.edge("visits", "sales").promote() == EdgeStatus.SUPPORTED


def test_break_cycles_refutes_weakest():
    g = sales_graph()
    g.add_edge("sales", "ad", evidence=[Evidence(kind=EvidenceKind.LLM_PRIOR, score=0.1)])
    assert not g.is_dag()
    assert g.break_cycles() == [("sales", "ad")]
    assert g.is_dag() and g.edge("sales", "ad").status == EdgeStatus.REFUTED
    assert ("sales", "ad") not in g.to_networkx().edges


def test_roundtrip(tmp_path):
    g = sales_graph()
    assert HCG.load(g.save(tmp_path / "g.json")) == g
    assert HCG.from_networkx(g.to_networkx(), "x").to_networkx().edges == g.to_networkx().edges


def test_serialize():
    g = sales_graph()
    assert "방문 -> 매출 (+): 방문이 구매로 전환된다" in to_edge_text(g)
    assert '"광고비", "방문", "+"' in to_compact_json(g)
