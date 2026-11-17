import networkx as nx
import pytest

from humm.bench.metrics import brier, ece, edge_f1, is_valid_adjustment, shd, sid

CHAIN = nx.DiGraph([("a", "b"), ("b", "c")])


def empty_like(g):
    e = nx.DiGraph()
    e.add_nodes_from(g.nodes)
    return e


def test_identical_graph_scores_zero():
    assert shd(CHAIN, CHAIN) == 0 and sid(CHAIN, CHAIN) == 0
    assert edge_f1(CHAIN, CHAIN)["directed"]["f1"] == 1


def test_shd_counts_reversal_once():
    assert shd(CHAIN, nx.DiGraph([("b", "a"), ("b", "c")])) == 1
    assert shd(CHAIN, empty_like(CHAIN)) == 2


def test_sid_empty_estimate_on_chain():
    # 부모 없음(∅)으로 조정하면 (b,a), (c,a), (c,b)에서 '효과 없음'을 잘못 추론하지 못한다
    assert sid(CHAIN, empty_like(CHAIN)) == 3


def test_sid_markov_equivalent_graphs_differ():
    # a<-b<-c 는 a->b->c 와 같은 동치류지만 개입 효과는 다르다: SHD는 2, SID는 0보다 크다
    rev = nx.DiGraph([("b", "a"), ("c", "b")])
    assert shd(CHAIN, rev) == 2 and sid(CHAIN, rev) > 0


def test_backdoor_adjustment():
    g = nx.DiGraph([("z", "x"), ("z", "y"), ("x", "y")])
    assert is_valid_adjustment(g, "x", "y", {"z"})
    assert not is_valid_adjustment(g, "x", "y", set())


def test_mediator_is_forbidden():
    g = nx.DiGraph([("x", "m"), ("m", "y")])
    assert not is_valid_adjustment(g, "x", "y", {"m"})


def test_probability_metrics():
    assert brier([1, 0], [1, 0]) == 0
    assert ece([0.9] * 10, [1] * 9 + [0]) == pytest.approx(0.0)
    assert ece([0.9] * 10, [1] * 5 + [0] * 5) == pytest.approx(0.4)
