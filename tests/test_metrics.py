import networkx as nx
import pytest

from hvme.bench.metrics import brier, ece, edge_f1, is_valid_adjustment, shd, sid

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
    # adjusting with no parents (∅) cannot wrongly infer 'no effect' for (b,a), (c,a), (c,b)
    assert sid(CHAIN, empty_like(CHAIN)) == 3


def test_sid_markov_equivalent_graphs_differ():
    # a<-b<-c is in the same equivalence class as a->b->c but its interventional effects differ: SHD is 2, SID is above 0
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
