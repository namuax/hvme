"""Symbolic causal engine cross-checks: textbook examples, closed-form solutions and DoWhy results."""
import logging

import networkx as nx
import numpy as np
import pytest

from hvme.engine import (BinarySCM, LinearSCM, adjustment_sets, d_separated, estimate, frontdoor_sets, identify, instruments,
                         is_frontdoor, is_valid_adjustment, local_markov, optimal_adjustment, refute)

SMOKING = nx.DiGraph([("U", "S"), ("U", "C"), ("S", "T"), ("T", "C")])  # smoking-tar-cancer, U unobserved
CONFOUNDED = nx.DiGraph([("Z", "X"), ("Z", "Y"), ("X", "Y"), ("W", "Y")])
IV = nx.DiGraph([("Z", "X"), ("U", "X"), ("U", "Y"), ("X", "Y")])
M_BIAS = nx.DiGraph([("A", "X"), ("A", "M"), ("B", "M"), ("B", "Y"), ("X", "Y")])


# ---------- d-separation ----------
def test_dsep_chain_fork_collider():
    assert d_separated(nx.DiGraph([("a", "b"), ("b", "c")]), "a", "c", "b")
    assert not d_separated(nx.DiGraph([("a", "b"), ("b", "c")]), "a", "c")
    assert d_separated(nx.DiGraph([("b", "a"), ("b", "c")]), "a", "c", "b")
    assert d_separated(nx.DiGraph([("a", "b"), ("c", "b")]), "a", "c")
    assert not d_separated(nx.DiGraph([("a", "b"), ("c", "b")]), "a", "c", "b")


def test_local_markov():
    lm = {v: (rest, pa) for v, rest, pa in local_markov(nx.DiGraph([("a", "b"), ("b", "c")]))}
    assert lm["c"] == ({"a"}, {"b"})


# ---------- Identification ----------
def test_backdoor_sets():
    assert is_valid_adjustment(CONFOUNDED, "X", "Y", {"Z"})
    assert not is_valid_adjustment(CONFOUNDED, "X", "Y", set())
    assert optimal_adjustment(CONFOUNDED, "X", "Y") == {"Z", "W"}
    assert adjustment_sets(CONFOUNDED, "X", "Y")[0] == {"Z"}


def test_m_bias_collider_not_adjusted():
    # M is a collider, so adjusting only for M introduces bias. The empty set is valid
    assert is_valid_adjustment(M_BIAS, "X", "Y", set())
    assert not is_valid_adjustment(M_BIAS, "X", "Y", {"M"})
    assert is_valid_adjustment(M_BIAS, "X", "Y", {"M", "A"})


def test_frontdoor_and_iv():
    assert is_frontdoor(SMOKING, "S", "C", {"T"})
    assert frontdoor_sets(SMOKING, "S", "C", latent={"U"}) == [{"T"}]
    assert instruments(IV, "X", "Y", latent={"U"}) == ["Z"]


def test_identify_strategies():
    assert identify(CONFOUNDED, "X", "Y").strategy == "backdoor"
    assert identify(SMOKING, "S", "C", latent={"U"}).strategy == "frontdoor"
    assert identify(IV, "X", "Y", latent={"U"}).strategy == "iv"
    assert identify(nx.DiGraph([("U", "X"), ("U", "Y"), ("X", "Y")]), "X", "Y", latent={"U"}).strategy == "not_identified"
    assert identify(CONFOUNDED, "Y", "X").strategy == "no_effect"


# ---------- SCM and counterfactuals ----------
def simple_xy():
    return BinarySCM(nx.DiGraph([("X", "Y")]), {"X": np.array([0.4]), "Y": np.array([0.3, 0.8])})


def test_binary_scm_is_exact_on_small_graphs():
    assert BinarySCM.random(SMOKING, seed=1).exact


def test_counterfactual_closed_forms():
    s = simple_xy()  # monotone threshold model + exogenous treatment: PN = (p1-p0)/p1, PS = (p1-p0)/(1-p0), ETT = ATE
    assert s.ate("X", "Y") == pytest.approx(0.5)
    assert s.pn("X", "Y") == pytest.approx(0.5 / 0.8)
    assert s.ps("X", "Y") == pytest.approx(0.5 / 0.7)
    assert s.ett("X", "Y") == pytest.approx(0.5)


def test_mediation_without_direct_edge():
    s = BinarySCM.random(nx.DiGraph([("X", "M"), ("M", "Y")]), seed=11)
    assert s.nde("X", "Y", ["M"]) == pytest.approx(0)
    assert s.nie("X", "Y", ["M"]) == pytest.approx(s.ate("X", "Y"))


def test_linear_total_effect_is_path_sum():
    g = nx.DiGraph([("X", "M"), ("M", "Y"), ("X", "Y")])
    l = LinearSCM(g, {("X", "M"): 2.0, ("M", "Y"): 3.0, ("X", "Y"): 1.0}, {v: 1.0 for v in g})
    assert l.total_effect("X", "Y") == pytest.approx(7.0)


# ---------- Estimation: does the data estimate's confidence interval contain the SCM truth ----------
def covers(est, truth, slack=0.01):
    return est.ci[0] - slack <= truth <= est.ci[1] + slack


def test_estimate_binary_backdoor_and_frontdoor():
    s = BinarySCM.random(CONFOUNDED, seed=3)
    assert covers(estimate(CONFOUNDED, s.sample(100_000, seed=4), "X", "Y", n_boot=50), s.ate("X", "Y"))
    s = BinarySCM.random(SMOKING, seed=1)
    df = s.sample(100_000, seed=2).drop(columns="U")
    assert covers(estimate(SMOKING, df, "S", "C", latent={"U"}, n_boot=50), s.ate("S", "C"))


def test_estimate_linear_backdoor_frontdoor_iv():
    for g, x, y, lat, seed in [(CONFOUNDED, "X", "Y", set(), 9), (SMOKING, "S", "C", {"U"}, 7), (IV, "X", "Y", {"U"}, 5)]:
        l = LinearSCM.random(g, seed=seed)
        df = l.sample(100_000, seed=seed + 1).drop(columns=list(lat))
        assert estimate(g, df, x, y, latent=lat, n_boot=0).value == pytest.approx(l.total_effect(x, y), abs=0.05)


def test_refuters_pass_on_correct_model():
    s = BinarySCM.random(CONFOUNDED, seed=3)
    assert refute(CONFOUNDED, s.sample(50_000, seed=4), "X", "Y").passed.all()


# ---------- DoWhy cross-check ----------
def test_matches_dowhy():
    dowhy = pytest.importorskip("dowhy")
    logging.disable(logging.WARNING)
    g = nx.DiGraph([("Z1", "X"), ("Z1", "Z2"), ("Z2", "Y"), ("X", "M"), ("M", "Y"), ("Z3", "X"), ("Z3", "Y")])
    l = LinearSCM.random(g, seed=1)
    df = l.sample(50_000, seed=2)
    model = dowhy.CausalModel(data=df, treatment="X", outcome="Y", graph=g)
    est = model.identify_effect(proceed_when_unidentifiable=True)
    assert is_valid_adjustment(g, "X", "Y", set(est.get_backdoor_variables()))
    ref = model.estimate_effect(est, method_name="backdoor.linear_regression").value
    assert estimate(g, df, "X", "Y", n_boot=0).value == pytest.approx(ref, abs=0.02)
