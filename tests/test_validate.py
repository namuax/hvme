import networkx as nx
import numpy as np
import pytest

from hvme.engine import BinarySCM, LinearSCM
from hvme.graph import HCG, EdgeStatus, Evidence, EvidenceKind
from hvme.validate import CITester, benchmark, check_structure, fisher_z, g_test, validate, validate_hcg

CHAIN = nx.DiGraph([("a", "b"), ("b", "c")])


@pytest.fixture(scope="module")
def binary_data():
    g = nx.DiGraph([("Z", "X"), ("Z", "Y"), ("X", "M"), ("M", "Y"), ("W", "Y")])
    scm = BinarySCM(g, {"Z": np.array([0.5]), "W": np.array([0.4]), "X": np.array([0.2, 0.8]), "M": np.array([0.1, 0.9]),
                        "Y": np.array([0.1, 0.4, 0.5, 0.9, 0.2, 0.5, 0.6, 0.95])})
    return g, scm.sample(8000, seed=1)


def edit(g, remove=(), add=()):
    h = g.copy()
    h.remove_edges_from(remove)
    h.add_edges_from(add)
    return h


def test_l1_structure():
    s = check_structure(nx.DiGraph([("a", "b"), ("b", "a")]), time={"a": 0, "b": 1})
    assert not s["is_dag"] and s["cycles"] and s["temporal_violations"] == [["b", "a"]]


def test_ci_tests_chain():
    l = LinearSCM(CHAIN, {("a", "b"): 1.0, ("b", "c"): 1.0}, {v: 1.0 for v in CHAIN})
    d = l.sample(3000, seed=0)
    assert fisher_z(d, "a", "c") < 1e-6 and fisher_z(d, "a", "c", ("b",)) > 0.01
    db = BinarySCM(CHAIN, {"a": np.array([0.5]), "b": np.array([0.2, 0.8]), "c": np.array([0.2, 0.8])}).sample(5000, seed=0)
    assert g_test(db, "a", "c") < 1e-6 and g_test(db, "a", "c", ("b",)) > 0.01
    assert CITester(db).method == "g_test" and CITester(d).method == "fisher_z"


def test_true_graph_passes_and_wrong_graph_warns(binary_data):
    g, df = binary_data
    assert validate(g, df, n_perm=50).verdict == "pass"
    wrong = validate(edit(g, remove=[("Z", "Y")]), df, n_perm=50)
    assert wrong.verdict == "warn"
    assert {"Z", "Y"} == set(wrong.missing_edge_candidates(1)[0][:2])


def test_spurious_edge_is_unsupported(binary_data):
    g, df = binary_data
    r = validate(edit(g, add=[("W", "M")]), df, n_perm=20)
    assert r.edges.set_index(["src", "dst"]).loc[("W", "M"), "supported"] == False  # noqa: E712


def test_apply_to_hcg_status_rules(binary_data):
    g, df = binary_data
    h = HCG.from_networkx(g, "demo")
    h.edge("X", "M").evidence.append(Evidence(kind=EvidenceKind.TEXT, score=0.8))
    h.add_edge("Y", "Z", evidence=[Evidence(kind=EvidenceKind.LLM_PRIOR, score=0.1)])  # weak edge that creates a cycle
    report, changes = validate_hcg(h, df, time={"Y": 2, "Z": 0}, n_perm=20)
    assert report.verdict == "pass" and report.structure["before_fix"]["cycles"]
    assert h.edge("Y", "Z").status == EdgeStatus.REFUTED
    assert h.edge("X", "M").status == EdgeStatus.VERIFIED  # text + data + pass
    assert h.edge("Z", "X").status == EdgeStatus.HYPOTHESIS  # data evidence only, so no directional evidence
    assert any(e.method.startswith("ci:") for e in h.edge("Z", "X").evidence) and len(changes) == 6


def test_l6_benchmark_with_oracle():
    from hvme.validate import sample_bnlearn
    pytest.importorskip("pgmpy")
    true = sample_bnlearn("asia", 200)[0]
    r = benchmark(lambda df: true.copy(), names=("asia",), n=200)
    assert r.loc[0, "shd"] == 0 and r.loc[0, "sid"] == 0
