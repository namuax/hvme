import networkx as nx
import numpy as np
import pandas as pd
import pytest

from hvme.bench.metrics import graph_report
from hvme.engine import BinarySCM
from hvme.graph import EdgeStatus
from hvme.graph.tabular import Knowledge, NoisyOracle, knowledge_from, llm_pairwise, pairwise_agreement, parse_tiers, pc, profile, tabular_to_hcg, to_dag

COLLIDER = nx.DiGraph([("X", "Z"), ("Y", "Z"), ("Z", "W")])


@pytest.fixture(scope="module")
def collider_data():
    scm = BinarySCM(COLLIDER, {"X": np.array([0.5]), "Y": np.array([0.4]), "Z": np.array([0.1, 0.6, 0.7, 0.95]), "W": np.array([0.2, 0.8])})
    return scm.sample(8000, seed=0)


def test_parse_tiers():
    text = 'think...\n```json\n{"tiers": [["a", "zz"], ["b", "a"], ["c"]]}\n```'
    assert parse_tiers(text, ["a", "b", "c"]) == [["a"], ["b"], ["c"]]
    assert parse_tiers("no json here", ["a"]) == []
    draft = "Tier 1: a, b\nTier 2: c\n\nLet me revise.\nTier 1: **a**\nTier 2: b and c"  # without JSON, read the last Tier group
    assert parse_tiers(draft, ["a", "b", "c"]) == [["a"], ["b", "c"]]


def test_knowledge_rules():
    k = Knowledge(tiers=[["a"], ["b"]], forbidden={("c", "b")}, required={("c", "a")})
    assert k.allows("a", "b") and not k.allows("b", "a") and not k.allows("c", "b")
    assert k.prefers("a", "b") and k.prefers("c", "a") and k.prefers("b", "c")


def test_pc_finds_vstructure_and_meek(collider_data):
    g = pc(collider_data, alpha=0.01)
    assert {("X", "Z"), ("Y", "Z")} <= g.directed and ("Z", "W") in g.directed  # v-structure + Meek R1
    assert g.oriented_by[("X", "Z")] == "vstruct" and g.oriented_by[("Z", "W")] == "meek"
    assert graph_report(COLLIDER, to_dag(g))["shd"] == 0


def test_pc_is_column_order_invariant(collider_data):
    a, b = pc(collider_data), pc(collider_data[["W", "Z", "Y", "X"]])
    assert a.directed == b.directed and a.undirected == b.undirected


def test_knowledge_orients_markov_equivalent_chain():
    chain = nx.DiGraph([("A", "B"), ("B", "C")])
    df = BinarySCM(chain, {"A": np.array([0.5]), "B": np.array([0.2, 0.8]), "C": np.array([0.2, 0.8])}).sample(5000, seed=1)
    assert pc(df).directed == set()  # data alone cannot determine direction
    g = pc(df, knowledge=Knowledge(tiers=[["A"], ["B"], ["C"]]))
    assert g.directed == {("A", "B"), ("B", "C")} and g.oriented_by[("A", "B")] == "knowledge"


def test_soft_mode_keeps_data_vstructure_against_wrong_prior(collider_data):
    wrong = Knowledge(tiers=[["Z"], ["X", "Y", "W"]])  # wrong prior knowledge that puts the collider as a cause
    assert {("X", "Z"), ("Y", "Z")} <= pc(collider_data, knowledge=wrong, mode="soft").directed
    assert not {("X", "Z"), ("Y", "Z")} & pc(collider_data, knowledge=wrong, mode="hard").directed


def test_noisy_oracle():
    o = NoisyOracle(COLLIDER, error=0.0)
    assert parse_tiers(o.complete(["q"])[0], list(COLLIDER)) == [[v] for v in nx.topological_sort(COLLIDER)]
    pw = llm_pairwise(o, [("X", "W"), ("W", "Y")], "test")
    assert pw.p_ab.round(2).tolist() == [0.85, 0.15]


def test_pairwise_with_api_style_provider():
    class Api:
        supports_logits = False
        name = "api"

        def complete(self, prompts, max_new_tokens=8):
            return ["A" for _ in prompts]
    assert llm_pairwise(Api(), [("a", "b")], "d").p_ab.tolist() == [1.0]


def test_pipeline_outputs_hcg(collider_data):
    r = tabular_to_hcg(collider_data, "demo", "toy", provider=NoisyOracle(COLLIDER, 0.0), n_boot=5, validate_kw={"n_perm": 20})
    assert set(r.dag.edges) == set(COLLIDER.edges)
    assert all(any(x.method.startswith("pc:") for x in e.evidence) for e in r.hcg.edges)
    assert r.report.verdict == "pass"
    assert r.hcg.edge("X", "Z").status == EdgeStatus.VERIFIED  # data (v-structure) and LLM ordering agree and the graph passes
    assert r.conflicts == []
    alone = tabular_to_hcg(collider_data, "demo", "toy", n_boot=0, validate_kw={"n_perm": 20})
    assert alone.hcg.edge("X", "Z").status == EdgeStatus.HYPOTHESIS  # not promoted with data evidence only
    wrong = tabular_to_hcg(collider_data, "demo", "toy", provider=NoisyOracle(COLLIDER.reverse(), 0.0), n_boot=0, validate_kw={"n_perm": 20})
    assert ("X", "Z", "vstruct") in wrong.conflicts
    assert set(profile(collider_data).kind) == {"binary"}


def test_pairwise_knowledge_and_agreement():
    pw = pd.DataFrame([{"a": "X", "b": "Z", "p_ab": 0.9}, {"a": "W", "b": "Z", "p_ab": 0.2}, {"a": "X", "b": "Y", "p_ab": 0.5}])
    k = knowledge_from(pairwise=pw)
    assert k.forbidden == {("Z", "X"), ("W", "Z")}
    assert pairwise_agreement(pw, COLLIDER) | {} == {"pairs_on_true_edges": 2, "correct": 2, "wrong": 0, "unsure": 0, "pair_accuracy": 1.0}


def test_hybrid_uses_only_confident_consensus(collider_data):
    from hvme.graph.tabular import strong_knowledge
    a = pd.DataFrame([{"a": "X", "b": "Z", "p_ab": 0.99}, {"a": "W", "b": "Z", "p_ab": 0.97}])
    b = pd.DataFrame([{"a": "X", "b": "Z", "p_ab": 0.96}, {"a": "W", "b": "Z", "p_ab": 0.40}])
    assert strong_knowledge([a, b], 0.95).forbidden == {("Z", "X")}  # only pairs both models are confident about
    wrong = Knowledge(forbidden={("X", "Z"), ("Y", "Z")})  # confident (wrong) knowledge that flips the data's v-structure
    assert ("Z", "X") in pc(collider_data, strong=wrong).directed
    assert ("X", "Z") in pc(collider_data, knowledge=wrong).directed  # given only as soft knowledge, the data wins


def test_default_mode_is_hybrid(collider_data):
    r = tabular_to_hcg(collider_data, "demo", "toy", provider=NoisyOracle(COLLIDER, 0.0), n_boot=0, validate_kw={"n_perm": 0})
    assert set(r.dag.edges) == set(COLLIDER.edges)
