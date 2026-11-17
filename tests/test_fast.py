import numpy as np
import pytest

from humm.fast import fit_temperature, metrics, routing_curve, top_ece, verify_items, pairwise_items
from humm.fast.calibrate import apply_temperature


def test_temperature_fixes_overconfidence():
    rng = np.random.default_rng(0)
    target = rng.integers(0, 2, 4000)
    true_logit = np.where(target == 0, 1.0, -1.0) + rng.normal(0, 1.5, 4000)  # 정답 쪽으로 약하게 기운 로짓
    logits = [np.array([z, -z]) * 4 for z in true_logit]  # 4배 과신
    before = metrics(np.stack(apply_temperature(logits, 1.0)), target)
    t = fit_temperature(logits, target)
    after = metrics(np.stack(apply_temperature(logits, t)), target)
    assert t > 2 and after["ece"] < before["ece"] / 3 and after["ece"] < 0.03 and after["acc"] == before["acc"]


def test_top_ece_perfect():
    p = np.array([[0.8, 0.2]] * 10)
    assert top_ece(p, np.array([0] * 8 + [1] * 2)) == pytest.approx(0.0)


def test_network_items():
    import networkx as nx
    g = nx.DiGraph([("smoke", "lung"), ("lung", "dysp"), ("asia", "tub")])
    pw = pairwise_items("asia", g)
    assert len(pw) == 3 and all(it["labels"] == ["A", "B"] for it in pw)
    for it in pw:  # 정답 보기가 참 방향과 맞는다
        a = it["prompt"].split("A. ")[1].split(" causes ")[0]
        assert (a in {u for u, _ in g.edges}) == (it["target"][0] == 1.0)
    vf = verify_items("asia", g, n=4)
    assert sum(it["target"][0] for it in vf) == len(vf) / 2


def test_routing_curve():
    fast = np.array([[0.95, 0.05], [0.55, 0.45], [0.6, 0.4]])
    slow = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2]])
    rows = {r["threshold"]: r for r in routing_curve(fast, slow, np.array([0, 1, 0]), [0.5, 0.7])}
    assert rows[0.5]["escalation"] == 0 and rows[0.7]["escalation"] == pytest.approx(2 / 3) and rows[0.7]["acc"] == 1.0


def test_metrics_mixed_options():
    probs = [np.array([0.9, 0.1]), np.array([0.1, 0.2, 0.6, 0.1]), np.array([0.3, 0.7])]
    m = metrics(probs, np.array([0, 2, 0]))
    assert m["acc"] == pytest.approx(2 / 3) and "auroc" not in m and m["n"] == 3
