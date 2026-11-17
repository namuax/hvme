"""Orchestrator v1: two demos, Rung router escalation, and the Humm-LM verification path (simulated remote generation)."""
import numpy as np
import pytest

from humm.graph.text.oracle import TextOracle
from humm.orchestrator import demo
from humm.orchestrator.core import Orchestrator, acquire
from humm.orchestrator.route import QueryRouter, lm_verify, verify_item


@pytest.fixture(scope="module")
def manufacturing():
    s = demo.scenario("manufacturing", n=6000, n_exp=1000)
    hcg, _ = acquire(s.data, s.corpus.docs, "manufacturing", TextOracle(s.corpus), s.column_aliases, "mf")
    return s, hcg


def test_demo_specs_truth():
    for d in demo.SPECS:
        s = demo.scenario(d, n=500, n_exp=100)
        t = s.truth()
        assert set(t) == {e[3] for e in s.expected} | {"nde"} and len(s.questions) == len(s.expected) == 6
    mf = demo.scenario("manufacturing", n=500, n_exp=100).truth()
    assert mf["ate"] < -0.05 and abs(mf["nde"]) < 1e-9  # the maintenance effect passes entirely through vibration


def test_manufacturing_parse_and_sign(manufacturing):
    s, hcg = manufacturing
    orch = Orchestrator(hcg, s.data, n_perm=0, n_boot=20, n_boot_scm=5)
    card = demo.scorecard([orch.ask(q) for q in s.questions], s)
    assert card.parsed_ok.all()
    assert card.value.iloc[0] < 0 and card.strategy.iloc[4] == "no_effect"


class FakeFast:
    def __init__(self, probs):
        self.probs = probs

    def option_probs(self, prompts, labels):
        return [self.probs] * len(prompts)


class FakeLM:
    """Mimics only complete (classification) and generate_raw (remote tool loop)."""
    class tok:
        @staticmethod
        def apply_chat_template(msgs, **kw):
            return msgs[0]["content"]

    batch_size = 4

    def __init__(self, text):
        self.text = text

    def complete(self, prompts, max_new_tokens=32):
        return ['{"rung": 3}'] * len(prompts)

    def generate_raw(self, texts, max_new_tokens, stop, *args):
        return [self.text] * len(texts)


def test_router_escalates_only_low_confidence():
    sure = QueryRouter(FakeFast([0.05, 0.9, 0.05]), FakeLM(""), threshold=0.8).classify(["q"])[0]
    assert (sure["rung"], sure["kind"], sure["router"]) == (2, "ate", "fast")
    unsure = QueryRouter(FakeFast([0.3, 0.4, 0.3]), FakeLM(""), threshold=0.8).classify(["q"])[0]
    assert (unsure["rung"], unsure["kind"], unsure["router"], unsure["fast_rung"]) == (3, "ett", "lm", 2)


def test_lm_verify_remote_path(manufacturing):
    s, hcg = manufacturing
    item = verify_item(hcg, s.data, "maintenance", "defect_rate")
    assert "maintenance" in item["context"] and "defect rate is high" in item["question"]
    v = lm_verify(FakeLM("The effect is negative.\nFinal answer: no"), hcg, s.data, "maintenance", "defect_rate", -0.11, (-0.13, -0.09))
    assert (v["lm_answer"], v["engine_sign"], v["agree"]) == ("no", "-", True)
    v = lm_verify(FakeLM("Final answer: yes"), hcg, s.data, "maintenance", "defect_rate", 0.01, (-0.02, 0.04))
    assert v["engine_sign"] is None and v["agree"] is None and np.isclose(v["engine_ate"], 0.01)


def test_router_keeps_causal_verbs(manufacturing):
    """A question with a causal verb stays ATE even if the router says Rung 1 (association); Korean demo questions had been turned into association."""
    s, hcg = manufacturing
    router = QueryRouter(FakeFast([0.98, 0.01, 0.01]), FakeLM(""), threshold=0.8)
    orch = Orchestrator(hcg, s.data, n_perm=0, n_boot=20, n_boot_scm=5, router=router)
    ans = orch.ask("설비 진동이 생산량을 줄이는가?")
    assert ans.analysis.query.kind == "ate" and ans.route["rung"] == 1
