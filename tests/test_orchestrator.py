import networkx as nx
import pytest

from humm.engine import BinarySCM
from humm.graph.schema import HCG
from humm.graph.text import TextOracle
from humm.orchestrator import Orchestrator, parse_rules
from humm.orchestrator.demo import QUESTIONS, marketing


@pytest.fixture(scope="module")
def scenario():
    return marketing(n=6000, n_exp=2000)


@pytest.fixture(scope="module")
def orch(scenario):
    s = scenario
    return Orchestrator.from_sources(s.data, s.corpus.docs, "marketing", TextOracle(s.corpus, error=0.0), s.column_aliases, "demo",
                                     experiments={"ad_spend": s.experiment}, n_perm=20, n_boot=50, n_boot_scm=5)


def test_fit_recovers_cpt():
    g = nx.DiGraph([("z", "x"), ("z", "y"), ("x", "y")])
    true = BinarySCM.random(g, seed=1)
    fit = BinarySCM.fit(g, true.sample(40000, seed=2))
    assert all(abs(fit.cpt[v] - true.cpt[v]).max() < 0.03 for v in g)
    assert abs(fit.ate("x", "y") - true.ate("x", "y")) < 0.03


def test_parse_rules():
    hcg = HCG(graph_id="t")
    for v, al in [("ad_spend", ["광고비"]), ("sales", ["매출", "매출액"]), ("web_traffic", ["웹 방문"])]:
        hcg.add_node(v, aliases=al)
    q = parse_rules("광고비가 매출에 주는 효과 중 웹 방문을 거치는 간접 효과는?", hcg)
    assert (q.kind, q.treatment, q.outcome, q.mediators) == ("mediation", "ad_spend", "sales", ["web_traffic"])
    assert parse_rules("매출액이 오른 것은 광고비 때문인가?", hcg).treatment == "ad_spend"
    assert parse_rules("광고비와 매출의 상관관계는 인과인가?", hcg).kind == "association"
    assert parse_rules("광고비를 쓰지 않았다면 매출은?", hcg).kind == "ett"


def test_graph_recovered(orch, scenario):
    assert orch.report.verdict == "pass"
    assert {e.key for e in orch.hcg.active_edges()} == set(scenario.graph.edges)


def test_demo_questions(orch, scenario):
    t = scenario.truth()
    answers = [orch.ask(q) for q in QUESTIONS]
    v = [a.analysis.estimate for a in answers]
    assert abs(v[0]["ate"]["value"] - t["ate"]) < 0.04
    assert v[1]["association"]["value"] > v[1]["ate"]["value"] + 0.05  # 성수기 교란으로 상관이 인과 효과보다 크다
    assert abs(v[2]["nie"]["value"] - t["nie"]) < 0.04 and abs(v[2]["nde"]["value"] - t["nde"]) < 0.05
    assert abs(v[3]["ett"]["value"] - t["ett"]) < 0.04
    assert answers[4].analysis.estimand.strategy == "no_effect" and v[4]["ate"]["value"] == 0
    assert v[5]["ate"]["ci"][1] < 0  # 할인은 만족도를 거쳐 이탈을 줄인다
    assert answers[0].analysis.experiment["consistent"]
    for a in answers:
        assert all(h in a.markdown for h in ("근거 경로", "그래프 민감도", "가정"))
        assert a.analysis.sensitivity_summary["scenarios"] >= 1 and a.analysis.assumptions
    assert "not_identified" in set(answers[0].analysis.sensitivity.strategy)  # 숨은 교란을 넣으면 식별이 깨진다


def test_parse_keeps_rule_variables():
    """LLM이 방향을 뒤집어도 변수는 질문 문자열 기준을 따른다. 유형 키워드가 없을 때만 LLM 유형을 쓴다."""
    class Flip:
        name = "flip"
        def complete(self, prompts, max_new_tokens=256):
            return ['{"kind": "ate", "treatment": "sales", "outcome": "ad_spend", "mediators": []}']
    from humm.orchestrator import parse
    hcg = HCG(graph_id="t")
    hcg.add_node("ad_spend", aliases=["광고비"])
    hcg.add_node("sales", aliases=["매출"])
    q = parse("광고비가 매출을 좌우하는가?", hcg, Flip())
    assert (q.treatment, q.outcome, q.parser) == ("ad_spend", "sales", "rules+flip")
    assert parse("광고비와 매출의 상관관계는?", hcg, Flip()).parser == "rules"


def test_check_narrative(orch):
    from humm.orchestrator.explain import check_narrative
    a = orch.ask(QUESTIONS[0]).analysis
    v = a.estimate["ate"]["value"]
    assert check_narrative(f"광고비를 늘리면 매출이 오른다. 효과는 {v:.3f}이다.", a) == []
    assert check_narrative("효과는 0.999이다.", a)  # 근거에 없는 숫자
    assert check_narrative(f"효과는 {v:.3f}입니다.", a)  # 합쇼체
    assert check_narrative(f"ad_spend 효과는 {v:.3f}이다.", a)  # 내부 이름


def test_to_plain():
    from humm.orchestrator.explain import to_plain
    cases = {"매출이 증가합니다.": "매출이 증가한다.", "같지 않습니다.": "같지 않다.", "변하지 않습니다.": "변하지 않는다.", "결과입니다.": "결과이다.",
             "볼 수 있습니다.": "볼 수 있다.", "다릅니다.": "다르다.", "확인됩니다.": "확인된다.", "유의하게 나타났습니다.": "유의하게 나타났다.",
             "이미 '~다'체이다.": "이미 '~다'체이다."}
    assert {k: to_plain(k) for k in cases} == cases
