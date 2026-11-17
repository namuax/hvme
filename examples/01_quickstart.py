"""GPU·LLM 없이 도는 빠른 시작: 합성 마케팅 데이터와 문서로 인과 그래프를 만들고, 검증·식별·추정·반사실·질문까지 돈다.

    pip install humm
    HUMM_HOME=./humm_work python examples/01_quickstart.py

텍스트에서 엣지를 뽑는 LLM 대신 정답 코퍼스를 아는 가상 LLM(TextOracle)을 쓴다. 실제 LLM은 llm="hf:Qwen/Qwen3.5-9B" 처럼 넘긴다.
"""
import json
import os
from pathlib import Path

os.environ.setdefault("HUMM_HOME", "humm_work")
from humm import api  # noqa: E402
from humm.graph.text import TextOracle  # noqa: E402
from humm.orchestrator import acquire  # noqa: E402
from humm.orchestrator.demo import marketing  # noqa: E402

work = Path(os.environ["HUMM_HOME"])
s = marketing(n=5000, n_exp=1000)  # 관측 데이터 5,000행, 실험 데이터 1,000행, 문서 12개(정답 DAG에서 생성)
files = s.save(work / "demo")

# 1) 데이터 + 문서 → 인과 그래프(HCG)
hcg, _ = acquire(s.data, s.corpus.docs, "marketing", TextOracle(s.corpus), s.column_aliases, "marketing")
hcg.save(api._path("marketing"))
g = api.graph_summary(hcg)
print("그래프: 노드", len(g["nodes"]), "· 지지된 엣지", sum(e["status"] == "supported" for e in g["edges"]), "· 반증된 엣지", sum(e["status"] == "refuted" for e in g["edges"]))

# 2) 데이터로 그래프 반증 검사
v = api.validate_run("marketing", files["csv"], n_perm=20)["summary"]
print("검증:", v["verdict"], "· LMC 위반율", round(v["lmc_violation_rate"], 3), "· 순열 p", v["permutation_p"])

# 3) 식별 → 추정 → 반사실
print("식별:", api.engine_identify("marketing", "ad_spend", "sales", files["csv"])["strategy"])
est = api.engine_estimate("marketing", files["csv"], "ad_spend", "sales", n_boot=50)
print("ATE:", round(est["value"], 4), "95% CI", [round(float(x), 4) for x in est["ci"]])
print("NIE(웹 유입 경유):", api.engine_counterfactual("marketing", files["csv"], "ad_spend", "sales", "nie", ["web_traffic"])["value"])

# 4) 질문 하나로 전 과정(실험 데이터로 교차 확인)
out = api.ask("광고비를 늘리면 매출이 오르는가?", "marketing", files["csv"], files["experiment_csv"], "ad_spend", n_perm=20)
print(json.dumps({k: out[k] for k in ("strategy", "answer") if k in out}, ensure_ascii=False, default=str, indent=1))
