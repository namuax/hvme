"""Quickstart that runs without a GPU or LLM: build a causal graph from synthetic marketing data and documents, then validate, identify, estimate, run counterfactuals and ask a question.

    pip install hvme
    HVME_HOME=./hvme_work python examples/01_quickstart.py

Instead of an LLM that extracts edges from text, a fake LLM that knows the ground-truth corpus (TextOracle) is used. For a real LLM, pass e.g. llm="hf:Qwen/Qwen3.5-9B".
"""
import json
import os
from pathlib import Path

os.environ.setdefault("HVME_HOME", "hvme_work")
from hvme import api  # noqa: E402
from hvme.graph.text import TextOracle  # noqa: E402
from hvme.orchestrator import acquire  # noqa: E402
from hvme.orchestrator.demo import marketing  # noqa: E402

work = Path(os.environ["HVME_HOME"])
s = marketing(n=5000, n_exp=1000)  # 5,000 observational rows, 1,000 experimental rows, 12 documents (generated from the true DAG)
files = s.save(work / "demo")

# 1) data + documents → causal graph (HCG)
hcg, _ = acquire(s.data, s.corpus.docs, "marketing", TextOracle(s.corpus), s.column_aliases, "marketing")
hcg.save(api._path("marketing"))
g = api.graph_summary(hcg)
print("graph: nodes", len(g["nodes"]), "· supported edges", sum(e["status"] == "supported" for e in g["edges"]), "· refuted edges", sum(e["status"] == "refuted" for e in g["edges"]))

# 2) falsify the graph against data
v = api.validate_run("marketing", files["csv"], n_perm=20)["summary"]
print("validation:", v["verdict"], "· LMC violation rate", round(v["lmc_violation_rate"], 3), "· permutation p", v["permutation_p"])

# 3) identify → estimate → counterfactual
print("identification:", api.engine_identify("marketing", "ad_spend", "sales", files["csv"])["strategy"])
est = api.engine_estimate("marketing", files["csv"], "ad_spend", "sales", n_boot=50)
print("ATE:", round(est["value"], 4), "95% CI", [round(float(x), 4) for x in est["ci"]])
print("NIE (via web traffic):", api.engine_counterfactual("marketing", files["csv"], "ad_spend", "sales", "nie", ["web_traffic"])["value"])

# 4) the whole pipeline from a single question (cross-checked with experiment data)
out = api.ask("광고비를 늘리면 매출이 오르는가?", "marketing", files["csv"], files["experiment_csv"], "ad_spend", n_perm=20)
print(json.dumps({k: out[k] for k in ("strategy", "answer") if k in out}, ensure_ascii=False, default=str, indent=1))
