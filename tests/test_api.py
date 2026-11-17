import json

import pytest

from humm import api
from humm.graph.text import TextOracle
from humm.orchestrator import acquire
from humm.orchestrator.demo import marketing


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    root = tmp_path_factory.mktemp("humm")
    s = marketing(n=5000, n_exp=1000)
    f = s.save(root / "demo")
    hcg, _ = acquire(s.data, s.corpus.docs, "marketing", TextOracle(s.corpus), s.column_aliases, "m")
    mp = pytest.MonkeyPatch()
    mp.setattr(api, "GRAPHS", root / "graphs")
    mp.setattr(api, "RESULTS", root / "results")
    hcg.save(api._path("m"))
    yield f
    mp.undo()


def test_api_flow(files):
    t = api.graph_from_table(files["csv"], "t")
    assert api.graph_list() == ["m", "t"] and t["verdict"] in ("pass", "warn")
    assert api.validate_run("m", files["csv"], n_perm=10)["summary"]["verdict"] == "pass"
    assert api.engine_identify("m", "ad_spend", "sales", files["csv"])["strategy"] == "backdoor"
    est = api.engine_estimate("m", files["csv"], "ad_spend", "sales", n_boot=20)
    assert est["ci"][0] > 0
    assert api.engine_counterfactual("m", files["csv"], "ad_spend", "sales", "nie", ["web_traffic"])["value"] > 0
    out = api.ask("광고비를 늘리면 매출이 오르는가?", "m", files["csv"], files["experiment_csv"], "ad_spend", n_perm=10)
    assert out["strategy"] == "backdoor" and out["experiment"]["consistent"]
    json.dumps(out, ensure_ascii=False, default=str)


def test_mcp_tools():
    pytest.importorskip("mcp")
    import asyncio

    from humm.mcp.server import mcp
    names = {t.name for t in asyncio.run(mcp.list_tools())}
    assert {"graph_from_table", "graph_from_text", "graph_merge", "validate_run", "engine_identify", "engine_estimate",
            "engine_refute", "engine_counterfactual", "ask"} <= names
