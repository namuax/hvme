import json

import pytest

from hvme import api
from hvme.graph.text import TextOracle
from hvme.orchestrator import acquire
from hvme.orchestrator.demo import marketing


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    root = tmp_path_factory.mktemp("hvme")
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


def test_api_errors(files):
    with pytest.raises(ValueError, match=r"unknown variable\(s\) \['revenue'\].*ad_spend"):
        api.engine_estimate("m", files["csv"], "ad_spend", "revenue")
    with pytest.raises(ValueError, match="unknown variable"):
        api.engine_counterfactual("m", files["csv"], "ad_spend", "sales", "nie", ["visits"])
    with pytest.raises(ValueError, match="unknown query"):
        api.engine_counterfactual("m", files["csv"], "ad_spend", "sales", "cate")
    with pytest.raises(FileNotFoundError, match="known graphs"):
        api.engine_identify("nope", "ad_spend", "sales")


def test_model_defaults():
    import os
    import subprocess
    import sys
    env = {k: v for k, v in os.environ.items() if k not in ("HVME_FAST", "HVME_ROUTER", "HVME_LM")}
    out = subprocess.run([sys.executable, "-c", "from hvme import api; print(api.FAST_PATH, api.ROUTER_PATH, api.LM_PATH)"],
                         env=env, capture_output=True, text=True, check=True).stdout.split()
    assert out == ["namuax/hvme-fast-4b", "namuax/hvme-router-4b", "namuax/hvme-lm-9b"]


def test_mcp_version():
    pytest.importorskip("mcp")
    import hvme
    from hvme.mcp.server import mcp
    assert mcp._mcp_server.create_initialization_options().server_version == hvme.__version__


def test_mcp_tools():
    pytest.importorskip("mcp")
    import asyncio

    from hvme.mcp.server import mcp
    names = {t.name for t in asyncio.run(mcp.list_tools())}
    assert {"graph_from_table", "graph_from_text", "graph_merge", "validate_run", "engine_identify", "engine_estimate",
            "engine_refute", "engine_counterfactual", "ask"} <= names
