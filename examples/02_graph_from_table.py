"""Build a causal graph from a table alone (PC, no LLM), then validate it against the same data.

    HVME_HOME=./hvme_work python examples/02_graph_from_table.py
"""
import os

os.environ.setdefault("HVME_HOME", "hvme_work")
from hvme import api  # noqa: E402
from hvme.orchestrator.demo import scenario  # noqa: E402

s = scenario("manufacturing", n=5000, n_exp=0)  # synthetic factory data from a known DAG
files = s.save(os.path.join(os.environ["HVME_HOME"], "demo"))

out = api.graph_from_table(files["csv"], "factory")  # pass llm="hf:Qwen/Qwen3.5-9B" to add LLM direction priors
print("verdict:", out["verdict"])
for e in out["edges"]:
    print(f'  {e["src"]:>18} -> {e["dst"]:<18} {e["status"]:<10} confidence {e["confidence"]:.2f}')
print("true edges:", sorted(s.graph.edges))
# Data alone fixes the skeleton but not every direction; the LLM prior and documents (example 01) supply the rest.
