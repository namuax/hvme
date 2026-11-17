import sys
import nbformat as nbf

cells = [
"""md:# 08. Thin end-to-end — MCP tools + orchestrator v0

This notebook connects the graph construction, validation and causal engine built so far into a single question-to-answer flow.

| Stage | What it does | Module |
|---|---|---|
| Query analysis | Question → type (ATE, correlation vs. causation, mediation, ETT), treatment, outcome and mediator variables. Rule-based; a general-purpose LLM, if available, produces a JSON query | `orchestrator.query` |
| Graph acquisition | Tabular data → HCG, documents → HCG, merge | `orchestrator.acquire` |
| Validation | L1 structure and L2 data falsification, plus L4 experiment comparison when experimental data exists | `validate_hcg`, `experiment_check` |
| Identification and estimation | Backdoor → frontdoor → IV, bootstrap intervals. Rung 3 uses a binary SCM fitted to the data | `engine`, `BinarySCM.fit` |
| Refutation | Placebo treatment, random common cause, data subset | `refute` |
| Graph sensitivity (L5) | Delete or flip each edge, restore falsified edges, add a hidden confounder → does the conclusion change? | `orchestrator.sensitivity` |
| Explanation | Answer, evidence paths (edge status, evidence, quotes) and list of assumptions. The LLM only writes the summary paragraph | `orchestrator.explain` |

The same functionality is exposed as `humm.api` (path-based functions) and `humm.mcp` (MCP server). Sections 1-3, 5 and 6 run on CPU; section 4 needs a GPU.""",
"""import pandas as pd
from IPython.display import Markdown, display

from humm.graph.text import TextOracle
from humm.orchestrator import Orchestrator
from humm.orchestrator.demo import QUESTIONS, marketing, scorecard
from humm.paths import DATA, GRAPHS, RESULTS

pd.set_option("display.max_colwidth", 60)
s = marketing()
files = s.save(DATA / "demo" / "marketing")
files""",
"""md:## 1. Demo scenario (marketing)

From a ground-truth SCM, generate 8,000 rows of observational data, 3,000 rows of an A/B test with randomly assigned ad spend, and 12 documents. Peak season is a **confounder** that moves ad spend, discounts and sales together, and part of the ad-spend effect **goes through web visits**. The answers to the 6 questions are computed exactly from the ground-truth SCM.""",
"""print("Ground-truth edges:", [(a, b, d["sign"]) for a, b, d in s.graph.edges(data=True)])
print("Questions:", *QUESTIONS, sep="\\n- ")
pd.Series(s.truth()).round(4)""",
"""md:## 2. (CPU) End to end with a simulated LLM

Text extraction is done by a simulated LLM (error rate 0.1); query analysis and explanation use only rules and templates. Graph acquisition and validation run once, then each question goes through identification → estimation → refutation → sensitivity → explanation.""",
"""orch = Orchestrator.from_sources(s.data, s.corpus.docs, "marketing", TextOracle(s.corpus, error=0.1), s.column_aliases, "marketing-demo",
                                 experiments={"ad_spend": s.experiment})
print("Validation verdict:", orch.report.verdict, "| matches ground-truth edges:", {e.key for e in orch.hcg.active_edges()} == set(s.graph.edges))
orch.changes""",
"""answers = [orch.ask(q) for q in QUESTIONS]
card = scorecard(answers, s)
card""",
"""md:`parsed_ok` says whether the question was interpreted as expected (type, treatment, outcome), and `covered` says whether the true value lies within the 95% interval. `alternatives` is the number of alternative graphs in the sensitivity analysis; `flipped` and `vanished` are the numbers of alternatives in which the sign of the conclusion flips or the effect becomes zero.

## 3. Full answers

For each question, the answer, evidence paths, refutations, graph sensitivity and assumptions are produced automatically.""",
"""for a in answers:
    display(Markdown(a.markdown + "\\n\\n---"))""",
"""md:## 4. (GPU) Graph construction + query analysis + explanation with Qwen3.5

In the same flow, replace the simulated LLM with a real model. A single model handles text extraction, normalization and counterfactual verification (graph acquisition), query analysis, and the summary paragraph of the explanation. Responses are recorded to `results/llm_cache/`, so they can be replayed without a GPU.

- Query analysis: variables are determined by expressions found in the question string (rules); the LLM is used only when no variable is found or the question has no type keyword.
- Summary paragraph: a paragraph with numbers not in the evidence, sentences not in the plain "~다" register, or internal names (variable ids, `_U`) is not included in the answer (`narrative_ok`). The raw text and check results are saved to `results/e2e/<model>_narratives.json`.

> Observed results: both models recovered the graph exactly (10 ground-truth edges, verdict pass), and the numbers matched the simulated-LLM run. However, in LLM query analysis both models flipped the direction of "Does ad spend affect customer churn?" (the 4B model also classified an ETT question as ATE), and 11 of 12 summary paragraphs failed the check because of formal-register endings, internal names or numbers not in the evidence. The two rules above were set from this result.
>
> After the prompt fixes, query interpretation is 6/6 for both models. Direction and internal-name problems disappeared from the summaries, but the models mostly wrote in the formal register despite instructions, so the check passed 1/6 for 4B and 0/6 for 9B. With a post-processing step that converts the formal register to the plain "~다" register (`explain.to_plain`), replaying the same recordings gives 5/6 for 4B and 6/6 for 9B.""",
"""gpu:import gc
import json

import torch

from humm.providers.cache import CachedProvider
from humm.providers.hf import HFProvider

MODELS = {
    "Qwen/Qwen3.5-4B": "cuda:0",
    "Qwen/Qwen3.5-9B": "cuda:1",
}
(RESULTS / "e2e").mkdir(parents=True, exist_ok=True)
cards = []
for model_id, device in MODELS.items():
    name = model_id.split("/")[-1]
    lm = CachedProvider(HFProvider(model_id, device=device, batch_size=8), RESULTS / "llm_cache" / f"{name}.json")
    o = Orchestrator.from_sources(s.data, s.corpus.docs, "marketing", lm, s.column_aliases, f"marketing-demo-{name}", llm=lm,
                                  experiments={"ad_spend": s.experiment})
    ans = [o.ask(q) for q in QUESTIONS]
    cards.append(scorecard(ans, s).assign(model=name, verdict=o.report.verdict,
                                          graph_exact={e.key for e in o.hcg.active_edges()} == set(s.graph.edges)))
    (RESULTS / "e2e" / f"{name}.md").write_text("\\n\\n---\\n\\n".join(a.markdown for a in ans), encoding="utf-8")
    (RESULTS / "e2e" / f"{name}_narratives.json").write_text(json.dumps([{"question": a.analysis.query.text, "draft": a.draft, "issues": a.issues} for a in ans],
                                                                        ensure_ascii=False, indent=1), encoding="utf-8")
    o.hcg.save(RESULTS / "validation" / f"marketing-demo-{name}.hcg.json")
    lm.save()
    del lm, o
    gc.collect()
    torch.cuda.empty_cache()

e2e = pd.concat(cards)
e2e.to_csv(RESULTS / "e2e_qwen35.csv", index=False)
e2e.groupby("model").agg(parsed_ok=("parsed_ok", "mean"), narrative_ok=("narrative_ok", "mean"), abs_error=("abs_error", "mean"),
                         covered=("covered", "mean"), verdict=("verdict", "first"), graph_exact=("graph_exact", "first"))""",
"""gpu:pd.DataFrame([{"model": m, **d} for m in ("Qwen3.5-4B", "Qwen3.5-9B")
              for d in json.loads((RESULTS / "e2e" / f"{m}_narratives.json").read_text(encoding="utf-8"))])""",
"""gpu:display(Markdown((RESULTS / "e2e" / "Qwen3.5-9B.md").read_text(encoding="utf-8").split("\\n\\n---\\n\\n")[2]))""",
"""md:## 5. (CPU) Replaying recorded responses

Replay the 9B responses recorded on the GPU and check that the result matches section 4. Before the GPU run there is no recording, so this raises a `KeyError`.""",
"""replay:from humm.providers.cache import ReplayProvider

rp = ReplayProvider(RESULTS / "llm_cache" / "Qwen3.5-9B.json", name="Qwen3.5-9B")
o = Orchestrator.from_sources(s.data, s.corpus.docs, "marketing", rp, s.column_aliases, "marketing-demo-Qwen3.5-9B", llm=rp,
                              experiments={"ad_spend": s.experiment})
replayed = scorecard([o.ask(q) for q in QUESTIONS], s)
gpu = pd.read_csv(RESULTS / "e2e_qwen35.csv").query("model == 'Qwen3.5-9B'").reset_index(drop=True)
print("Values match the GPU run:", (replayed.value.round(4).values == gpu.value.round(4).values).all())
print("Summary check passed - GPU recording:", int(gpu.narrative_ok.sum()), "/ replay:", int(replayed.narrative_ok.sum()))  # changing post-processing such as register cleanup affects only the replay side
replayed""",
"""md:## 6. MCP server

`humm.mcp` exposes the `humm.api` functions as MCP tools. Register it in Claude Desktop or Claude Code as below, and an agent can call the tools to build graphs and answer questions (`pip install -e ".[mcp]"`).

```json
{"mcpServers": {"humm": {"command": "python", "args": ["-m", "humm.mcp"], "cwd": "<path to the humm repository>",
                         "env": {"HUMM_LLM": "hf:Qwen/Qwen3.5-9B@cuda:1"}}}}
```

| MCP tool | Module | Purpose |
|---|---|---|
| `graph_from_table`, `graph_from_text`, `graph_merge`, `graph_get`, `graph_list` | `humm.graph.*` | Build, merge and retrieve graphs |
| `validate_run` | `humm.validate.run` | L1/L2 validation, edge status update |
| `engine_identify`, `engine_estimate`, `engine_refute`, `engine_counterfactual` | `humm.engine.*` | Identification, estimation, refutation, counterfactuals |
| `ask` | (orchestrator) | Handle one question end to end |

Below is an example of calling the same functions directly from the notebook (no LLM, using the graph from section 2).""",
"""from humm import api
from humm.mcp.server import mcp

orch.hcg.save(GRAPHS / "marketing-demo.hcg.json")
print("MCP tools:", [t.name for t in await mcp.list_tools()])
print(api.engine_identify("marketing-demo", "ad_spend", "sales", files["csv"]))
print(api.engine_counterfactual("marketing-demo", files["csv"], "ad_spend", "sales", "nie", ["web_traffic"]))
print(api.ask(QUESTIONS[0], "marketing-demo", files["csv"], files["experiment_csv"], "ad_spend")["answer"])""",
"""md:## Result files

Files written by this notebook.

| File | Required | Notes |
|---|---|---|
| `results/e2e_qwen35.csv` | yes | Section 4: per-model scorecard of the questions |
| `results/e2e/Qwen3.5-4B.md`, `Qwen3.5-9B.md` | yes | Section 4: full answers per question (including LLM summary paragraphs that passed the check) |
| `results/e2e/Qwen3.5-4B_narratives.json`, `Qwen3.5-9B_narratives.json` | yes | Section 4: raw summary paragraphs and check results |
| `results/llm_cache/Qwen3.5-4B.json`, `Qwen3.5-9B.json` | yes | Recorded LLM responses (appended to the ones from notebook 07); used for the replay in section 5 |
| `results/validation/marketing-demo-Qwen3.5-4B.hcg.json`, `marketing-demo-Qwen3.5-9B.hcg.json` | optional | Validated HCG per model |

`data/demo/` and `data/graphs/marketing-demo.hcg.json` are regenerated by the notebook, so they need not be kept.""",
]

n = nbf.v4.new_notebook()
out = []
for c in cells:
    tag = next((t for t in ("md", "gpu", "replay") if c.startswith(t + ":")), None)
    body = c[len(tag) + 1:] if tag else c
    cell = nbf.v4.new_markdown_cell(body) if tag == "md" else nbf.v4.new_code_cell(body)
    cell.metadata["tags"] = [tag] if tag in ("gpu", "replay") else []
    out.append(cell)
n.cells = out
n.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(n, sys.argv[1])
