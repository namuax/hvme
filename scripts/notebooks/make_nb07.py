import sys
import nbformat as nbf

cells = [
"""md:# 07. Text → causal graph — extraction, normalization, aggregation, counterfactual verification

This notebook builds a causal graph (HCG) from unstructured text such as reports and documents, and merges it with a graph built from tabular data.

| Stage | What it does | Module |
|---|---|---|
| T1 | Split documents into chunks at sentence boundaries | `chunk` |
| T2-T3 | For each chunk, the LLM extracts relation claims (causal, correlational, no effect; increase/decrease; hedged or not) as JSON | `extract_claims` |
| T5 | Group different expressions of the same variable under a canonical name | `canonicalize` |
| T6 | Count claims per document to produce edge candidates and scores. Opposite-direction and no-effect claims count as counter-evidence | `aggregate` |
| T7 | Ask "if the cause changes, does the effect change?" in the forward and reverse directions, and attach the probabilities as LLM evidence (not used as a filter) | `verify_edges` |
| Merge | Attach the text HCG to the tabular HCG and validate with data | `merge_hcg`, `validate_hcg` |

Performance can only be measured against a ground-truth graph, so we evaluate on a **synthetic corpus**. Documents are generated from a ground-truth DAG and mixed with synonyms, several sentence patterns, correlational statements (distractors), no-effect statements, hedged expressions and wrong claims. Sections 1-4 and 6 run on CPU (simulated LLM or replay of recorded responses); section 5 needs a GPU.""",
"""import networkx as nx
import pandas as pd

from humm.engine import BinarySCM
from humm.graph.tabular import tabular_to_hcg
from humm.graph.text import TEXT_VOCAB, TextOracle, evaluate, link_nodes, make_corpus, merge_hcg, text_to_hcg
from humm.paths import RESULTS
from humm.validate import validate_hcg

pd.set_option("display.max_colwidth", 80)
DOMAINS, LANGS = ["marketing", "manufacturing", "it_ops"], ["ko", "en"]
corpora = {(d, l): make_corpus(d, l, seed=1) for d in DOMAINS for l in LANGS}""",
"""md:## 1. Synthetic corpus

3 domains × 2 languages, with 12 documents per corpus. Each of the 9 ground-truth edges appears in 3 documents, and 15% of documents contain wrong claims (reversed direction, nonexistent edge). The ground-truth edges follow a plausible causal order for the domain.""",
"""c = corpora[("marketing", "ko")]
print("Ground-truth edges:", [(a, b, d["sign"]) for a, b, d in c.graph.edges(data=True)])
print("\\nExample document:", c.docs[0]["text"])
pd.DataFrame([{"sentence": s, **{k: n[k] for k in ("cause", "effect", "relation", "polarity", "hedged")}, "wrong claim": n.get("wrong", False)}
              for s, n in list(c.annotations.items())[:12]])""",
"""md:## 2. The pipeline step by step (simulated LLM, error rate 0.1)

`TextOracle` is a simulated LLM that knows the ground truth but errs on purpose. In extraction it drops claims or flips directions, in normalization it wrongly separates expressions, and in counterfactual verification it flips answers.""",
"""r = text_to_hcg(c.docs, "marketing-ko", "marketing", TextOracle(c, 0.1, seed=3))
print("Extracted claims:", len(r.claims), "| canonical variables:", len(set(r.mapping.values())))
r.claims[["doc_id", "cause", "effect", "relation", "polarity", "hedged"]].head(8)""",
"""pd.Series(r.mapping, name="canonical name").to_frame().sort_values("canonical name")""",
"""r.candidates[["src", "dst", "support", "contra", "correlational", "score", "sign"]].merge(r.edges[["src", "dst", "p_fwd", "p_rev"]], how="left", on=["src", "dst"])""",
"""md:The aggregation score floor (default 0.55) keeps only edges that at least two different documents state in the same direction. An edge claimed by a single wrong document is filtered out here. The counterfactual verification probability (p_fwd) does not filter edges; it is attached as LLM evidence. A final edge has two kinds of evidence, text and LLM, so it becomes supported.

> Filtering edges by counterfactual verification dropped the 9B model's F1 from 0.82 to 0.51. The mean p_fwd of true and false candidates was 0.48 and 0.51, which is not separable: the LLM's counterfactual judgment reflects world knowledge rather than what the text says. The filtering criterion was therefore changed to the cross-document aggregation score.""",
"""print(evaluate(list(zip(r.edges.src, r.edges.dst)), r.mapping, c))
[(e.key, e.sign, e.status.value, round(e.confidence, 2)) for e in r.hcg.edges]""",
"""md:## 3. Error-rate experiment (simulated LLM)

Vary the extraction and normalization error rate and look at the average performance over the six corpora. We compare an aggregation score floor of 0.4 (a single document suffices) with 0.55 (at least two documents). The simulated LLM answers counterfactual verification with knowledge of the ground truth, so the verification-filter setting is not compared here.""",
"""rows = []
for (d, l), corp in corpora.items():
    for err in (0.0, 0.1, 0.2, 0.3):
        for ms in (0.4, 0.55):
            res = text_to_hcg(corp.docs, f"{d}-{l}", d, TextOracle(corp, err, seed=3), min_score=ms)
            rows.append({"domain": d, "lang": l, "error": err, "min_score": ms, **evaluate(list(zip(res.edges.src, res.edges.dst)), res.mapping, corp)})
oracle_res = pd.DataFrame(rows)
oracle_res.groupby(["error", "min_score"])[["precision", "recall", "f1", "reversed", "group_purity", "var_split"]].mean().round(3)""",
"""md:## 4. (GPU) Real LLMs — Qwen3.5

Process the six corpora with two models and compare with the ground-truth graph. We measure the old setting (score floor 0.4 + filtering with a verification threshold of 0.5) alongside the new default (score floor 0.55, verification used only as evidence). Responses are recorded to `results/llm_cache/`, so thresholds can be varied and re-evaluated by replay without a GPU.

> Replay analysis (9B): the new default gives precision 0.94, recall 0.87 and F1 0.90, with no reversed edges. The old setting gave F1 0.51.""",
"""import gc

import torch

from humm.providers.cache import CachedProvider
from humm.providers.hf import HFProvider

MODELS = {
    "Qwen/Qwen3.5-4B": "cuda:0",
    "Qwen/Qwen3.5-9B": "cuda:1",
}
llm_rows = []
for model_id, device in MODELS.items():
    lm = CachedProvider(HFProvider(model_id, device=device, batch_size=8), RESULTS / "llm_cache" / f"{model_id.split('/')[-1]}.json")
    for (d, l), corp in corpora.items():
        for setting, kw in {"old: score≥0.4 + verify filter": {"min_score": 0.4, "verify_threshold": 0.5}, "default: score≥0.55": {}}.items():
            res = text_to_hcg(corp.docs, f"{d}-{l}", d, lm, **kw)
            llm_rows.append({"model": model_id, "domain": d, "lang": l, "setting": setting, "claims": len(res.claims),
                             **evaluate(list(zip(res.edges.src, res.edges.dst)), res.mapping, corp)})
    lm.save()
    del lm
    gc.collect()
    torch.cuda.empty_cache()

llm_res = pd.DataFrame(llm_rows)
llm_res.to_csv(RESULTS / "text_llm.csv", index=False)
llm_res.groupby(["model", "setting"])[["claims", "precision", "recall", "f1", "reversed", "group_purity", "unmapped_phrases"]].mean().round(3)""",
"""llm_res.pivot_table(index=["model", "setting"], columns=["lang", "domain"], values="f1").round(2)""",
"""md:## 5. (CPU) Merging text and data (replay)

Generate tabular data from the marketing corpus's ground-truth DAG (PC only, no LLM) and merge it with the text HCG built by the 9B model. Nodes of the two graphs are linked through an alias dictionary. When opposite-direction edges coexist, the one with stronger **orientation evidence** (text, LLM, expert, and PC's v-structures and Meek rules) is kept. A direction PC chose arbitrarily does not count as orientation evidence.""",
"""from humm.providers.cache import ReplayProvider

c = corpora[("marketing", "ko")]
df = BinarySCM.random(c.graph, seed=5).sample(8000, seed=1)[sorted(c.graph.nodes)]
tab = tabular_to_hcg(df, "marketing-data", n_boot=10, validate_kw={"n_perm": 50})
txt = text_to_hcg(c.docs, "marketing-text", "marketing", ReplayProvider(RESULTS / "llm_cache" / "Qwen3.5-9B.json", name="9B"))
link = link_nodes(txt.hcg, tab.hcg, {v[0]: v[1] + v[2] for v in TEXT_VOCAB["marketing"]})
merged = merge_hcg(txt.hcg, tab.hcg, link, "marketing-merged")
report, changes = validate_hcg(merged, df, n_perm=50)
print("Links:", link)
print("Verdict:", report.summary()["verdict"], "| edges with data only:", len(tab.hcg.active_edges()), "| active edges after merge:", len(merged.active_edges()))
changes.assign(true_edge=[tuple(e.split("→")) in set(c.graph.edges) for e in changes.edge])""",
"""report.save(RESULTS / "validation" / "marketing-merged.json")
merged.save(RESULTS / "validation" / "marketing-merged.hcg.json")
changes.groupby("after").edge.count()""",
"""md:## Result files

Files written by this notebook.

| File | Required | Notes |
|---|---|---|
| `results/text_llm.csv` | yes | Section 4: edge F1 and normalization quality per model and corpus |
| `results/llm_cache/Qwen3.5-4B.json`, `Qwen3.5-9B.json` | yes | Recorded LLM responses (appended to the ones from notebook 06); replay them to experiment with thresholds |
| `results/validation/marketing-merged.json`, `marketing-merged.hcg.json` | yes | Section 5: validation report and HCG of the merged graph |""",
]

n = nbf.v4.new_notebook()
n.cells = [nbf.v4.new_markdown_cell(c[3:]) if c.startswith("md:") else nbf.v4.new_code_cell(c) for c in cells]
n.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(n, sys.argv[1])
