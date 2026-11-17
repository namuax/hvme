# HCG: the causal graph format

An HCG is a JSON document validated by pydantic (`hvme.graph.schema.HCG`). It is the single object that graph construction writes, validation updates and the engine reads.

```json
{
  "graph_id": "marketing",
  "nodes": [
    {"id": "ad_spend", "label": "광고비", "type": "variable", "dtype": "binary",
     "aliases": ["광고 예산", "마케팅 지출"], "source": [{"kind": "table", "ref": "data.csv"}]}
  ],
  "edges": [
    {"src": "ad_spend", "dst": "sales", "sign": "+", "lag": 0, "mechanism": "",
     "evidence": [{"kind": "data", "score": 0.99, "method": "pc"},
                  {"kind": "text", "score": 0.9, "ref": "doc-03", "span": "광고비를 늘리자 매출이 올랐다"}],
     "confidence": 0.993, "status": "supported"}
  ],
  "assumptions": ["no unobserved confounding between ad_spend and sales"],
  "provenance": {"created": "2026-11-17", "pipeline": "hvme 0.1.0"}
}
```

## Nodes

| Field | Values |
|---|---|
| `id` | identifier used in data columns and engine calls |
| `label`, `aliases` | names used in documents and questions; questions are matched to nodes through them |
| `type` | `variable` or `event` |
| `dtype` | `binary`, `categorical`, `continuous`, `unknown` |
| `source` | where the node came from: `text`, `table`, `llm`, `expert` |

## Edges

| Field | Values |
|---|---|
| `sign` | `+`, `-`, `0`, `?` |
| `lag` | time lag in steps (0 = same period) |
| `evidence` | list of `{kind, score, ref, span, method}`; kinds include data tests, text spans, LLM priors, experiments and expert input |
| `confidence` | 0–1, combined from the evidence |
| `status` | `hypothesis` → `supported` or `refuted` after validation, and further states for edges confirmed by experiments or by an expert |

An edge supported only by an LLM prior keeps that single evidence kind, so it can be reviewed before anyone relies on it.

## Rules

- The graph of edges that are not `refuted` must be acyclic.
- Edge endpoints must be node ids.
- `refuted` edges stay in the file with their evidence; the engine ignores them.

Use `HCG.load(path)`, `hcg.save(path)` and `hvme.api.graph_summary(hcg)`.
