---
name: humm-graph-from-table
description: Use to build a causal graph (HCG) from a CSV or DB table. For requests like "build the causal structure from this data", "find causal relations among these variables", or "causal analysis of this CSV" (Korean such as "CSV 인과 분석" is supported), it saves a graph that went through PC discovery, LLM pairwise priors, and data validation.
---

## When to use

- Use it when you have structured data (CSV) and no saved graph exists (`graph_list`), or a new one must be built.
- Use it in the graph-acquisition step before answering a causal question.

## Procedure

1. State assumptions: confirm the column names, meanings, and measurement times with the user, and write down the unit of analysis (customer, day, region). The engine's estimation and counterfactuals assume 0/1 binary columns, so ask whether continuous or categorical columns need binarizing.
2. Build the graph: call `humm.graph_from_table(csv, graph_id, domain="<description of the field>", alpha=0.01, mode="hybrid", llm="")`. `mode="hybrid"` mixes LLM pairwise priors into PC discovery. If `llm` is empty, the environment variable `HUMM_LLM` is used.
3. Check the graph: show the returned `edges` to the user and point out any direction that conflicts with domain common sense (for example, the outcome pointing at its cause). To view it again, use `humm.graph_get(graph_id)`.
4. If documents exist, build a text graph with `humm-graph-from-text` and merge it with `humm.graph_merge(text_graph, table_graph, graph_id, column_aliases)`.
5. Always hand off to `humm-validate-dag` before using the graph.

## Hume principles checklist

- [ ] Never assert causation from correlation alone.
- [ ] Always check for hidden confounding.
- [ ] State counterfactuals only on top of explicit assumptions.
- [ ] Confirm that an edge found by PC is a conditional-independence pattern, and that its direction rests on prior knowledge and time order.
- [ ] Ask the user about important variables missing from the data (unmeasured confounder candidates).

## Interpreting results

- `nodes`: variable ids, `label`, and `aliases`.
- `edges[]`: `src`→`dst`, `sign`, `status` (hypothesis/supported/verified/refuted), `confidence`, and `evidence` (data/llm_prior/text/expert/experiment).
- Treat an edge whose `evidence` is only `llm_prior` as a direction judgment with weak data support.
- `verdict`: the L1 and L2 validation verdict at build time (pass/warn/fail).
- `conflicts`: the list of edges where the data and the LLM prior disagree; these need user confirmation.

## Limitations to disclose

- Say that observational data alone cannot distinguish directions within a Markov equivalence class, so directions rest on prior-knowledge assumptions.
- If `conflicts` is non-empty or `verdict` is not pass, say so first.
- Unmeasured confounders do not appear in the graph, so say that later conclusions rest on that assumption.

## Handing off to other Skills

- Next step → `humm-validate-dag`
- Adding document evidence → `humm-graph-from-text`
- Checking many edge directions → `humm-fast-judge` (`fast_noul`)
