---
name: humm-graph-from-text
description: Use to extract causal structure from documents such as reports, papers, logs, or meeting notes and build an HCG. For requests like "summarize the cause-effect relations in this report" or "build a causal graph from these documents" (Korean such as "문서로 인과 그래프 만들어줘" is supported), it runs causal-claim extraction, normalization, cross-document aggregation, and counterfactual verification.
---

## When to use

- Use it when the causal evidence exists only in documents, or when you want to add domain knowledge (directions, hidden variables) to a data graph.
- An LLM is required, so the `llm` argument or `HUMM_LLM` must be set.

## Procedure

1. State assumptions: write down the document's source, date, and reliability, and first tell the user that claims in documents are only "claims", not verified causation.
2. Prepare documents: provide either JSONL (one `{"id", "text"}` per line) or the path to a folder of `.txt`/`.md` files.
3. Build the graph: call `humm.graph_from_text(docs, graph_id, domain="<field>", llm="")`.
4. Check the graph: look at the returned `edges` and `claims` (the number of extracted claims), and use `nodes[].aliases` to check whether the same concept was split into different nodes.
5. If data exists, merge into the tabular graph with `humm.graph_merge(text_graph, table_graph, graph_id, column_aliases={"<column>": ["name in document", ...]})`. Check how nodes were linked in the returned `link`.
6. Hand off to `humm-validate-dag` before using the graph (if there is no data, say that only L1 structural validation is possible).

## Hume principles checklist

- [ ] Never assert causation from correlation alone.
- [ ] Always check for hidden confounding.
- [ ] State counterfactuals only on top of explicit assumptions.
- [ ] Check the quotes (`quote`) to make sure that "rose together" in a document was not promoted to a causal edge.
- [ ] Mark an edge that appears in only one document as single-source.

## Interpreting results

- If `edges[].evidence` contains `text`, the edge is backed by documents, and `confidence` reflects cross-document aggregation and counterfactual verification.
- An edge with `status` `hypothesis` has not yet been confirmed by data.
- If `claims` is small, extraction was thin, so suggest adding more documents.
- `link` in the merge result is the mapping between text nodes and data columns; fix wrongly joined pairs with `column_aliases`.

## Limitations to disclose

- Text edges rest on the authors' claims and LLM extraction; say they are hypotheses until validated against data.
- Tell the user that text nodes with no data column are treated as unobserved (latent) variables in later analysis.
- State which LLM was used for extraction and the scope of the documents.

## Handing off to other Skills

- Next step → `humm-validate-dag`
- If there is no data graph → `humm-graph-from-table`
- Checking many edge judgments → `humm-fast-judge`
