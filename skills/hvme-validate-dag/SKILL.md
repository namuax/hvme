---
name: hvme-validate-dag
description: Use to validate a built causal graph against data before using it for analysis. For requests like "check whether the graph is right", "validate the DAG", or "does this structure fit the data?" (Korean such as "DAG 검증" is supported), it runs L1 structure checks and L2 conditional-independence falsification, updates edge statuses, and leaves a validation.json.
---

## When to use

- Always use it right after `hvme-graph-from-table`, `hvme-graph-from-text`, or `graph_merge`, before estimation or counterfactuals.
- Use it again after the data changes or the graph is edited by hand.

## Procedure

1. State assumptions: write down whether the data used for validation comes from the same population as the data the graph was built on.
2. Check the graph: use `hvme.graph_get(graph_id)` to see whether the nodes match the CSV columns (nodes without a column are skipped in the tests).
3. Validate: call `hvme.validate_run(graph_id, csv, n_perm=100)`. The result is stored on the graph and written to `results/validation/<graph_id>.json`.
4. Handle violations: show `missing_edge_candidates` to the user and decide whether to add a missing edge or a confounding variable. If you fix the graph, rebuild it and repeat step 3.
5. If experimental data exists, tell the user that `hvme.ask(..., experiment_csv, experiment_treatment)` adds the L4 experiment comparison.

## Hume principles checklist

- [ ] Never assert causation from correlation alone.
- [ ] Always check for hidden confounding.
- [ ] State counterfactuals only on top of explicit assumptions.
- [ ] Confirm that passing validation means "not falsified", not proof that the graph is true.
- [ ] Check whether any edge that changed to `refuted` lies on the question's treatment→outcome path.

## Interpreting results

- `summary.verdict`: `pass` (violations are at chance level, no strong violations, and the fit is better than random graphs), `warn`, or `fail` (a cycle or time-order violation).
- `summary.is_dag`, `temporal_violations`: the L1 structure checks.
- `summary.lmc_tests`, `lmc_violation_rate`, `excess_violation_p`, `strong_violations`: the L2 local Markov condition tests.
- `summary.permutation_p`, `permutation_informative`: the fit relative to random graphs; with few tests it is uninformative.
- `summary.edges_tested`, `edges_unsupported`: whether the data supports each edge.
- `missing_edge_candidates`: a list of `[node, other, given, p]`, pairs the graph says are independent but the data shows to be dependent.
- `changes`: the log of edge status changes.
- `notes`: cautions recorded during validation.

## Limitations to disclose

- Say that the L2 tests only look at independence relations among observed columns and cannot distinguish other graphs with the same independence structure (equivalence class).
- With a small sample, violations can be missed; say so together with `n_samples`.
- If `verdict` is `warn` or `fail`, attach that fact to every later conclusion.

## Handing off to other Skills

- Pass → `hvme-identify-estimate` or `hvme-counterfactual`
- Many violations → `hvme-graph-from-table` (rebuild after adding confounding variables or edges)
- Reporting → `hvme-explain`
