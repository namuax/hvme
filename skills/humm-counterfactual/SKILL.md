---
name: humm-counterfactual
description: Use for Rung 3 questions such as "what if we had done X back then?", "would the outcome have been different without X?", or "how much of the effect went through M?" (Korean such as "만약 그때 ~했다면?" is supported). It computes ETT, NDE, NIE, PN, and PS on a binary SCM fitted to the data and reports them together with their assumptions.
---

## When to use

- Use it when `humm-causal-question` classified the question as Rung 3 (`kind="ett"` or `"mediation"`), or when the user asks for probabilities of necessity or sufficiency (PN, PS).
- Use it only when the treatment, outcome, and mediator variables are all 0/1 binary columns in the data.

## Procedure

1. State assumptions: write out, as sentences, the factual world (what actually happened) and the counterfactual world (the changed treatment).
2. Check the graph: use `humm.graph_get(graph_id)` to see whether the ancestors of the treatment, outcome, and mediators are in the data, and first confirm with `humm.engine_identify(graph_id, treatment, outcome, csv)` that the intervention effect is identified.
3. Compute the counterfactual: call `humm.engine_counterfactual(graph_id, csv, treatment, outcome, query="ett", mediators=None)`. `query` is one of `ate`, `ett`, `nde`, `nie`, `pn`, `ps`; `nde` and `nie` need `mediators=[...]`.
4. If you need intervals or sensitivity, call `humm.ask(question, graph_id, csv)` to get `estimates` (ett, or nde and nie, plus ate_scm) and `sensitivity_table`.
5. If identification fails (`strategy="not_identified"`), do not report a counterfactual number.

## Hume principles checklist

- [ ] Never assert causation from correlation alone.
- [ ] Always check for hidden confounding.
- [ ] State counterfactuals only on top of explicit assumptions.
- [ ] Put a sentence in the answer saying the counterfactual value relies on SCM mechanism assumptions (thresholds, monotonicity, independent noise per node).
- [ ] For mediation decomposition, confirm the assumption that there is no confounder between mediator and outcome that is affected by the treatment.
- [ ] Check that the ETT sign is "actual minus counterfactual" and matches the direction of the user's question.

## Interpreting results

- `engine_counterfactual`: `query`, `value`, `exact` (true means exact computation, false means a Monte Carlo approximation), and `nodes` (the nodes used in the SCM).
- `ask` result: if `kind` is `ett`, read `estimates.ett`; if `mediation`, read `estimates.nde` and `estimates.nie`; `estimates.ate` and `estimates.ate_scm` are present in both cases, and each item has `value` and `ci`. Also read `mediators`, `assumptions`, and `graph_sensitivity`.
- `ask` does not handle `pn` and `ps`, so get them only from `engine_counterfactual`; they have no interval.

## Limitations to disclose

- Always say that a counterfactual is a quantity that observational data alone does not identify, and that it holds only under the fitted binary SCM's assumptions.
- The engine supports only binary variables, so for a continuous-variable question either state the binarization rule or answer "not identifiable + the data needed".
- If `exact=false`, say the value is an approximation.
- Say whether mediation or ETT identification breaks under a hidden-confounder alternative (`graph_sensitivity.breaks_with_hidden_confounder`).

## Handing off to other Skills

- Checking the result → `humm-refute`
- Reporting → `humm-explain`
- If only the intervention effect is needed → `humm-identify-estimate`
