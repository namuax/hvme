---
name: humm-identify-estimate
description: Use to compute a treatment or intervention effect (ATE). For Rung 2 questions like "how much will sales rise if we raise ad spend?", "what is the effect of this policy?", or "the impact of X on Y" (Korean such as "X가 Y에 미치는 영향" is supported), it runs an identifiability check, then estimation, and returns a 95% confidence interval.
---

## When to use

- Use it for questions that `humm-causal-question` classified as Rung 2 (`kind="ate"`).
- Use it when you have a validated graph (passed `humm-validate-dag`) and a CSV with 0/1 binary columns.

## Procedure

1. State assumptions: write down the treatment (what 1/0 mean), the outcome, and the population.
2. Check the graph: use `humm.graph_get(graph_id)` to check the treatment→outcome path and the arrows into the treatment (confounding paths).
3. Identify: call `humm.engine_identify(graph_id, treatment, outcome, csv)`. If you pass `csv`, nodes missing from the data are treated as unobserved variables.
4. If `strategy` is `not_identified`, stop estimation and answer "not identifiable + the data needed" (say which variables, if measured, would open a backdoor, front-door, or IV route).
5. Estimate: call `humm.engine_estimate(graph_id, csv, treatment, outcome, n_boot=200)`.
6. If you need an end-to-end answer, call `humm.ask(question, graph_id, csv)` once to get estimation, refutation, sensitivity, and assumptions together.
7. Independent check: use `humm.lm_verify(graph_id, csv, treatment, outcome)` to see whether Humm-LM's answer and the engine's sign agree.

## Hume principles checklist

- [ ] Never assert causation from correlation alone.
- [ ] Always check for hidden confounding.
- [ ] State counterfactuals only on top of explicit assumptions.
- [ ] Check that the adjustment set contains no consequence of the treatment (mediators or colliders).
- [ ] If the interval contains 0, do not assert that there is or is not an effect.

## Interpreting results

- `engine_identify`: `strategy` (backdoor/frontdoor/iv/no_effect/not_identified), `variables` (adjustment, mediator, and instrument variables), `expression` (the estimand), and `latent` (unobserved nodes).
- `engine_estimate`: `value` (the ATE, the difference in P(outcome=1) between treatment 1 and 0), `se`, `ci` (bootstrap 95% interval), `method`, and `strategy`.
- `ask` result: `estimates.ate.value` and `estimates.ate.ci`, `adjustment`, `expression`, `paths` (evidence paths), `backdoor_paths`, `refutation`, `sensitivity_table`, `graph_sensitivity`, `assumptions`, and `answer`.
- `lm_verify`: if `agree` is false, the engine and Humm-LM disagree on the sign, so re-examine the graph and the arguments. `trace` holds the LM's tool-call process.
- `strategy="no_effect"` means the graph has no causal path, and the conclusion rests entirely on the graph structure.

## Limitations to disclose

- Always state the `assumptions` (the adjustment set blocks all confounding paths, positivity, SUTVA, unvalidated edges).
- Also state the confidence interval and `graph_sensitivity` (the number of alternatives with a flipped sign, and whether identification breaks under a hidden confounder).
- If the effect is not identifiable, give no number and answer "not identifiable + the data needed".
- If `lm_verify.agree=false`, do not hide that fact.

## Handing off to other Skills

- Checking the result → `humm-refute`
- If the question turns into a counterfactual or mediation question → `humm-counterfactual`
- Reporting → `humm-explain`
