---
name: humm-refute
description: Use to check whether an estimated causal effect is trustworthy. For requests like "can I trust this result?", "check the effect", "refutation tests", or "sensitivity analysis" (Korean such as "이 결과 믿어도 돼?" is supported), it runs placebo, random-common-cause, and data-subset refutations and checks graph sensitivity (L5).
---

## When to use

- Use it right after getting a number from `humm-identify-estimate` or `humm-counterfactual`, before reporting.
- Use it when the user asks how robust the result is.

## Procedure

1. State assumptions: write down which estimate (treatment, outcome, identification strategy) is being checked.
2. Check the graph: use `humm.graph_get(graph_id)` to confirm that the graph under test is the same one used for the estimate.
3. Refute: call `humm.engine_refute(graph_id, csv, treatment, outcome)`.
4. Graph sensitivity: look at `sensitivity_table` and `graph_sensitivity` in `humm.ask(question, graph_id, csv)` (alternative graphs: edge deletion, reversal, restoration, and adding a hidden confounder `_U`).
5. Independent check: look at `agree` from `humm.lm_verify(graph_id, csv, treatment, outcome)`.
6. If a test fails, go back to the graph-acquisition step and add confounding variables (`humm-graph-from-table` → `humm-validate-dag`).

## Hume principles checklist

- [ ] Never assert causation from correlation alone.
- [ ] Always check for hidden confounding.
- [ ] State counterfactuals only on top of explicit assumptions.
- [ ] Do not rephrase "the refutations passed" as "causation has been proven".
- [ ] Check whether identification breaks in the hidden-confounder scenario (`breaks_with_hidden_confounder`).

## Interpreting results

- Each `engine_refute` row has `test` (placebo_treatment / random_common_cause / data_subset), `expected`, `new_value`, `estimate`, `tolerance`, and `passed`.
  - placebo_treatment: with the treatment shuffled, the effect should be near 0.
  - random_common_cause: adding a random common cause should leave the value unchanged.
  - data_subset: an 80% subsample should give the same value.
- `sensitivity_table` rows have `scenario`, `strategy`, `variables`, and `value`; `strategy="not_identified"` means identification breaks in that alternative.
- `graph_sensitivity`: `alternative_graphs`, `sign_flipped`, `effect_vanished`, and `breaks_with_hidden_confounder`. If `sign_flipped` is not 0, the conclusion rests on particular edges.

## Limitations to disclose

- Say that refutation tests catch errors in the estimation procedure but cannot rule out graph errors or hidden confounding.
- If any alternative graph flips the sign, name that scenario.
- Do not hide failed tests; lower the confidence in the conclusion accordingly.

## Handing off to other Skills

- Pass → `humm-explain`
- Fail → `humm-graph-from-table` / `humm-validate-dag`
