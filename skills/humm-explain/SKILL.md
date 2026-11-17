---
name: humm-explain
description: Use when reporting causal analysis results to the user. For requests like "summarize the results", "write it up as a report", or "explain why you reached that conclusion" (Korean such as "결과 정리해줘", "보고서로 써줘" is supported), it builds a report from the markdown of ask that includes the evidence paths, assumptions, confidence, and graph sensitivity (L5), leaving none out.
---

## When to use

- Always use it when writing the final answer after getting numbers from other humm Skills.
- Use it to check whether an already-written answer is missing evidence or assumptions.

## Procedure

1. State assumptions: put the question, graph (graph_id), data (csv), and validation verdict at the top.
2. Check the graph: use `humm.graph_get(graph_id)` to re-check the `status` and `confidence` of the edges on the evidence paths.
3. Get the result: call `humm.ask(question, graph_id, csv)` and take `markdown`. If experimental data exists, pass `experiment_csv` and `experiment_treatment` to add the L4 comparison.
4. Check the required items: confirm that `markdown` contains all four of the following. If any is missing, do not issue the report; call the relevant tool again. (The section headings below are the literal Korean strings that the tool emits.)
   - Evidence paths (`### 근거 경로`, field `paths`)
   - Assumptions (`### 가정`, field `assumptions`)
   - Confidence (`ci` in `estimates`, edge `confidence`, validation `verdict`, `passed` in `refutation`)
   - Graph sensitivity (`### 그래프 민감도 (L5)`, fields `sensitivity_table` and `graph_sensitivity`)
5. Summary paragraph: if `narrative` is non-empty, use it. If `issues` is present, discard the LLM summary and use only `answer` and the tables. Copy only numbers that appear in the result.
6. If a figure is needed, draw the evidence path as a text diagram (`A → B → C`) from the edges in `paths`.

## Hume principles checklist

- [ ] Never assert causation from correlation alone.
- [ ] Always check for hidden confounding.
- [ ] State counterfactuals only on top of explicit assumptions.
- [ ] Check that every number in the report is a value that appears in the tool output.
- [ ] Do not read the two ends of a 95% interval as effect sizes.
- [ ] If a conclusion rests on an edge that is not `verified`, name that edge.

## Interpreting results

- `answer`: a one-line answer built from the engine result; use it as the first sentence of the report.
- `kind`, `identification`, `strategy`, `expression`, `adjustment`: the question type and the identification method.
- `estimates`: `{name: {"value", "ci"}}`; depending on the type it contains `ate`, `association`, `ett`, `nde`, `nie`, and `ate_scm`.
- `paths`: for each path, a list of edges (`edge`, `status`, `confidence`, `sign`, `evidence`, `quote`).
- `backdoor_paths`: candidate confounding paths blocked by adjustment.
- `refutation`: refutation test rows (`test`, `expected`, `new_value`, `passed`).
- `graph_sensitivity`, `sensitivity_table`: the number of alternative graphs, sign flips, effect vanishing, and whether identification breaks under a hidden confounder.
- `experiment`: the L4 experiment comparison (including `consistent`); present only if experimental data was provided.
- `parser`: how the query was parsed (`rules`, `rules+<llm>`, and for v1 `rules+rung:fast` or `rules+rung:lm`).
- `route` and `lm_check` (`ask_v1` only): the Rung classification (confidence, fast/lm) and the independent Humm-LM solution (`lm_answer`, `agree`). If `agree` is false, write that the graph and assumptions should be re-examined before concluding; if null, write that judgment is withheld. In the markdown these appear in the "Humm-LM 검증" (Humm-LM verification) section.

## Limitations to disclose

- Include the `assumptions` list with its original numbering.
- Always state the confidence interval, the validation verdict (`pass`/`warn`/`fail`), the number of failed refutations, and the number of alternative graphs whose sign flipped.
- If the effect is not identifiable, write "not identifiable + the data needed (variables to measure, experiments)" as the answer.
- State that counterfactual and mediation numbers are values under binary SCM assumptions.

## Handing off to other Skills

- If the evidence is insufficient → `humm-refute` or `humm-validate-dag`
- New question → `humm-causal-question`
