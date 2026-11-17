---
name: humm-causal-question
description: Entry-point Skill for causal questions. Use it first when the user asks "why did X happen?", "what happens if we do X?", "is X the cause of Y?", or "what if we had done X?" (Korean questions such as "왜 ~했나?", "~하면 어떻게 돼?" are supported). It classifies the question into Rung 1/2/3, picks the processing path, and stops correlation from being presented as causation.
---

## When to use

- Use it when the question involves a cause, an effect, an intervention, or a counterfactual ("If we raise ad spend, will sales go up?", "Is the rise in churn because of price?").
- Use it when the question type is unclear, before choosing another humm Skill.
- Bulk yes/no judgments that need no numeric answer go to `humm-fast-judge`, not this Skill.

## Procedure

1. State assumptions: restate the treatment (candidate cause), outcome, and conditions in the user's words, and check whether data (CSV) and a graph (graph_id) exist.
2. Check the graph: find saved graphs with `humm.graph_list()`. If none exists, build one first with `humm-graph-from-table` or `humm-graph-from-text`. If one exists, use `humm.graph_get(graph_id)` to see whether the question's variables are nodes.
3. Classify: call `humm.classify_question(question, graph_id)`. If `router` is `"lm"` or `confidence` is low, do not trust the classification as is; confirm `treatment` and `outcome` with the user.
4. Choose the path by Rung.
   - Rung 1 (`kind="association"`): call `humm.ask(question, graph_id, csv)` to get the observed difference and the confounder-adjusted effect together, and compare them.
   - Rung 2 (`kind="ate"`): hand off to `humm-identify-estimate`.
   - Rung 3 (`kind="ett"` or `"mediation"`, or PN/PS-style "what if we had done X then"): hand off to `humm-counterfactual`.
   - To handle everything end to end in one call, use `humm.ask_v1(question, graph_id, csv)`. For a question with no type keyword, it decides the type by Rung classification, and the answer carries `route` (the classification) and `lm_check` (a comparison of the independent Humm-LM solution against the engine's sign).
5. If there is no data, do not invent numbers. Give a qualitative answer from the graph paths alone (whether a path exists, and its sign), or say which data is needed.

## Hume principles checklist

- [ ] Never assert causation from correlation alone.
- [ ] Always check for hidden confounding.
- [ ] State counterfactuals only on top of explicit assumptions.
- [ ] If the question itself presupposes causation ("prices rose because of X"), turn that presupposition back into something to be checked.
- [ ] Check that the direction of treatment and outcome matches the question (`treatment` and `outcome` from `classify_question`).

## Interpreting results

- `rung` (1/2/3) and `kind` (ate/association/mediation/ett) in the `classify_question` result decide the path.
- If `mediators` is non-empty, it is a mediation question, so treat it as Rung 3.
- `router="fast"` means the Humm-Fast classifier is confident; `"lm"` means confidence was low and the question was escalated.
- In the `ask` result on the Rung 1 path, read the gap between `estimates.association` (observed difference) and `estimates.ate` (adjusted effect) as the share due to confounding paths (see `backdoor_paths`).

## Limitations to disclose

- The classification is an estimate; report `confidence` with it.
- Do not answer a Rung 1 question with "it is causal"; say the adjusted effect holds only under the graph's assumptions.
- If the question's variables cannot be found in the graph, answer "not identifiable + the data needed (columns for those variables, relevant documents)".

## Handing off to other Skills

- Rung 2 → `humm-identify-estimate`
- Rung 3 → `humm-counterfactual`
- Graph missing or stale → `humm-graph-from-table` / `humm-graph-from-text` → `humm-validate-dag`
- Final report → `humm-explain`
