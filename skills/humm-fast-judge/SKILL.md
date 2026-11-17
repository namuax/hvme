---
name: humm-fast-judge
description: Use to process many yes/no, multiple-choice, or scale judgments quickly. For requests like "judge whether all these edge directions are right", "pick one of these options", or "rate these 1 to 5" (Korean such as "1~5점으로 매겨줘" is supported), it calls Humm-Fast (fast_noul, fast_choice, fast_score) and escalates low-confidence results to a higher path.
---

## When to use

- Use it when there are many judgments of the same form, such as pairwise edge judgments, counterfactual sentence checks, or query classification.
- Do not use it for questions that need a numeric effect or a counterfactual computation (use `humm-identify-estimate` or `humm-counterfactual` for those).

## Procedure

1. State assumptions: write down the meaning of the proposition, options, or scale to be judged, and put the needed background (graph summary, domain) into `context`.
2. Check the graph: for a judgment about the graph, copy the relevant edges and nodes from `humm.graph_get(graph_id)` into `context`.
3. Choose the tool call.
   - Yes/no: `humm.fast_noul(proposition, context="")`
   - Choice: `humm.fast_choice(question, options=[...], context="")`
   - Scale: `humm.fast_score(question, levels=5, context="")`
4. Escalate: do not use results where `escalate` is true. If a numeric computation is needed, escalate to `humm.ask`; if it is a judgment over a graph and data, escalate to `humm.lm_verify(graph_id, csv, treatment, outcome)`; if neither works, tell the user the judgment cannot be made.
5. Count the escalation rate and report it to the user along with the results.

## Hume principles checklist

- [ ] Never assert causation from correlation alone.
- [ ] Always check for hidden confounding.
- [ ] State counterfactuals only on top of explicit assumptions.
- [ ] Do not mix `escalate=true` results into the final answer.
- [ ] A Fast judgment is not data-backed verification evidence, so never use it as grounds to raise an edge's status to `verified`.

## Interpreting results

- `fast_noul`: `p_yes` (probability of yes), `confidence`, `escalate`.
- `fast_choice`: `choice` (the chosen option), `probs` (probability per option), `confidence`, `escalate`.
- `fast_score`: `score` (representative score), `dist` (probability per score), `confidence`, `escalate`.
- `confidence` is a calibrated value, so read it like a probability, but calibration may be less accurate on out-of-distribution items.

## Limitations to disclose

- Say that Humm-Fast is a single-pass classifier and does not compute anything.
- Report `confidence` with every judgment, and state how many items were escalated.
- Warn that accuracy can vary by a few points on sentence forms that differ from the training distribution (external sets).

## Handing off to other Skills

- Escalated causal questions → `humm-causal-question` or `humm-identify-estimate`
- Reflecting edge judgments in the graph → `humm-validate-dag`
