"""기준선 프롬프트. Noul(로짓) 방식은 답 토큰 하나만 읽고, CoT 방식은 CausalCoT(Jin et al., 2023) 단계를 따라 생성한다."""

NOUL_SUFFIX = "\n\nAnswer with Yes or No only."

CAUSAL_COT = """{question}

Solve this step by step.
Step 1. Extract the causal graph: list the variables and the direct causal edges.
Step 2. Classify the query type (e.g., marginal, conditional, ATE, ATT, NDE, NIE, backdoor adjustment set, collider bias, counterfactual).
Step 3. Formalize the query in symbolic form (e.g., E[Y | do(X=1)] - E[Y | do(X=0)]).
Step 4. Collect the available data (probabilities) from the question.
Step 5. Derive an estimand from the causal graph using causal inference rules.
Step 6. Compute the estimand with the data and answer the question.

End with a line in the form "Final answer: yes" or "Final answer: no"."""

CORR2CAUSE_NOUL = """{input}

Does the premise imply the hypothesis?"""


def cladder_noul(row) -> str:
    return row["prompt"] + NOUL_SUFFIX


def cladder_cot(row) -> str:
    return CAUSAL_COT.format(question=row["prompt"])


def corr2cause_noul(row) -> str:
    return CORR2CAUSE_NOUL.format(input=row["input"]) + NOUL_SUFFIX


def corr2cause_cot(row) -> str:
    return CAUSAL_COT.format(question=CORR2CAUSE_NOUL.format(input=row["input"]))
