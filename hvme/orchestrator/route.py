"""The two model stages of orchestrator v1.

1. Query analysis: Hvme-Fast picks the question's Rung (1 association, 2 intervention, 3 counterfactual) in a single forward pass. If the confidence (max option probability) is
   below threshold, Hvme-LM is asked again for JSON. Variables (treatment, outcome, mediators) are decided from expressions in the question string, as in v0.
5. Verification: Hvme-LM (tool mode) reads the probability table fitted from the graph and data as a Gym context and solves "does the treatment raise the outcome?" independently.
   The answer is compared with the sign of the engine-estimated ATE to report agreement. The LM does not produce numbers itself; it computes them with the causal_engine tool.
"""
import re

import networkx as nx
import numpy as np
import pandas as pd

from ..bench.runner import parse_yes_no
from ..graph.schema import HCG
from ..gym.formats import render
from ..gym.verbalize import Lexicon, context, describe_cpt, describe_graph
from .analysis import fit_scm, latent_of

RUNG_KIND = {1: "association", 2: "ate", 3: "ett"}
RUNG_OPTIONS = ["Rung 1 (association): what is observed together, correlations, conditional probabilities",
                "Rung 2 (intervention): what happens if we actively do or change something",
                "Rung 3 (counterfactual): what would have happened for the same cases under a different past action, or effects through a mediator"]
RUNG_CONTEXT = "Classify the causal question below by the rung of Pearl's ladder of causation it asks about."

LM_PROMPT = """Classify the causal question by the rung of Pearl's ladder of causation.
- 1: association (correlation, observed co-occurrence, conditional probability)
- 2: intervention (effect of actively changing something, "if we do X")
- 3: counterfactual (what would have happened otherwise for the same cases, or direct/indirect effects through a mediator)

Question: {question}

Answer with one JSON object only: {{"rung": 1}}"""


def rung_item(question: str) -> dict:
    return {"context": RUNG_CONTEXT, "question": f"Question: {question}", "options": RUNG_OPTIONS, "kind": "choice", "lang": "en"}


class QueryRouter:
    """fast: FastJudge (or an option_probs provider); lm: a provider with complete (if absent, the Fast answer is used as is)."""

    def __init__(self, fast, lm=None, threshold: float = 0.8):
        self.fast, self.lm, self.threshold = fast, lm, threshold

    def fast_probs(self, questions: list[str]) -> np.ndarray:
        return np.array(self.fast.option_probs([render(rung_item(q)) for q in questions], ["A", "B", "C"]), float)

    def lm_rungs(self, questions: list[str]) -> list[int | None]:
        outs = self.lm.complete([LM_PROMPT.format(question=q) for q in questions], max_new_tokens=32)
        found = [re.search(r'"rung"\s*:\s*([123])', o) for o in outs]
        return [int(m.group(1)) if m else None for m in found]

    def classify(self, questions: list[str]) -> list[dict]:
        p = self.fast_probs(questions)
        low = [i for i in range(len(questions)) if p[i].max() < self.threshold and self.lm is not None]
        slow = dict(zip(low, self.lm_rungs([questions[i] for i in low]))) if low else {}
        rows = []
        for i, q in enumerate(questions):
            fast_rung = int(p[i].argmax()) + 1
            rung = slow.get(i) or fast_rung
            rows.append({"question": q, "rung": rung, "kind": RUNG_KIND[rung], "confidence": float(p[i].max()), "fast_rung": fast_rung,
                         "probs": p[i].round(4).tolist(), "router": "lm" if slow.get(i) else "fast"})
        return rows


def lexicon(hcg: HCG, nodes) -> Lexicon:
    """Node id -> binary variable lexicon (1 = high, 0 = low). English names replace underscores in the id with spaces."""
    label = {n.id: n.label or n.id for n in hcg.nodes}
    return Lexicon({v: (label[v], "높음", "낮음", v.replace("_", " "), "high", "low") for v in nodes}, "en")


def verify_item(hcg: HCG, df: pd.DataFrame, x: str, y: str) -> dict:
    """Fit a binary SCM on the observed ancestors of treatment and outcome and build a Gym intervention item (tool mode).
    Unobserved ancestors are left out of the context (the LM answer then cannot see hidden confounding; they are recorded in latent)."""
    g = hcg.to_networkx()
    scm = fit_scm(g, df, [x, y])
    lex = lexicon(hcg, scm.order)
    ctx = context(lex, describe_graph(scm.graph, lex), describe_cpt(scm, lex))
    q = f"If we intervene to set {lex.name(x)} to high instead of low, does the probability that {lex.is_(y, 1)} increase?"
    hidden = sorted((nx.ancestors(g, x) | nx.ancestors(g, y)) & set(latent_of(g, df)))
    return {"context": ctx, "question": q, "options": [], "kind": "noul", "lang": "en", "nodes": scm.order, "latent": hidden}


def lm_verify(lm, hcg: HCG, df: pd.DataFrame, x: str, y: str, engine_ate: float, engine_ci=None, max_new_tokens: int = 1536) -> dict:
    """lm: HFProvider (Hvme-LM). If the engine CI contains 0, the sign is undetermined and agree is None."""
    from ..lm.sample import generate_tools

    item = verify_item(hcg, df, x, y)
    out = generate_tools(lm, [render(item, "tool")], tools=True, max_new_tokens=max_new_tokens)[0]
    yes = parse_yes_no(out)
    sign = None if engine_ci is not None and engine_ci[0] <= 0 <= engine_ci[1] else engine_ate > 0
    agree = None if yes < 0 or sign is None else bool(yes) == sign
    calls = re.findall(r"<tool_response>\n(.*?)\n</tool_response>", out, re.S)
    return {"lm_answer": {1: "yes", 0: "no", -1: None}[yes], "engine_ate": float(engine_ate), "engine_sign": None if sign is None else ("+" if sign else "-"),
            "agree": agree, "n_tool_calls": len(calls), "tool_results": calls, "trace": out, "prompt_nodes": item["nodes"], "latent": item["latent"]}


def verify_markdown(v: dict) -> str:
    state = {True: "일치", False: "불일치", None: "판단 보류"}[v["agree"]]
    why = " 엔진 95% 구간이 0을 포함해 부호를 비교하지 않았다." if v["engine_sign"] is None else ""
    hid = f" 관측되지 않는 조상 {', '.join(v['latent'])}은 LM 문맥에 없다." if v["latent"] else ""
    return (f"### Hvme-LM 검증\n\nHvme-LM(tool 모드)이 같은 그래프와 데이터로 적합한 확률표를 읽고 따로 풀었다. "
            f"답: **{v['lm_answer']}**, 엔진 ATE {v['engine_ate']:+.3f} → **{state}**.{why}{hid} 도구 호출 {v['n_tool_calls']}번.\n")
