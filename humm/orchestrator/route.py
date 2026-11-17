"""오케스트레이터 v1의 두 모델 단계 (02_proposal.md 7장).

① 질의 분석: Humm-Fast가 질문의 Rung(1 연관·2 개입·3 반사실)을 한 번의 순전파로 고른다. 확신도(최대 보기 확률)가
   threshold보다 낮으면 Humm-LM에 JSON으로 다시 묻는다. 변수(처치·결과·매개)는 v0처럼 질문 문자열의 표현으로 정한다.
⑤ 검증: Humm-LM(tool 모드)이 그래프와 데이터로 적합한 확률표를 Gym 문맥으로 읽고 "처치가 결과를 올리는가"를 따로 푼다.
   엔진 추정 ATE의 부호와 비교해 일치 여부를 붙인다. LM은 수치를 직접 만들지 않고 causal_engine 도구로 계산한다.
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
    """fast: FastJudge(또는 option_probs 공급자), lm: complete가 있는 공급자(없으면 Fast 답을 그대로 쓴다)."""

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
    """노드 id → 이진 변수 어휘(1 = high, 0 = low). 영어 이름은 id의 밑줄을 띄어 쓴다."""
    label = {n.id: n.label or n.id for n in hcg.nodes}
    return Lexicon({v: (label[v], "높음", "낮음", v.replace("_", " "), "high", "low") for v in nodes}, "en")


def verify_item(hcg: HCG, df: pd.DataFrame, x: str, y: str) -> dict:
    """처치·결과의 관측된 조상으로 이진 SCM을 적합해 Gym 개입 문항(tool 모드)으로 만든다.
    관측되지 않는 조상은 문맥에서 빠진다(그 경우 LM 답은 숨은 교란을 보지 못한다, latent에 남긴다)."""
    g = hcg.to_networkx()
    scm = fit_scm(g, df, [x, y])
    lex = lexicon(hcg, scm.order)
    ctx = context(lex, describe_graph(scm.graph, lex), describe_cpt(scm, lex))
    q = f"If we intervene to set {lex.name(x)} to high instead of low, does the probability that {lex.is_(y, 1)} increase?"
    hidden = sorted((nx.ancestors(g, x) | nx.ancestors(g, y)) & set(latent_of(g, df)))
    return {"context": ctx, "question": q, "options": [], "kind": "noul", "lang": "en", "nodes": scm.order, "latent": hidden}


def lm_verify(lm, hcg: HCG, df: pd.DataFrame, x: str, y: str, engine_ate: float, engine_ci=None, max_new_tokens: int = 1536) -> dict:
    """lm: HFProvider(Humm-LM). 엔진 CI가 0을 포함하면 부호가 정해지지 않아 agree는 None이다."""
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
    return (f"### Humm-LM 검증\n\nHumm-LM(tool 모드)이 같은 그래프와 데이터로 적합한 확률표를 읽고 따로 풀었다. "
            f"답: **{v['lm_answer']}**, 엔진 ATE {v['engine_ate']:+.3f} → **{state}**.{why}{hid} 도구 호출 {v['n_tool_calls']}번.\n")
