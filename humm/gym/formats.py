"""Hume-Gym 문항을 모델 입력으로 바꾸고, 모델 출력을 채점한다.

mode="direct": 답 토큰 하나만 읽는 형식(Hume-Fast의 Noul·Choice·Score, option_probs 평가)
mode="cot":    단계별 풀이 후 'Final answer: ...' 로 끝내는 형식(Hume-LM SFT·GRPO, 생성형 평가)
mode="tool":   cot와 같되 수치 계산은 causal_engine 도구 호출로 한다(Hume-LM SFT v2, humm.lm.tools)
"""
import json
import re

LETTERS = "ABCDEFGH"

DIRECT = {
    ("noul", "ko"): "예 또는 아니오로만 답하라.", ("noul", "en"): "Answer with Yes or No only.",
    ("choice", "ko"): "보기의 알파벳 하나로만 답하라.", ("choice", "en"): "Answer with the option letter only.",
    ("score", "ko"): "단계 번호 하나로만 답하라.", ("score", "en"): "Answer with the level number only.",
}
FINAL_FORMAT = {"noul": "yes 또는 no|yes or no", "choice": "보기 알파벳|the option letter", "score": "단계 번호|the level number",
                "prob": "위에서 요청한 답 형식|the answer format asked for above"}  # prob(v3.7): 값을 JSON으로 묻는 문항
COT = {
    "ko": "단계별로 풀어라. (1) 변수와 인과 그래프를 정리하고, (2) 질의 유형을 판단하고, (3) 필요한 추정식을 세우고, (4) 주어진 확률로 계산한 뒤 답하라. "
          "마지막 줄은 반드시 'Final answer: <{fmt}>' 형식으로 쓴다.",
    "en": "Solve this step by step: (1) list the variables and the causal graph, (2) identify the query type, (3) derive the estimand, "
          "(4) compute it with the given probabilities and answer. End with a line in the form 'Final answer: <{fmt}>'.",
}

TOOL = {
    "ko": "단계별로 풀어라. (1) 변수와 상태, 인과 그래프를 정리하고, (2) 질의 유형과 비교 대상(처치·기준·결과의 상태)을 판단하고, (3) 추정식을 세운 뒤, "
          "(4) 확률 계산이 필요하면 causal_engine 도구에 문맥의 확률을 그대로 넘겨 값을 구하고 그 값으로 답하라. 마지막 줄은 반드시 'Final answer: <{fmt}>' 형식으로 쓴다.",
    "en": "Solve this step by step: (1) list the variables with their states and the causal graph, (2) identify the query type and what is compared "
          "(treatment, baseline and outcome states), (3) derive the estimand, (4) if probabilities must be computed, pass the given probabilities unchanged "
          "to the causal_engine tool and answer from its value. End with a line in the form 'Final answer: <{fmt}>'.",
}


def options_block(item: dict) -> str:
    keys = LETTERS if item["kind"] == "choice" else [str(i + 1) for i in range(len(item["options"]))]
    return "\n".join(f"{k}. {o}" for k, o in zip(keys, item["options"]))


def render(item: dict, mode: str = "direct") -> str:
    fmt = FINAL_FORMAT[item["kind"]].split("|")[0 if item["lang"] == "ko" else 1]
    instr = DIRECT[(item["kind"], item["lang"])] if mode == "direct" else {"cot": COT, "tool": TOOL}[mode][item["lang"]].format(fmt=fmt)
    body = f"{item['context']}\n\n{item['question']}"
    opts = options_block(item)
    return f"{body}\n{opts}\n\n{instr}" if opts else f"{body}\n\n{instr}"


def option_labels(item: dict) -> list[str]:
    """direct 모드에서 확률을 읽을 답 토큰. noul은 [참, 거짓] 순서이다."""
    return {"noul": ["예", "아니오"] if item["lang"] == "ko" else ["Yes", "No"],
            "choice": list(LETTERS[: len(item["options"])]),
            "score": [str(i + 1) for i in range(len(item["options"]))]}[item["kind"]]


def target_index(item: dict) -> int:
    """option_labels 안에서 정답의 위치."""
    return {"noul": 1 - item["label"], "choice": item["label"], "score": item["label"] - 1}[item["kind"]]


def target_text(item: dict) -> str:
    """cot 모드에서 'Final answer:' 뒤에 와야 할 정답."""
    if item["kind"] == "prob":
        return json.dumps(item["meta"]["answer"])
    return {"noul": "yes" if item["label"] else "no", "choice": LETTERS[item["label"]], "score": str(item["label"])}[item["kind"]]


FINAL = re.compile(r"final answer\s*(?:is)?\s*[:：]?\s*[*'\"<(\[]*\s*(yes|no|예|아니오|[A-H]|\d+)\b", re.I)
NORMAL = {"예": "yes", "아니오": "no"}


def parse_final(text: str) -> str | None:
    hits = FINAL.findall(text)
    return NORMAL.get(hits[-1], hits[-1].lower() if hits[-1].lower() in ("yes", "no") else hits[-1].upper()) if hits else None


JSON_ANSWER = re.compile(r'"ANSWER"\s*:\s*"\s*(yes|no)', re.I)
JSON_PROB = re.compile(r'"PROB"\s*:\s*"?\s*(-?[0-9]*\.?[0-9]+)', re.I)


def prob_correct(item: dict, text: str) -> bool:
    """prob 문항: 마지막 'Final answer:' 뒤 JSON의 PROB가 정답과 0.0005 안이고, 예/아니오를 물으면 ANSWER도 같다."""
    i = text.lower().rfind("final answer:")
    tail = text[i:] if i >= 0 else text
    p, a = JSON_PROB.findall(tail), JSON_ANSWER.findall(tail)
    want = item["meta"]["answer"]
    return bool(p) and abs(float(p[-1]) - float(want["PROB"])) < 0.0005 and ("ANSWER" not in want or bool(a) and a[-1].lower() == want["ANSWER"].lower())


def reward(item: dict, text: str) -> float:
    """검증 가능 보상(RLVR): 마지막 'Final answer'가 정답과 같으면 1, 아니면 0."""
    return float(prob_correct(item, text)) if item["kind"] == "prob" else float(parse_final(text) == target_text(item))
