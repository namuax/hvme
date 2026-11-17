"""Hume-Fast v2 (Step 10): System 2 → 1 증류.

Hume-LM(9B, tool 모드)이 같은 문항을 N번 풀고, 검증을 통과한 답의 경험 분포를 Fast의 목표 분포로 쓴다(04_hume_fast.md 3장).
- 원천 문항: Gym 문항의 문맥(context)을 LLM이 자연스러운 글로 다시 쓴 것. 다시 쓰면 생성기 라벨이 그대로 맞는다는 보장이 없어서(숫자 누락·관계 변형),
  라벨 대신 느린 경로(Hume-LM + 엔진)가 그 글을 풀어 낸 답을 쓴다. 숫자가 빠지거나 새로 생긴 글은 버린다(numbers_kept).
- 큰 표면 원천(story, 노트북 20): 예/아니오 문항 전체(문맥 + 질문)를 다른 분야의 이야기로 다시 쓴다. 변수·상태 이름과 질문 문장이 바뀌어
  질문 극성이 뒤집힐 수 있으므로 생성기 라벨은 진단용으로만 남기고(meta.orig_label), 목표는 느린 경로의 답만 쓴다.
- 검증(verified): 최종 답이 있고, 반복 루프가 없고, 도구를 불렀다면 마지막 엔진 결과가 오류가 아니다. 엔진 값은 기호 계산이므로 이 조건을 통과한 답만 센다.
- 목표: 검증된 답의 빈도 + alpha 평활. 검증된 풀이가 없는 문항은 버린다.
"""
import re

import numpy as np

from ..gym.formats import option_labels, parse_final, render
from ..lm.data import looping

PARAPHRASE = ("Rewrite the following description of a hypothetical world as a natural passage, as a person would explain it in a short report or story. "
              "Keep every variable name, every state name, every causal relation and every number exactly as given, and do not add, drop or change any fact. "
              "Do not answer any question and do not add conclusions. Output only the rewritten passage.\n\n{context}")
STORY = ("Below is a yes/no causal reasoning problem. Rewrite the whole problem so that it is set in a different real-world field of your choice "
         "(for example medicine, farming, schooling, economics, engineering, ecology or sports). Give every variable a new meaningful name and give its states "
         "fitting names, and write the conditions as a natural passage. Keep the same causal relations, every number exactly as given, and a question that asks "
         "about the same quantity and still has a yes or no answer. Do not solve the problem and do not add facts or numbers. "
         "Output exactly two parts in this form:\nPassage: <the rewritten conditions>\nQuestion: <the rewritten question>\n\n{context}\n\n{question}")
# story2(노트북 21): 분야·글 형식을 문항마다 돌려 표면을 더 넓히고, 질문을 한 번만 쓰게 한다(노트북 20 탈락의 80%가 질문을 여러 번 다르게 쓴 것)
FIELDS = ("medicine", "farming", "schooling", "economics", "manufacturing", "ecology", "sports", "public health", "software operations", "retail",
          "transportation", "energy")
FORMS = ("a short news report", "a lab notebook entry", "a textbook exercise", "a consultant's memo", "a field study summary")
STORY2 = ("Below is a yes/no causal reasoning problem. Rewrite the whole problem as {form} set in {field}. Give every variable a new meaningful name "
          "and give its states fitting names, and write the conditions in the voice of that kind of text. Keep the same causal relations, every number exactly as given, "
          "and a question that asks about the same quantity and still has a yes or no answer. Do not solve the problem and do not add facts or numbers. "
          "Write the question only once and stop right after it. Output exactly two parts in this form:\nPassage: <the rewritten conditions>\nQuestion: <the rewritten question>"
          "\n\n{context}\n\n{question}")
STORY_PARTS = re.compile(r"Passage:\s*(.+?)\s*Question:", re.S)
NUMBER = re.compile(r"\d+(?:\.\d+)?")


def paraphrase_prompt(item: dict) -> str:
    return PARAPHRASE.format(context=item["context"])


def story_prompt(item: dict) -> str:
    return STORY.format(context=item["context"], question=item["question"])


def story2_prompt(item: dict, i: int) -> str:
    """i번째 문항의 분야·형식(FIELDS × FORMS를 i로 돌린다)."""
    return STORY2.format(field=FIELDS[i % len(FIELDS)], form=FORMS[i // len(FIELDS) % len(FORMS)], context=item["context"], question=item["question"])


def storied(item: dict, text: str, counts: bool = True) -> dict | None:
    """story 출력(Passage/Question)으로 바꾼 문항. counts=False면 % 없는 12 이하 정수(변수 개수 'five' 등)는 숫자 비교에서 뺀다. 두 부분이 없거나, 질문을 여러 번 다르게 썼거나, 문맥 + 질문의 숫자 집합이 원문과 다르면 None."""
    m, qs = STORY_PARTS.search(text), {q.strip() for q in re.findall(r"Question:[ \t]*(.+)", text)}
    ctx, q = (m.group(1).strip(), next(iter(qs))) if m and len(qs) == 1 else ("", "")
    ok = ctx and q and numbers_kept(f"{item['context']} {item['question']}", f"{ctx} {q}", counts)
    return item | {"id": f"{item['id']}-s", "context": ctx, "question": q,
                   "meta": item.get("meta", {}) | {"orig_context": item["context"], "orig_question": item["question"], "orig_label": item["label"]}} if ok else None


def numbers_kept(src: str, text: str, counts: bool = True) -> bool:
    """다시 쓴 글의 숫자(확률·값) 집합이 원문과 같은지. 빠진 숫자도, 지어낸 숫자(없던 확률표)도 안 된다(중복 횟수는 보지 않는다).
    counts=False면 뒤에 %가 없는 12 이하 정수는 보지 않는다(변수 개수를 글자로 쓴 경우)."""
    nums = lambda t: {m.group() for m in NUMBER.finditer(t) if counts or "." in m.group() or t[m.end():m.end() + 1] == "%" or int(m.group()) > 12}
    return nums(src) == nums(text)


def paraphrased(item: dict, text: str) -> dict | None:
    """다시 쓴 문맥으로 바꾼 문항. 숫자가 빠졌으면 None. 질문·보기·라벨은 원 문항 그대로 둔다(라벨은 비교용)."""
    text = text.strip()
    return item | {"id": f"{item['id']}-p", "context": text, "meta": item.get("meta", {}) | {"orig_context": item["context"]}} \
        if text and numbers_kept(item["context"], text) else None


def verified(output: str) -> str | None:
    """검증을 통과한 풀이의 최종 답(parse_final), 아니면 None."""
    calls = re.findall(r"<tool_response>\n(.*?)\n</tool_response>", output, flags=re.S)
    ok = not looping(output) and not (calls and calls[-1].startswith("error"))
    return parse_final(output) if ok else None


def answer_index(item: dict, answer: str) -> int | None:
    """parse_final 답 → option_labels 안의 위치."""
    keys = {"noul": ["yes", "no"], "choice": [x.upper() for x in option_labels(item)], "score": option_labels(item)}[item["kind"]]
    return keys.index(answer) if answer in keys else None


def target_from(item: dict, answers: list[str | None], alpha: float = 0.1) -> list[float] | None:
    """검증된 답들의 빈도 + alpha 평활 분포. 쓸 수 있는 답이 없으면 None."""
    idx = [i for i in (answer_index(item, a) for a in answers if a) if i is not None]
    if not idx:
        return None
    c = np.bincount(idx, minlength=len(option_labels(item))) + alpha
    return (c / c.sum()).tolist()


def distill_items(items: list[dict], samples: list[dict], alpha: float = 0.1) -> list[dict]:
    """samples: humm.lm.jobs sample 출력 행({id, output, ...}). Fast 학습 문항(from_gym 모양)에 증류 목표와 검증 통계를 붙인다."""
    by_id = {}
    for s in samples:
        by_id.setdefault(s["id"], []).append(verified(s["output"]))
    out = []
    for d in items:
        answers = by_id.get(d["id"], [])
        t = target_from(d, answers, alpha)
        if t is not None:
            out.append({"id": d["id"], "prompt": render(d, "direct"), "labels": option_labels(d), "target": t, "kind": d["kind"], "task": d["query_type"],
                        "lang": d["lang"], "source": "distill", "n_samples": len(answers), "n_verified": sum(a is not None for a in answers)})
    return out
