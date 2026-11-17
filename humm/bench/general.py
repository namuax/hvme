"""일반 능력 점검(MATH-500·KMMLU): 인과 학습이 일반 능력을 깎지 않았는지 본다(Step 9 완료 기준: 기반 대비 하락 2pt 이내).

채점은 기반 모델과 학습 모델에 똑같이 적용하는 단순 규칙이다. 절대 점수보다 두 모델의 차이를 본다.
"""
import re

MATH_PROMPT = "{problem}\n\nSolve the problem step by step, then write the final answer in \\boxed{{}}."
KMMLU_PROMPT = "{question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n\n단계별로 짧게 생각한 뒤 마지막 줄에 'Final answer: <A|B|C|D>' 형식으로 답하라."
CHOICE = re.compile(r"final answer\s*[:：]?\s*\**\s*\(?([ABCD])\b", re.I)


def boxed(text: str) -> str | None:
    """마지막 \\boxed{...}의 내용(중괄호 짝을 맞춘다)."""
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    depth, j = 0, i + len("\\boxed")
    for k in range(j, len(text)):
        depth += {"{": 1, "}": -1}.get(text[k], 0)
        if depth == 0:
            return text[j + 1:k]
    return None


def norm_math(s: str) -> str:
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    for a, b in [("\\left", ""), ("\\right", ""), ("\\!", ""), ("\\,", ""), ("dfrac", "frac"), ("tfrac", "frac"), ("^\\circ", ""), ("^{\\circ}", ""),
                 ("\\%", ""), ("$", ""), (" ", "")]:
        s = s.replace(a, b)
    s = s.rstrip(".")
    return re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", s)


def math_correct(out: str, answer: str) -> bool:
    b = boxed(out)
    return b is not None and norm_math(b) == norm_math(str(answer))


def parse_choice(out: str) -> int:
    """'Final answer: X'의 보기 번호 1~4, 없으면 -1."""
    hits = CHOICE.findall(out)
    return "ABCD".index(hits[-1].upper()) + 1 if hits else -1


PROB = re.compile(r'\{[^{}]*"prob"\s*:\s*"?(-?[0-9.]+)"?[^{}]*\}', re.I)


def caugym_correct(out: str, answer: str) -> bool:
    """CauGym 채점(eval/sample_evaluate.py): 출력의 JSON 중 PROB 하나라도 정답과 0.0005 안이면 맞다. insufficient는 LACK_CONDITION을 썼는지 본다."""
    if answer == "LACK_CONDITION":
        return "LACK_CONDITION" in out
    gold = float(PROB.search(answer).group(1))
    return any(abs(float(v) - gold) < 0.0005 for v in PROB.findall(re.sub(r"\s+", "", out)) if v.count(".") <= 1 and v not in ("-", "."))
