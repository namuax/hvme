"""General-ability check (MATH-500, KMMLU): verify that causal training did not erode general ability (criterion: at most 2pt drop from the base model).

Scoring is a simple rule applied identically to the base and trained models. The difference between the two matters more than absolute scores.
"""
import re

MATH_PROMPT = "{problem}\n\nSolve the problem step by step, then write the final answer in \\boxed{{}}."
KMMLU_PROMPT = "{question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n\n단계별로 짧게 생각한 뒤 마지막 줄에 'Final answer: <A|B|C|D>' 형식으로 답하라."
CHOICE = re.compile(r"final answer\s*[:：]?\s*\**\s*\(?([ABCD])\b", re.I)


def boxed(text: str) -> str | None:
    """Contents of the last \\boxed{...} (matching braces)."""
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
    """Option number 1-4 from 'Final answer: X', or -1 if absent."""
    hits = CHOICE.findall(out)
    return "ABCD".index(hits[-1].upper()) + 1 if hits else -1


PROB = re.compile(r'\{[^{}]*"prob"\s*:\s*"?(-?[0-9.]+)"?[^{}]*\}', re.I)


def caugym_correct(out: str, answer: str) -> bool:
    """CauGym scoring: correct if any PROB in the output JSON is within 0.0005 of the answer. For insufficient, checks whether LACK_CONDITION was written."""
    if answer == "LACK_CONDITION":
        return "LACK_CONDITION" in out
    gold = float(PROB.search(answer).group(1))
    return any(abs(float(v) - gold) < 0.0005 for v in PROB.findall(re.sub(r"\s+", "", out)) if v.count(".") <= 1 and v not in ("-", "."))
