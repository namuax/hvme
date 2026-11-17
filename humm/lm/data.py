"""SFT 예제 구성(torch 불필요): 질의 유형 가중 표본, 풀이 템플릿 예제, 반복 루프 판정."""
import re

import numpy as np
import pandas as pd

from ..gym.formats import render
from .solutions import solve
from .tools import parse_call

# 구조 추론형과 4B가 약한 nie를 더 많이 뽑는다 (reports/2026-10-25_gym_baseline.md 3·4장)
WEIGHTS = {"corr2cause": 2.0, "identify": 2.0, "backadj": 2.0, "nie": 1.5, "dsep": 1.2}
# SFT v2(Gym v3): 1차에서 CLadder로 이어지지 않은 반사실형(ett·det_cf)과 구조 추론형을 더 뽑는다 (reports/2026-10-28_hume_lm_sft.md 3장)
WEIGHTS_V3 = {"corr2cause": 1.5, "identify": 1.5, "backadj": 1.5, "det_cf": 1.5, "ett": 1.3, "nie": 1.3, "dsep": 1.2}
# SFT v2.1(Gym v3.1): v2에서 퇴보한 구조 추론형을 유형당 약 1,350개로 늘린다 (reports/2026-10-28_hume_lm_sft_v2.md 2.2)
WEIGHTS_V31 = {"corr2cause": 2.0, "identify": 2.0, "backadj": 2.0, "dsep": 2.0, "det_cf": 1.3, "ett": 1.2, "nie": 1.2}
# GRPO(Gym v3.2): v2.1이 CLadder·Gym에서 틀린 유형(ett·nde·nie·backadj·corr2cause·marginal)에서 보상 신호가 나온다 (reports/2026-10-29_hume_lm_sft_v21.md 3·4절)
WEIGHTS_GRPO = {"ett": 2.0, "nde": 2.0, "nie": 2.0, "backadj": 2.0, "corr2cause": 2.0, "marginal": 1.5, "correlation": 0.7, "cde": 0.5, "det_cf": 0.5, "ate_score": 0.5}


def pick(items: list[dict], n: int, weights: dict = WEIGHTS, seed: int = 0) -> list[dict]:
    """질의 유형별 가중치에 맞춰 n개를 뽑는다(유형 안에서는 무작위)."""
    rng = np.random.default_rng(seed)
    by = pd.Series(range(len(items))).groupby([d["query_type"] for d in items]).apply(list).to_dict()
    w = pd.Series({t: weights.get(t, 1.0) for t in by})
    quota = (w / w.sum() * n).round().astype(int)
    idx = [i for t, k in quota.items() for i in rng.permutation(by[t])[: min(k, len(by[t]))]]
    return [items[i] for i in rng.permutation(idx)]


def template_examples(items: list[dict]) -> list[dict]:
    return [{"id": d["id"], "prompt": render(d, "cot"), "response": solve(d), "task": d["query_type"], "source": "template"} for d in items]


def looping(text: str, min_repeat: int = 4) -> bool:
    """같은 줄(10자 이상)이 min_repeat번 이상 나오면 반복 루프로 본다. 도구 호출 표기 줄('</parameter>' 등)은 세지 않는다."""
    lines = [s.strip() for s in text.splitlines() if len(s.strip()) >= 10 and not s.strip().startswith("<")]
    return any(lines.count(s) >= min_repeat for s in set(lines))


def error_repeats(out: str, start: int = 0) -> int:
    """오류를 받은 호출을 바로 다음 호출에서 같은 인자로 다시 보낸 횟수. start: 앞의 몇 호출(주어진 앞부분)은 다시 보낸 쪽으로 세지 않는다."""
    calls, resps = re.findall(r"<tool_call>(.*?)</tool_call>", out, re.S), re.findall(r"<tool_response>\n(.*?)</tool_response>", out, re.S)
    return sum(r.startswith("error") and a.strip() == b.strip() for i, (a, b, r) in enumerate(zip(calls, calls[1:], resps)) if i + 1 >= start)


STRUCT = ("dsep", "backadj", "identify", "corr2cause")


def _canon(line: str) -> frozenset:
    """한 줄을 쉼표·and·콜론으로 나눈 조각 집합. 경로·엣지·변수 나열 순서가 달라도 같은 줄로 본다."""
    return frozenset(c.strip(" .") for c in re.split(r",\s*|\s+and\s+|:\s+", " ".join(line.split())) if c.strip(" ."))


def process(d: dict, out: str) -> float:
    """구조 유형 풀이의 과정 점수: 엔진 풀이의 3단계 이후 핵심 줄(보기 고르기·Final 제외) 중 출력에 있는 비율."""
    if "call" in d["meta"]:  # v3.3 corr2cause: 판정은 도구가 하므로 인자(가설 종류·변수, 관계 목록)를 맞게 옮겼는지 본다
        c, ref = parse_call(out) or {}, d["meta"]["call"]
        return (c.get("hypothesis") == ref["hypothesis"] and [c.get("a"), c.get("b")] == [ref["a"], ref["b"]]) / 2 + \
            (sorted(map(str, c.get("relations") or [])) == sorted(ref["relations"])) / 2
    ref = solve(d).splitlines()
    i = next(k for k, l in enumerate(ref) if l.startswith("3)"))
    keys = [_canon(l) for l in ref[i:] if l.strip() and not l.startswith("Final") and not re.match(r"\s*(해당하는 보기|Matching option)", l)]
    lines = {_canon(l) for l in out.splitlines()}
    return sum(k in lines for k in keys) / len(keys)
