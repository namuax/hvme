"""기준선 실행기. Noul 방식(보기 토큰 확률)과 CoT 방식(생성 후 답 파싱)을 같은 결과 형식으로 낸다."""
import re
from pathlib import Path

import pandas as pd

from ..paths import RESULTS
from .metrics import brier, ece

YES_NO = ["Yes", "No"]
FINAL = re.compile(r"final answer\s*(?:is)?\s*[:：]?\s*\**\s*(yes|no)\b", re.I)
ANSWER = re.compile(r"\banswer\s*(?:is)?\s*[:：]?\s*\**\s*(yes|no)\b", re.I)


def parse_yes_no(text: str) -> int:
    """'Final answer: yes/no'를 우선 찾고, 없으면 'answer is yes/no' 형식을 찾는다. 둘 다 없으면 -1(파싱 실패).

    본문에 섞인 아무 yes/no 단어는 답으로 쓰지 않는다. 잘린 풀이를 오답이 아니라 파싱 실패로 집계하기 위해서다.
    """
    hits = FINAL.findall(text) or ANSWER.findall(text)
    return {"yes": 1, "no": 0}[hits[-1].lower()] if hits else -1


def run_noul(provider, df: pd.DataFrame, prompt_fn) -> pd.DataFrame:
    probs = provider.option_probs([prompt_fn(r) for _, r in df.iterrows()], YES_NO)
    p_yes = [p[0] for p in probs]
    return df.assign(p_yes=p_yes, pred=[int(p >= 0.5) for p in p_yes], mode="noul", model=provider.name)


def run_cot(provider, df: pd.DataFrame, prompt_fn, max_new_tokens: int = 1024) -> pd.DataFrame:
    outs = provider.complete([prompt_fn(r) for _, r in df.iterrows()], max_new_tokens=max_new_tokens)
    return df.assign(output=outs, pred=[parse_yes_no(o) for o in outs], mode="cot", model=provider.name)


def summarize(res: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """정확도·F1(양성=1)·파싱 실패율, Noul 결과면 Brier·ECE까지."""
    def stats(d):
        tp = ((d.pred == 1) & (d.label == 1)).sum()
        p = tp / max((d.pred == 1).sum(), 1)
        r = tp / max((d.label == 1).sum(), 1)
        pos = d.label.mean()
        row = {"n": len(d), "acc": (d.pred == d.label).mean(), "majority": max(pos, 1 - pos), "f1": 2 * p * r / max(p + r, 1e-12),
               "parse_fail": (d.pred == -1).mean()}
        has_p = "p_yes" in d and d.p_yes.notna().all()
        return row | ({"brier": brier(d.p_yes, d.label), "ece": ece(d.p_yes, d.label)} if has_p else {})

    groups = ["model", "mode"] + (by or [])
    return pd.DataFrame([dict(zip(groups, key)) | stats(d) for key, d in res.groupby(groups)])


def save(res: pd.DataFrame, name: str, out_dir: str | Path = RESULTS) -> Path:
    path = Path(out_dir) / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(path, index=False)
    return path
