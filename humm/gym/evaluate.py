"""Humm-Gym 문항으로 모델을 평가한다. direct는 답 토큰 확률(Humm-Fast 관점), cot는 풀이 생성 후 채점(Humm-LM 관점)."""
import numpy as np
import pandas as pd

from ..bench.metrics import ece
from .formats import option_labels, parse_final, render, reward, target_index


def eval_direct(provider, items: list[dict]) -> pd.DataFrame:
    """보기 토큰이 같은 문항끼리 묶어 option_probs를 한 번에 계산한다."""
    df = pd.DataFrame(items).assign(labels=[tuple(option_labels(d)) for d in items], target=[target_index(d) for d in items])
    out = []
    for labels, g in df.groupby("labels", sort=False):
        probs = np.array(provider.option_probs([render(d, "direct") for d in g.to_dict("records")], list(labels)))
        out.append(g.assign(probs=list(probs), pred=probs.argmax(1), p_target=probs[np.arange(len(g)), g.target]))
    res = pd.concat(out).sort_index()
    return res.assign(correct=res.pred == res.target, mode="direct", model=provider.name)


def eval_cot(provider, items: list[dict], max_new_tokens: int = 4096) -> pd.DataFrame:
    outs = provider.complete([render(d, "cot") for d in items], max_new_tokens=max_new_tokens)
    df = pd.DataFrame(items).assign(output=outs)
    return df.assign(answer=[parse_final(o) for o in outs], correct=[bool(reward(d, o)) for d, o in zip(items, outs)], mode="cot", model=provider.name)


def summarize(res: pd.DataFrame, by=("query_type",)) -> pd.DataFrame:
    """정확도, (cot) 파싱 실패율, (direct·noul) 확률 보정 ECE."""
    def stats(d):
        row = {"n": len(d), "acc": d.correct.mean()}
        cot = d[d["mode"] == "cot"] if "answer" in d else d.iloc[:0]
        row |= {"parse_fail": cot.answer.isna().mean()} if len(cot) else {}
        noul = d[(d.kind == "noul") & d.p_target.notna()] if "p_target" in d else d.iloc[:0]
        row |= {"ece_noul": ece(np.where(noul.target == 0, noul.p_target, 1 - noul.p_target), noul.label)} if len(noul) else {}
        return row
    keys = ["model", "mode", *by]
    return pd.DataFrame([dict(zip(keys, k)) | stats(d) for k, d in res.groupby(keys)])
