"""보정: 지표(정확도·top-1 ECE·Brier·NLL·AUROC)와 temperature scaling."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax


def ece_conf(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    """신뢰도 구간별 |정답률 − 평균 신뢰도|의 가중 평균."""
    idx = np.minimum((conf * bins).astype(int), bins - 1)
    return float(sum(abs(correct[idx == b].mean() - conf[idx == b].mean()) * (idx == b).mean() for b in np.unique(idx)))


def top_ece(probs, target, bins: int = 10) -> float:
    """다중 보기용 ECE: 예측 보기의 확률(신뢰도)로 구간을 나눈다. probs는 행렬 또는 보기 수가 다른 배열들의 목록이다."""
    conf = np.array([np.max(p) for p in probs])
    correct = np.array([int(np.argmax(p)) == t for p, t in zip(probs, target)], float)
    return ece_conf(conf, correct, bins)


def reliability(probs, target, bins: int = 10) -> pd.DataFrame:
    conf = np.array([np.max(p) for p in probs])
    correct = np.array([int(np.argmax(p)) == t for p, t in zip(probs, target)])
    idx = np.minimum((conf * bins).astype(int), bins - 1)
    return pd.DataFrame({"bin": idx, "conf": conf, "correct": correct}).groupby("bin").agg(n=("conf", "size"), confidence=("conf", "mean"),
                                                                                           accuracy=("correct", "mean")).reset_index()


def auroc(score: np.ndarray, y: np.ndarray) -> float:
    """이진 AUROC(순위 기반). 한쪽 클래스만 있으면 NaN."""
    pos, neg = score[y == 1], score[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    ranks = pd.Series(np.r_[pos, neg]).rank().to_numpy()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def metrics(probs, target) -> dict:
    """probs: 문항별 보기 확률(보기 수가 달라도 된다), target: 정답 보기 위치.
    모든 문항이 보기 2개(noul·이진 choice)이면 첫 보기 확률의 AUROC도 낸다."""
    probs, target = [np.asarray(p, float) for p in probs], np.asarray(target, int)
    p_t = np.array([p[t] for p, t in zip(probs, target)])
    brier = np.array([((p - np.eye(len(p))[t]) ** 2).sum() for p, t in zip(probs, target)])
    out = {"n": len(target), "acc": float(np.mean([int(np.argmax(p)) == t for p, t in zip(probs, target)])), "ece": top_ece(probs, target),
           "brier": float(brier.mean()), "nll": float(-np.log(np.clip(p_t, 1e-12, 1)).mean())}
    binary = all(len(p) == 2 for p in probs)
    return out | ({"auroc": auroc(np.array([p[0] for p in probs]), (target == 0).astype(int))} if binary else {})


def fit_temperature(logits: list[np.ndarray], target: np.ndarray) -> float:
    """검증셋 NLL을 최소화하는 온도 T (logits / T). 보기 수가 서로 다른 문항을 함께 넣을 수 있다."""
    def nll(log_t):
        t = np.exp(log_t)
        return -np.mean([log_softmax(z / t)[k] for z, k in zip(logits, target)])
    return float(np.exp(minimize_scalar(nll, bounds=(np.log(0.05), np.log(20)), method="bounded").x))


def apply_temperature(logits: list[np.ndarray], t: float) -> list[np.ndarray]:
    return [softmax(z / t) for z in logits]
