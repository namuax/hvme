"""조건부 독립 검정. 이산 데이터는 G-검정(층별 우도비), 연속 데이터는 Fisher-z(편상관)를 쓴다."""
import numpy as np
import pandas as pd
from scipy import stats


def is_discrete(df: pd.DataFrame, cols, max_levels: int = 8) -> bool:
    return all(df[c].nunique() <= max_levels for c in cols)


def fisher_z(df: pd.DataFrame, a: str, b: str, s=()) -> float:
    x = df[[a, b, *s]].to_numpy(float)
    prec = np.linalg.pinv(np.corrcoef(x, rowvar=False))
    r = np.clip(-prec[0, 1] / np.sqrt(prec[0, 0] * prec[1, 1]), -0.999999, 0.999999)
    z = np.arctanh(r) * np.sqrt(max(len(df) - len(s) - 3, 1))
    return float(2 * stats.norm.sf(abs(z)))


def _codes(col: pd.Series) -> tuple[np.ndarray, int]:
    c, uniq = pd.factorize(col, sort=True)
    return c, len(uniq)


def g_test_codes(a: tuple, b: tuple, s: list) -> float:
    """정수 코드로 된 변수(코드 배열, 범주 수)로 G-검정을 한다. 모든 층의 분할표를 bincount 한 번으로 만든다."""
    (xa, ka), (xb, kb) = a, b
    z, nz = np.zeros(len(xa), np.int64), 1
    for c, k in s:
        z, nz = z * k + c, nz * k
    t = np.bincount((z * ka + xa) * kb + xb, minlength=nz * ka * kb).reshape(nz, ka, kb).astype(float)
    t = t[t.sum((1, 2)) > 0]
    e = t.sum(2, keepdims=True) * t.sum(1, keepdims=True) / t.sum((1, 2), keepdims=True)
    m = t > 0
    g = 2 * (t[m] * np.log(t[m] / e[m])).sum()
    dof = ((((t.sum(2) > 0).sum(1) - 1) * ((t.sum(1) > 0).sum(1) - 1)).clip(0)).sum()
    return float(stats.chi2.sf(g, dof)) if dof else 1.0


def g_test(df: pd.DataFrame, a: str, b: str, s=()) -> float:
    """층(조건 변수 조합)마다 a×b 분할표의 G 통계량과 자유도를 더한다."""
    return g_test_codes(_codes(df[a]), _codes(df[b]), [_codes(df[c]) for c in s])


class CITester:
    """같은 검정을 반복하지 않도록 결과를 저장한다. (a, b, S)와 (b, a, S)는 같은 검정이다."""

    def __init__(self, df: pd.DataFrame, method: str = "auto"):
        self.df = df
        self.method = method if method != "auto" else ("g_test" if is_discrete(df, df.columns) else "fisher_z")
        self.codes = {c: _codes(df[c]) for c in df.columns} if self.method == "g_test" else {}
        self.fn = {"g_test": lambda d, a, b, s: g_test_codes(self.codes[a], self.codes[b], [self.codes[c] for c in s]), "fisher_z": fisher_z}[self.method]
        self.cache = {}

    def __call__(self, a: str, b: str, s=()) -> float:
        key = (frozenset((a, b)), frozenset(s))
        if key not in self.cache:
            self.cache[key] = self.fn(self.df, a, b, tuple(sorted(s)))
        return self.cache[key]
