"""T6 집계: 문서 여러 개의 주장을 모아 엣지 후보와 신뢰도를 만든다.

- 같은 문서의 같은 방향 주장은 한 번만 센다(문서가 증거의 단위).
- 인과 주장 가중치: 단정 1.0, 불확실 표현 0.5. 상관 주장은 방향 근거로 쓰지 않는다.
- 반대 방향 주장과 '영향 없음' 주장은 반대 증거(contra)이다.
- score = support / (support + contra + 1)  (라플라스 평활: 문서 하나뿐인 주장은 0.5)
"""
import pandas as pd

COLS = ["src", "dst", "support", "contra", "correlational", "score", "sign", "docs", "quotes"]


def aggregate(claims: pd.DataFrame, min_support: float = 1.0, min_score: float = 0.4) -> pd.DataFrame:
    if claims.empty:
        return pd.DataFrame(columns=COLS)
    c = claims[claims.cause_c != claims.effect_c].assign(w=lambda d: d.hedged.map({True: 0.5, False: 1.0}))
    causal = c[c.relation == "causal"].groupby(["cause_c", "effect_c", "doc_id"], as_index=False).agg(w=("w", "max"), sign=("polarity", "first"), quote=("quote", "first"))
    fwd = causal.groupby(["cause_c", "effect_c"]).agg(support=("w", "sum"), docs=("doc_id", list), quotes=("quote", list),
                                                      sign=("sign", lambda s: s[s != "?"].mode().iloc[0] if (s != "?").any() else "?"))
    pair = lambda a, b: tuple(sorted((a, b)))
    none = c[c.relation == "no_effect"].assign(p=lambda d: [pair(a, b) for a, b in zip(d.cause_c, d.effect_c)]).groupby("p").doc_id.nunique()
    corr = c[c.relation == "correlational"].assign(p=lambda d: [pair(a, b) for a, b in zip(d.cause_c, d.effect_c)]).groupby("p").doc_id.nunique()
    rows = []
    for (a, b), r in fwd.iterrows():
        rev = fwd.support.get((b, a), 0.0)
        contra = rev + none.get(pair(a, b), 0)
        if r.support > rev and r.support >= min_support:
            rows.append({"src": a, "dst": b, "support": r.support, "contra": contra, "correlational": int(corr.get(pair(a, b), 0)),
                         "score": r.support / (r.support + contra + 1), "sign": r.sign, "docs": r.docs, "quotes": r.quotes})
    out = pd.DataFrame(rows, columns=COLS)
    return out[out.score >= min_score].sort_values("score", ascending=False).reset_index(drop=True)
