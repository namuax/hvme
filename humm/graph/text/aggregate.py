"""T6 aggregation: collect claims from multiple documents into edge candidates with confidence.

- Claims in the same direction within one document are counted once (the document is the unit of evidence).
- Causal claim weights: assertive 1.0, hedged 0.5. Correlation claims are not used as evidence of direction.
- Opposite-direction claims and 'no effect' claims are counter-evidence (contra).
- score = support / (support + contra + 1)  (Laplace smoothing: a claim from a single document scores 0.5)
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
