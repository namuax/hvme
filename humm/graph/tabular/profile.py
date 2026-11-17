"""S1 변수 프로파일링: 열마다 유형·범주·결측을 요약하고, LLM에 줄 변수 설명 텍스트를 만든다."""
import pandas as pd


def profile(df: pd.DataFrame, max_levels: int = 8) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        k = df[c].nunique()
        kind = "binary" if k <= 2 else ("categorical" if k <= max_levels else "continuous")
        summary = ", ".join(map(str, sorted(df[c].dropna().unique())[:max_levels])) if kind != "continuous" else f"mean {df[c].mean():.3g}, sd {df[c].std():.3g}"
        rows.append({"variable": c, "kind": kind, "levels": k, "missing": float(df[c].isna().mean()), "summary": summary})
    return pd.DataFrame(rows)


def describe_variables(prof: pd.DataFrame, descriptions: dict | None = None) -> str:
    """'- name (kind): 설명' 형식의 줄 목록. 설명이 없으면 이름만 쓴다."""
    descriptions = descriptions or {}
    return "\n".join(f"- {r.variable} ({r.kind}){': ' + descriptions[r.variable] if r.variable in descriptions else ''}" for r in prof.itertuples())
