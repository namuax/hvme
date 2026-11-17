"""S1 variable profiling: summarize type, categories and missing values per column, and build the variable description text given to the LLM."""
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
    """List of lines in the form '- name (kind): description'. Without a description, only the name is written."""
    descriptions = descriptions or {}
    return "\n".join(f"- {r.variable} ({r.kind}){': ' + descriptions[r.variable] if r.variable in descriptions else ''}" for r in prof.itertuples())
