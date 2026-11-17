"""S2/S4 LLM prior knowledge.

- llm_order: ask for the causal order tiers of all variables at once. More stable than asking edge by edge (Vashishtha et al., Causal Order).
- llm_pairwise: for edges whose direction the data could not decide, ask which of the two directions holds (Choice format; models that expose logits return probabilities).
"""
import json
import re

import pandas as pd

from .pc import Knowledge

ORDER_PROMPT = """You are an expert in {domain}. Below are the variables of a dataset.
{variables}

Arrange ALL variables into causal tiers: a variable may only cause variables in later tiers. Put root causes (exogenous factors) in the first tier and final outcomes in the last tier.
Use at least {min_tiers} tiers and put at most {max_size} variables in one tier. Do not sort alphabetically; order by causal mechanism.
You may reason step by step, but you MUST finish with the answer as a JSON object in a ```json code block``` of the form {{"tiers": [["var1", "var2"], ["var3"]]}}. Use every variable name exactly as written, once."""

PAIR_PROMPT = """You are an expert in {domain}.
- {a}{da}
- {b}{db}
Which causal direction between these two variables is more plausible?
A. {a} causes {b}
B. {b} causes {a}
Answer with the option letter only."""

JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)
JSON_ANY = re.compile(r"(\{\s*\"tiers\".*\})", re.S)


TIER_LINE = re.compile(r"Tier\s*(\d+)\s*[:：]\s*(.+)", re.I)


def _tier_lines(text: str) -> list[list[str]]:
    """Fallback parser when there is no JSON: reads 'Tier 1: a, b' lines. If several drafts appear, uses the last group, where numbering restarts at 1."""
    runs = []
    for num, body in TIER_LINE.findall(text):
        runs += [[]] if int(num) == 1 or not runs else []
        runs[-1].append([w.strip(" *`'\".()") for w in re.split(r",|\band\b", body)])
    return runs[-1] if runs else []


def parse_tiers(text: str, variables: list[str]) -> list[list[str]]:
    """Read tiers from the last JSON in the response, else from 'Tier k:' lines. Unknown names are dropped; if a name appears twice, only the first is kept."""
    blocks = JSON_BLOCK.findall(text) or JSON_ANY.findall(text)
    tiers = json.loads(blocks[-1]).get("tiers", []) if blocks else _tier_lines(text)
    seen, out = set(), []
    for tier in tiers:
        t = [v for v in tier if v in variables and v not in seen]
        seen |= set(t)
        out += [t] if t else []
    return out


def llm_order(provider, variables: list[str], domain: str, variable_text: str, max_new_tokens: int = 4096) -> tuple[list[list[str]], str]:
    n = len(variables)
    prompt = ORDER_PROMPT.format(domain=domain, variables=variable_text, min_tiers=max(3, n // 4), max_size=max(3, -(-n // 3)))
    raw = provider.complete([prompt], max_new_tokens=max_new_tokens)[0]
    return parse_tiers(raw, variables), raw


def llm_pairwise(provider, pairs: list[tuple[str, str]], domain: str, descriptions: dict | None = None) -> pd.DataFrame:
    """Return p(a -> b) for each pair (a, b). Providers without logits (API) have their generated answer read as 0/1."""
    d = descriptions or {}
    prompts = [PAIR_PROMPT.format(domain=domain, a=a, b=b, da=f": {d[a]}" if a in d else "", db=f": {d[b]}" if b in d else "") for a, b in pairs]
    probs = (provider.option_probs(prompts, ["A", "B"]) if getattr(provider, "supports_logits", False)
             else [[float(o.strip()[:1].upper() == "A"), float(o.strip()[:1].upper() == "B")] for o in provider.complete(prompts, max_new_tokens=8)])
    return pd.DataFrame([{"a": a, "b": b, "p_ab": p[0] / max(p[0] + p[1], 1e-9)} for (a, b), p in zip(pairs, probs)])


def knowledge_from(tiers: list[list[str]] | None = None, pairwise: pd.DataFrame | None = None, threshold: float = 0.6) -> Knowledge:
    """Combine order tiers and pairwise answers into background knowledge. A pairwise answer forbids the opposite direction only when its confidence is at least threshold."""
    pw = pairwise if pairwise is not None else pd.DataFrame(columns=["a", "b", "p_ab"])
    forbidden = {(r.b, r.a) for r in pw.itertuples() if r.p_ab >= threshold} | {(r.a, r.b) for r in pw.itertuples() if r.p_ab <= 1 - threshold}
    return Knowledge(tiers=tiers or [], forbidden=forbidden)


def pairwise_agreement(pw: pd.DataFrame, true, threshold: float = 0.6) -> dict:
    """Over pairs present in the true graph only: the fractions of confident (threshold) LLM answers that are right and wrong."""
    rows = [(r.p_ab if (r.a, r.b) in true.edges else 1 - r.p_ab) for r in pw.itertuples() if true.has_edge(r.a, r.b) or true.has_edge(r.b, r.a)]
    p = pd.Series(rows, dtype=float)
    return {"pairs_on_true_edges": len(p), "correct": int((p >= threshold).sum()), "wrong": int((p <= 1 - threshold).sum()),
            "unsure": int(((p > 1 - threshold) & (p < threshold)).sum()), "pair_accuracy": float((p >= threshold).mean()) if len(p) else 0.0}


def order_agreement(tiers: list[list[str]], true) -> dict:
    """How well the LLM order agrees with true graph edges. consistent: cause in an earlier tier, reversed: flipped, same_tier: same tier, missing: variable not in any tier."""
    t = {v: i for i, tier in enumerate(tiers) for v in tier}
    edges = list(true.edges)
    cnt = {"consistent": 0, "reversed": 0, "same_tier": 0, "missing": 0}
    for u, v in edges:
        key = "missing" if u not in t or v not in t else ("consistent" if t[u] < t[v] else "reversed" if t[u] > t[v] else "same_tier")
        cnt[key] += 1
    return {"edges": len(edges), "tiers": len(tiers), **cnt, "accuracy": cnt["consistent"] / max(len(edges), 1)}


def strong_knowledge(pws: list[pd.DataFrame], tau: float) -> Knowledge:
    """'Certain' background knowledge for hybrid mode: the opposite direction is forbidden only for pairs where all models are at least tau confident in the same direction."""
    m = pd.concat([pw.set_index(["a", "b"]).p_ab.rename(i) for i, pw in enumerate(pws)], axis=1)
    fwd, rev = (m >= tau).all(axis=1), (m <= 1 - tau).all(axis=1)
    return Knowledge(forbidden={(b, a) for a, b in m.index[fwd]} | {(a, b) for a, b in m.index[rev]})


def average_pairwise(pws: list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(pws).groupby(["a", "b"], as_index=False, sort=True).p_ab.mean()
