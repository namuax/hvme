"""S2·S4 LLM 사전지식.

- llm_order: 변수 전체의 인과 순서 계층(tier)을 한 번에 묻는다. 엣지를 하나씩 묻는 것보다 안정적이다(Vashishtha et al., Causal Order).
- llm_pairwise: 데이터로 방향을 정하지 못한 엣지만 골라 두 방향 중 어느 쪽인지 묻는다(Choice 형식, 로짓을 주는 모델은 확률로 받는다).
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
    """JSON이 없을 때의 대체 파서: 'Tier 1: a, b' 줄을 읽는다. 초안이 여러 번 나오면 번호가 1로 다시 시작하는 마지막 묶음을 쓴다."""
    runs = []
    for num, body in TIER_LINE.findall(text):
        runs += [[]] if int(num) == 1 or not runs else []
        runs[-1].append([w.strip(" *`'\".()") for w in re.split(r",|\band\b", body)])
    return runs[-1] if runs else []


def parse_tiers(text: str, variables: list[str]) -> list[list[str]]:
    """응답에서 마지막 JSON의 tiers를 읽고, 없으면 'Tier k:' 줄을 읽는다. 없는 이름은 버리고, 같은 이름이 두 번 나오면 처음 것만 쓴다."""
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
    """각 쌍 (a, b)에 대해 p(a → b)를 돌려준다. 로짓을 못 주는 공급자(API)는 생성 답을 0/1로 읽는다."""
    d = descriptions or {}
    prompts = [PAIR_PROMPT.format(domain=domain, a=a, b=b, da=f": {d[a]}" if a in d else "", db=f": {d[b]}" if b in d else "") for a, b in pairs]
    probs = (provider.option_probs(prompts, ["A", "B"]) if getattr(provider, "supports_logits", False)
             else [[float(o.strip()[:1].upper() == "A"), float(o.strip()[:1].upper() == "B")] for o in provider.complete(prompts, max_new_tokens=8)])
    return pd.DataFrame([{"a": a, "b": b, "p_ab": p[0] / max(p[0] + p[1], 1e-9)} for (a, b), p in zip(pairs, probs)])


def knowledge_from(tiers: list[list[str]] | None = None, pairwise: pd.DataFrame | None = None, threshold: float = 0.6) -> Knowledge:
    """순서 계층과 쌍별 답을 배경지식으로 합친다. 쌍별 답은 threshold 이상 확신할 때만 반대 방향을 금지한다."""
    pw = pairwise if pairwise is not None else pd.DataFrame(columns=["a", "b", "p_ab"])
    forbidden = {(r.b, r.a) for r in pw.itertuples() if r.p_ab >= threshold} | {(r.a, r.b) for r in pw.itertuples() if r.p_ab <= 1 - threshold}
    return Knowledge(tiers=tiers or [], forbidden=forbidden)


def pairwise_agreement(pw: pd.DataFrame, true, threshold: float = 0.6) -> dict:
    """참 그래프에 있는 쌍만 놓고, LLM이 확신(threshold)한 답이 맞은 비율과 틀린 비율."""
    rows = [(r.p_ab if (r.a, r.b) in true.edges else 1 - r.p_ab) for r in pw.itertuples() if true.has_edge(r.a, r.b) or true.has_edge(r.b, r.a)]
    p = pd.Series(rows, dtype=float)
    return {"pairs_on_true_edges": len(p), "correct": int((p >= threshold).sum()), "wrong": int((p <= 1 - threshold).sum()),
            "unsure": int(((p > 1 - threshold) & (p < threshold)).sum()), "pair_accuracy": float((p >= threshold).mean()) if len(p) else 0.0}


def order_agreement(tiers: list[list[str]], true) -> dict:
    """LLM 순서가 참 그래프 엣지와 얼마나 맞는가. consistent: 원인이 앞 계층, reversed: 뒤집힘, same_tier: 같은 계층, missing: 계층에 없는 변수."""
    t = {v: i for i, tier in enumerate(tiers) for v in tier}
    edges = list(true.edges)
    cnt = {"consistent": 0, "reversed": 0, "same_tier": 0, "missing": 0}
    for u, v in edges:
        key = "missing" if u not in t or v not in t else ("consistent" if t[u] < t[v] else "reversed" if t[u] > t[v] else "same_tier")
        cnt[key] += 1
    return {"edges": len(edges), "tiers": len(tiers), **cnt, "accuracy": cnt["consistent"] / max(len(edges), 1)}


def strong_knowledge(pws: list[pd.DataFrame], tau: float) -> Knowledge:
    """혼합 모드의 '확실한' 배경지식: 모든 모델이 같은 방향을 tau 이상 확신한 쌍만 반대 방향을 금지한다."""
    m = pd.concat([pw.set_index(["a", "b"]).p_ab.rename(i) for i, pw in enumerate(pws)], axis=1)
    fwd, rev = (m >= tau).all(axis=1), (m <= 1 - tau).all(axis=1)
    return Knowledge(forbidden={(b, a) for a, b in m.index[fwd]} | {(a, b) for a, b in m.index[rev]})


def average_pairwise(pws: list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(pws).groupby(["a", "b"], as_index=False, sort=True).p_ab.mean()
