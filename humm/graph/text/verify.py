"""T7 반사실 검증: 엣지 후보마다 '원인이 바뀌면 결과가 바뀌는가'를 정방향·역방향으로 묻는다(Noul 형식)."""
import pandas as pd

VERIFY_PROMPT = """You are an expert in {domain}.
Question: if {cause} were changed while everything else stayed the same, would {effect} change as a result?
Answer with Yes or No only."""


def _yes_prob(provider, prompts) -> list[float]:
    if getattr(provider, "supports_logits", False):
        return [p[0] / max(p[0] + p[1], 1e-9) for p in provider.option_probs(prompts, ["Yes", "No"])]
    return [float(o.strip().lower().startswith("yes")) for o in provider.complete(prompts, max_new_tokens=8)]


def verify_edges(provider, edges: pd.DataFrame, domain: str) -> pd.DataFrame:
    """p_fwd: 원인 → 결과 반사실 의존 확률, p_rev: 반대 방향."""
    if edges.empty:
        return pd.DataFrame(columns=["src", "dst", "p_fwd", "p_rev"])
    fwd = _yes_prob(provider, [VERIFY_PROMPT.format(domain=domain, cause=a, effect=b) for a, b in zip(edges.src, edges.dst)])
    rev = _yes_prob(provider, [VERIFY_PROMPT.format(domain=domain, cause=b, effect=a) for a, b in zip(edges.src, edges.dst)])
    return pd.DataFrame({"src": edges.src, "dst": edges.dst, "p_fwd": fwd, "p_rev": rev})
