"""T5 canonicalization: group multiple expressions of the same variable under one canonical name."""
import json
import re

CANON_PROMPT = """The phrases below were extracted from documents in the domain: {domain}.
Some phrases refer to the same variable (synonyms, paraphrases, abbreviations, or different wording of the same measure).
Group them so that each group is exactly one variable, and give each group a short canonical name.
Return only a JSON object in a ```json code block``` of the form {{"groups": [{{"name": "...", "members": ["...", "..."]}}]}}. Every phrase must appear in exactly one group.

Phrases:
{phrases}"""

JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)
JSON_ANY = re.compile(r"(\{\s*\"groups\".*\})", re.S)


def norm(p: str) -> str:
    return re.sub(r"[^\w가-힣]+", " ", p.lower()).strip()


def parse_groups(raw: str, phrases: list[str]) -> dict:
    """phrase -> canonical name. A phrase missing from the response is its own name. Groups with the same name are merged."""
    blocks = JSON_BLOCK.findall(raw) or JSON_ANY.findall(raw)
    groups = json.loads(blocks[-1]).get("groups", []) if blocks else []
    by_norm = {norm(p): p for p in phrases}
    out = {}
    for grp in groups:
        name = str(grp.get("name", "")).strip()
        for m in grp.get("members", []):
            p = by_norm.get(norm(str(m)))
            if p and p not in out and name:
                out[p] = name
    return out | {p: p for p in phrases if p not in out}


def canonicalize(provider, phrases: list[str], domain: str, max_new_tokens: int = 2048) -> tuple[dict, str]:
    """Expressions that differ only in notation (case, whitespace, symbols) are merged first; the LLM groups the rest."""
    uniq = sorted(set(phrases), key=lambda p: (norm(p), p))  # always pick the same representative among notation variants (same prompt on every run)
    first = {}
    for p in uniq:
        first.setdefault(norm(p), p)
    reps = sorted(set(first.values()))
    raw = provider.complete([CANON_PROMPT.format(domain=domain, phrases="\n".join(f"- {p}" for p in reps))], max_new_tokens=max_new_tokens)[0]
    rep_map = parse_groups(raw, reps)
    return {p: rep_map[first[norm(p)]] for p in uniq}, raw
