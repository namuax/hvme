"""T5 정규화: 같은 변수를 가리키는 여러 표현을 하나의 정규 이름으로 묶는다."""
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
    """phrase → 정규 이름. 응답에 없는 표현은 자기 자신을 이름으로 쓴다. 같은 이름의 그룹은 합친다."""
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
    """표기만 다른 표현(대소문자·공백·기호)은 먼저 합치고, 나머지는 LLM이 묶는다."""
    uniq = sorted(set(phrases), key=lambda p: (norm(p), p))  # 표기 변형 중 대표를 항상 같게 고른다(실행마다 같은 프롬프트)
    first = {}
    for p in uniq:
        first.setdefault(norm(p), p)
    reps = sorted(set(first.values()))
    raw = provider.complete([CANON_PROMPT.format(domain=domain, phrases="\n".join(f"- {p}" for p in reps))], max_new_tokens=max_new_tokens)[0]
    rep_map = parse_groups(raw, reps)
    return {p: rep_map[first[norm(p)]] for p in uniq}, raw
