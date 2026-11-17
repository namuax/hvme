"""T1/T2/T3 document splitting and causal claim extraction."""
import json
import re

import pandas as pd

EXTRACT_PROMPT = """Extract every statement that relates two variables from the text below.
For each statement return an object with:
- "cause": the first variable, copied as it appears in the text (noun phrase only; for Korean drop particles such as 이/가/은/는/을/를/와/과/도/의)
- "effect": the second variable, copied the same way
- "relation": "causal" if the text says the cause changes the effect, "correlational" if it only says the two are associated, "no_effect" if it says there is no effect
- "polarity": "+" if the effect increases, "-" if it decreases, "?" if unclear
- "hedged": true if the statement is uncertain (may, could, possibly, ~할 수도 있다), otherwise false
- "quote": the sentence
For "X is driven by Y" or "X는 Y 때문에 올라간다", the cause is Y. Ignore sentences that do not relate two variables.
Return only a JSON object in a ```json code block``` of the form {{"claims": [...]}}.

Text:
{text}"""

SENT = re.compile(r"(?<=[.!?。])\s+")
JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)
JSON_ANY = re.compile(r"(\{\s*\"claims\".*\})", re.S)
RELATIONS = {"causal", "correlational", "no_effect"}


def chunk(docs: list[dict], max_chars: int = 1500) -> list[dict]:
    """Split a document at sentence boundaries into pieces of at most max_chars."""
    out = []
    for d in docs:
        buf, k = "", 0
        for s in SENT.split(d["text"]):
            if buf and len(buf) + len(s) > max_chars:
                out.append({"doc_id": d["id"], "chunk_id": f"{d['id']}#{k}", "text": buf.strip()})
                buf, k = "", k + 1
            buf += " " + s
        out.append({"doc_id": d["id"], "chunk_id": f"{d['id']}#{k}", "text": buf.strip()})
    return out


CHANGE_EN = re.compile(r"^(?:higher|lower|increased|decreased|increasing|decreasing|rising|falling|more|less|greater|reduced)\s+", re.I)
CHANGE_KO = re.compile(r"\s*(?:증가|감소|상승|하락|증대|저하)$")


def clean(phrase: str) -> str:
    """Keep only the variable name: strip surrounding symbols and increase/decrease wording such as 'higher ...' or '... 증가' (polarity carries the direction of change)."""
    p = re.sub(r"\s+", " ", str(phrase).strip(" \"'`.,;:()[]")).strip()
    return CHANGE_KO.sub("", CHANGE_EN.sub("", p)).strip() or p


def parse_claims(raw: str) -> list[dict]:
    blocks = JSON_BLOCK.findall(raw) or JSON_ANY.findall(raw)
    claims = json.loads(blocks[-1]).get("claims", []) if blocks else []
    out = []
    for c in claims:
        rel = str(c.get("relation", "")).lower().replace("-", "_")
        pol = str(c.get("polarity", "?"))
        if c.get("cause") and c.get("effect") and rel in RELATIONS:
            out.append({"cause": clean(c["cause"]), "effect": clean(c["effect"]), "relation": rel, "polarity": pol if pol in "+-?" else "?",
                        "hedged": bool(c.get("hedged", False)), "quote": str(c.get("quote", ""))})
    return out


def extract_claims(provider, chunks: list[dict], max_new_tokens: int = 2048) -> tuple[pd.DataFrame, list[str]]:
    raws = provider.complete([EXTRACT_PROMPT.format(text=c["text"]) for c in chunks], max_new_tokens=max_new_tokens)
    rows = [{"doc_id": c["doc_id"], "chunk_id": c["chunk_id"], **x} for c, raw in zip(chunks, raws) for x in parse_claims(raw)]
    cols = ["doc_id", "chunk_id", "cause", "effect", "relation", "polarity", "hedged", "quote"]
    return pd.DataFrame(rows, columns=cols), raws
