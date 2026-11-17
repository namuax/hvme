"""Query analysis: question -> (kind, treatment, outcome, mediators). Rule-based by default; if a general-purpose LLM is given, it is asked for JSON."""
import json
import re
from dataclasses import dataclass, field, replace

from ..graph.schema import HCG
from ..graph.text.canonical import norm

KINDS = ("ate", "association", "mediation", "ett")
KEYWORDS = [("mediation", r"간접|직접 ?효과|거치|거쳐|매개|경유|indirect|direct effect|mediat"),
            ("ett", r"않았다면|안 했다면|없었다면|했다면|였다면|would have|had not|hadn't"),
            ("association", r"상관|연관|함께 움직|correlat|associat"),
            ("ate", r"영향|효과|늘리|줄이|낮추|높이|올리|오르|낮아지|늘어나")]  # causal verbs: keep the router from overriding the default ate
CAUSE_LAST = r"때문|탓|덕분|because of|due to|caused by"  # "Y rose because of X?" -> the cause comes last

PARSE_PROMPT = """You map a causal question onto the variables of a causal graph.

Variables (id: names):
{variables}

Question: {question}

Question kinds:
- ate: effect on the outcome of intervening on the treatment
- association: whether the observed correlation between two variables reflects causation
- mediation: how much of the effect of the treatment goes through mediator variables
- ett: counterfactual effect of the treatment on units that actually received it

Answer with one JSON object only, using variable ids:
{{"kind": "...", "treatment": "...", "outcome": "...", "mediators": []}}"""


@dataclass
class Query:
    text: str
    kind: str
    treatment: str
    outcome: str
    mediators: list = field(default_factory=list)
    parser: str = "rules"


def mentions(text: str, hcg: HCG) -> list[str]:
    """Return variable ids mentioned in the question in order of first appearance. Longer expressions are matched first; a match overlapping an already matched span is skipped."""
    t = norm(text)
    names = sorted({(norm(s), n.id) for n in hcg.nodes for s in {n.id, n.label, *n.aliases} if norm(s)}, key=lambda x: (-len(x[0]), x))
    taken, first = [], {}
    for s, nid in names:
        for m in re.finditer(re.escape(s), t):
            free = all(m.end() <= a or b <= m.start() for a, b in taken)
            taken += [(m.start(), m.end())] * free
            first[nid] = min(first.get(nid, m.start()), m.start()) if free else first.get(nid, len(t) + 1)
    return sorted((k for k, v in first.items() if v <= len(t)), key=first.get)


def parse_rules(text: str, hcg: HCG) -> Query:
    vs = mentions(text, hcg)
    assert len(vs) >= 2, f"fewer than two graph variables found in the question: {vs}"
    kind = next((k for k, pat in KEYWORDS if re.search(pat, text, re.I)), "ate")
    x, y = (vs[1], vs[0]) if re.search(CAUSE_LAST, text, re.I) else (vs[0], vs[1])
    return Query(text, kind, x, y, vs[2:] if kind == "mediation" else [])


def parse_llm(text: str, hcg: HCG, llm) -> Query | None:
    """None if the LLM answer is not JSON or refers to a variable not in the graph (falls back to rules)."""
    variables = "\n".join(f"- {n.id}: {', '.join(sorted({n.label, *n.aliases} - {''}))}" for n in hcg.nodes)
    raw = llm.complete([PARSE_PROMPT.format(variables=variables, question=text)], max_new_tokens=256)[0]
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        d = json.loads(m.group()) if m else {}
    except json.JSONDecodeError:
        d = {}
    ids = {n.id for n in hcg.nodes}
    ok = d.get("kind") in KINDS and d.get("treatment") in ids and d.get("outcome") in ids and set(d.get("mediators") or []) <= ids
    return Query(text, d["kind"], d["treatment"], d["outcome"], list(d.get("mediators") or []), getattr(llm, "name", "llm")) if ok else None


def parse(text: str, hcg: HCG, llm=None) -> Query:
    """Variables (treatment, outcome, mediators) are decided by rules, from expressions that actually appear in the question string. The LLM is used in two cases only.
    - fewer than two variables found in the question: use the whole LLM interpretation
    - no kind keyword (mediation/counterfactual/association): use only the kind chosen by the LLM
    (Observed on GPU: Qwen3.5 4B and 9B both flipped the direction of '고객 이탈률이 광고비에 영향을 주는가'.)"""
    if len(mentions(text, hcg)) < 2:
        return (llm and parse_llm(text, hcg, llm)) or parse_rules(text, hcg)
    q = parse_rules(text, hcg)
    keyword = any(re.search(pat, text, re.I) for _, pat in KEYWORDS)
    lq = None if keyword or llm is None else parse_llm(text, hcg, llm)
    return replace(q, kind=lq.kind, parser=f"rules+{lq.parser}") if lq else q
