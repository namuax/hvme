"""질의 분석: 질문 → (유형, 처치, 결과, 매개). 규칙 기반이 기본이고, 범용 LLM을 주면 JSON으로 묻는다."""
import json
import re
from dataclasses import dataclass, field, replace

from ..graph.schema import HCG
from ..graph.text.canonical import norm

KINDS = ("ate", "association", "mediation", "ett")
KEYWORDS = [("mediation", r"간접|직접 ?효과|거치|거쳐|매개|경유|indirect|direct effect|mediat"),
            ("ett", r"않았다면|안 했다면|없었다면|했다면|였다면|would have|had not|hadn't"),
            ("association", r"상관|연관|함께 움직|correlat|associat"),
            ("ate", r"영향|효과|늘리|줄이|낮추|높이|올리|오르|낮아지|늘어나")]  # 인과 동사: 기본 ate를 라우터가 덮어쓰지 않게
CAUSE_LAST = r"때문|탓|덕분|because of|due to|caused by"  # "Y가 오른 것은 X 때문인가" → 원인이 뒤에 나온다

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
    """질문에 나오는 변수 id를 처음 등장한 순서대로 돌려준다. 긴 표현부터 맞추고 이미 맞춘 구간과 겹치면 건너뛴다."""
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
    assert len(vs) >= 2, f"질문에서 그래프 변수를 두 개 이상 찾지 못했다: {vs}"
    kind = next((k for k, pat in KEYWORDS if re.search(pat, text, re.I)), "ate")
    x, y = (vs[1], vs[0]) if re.search(CAUSE_LAST, text, re.I) else (vs[0], vs[1])
    return Query(text, kind, x, y, vs[2:] if kind == "mediation" else [])


def parse_llm(text: str, hcg: HCG, llm) -> Query | None:
    """LLM 답이 JSON이 아니거나 그래프에 없는 변수를 가리키면 None(규칙 기반으로 넘어간다)."""
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
    """변수(처치·결과·매개)는 질문 문자열에 실제로 나온 표현(규칙)으로 정한다. LLM은 두 경우에만 쓴다.
    - 질문에서 변수를 두 개 이상 찾지 못했을 때: LLM 해석 전체를 쓴다
    - 유형 키워드(매개·반사실·상관)가 없을 때: LLM이 고른 유형만 쓴다
    (2026-10-25 GPU: '고객 이탈률이 광고비에 영향을 주는가'의 방향을 Qwen3.5 4B·9B가 모두 뒤집었다. reports/2026-10-25_e2e.md)"""
    if len(mentions(text, hcg)) < 2:
        return (llm and parse_llm(text, hcg, llm)) or parse_rules(text, hcg)
    q = parse_rules(text, hcg)
    keyword = any(re.search(pat, text, re.I) for _, pat in KEYWORDS)
    lq = None if keyword or llm is None else parse_llm(text, hcg, llm)
    return replace(q, kind=lq.kind, parser=f"rules+{lq.parser}") if lq else q
