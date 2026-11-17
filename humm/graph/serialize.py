"""HCG를 LLM 프롬프트용 텍스트로 바꾼다. v1은 '엣지 리스트 + 메커니즘 설명' 직렬화를 쓴다(02_proposal.md 6장 참고)."""
import json

from .schema import HCG


def to_edge_text(g: HCG, with_confidence: bool = False) -> str:
    """'광고비 -> 매출 (+): 메커니즘' 형식의 줄 목록."""
    name = {n.id: n.label or n.id for n in g.nodes}
    lines = []
    for e in g.active_edges():
        conf = f" [conf={e.confidence:.2f}]" if with_confidence else ""
        mech = f": {e.mechanism}" if e.mechanism else ""
        lines.append(f"{name[e.src]} -> {name[e.dst]} ({e.sign}){conf}{mech}")
    return "\n".join(lines)


def to_compact_json(g: HCG) -> str:
    """토큰을 아끼는 최소 JSON: 노드 이름과 [원인, 결과, 부호] 리스트."""
    name = {n.id: n.label or n.id for n in g.nodes}
    return json.dumps(
        {"nodes": list(name.values()), "edges": [[name[e.src], name[e.dst], e.sign] for e in g.active_edges()]},
        ensure_ascii=False,
    )


def to_prompt_block(g: HCG) -> str:
    """질의 앞에 붙이는 그래프 블록."""
    return f"[인과 그래프: {g.graph_id}]\n{to_edge_text(g)}\n[가정]\n" + "\n".join(f"- {a}" for a in g.assumptions)
