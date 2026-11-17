"""Convert an HCG to text for LLM prompts. v1 uses an 'edge list + mechanism description' serialization."""
import json

from .schema import HCG


def to_edge_text(g: HCG, with_confidence: bool = False) -> str:
    """List of lines in the form 'ad_spend -> sales (+): mechanism'."""
    name = {n.id: n.label or n.id for n in g.nodes}
    lines = []
    for e in g.active_edges():
        conf = f" [conf={e.confidence:.2f}]" if with_confidence else ""
        mech = f": {e.mechanism}" if e.mechanism else ""
        lines.append(f"{name[e.src]} -> {name[e.dst]} ({e.sign}){conf}{mech}")
    return "\n".join(lines)


def to_compact_json(g: HCG) -> str:
    """Token-saving minimal JSON: node names and a list of [cause, effect, sign]."""
    name = {n.id: n.label or n.id for n in g.nodes}
    return json.dumps(
        {"nodes": list(name.values()), "edges": [[name[e.src], name[e.dst], e.sign] for e in g.active_edges()]},
        ensure_ascii=False,
    )


def to_prompt_block(g: HCG) -> str:
    """Graph block prepended to a query."""
    return f"[인과 그래프: {g.graph_id}]\n{to_edge_text(g)}\n[가정]\n" + "\n".join(f"- {a}" for a in g.assumptions)
