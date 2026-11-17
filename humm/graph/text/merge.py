"""Merge a text HCG and a tabular (data) HCG. Links nodes of the same variable and pools edge evidence."""
from difflib import SequenceMatcher

from ...validate.report import DATA_DIRECTION_METHODS, DIRECTION_KINDS
from ..schema import HCG, Edge, EdgeStatus, KIND_WEIGHT
from .canonical import norm


def direction_score(e: Edge) -> float:
    """noisy-OR over direction-supporting evidence only. Data evidence that only speaks to adjacency (skeleton, tests, arbitrary extension) is excluded;
    only text, LLM, expert and experiment evidence, plus directions PC fixed via v-structures and Meek rules, are counted."""
    miss = 1.0
    for x in e.evidence:
        if x.kind in DIRECTION_KINDS or x.method in DATA_DIRECTION_METHODS:
            miss *= 1 - KIND_WEIGHT[x.kind] * x.score
    return 1 - miss


def link_nodes(text: HCG, table: HCG, column_aliases: dict | None = None, cutoff: float = 0.85) -> dict:
    """text node id -> tabular node id. A text node is linked if its name/alias equals, or is similar enough to, a column name (or column alias)."""
    names = {n.id: {norm(n.id), norm(n.label)} | {norm(a) for a in (column_aliases or {}).get(n.id, [])} for n in table.nodes}
    out = {}
    for n in text.nodes:
        mine = {norm(n.id)} | {norm(a) for a in n.aliases}
        score = {c: max(SequenceMatcher(None, a, b).ratio() for a in mine for b in ns) for c, ns in names.items()}
        best = max(score, key=score.get)
        out |= {n.id: best} if score[best] >= cutoff else {}
    return out


def merge_hcg(text: HCG, table: HCG, link: dict, graph_id: str = "merged") -> HCG:
    """Merge using tabular node names as the reference. Unlinked text nodes are added as is (they may be unobserved variables).
    Identical edges have their evidence pooled and their status reset to hypothesis so they are judged again. Refuted edges are not carried over."""
    out = HCG(graph_id=graph_id)
    for n in table.nodes:
        out.add_node(n.id, **n.model_dump(exclude={"id"}))
    for n in text.nodes:
        if n.id in link:
            node = out.node(link[n.id])
            node.aliases = sorted(set(node.aliases) | set(n.aliases) | {n.id} - {node.id})
            node.source = node.source + n.source
        else:
            out.add_node(n.id, **n.model_dump(exclude={"id"}))
    for e in table.active_edges() + text.active_edges():
        u, v = link.get(e.src, e.src) if e in text.edges else e.src, link.get(e.dst, e.dst) if e in text.edges else e.dst
        if any(x.key == (u, v) for x in out.edges):
            out.edge(u, v).evidence += e.evidence
            out.edge(u, v).sign = e.sign if out.edge(u, v).sign == "?" else out.edge(u, v).sign
        else:
            out.add_edge(u, v, sign=e.sign, mechanism=e.mechanism, evidence=list(e.evidence))
    for e in out.edges:
        e.status = EdgeStatus.HYPOTHESIS
    for e in list(out.edges):  # if opposite-direction edges coexist, keep only the one with stronger direction evidence and move adjacency evidence onto it
        back = next((x for x in out.edges if x.key == (e.dst, e.src)), None)
        if back and e.status != EdgeStatus.REFUTED and back.status != EdgeStatus.REFUTED:
            keep, drop = (e, back) if direction_score(e) >= direction_score(back) else (back, e)
            keep.evidence += [x for x in drop.evidence if not (x.kind in DIRECTION_KINDS or x.method in DATA_DIRECTION_METHODS)]
            drop.status = EdgeStatus.REFUTED
    for e in out.active_edges():
        e.update_confidence()
        e.promote()
    out.break_cycles()
    return out
