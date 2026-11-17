"""텍스트 HCG와 정형(데이터) HCG를 합친다. 같은 변수의 노드를 이어 붙이고 엣지 증거를 모은다."""
from difflib import SequenceMatcher

from ...validate.report import DATA_DIRECTION_METHODS, DIRECTION_KINDS
from ..schema import HCG, Edge, EdgeStatus, KIND_WEIGHT
from .canonical import norm


def direction_score(e: Edge) -> float:
    """방향을 뒷받침하는 증거만의 noisy-OR. 인접성만 말해 주는 데이터 증거(골격·검정·임의 확장)는 빼고,
    텍스트·LLM·전문가·실험 증거와 PC가 v-구조·Meek 규칙으로 정한 방향만 센다."""
    miss = 1.0
    for x in e.evidence:
        if x.kind in DIRECTION_KINDS or x.method in DATA_DIRECTION_METHODS:
            miss *= 1 - KIND_WEIGHT[x.kind] * x.score
    return 1 - miss


def link_nodes(text: HCG, table: HCG, column_aliases: dict | None = None, cutoff: float = 0.85) -> dict:
    """텍스트 노드 id → 정형 노드 id. 텍스트 노드의 이름·별칭이 열 이름(또는 열 별칭)과 같거나 충분히 비슷하면 잇는다."""
    names = {n.id: {norm(n.id), norm(n.label)} | {norm(a) for a in (column_aliases or {}).get(n.id, [])} for n in table.nodes}
    out = {}
    for n in text.nodes:
        mine = {norm(n.id)} | {norm(a) for a in n.aliases}
        score = {c: max(SequenceMatcher(None, a, b).ratio() for a in mine for b in ns) for c, ns in names.items()}
        best = max(score, key=score.get)
        out |= {n.id: best} if score[best] >= cutoff else {}
    return out


def merge_hcg(text: HCG, table: HCG, link: dict, graph_id: str = "merged") -> HCG:
    """정형 노드 이름을 기준으로 합친다. 연결되지 않은 텍스트 노드는 그대로 추가한다(관측되지 않은 변수일 수 있다).
    같은 엣지는 증거를 합치고 상태를 hypothesis로 되돌려 다시 판정받게 한다. 반증된 엣지는 넘기지 않는다."""
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
    for e in list(out.edges):  # 반대 방향 엣지가 함께 있으면 방향 증거가 강한 쪽만 남기고, 인접성 증거는 그쪽으로 옮긴다
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
