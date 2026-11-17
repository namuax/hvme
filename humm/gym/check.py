"""저장된 문항의 정답 재검증. 문항에 남긴 그래프·확률표·메타만으로 엔진 계산을 다시 해 label과 value를 대조한다."""
import networkx as nx
import numpy as np

from ..engine import BinarySCM, d_separated, identify, is_valid_adjustment
from ..engine.discovery import holds, mec_of
from .graphs import mec
from .queries import SCORE_EDGES, STRATEGIES, _rel


def _scm(d):
    g = nx.DiGraph([tuple(e) for e in d["graph"]])
    g.add_nodes_from(d["meta"]["cpt"])
    return BinarySCM(g, {v: np.array(p) for v, p in d["meta"]["cpt"].items()})


VALUE = {
    "marginal": lambda m, c: m.prob(c["y"]) - 0.5,
    "correlation": lambda m, c: m.prob(c["y"], given={c["x"]: 1}) - m.prob(c["y"], given={c["x"]: 0}),
    "ate": lambda m, c: m.ate(c["x"], c["y"]),
    "cde": lambda m, c: m.cde(c["x"], c["y"], c["fix"]),
    "ett": lambda m, c: m.ett(c["x"], c["y"]),
    "nde": lambda m, c: m.nde(c["x"], c["y"], c["med"]),
    "nie": lambda m, c: m.nie(c["x"], c["y"], c["med"]),
}


def _bits(q: dict) -> dict:
    """JSON으로 저장된 v3 엔진 질의(meta.q)의 값을 정수로 되돌린다."""
    return {k: ({a: int(b) for a, b in v.items()} if isinstance(v, dict) else v) for k, v in q.items()}


def recompute_label(d: dict) -> int:
    t, c = d["query_type"], d["meta"]
    g = nx.DiGraph([tuple(e) for e in d["graph"]])
    if t == "ancestry":  # v3.8: 부모·조상
        from .ancestry import recompute
        return recompute(d)
    if t == "graphid":  # v3.9: 앞문·도구변수·교란 쌍 식별
        from .graphid import recompute
        return recompute(d)
    if d["kind"] == "prob":  # v3.7: 저장한 엔진 호출을 다시 돌린다
        from .prob import recompute
        return recompute(d)
    if "q" in c:  # v3 수치 질의: 값 = 처치 세계 − 기준 세계
        from ..lm.tools import evaluate
        return int((evaluate(_scm(d), _bits(c["q"]))[0] > 0) == (c["direction"] == "up"))
    if t == "det_cf":
        from .v3 import det_values
        rules = {v: (op, [tuple(l) for l in lits]) for v, (op, lits) in c["rules"].items()}
        order = list(nx.topological_sort(g))
        return int(det_values(order, rules, c["exo"], {c["x"]: 1 - c["fact"][c["x"]]})[c["y"]] == c["ask"])
    if t == "dsep" and c.get("asks") == "dependent":
        return int(not d_separated(g, c["a"], c["b"], c["z"]))
    if t == "backadj" and "methods" in c:
        return int(is_valid_adjustment(g, c["x"], c["y"], c["methods"][0]))
    if t in VALUE:
        v = VALUE[t](_scm(d), c)
        return int((v > 0) == (c["direction"] == "up"))
    if t == "ate_score":
        return int(np.digitize(_scm(d).ate(c["x"], c["y"]), SCORE_EDGES)) + 1
    if t == "dsep":
        return int(d_separated(g, c["a"], c["b"], c["z"]))
    if t == "backadj":
        return int(is_valid_adjustment(g, c["x"], c["y"], c["z"]))
    if t == "identify":
        g.add_nodes_from(d["latent"])
        return c["option_order"].index(identify(g, c["x"], c["y"], set(d["latent"])).strategy)
    if t == "corr2cause" and "call" in c:  # v3.3: 변수 6개까지, 원 벤치 가설 정의
        g.add_nodes_from(c["names"])
        return int(all(holds(c["hypothesis"], m, c["a"], c["b"]) for m in mec_of(g)))
    if t == "corr2cause":
        g.add_nodes_from(c["names"])
        return int(all(_rel(c["hypothesis"], m, c["a"], c["b"]) for m in mec(g)))
    raise KeyError(t)


def audit(items: list[dict]) -> list[str]:
    """정답이 재계산 결과와 다른 문항의 id 목록. 비어 있어야 한다."""
    return [d["id"] for d in items if recompute_label(d) != d["label"]]
