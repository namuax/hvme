"""Humm-Fast 학습·평가 문항.

문항 = {"id", "prompt", "labels"(답 토큰), "target"(labels 위의 목표 분포), "kind"(noul·choice·score), "task", "lang", "source"}.
- gym: Humm-Gym direct 형식(정답 라벨, 원-핫)
- pairwise: 엣지 방향 Choice. Step 5 정형 파이프라인의 쌍별 질의(PAIR_PROMPT)와 같은 문장이라 학습한 모델을 그대로 끼울 수 있다
- verify: 반사실 의존 Noul. Step 6 텍스트 파이프라인의 L3 검증(VERIFY_PROMPT)과 같은 문장
target에 원-핫이 아닌 분포를 넣으면 연성 라벨로 학습한다(손실은 같은 교차 엔트로피).
"""
import networkx as nx
import numpy as np

from ..graph.tabular.networks import NETWORKS
from ..graph.tabular.prior import PAIR_PROMPT
from ..graph.text.verify import VERIFY_PROMPT
from ..gym.formats import option_labels, render, target_index


def onehot(k: int, n: int) -> list[float]:
    return [float(i == k) for i in range(n)]


def from_gym(items: list[dict]) -> list[dict]:
    return [{"id": d["id"], "prompt": render(d, "direct"), "labels": option_labels(d), "target": onehot(target_index(d), len(option_labels(d))),
             "kind": d["kind"], "task": d["query_type"], "lang": d["lang"], "source": "gym"} for d in items]


def _desc(name: str, v: str) -> str:
    return NETWORKS[name]["descriptions"].get(v, "")


def pairwise_items(name: str, g: nx.DiGraph, seed: int = 0) -> list[dict]:
    """참 엣지마다 방향 질의 하나. 두 변수를 쓰는 순서는 무작위라 정답이 A·B 반반이다."""
    rng = np.random.default_rng(seed)
    out = []
    for i, (u, v) in enumerate(sorted(g.edges)):
        a, b = (u, v) if rng.random() < 0.5 else (v, u)
        da, db = _desc(name, a), _desc(name, b)
        prompt = PAIR_PROMPT.format(domain=NETWORKS[name]["domain"], a=a, b=b, da=f": {da}" if da else "", db=f": {db}" if db else "")
        out.append({"id": f"pair-{name}-{i:04d}", "prompt": prompt, "labels": ["A", "B"], "target": onehot(int(a != u), 2),
                    "kind": "choice", "task": "pairwise", "lang": "en", "source": name})
    return out


def verify_items(name: str, g: nx.DiGraph, n: int = 200, seed: int = 0) -> list[dict]:
    """조상 → 자손이면 '예'(반사실 의존), 자손 → 조상이거나 서로 무관하면 '아니오'. 예·아니오 반반으로 뽑는다."""
    rng = np.random.default_rng(seed)
    anc = {v: nx.ancestors(g, v) for v in g}
    yes = sorted((a, b) for b in g for a in anc[b])
    no = sorted((a, b) for a in g for b in g if a != b and a not in anc[b])
    k = min(n // 2, len(yes), len(no))
    pick = [yes[i] for i in rng.choice(len(yes), k, replace=False)] + [no[i] for i in rng.choice(len(no), k, replace=False)]
    name_of = lambda v: _desc(name, v) or v
    return [{"id": f"verify-{name}-{i:04d}", "prompt": VERIFY_PROMPT.format(domain=NETWORKS[name]["domain"], cause=name_of(a), effect=name_of(b)),
             "labels": ["Yes", "No"], "target": onehot(int(i >= k), 2), "kind": "noul", "task": "verify", "lang": "en", "source": name}
            for i, (a, b) in enumerate(pick)]


def network_items(names, graphs: dict, n_verify: int = 200, seed: int = 0) -> list[dict]:
    return [it for name in names for it in pairwise_items(name, graphs[name], seed) + verify_items(name, graphs[name], n_verify, seed)]
