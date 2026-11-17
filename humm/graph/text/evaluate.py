"""텍스트 파이프라인 평가: 합성 코퍼스의 정답 그래프와 비교한다."""
from collections import Counter

import networkx as nx

from .canonical import norm


def gold_ids(mapping: dict, corpus) -> dict:
    """정규 이름 → 정답 id (그 이름으로 묶인 표현들의 정답 id 중 다수)."""
    alias = {norm(s): i for s, i in corpus.aliases.items()}
    votes = {}
    for phrase, name in mapping.items():
        votes.setdefault(name, Counter())[alias.get(norm(phrase))] += 1
    return {name: c.most_common(1)[0][0] for name, c in votes.items()}


def evaluate(edges, mapping: dict, corpus) -> dict:
    """edges: (src, dst) 정규 이름 쌍의 목록. 방향까지 맞힌 엣지 기준 정밀도·재현율과 정규화 품질을 낸다."""
    gid = gold_ids(mapping, corpus)
    pred = {(gid.get(a), gid.get(b)) for a, b in edges}
    true = set(corpus.graph.edges)
    tp = len(pred & true)
    p, r = tp / max(len(pred), 1), tp / max(len(true), 1)
    alias = {norm(s): i for s, i in corpus.aliases.items()}
    groups = {}
    for phrase, name in mapping.items():
        groups.setdefault(name, set()).add(alias.get(norm(phrase)))
    covered = Counter(i for ids in groups.values() for i in ids if i)
    reversed_ = len({(b, a) for a, b in pred} & true)
    known = [i - {None} for i in groups.values() if i - {None}]  # 정답 사전에 없는 표현(예: 수식어가 붙은 표현)은 순도 계산에서 뺀다
    return {"pred": len(pred), "true": len(true), "tp": tp, "reversed": reversed_, "precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-12),
            "group_purity": sum(len(i) == 1 for i in known) / max(len(known), 1),
            "var_split": sum(n > 1 for n in covered.values()) / max(len(covered), 1),
            "unmapped_phrases": sum(alias.get(norm(ph)) is None for ph in mapping) / max(len(mapping), 1)}
