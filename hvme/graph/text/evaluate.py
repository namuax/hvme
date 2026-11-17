"""Text pipeline evaluation: compare against the ground-truth graph of the synthetic corpus."""
from collections import Counter

import networkx as nx

from .canonical import norm


def gold_ids(mapping: dict, corpus) -> dict:
    """canonical name -> true id (the majority true id among the expressions grouped under that name)."""
    alias = {norm(s): i for s, i in corpus.aliases.items()}
    votes = {}
    for phrase, name in mapping.items():
        votes.setdefault(name, Counter())[alias.get(norm(phrase))] += 1
    return {name: c.most_common(1)[0][0] for name, c in votes.items()}


def evaluate(edges, mapping: dict, corpus) -> dict:
    """edges: list of (src, dst) canonical-name pairs. Reports precision/recall on edges with the correct direction, and canonicalization quality."""
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
    known = [i - {None} for i in groups.values() if i - {None}]  # expressions absent from the ground-truth dictionary (e.g. with modifiers attached) are excluded from the purity computation
    return {"pred": len(pred), "true": len(true), "tp": tp, "reversed": reversed_, "precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-12),
            "group_purity": sum(len(i) == 1 for i in known) / max(len(known), 1),
            "var_split": sum(n > 1 for n in covered.values()) / max(len(covered), 1),
            "unmapped_phrases": sum(alias.get(norm(ph)) is None for ph in mapping) / max(len(mapping), 1)}
