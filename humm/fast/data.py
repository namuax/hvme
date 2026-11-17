"""Humm-Fast training and evaluation items.

Item = {"id", "prompt", "labels"(answer tokens), "target"(target distribution over labels), "kind"(noul·choice·score), "task", "lang", "source"}.
- gym: Humm-Gym direct format (hard label, one-hot)
- pairwise: edge-direction Choice. Same wording as the pairwise query (PAIR_PROMPT) of the tabular pipeline, so a trained model can be plugged in directly
- verify: counterfactual-dependence Noul. Same wording as the L3 verification (VERIFY_PROMPT) of the text pipeline
A non-one-hot distribution in target trains as a soft label (same cross-entropy loss).
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
    """One direction query per true edge. The order of the two variables is random, so the answer is split evenly between A and B."""
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
    """Ancestor → descendant gives 'yes' (counterfactual dependence); descendant → ancestor or unrelated gives 'no'. Sampled half yes, half no."""
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
