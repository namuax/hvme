"""L4 experiment comparison: compare the effect from randomized experiment (A/B test) data with the estimate from observational data.

If the experiment randomized treatment x, E[y|x=1] - E[y|x=0] is the interventional effect.
If the observational estimate overlaps the experimental confidence interval, that is evidence the graph (adjustment set) is right for this effect,
and experimental evidence is attached to the direct x→y edge if there is one.
"""
import numpy as np
import pandas as pd

from ..graph.schema import HCG, Evidence, EvidenceKind


def experiment_effect(exp: pd.DataFrame, x: str, y: str, n_boot: int = 200, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    diff = lambda d: float(d.loc[d[x] == 1, y].mean() - d.loc[d[x] == 0, y].mean())
    boots = [diff(exp.iloc[rng.integers(0, len(exp), len(exp))]) for _ in range(n_boot)]
    return {"value": diff(exp), "ci": tuple(float(v) for v in np.quantile(boots, [0.025, 0.975])), "n": len(exp)}


def experiment_check(hcg: HCG, exp: pd.DataFrame, x: str, y: str, observed_ci: tuple, **kw) -> dict:
    """observed_ci: 95% confidence interval of the observational estimate. consistent if the two intervals overlap."""
    e = experiment_effect(exp, x, y, **kw)
    consistent = observed_ci[0] <= e["ci"][1] and e["ci"][0] <= observed_ci[1]
    edge = next((d for d in hcg.active_edges() if d.key == (x, y)), None)
    score = 1.0 if consistent and e["ci"][0] * e["ci"][1] > 0 else 0.0  # supports the edge only if the interval excludes 0 and agrees with the observational estimate
    if edge is not None and score:
        edge.evidence = [d for d in edge.evidence if d.method != "exp:rct"] + [Evidence(kind=EvidenceKind.EXPERIMENT, score=score, method="exp:rct", ref=f"n={e['n']}")]
        edge.update_confidence()
    return e | {"observed_ci": tuple(observed_ci), "consistent": consistent, "edge_evidence": bool(edge is not None and score)}
