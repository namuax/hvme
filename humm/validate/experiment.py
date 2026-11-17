"""L4 실험 대조: 무작위 실험(A/B 테스트) 데이터의 효과와 관찰 데이터 추정치를 비교한다.

실험이 처치 x를 무작위로 배정했다면 E[y|x=1] - E[y|x=0]가 곧 개입 효과이다.
관찰 추정치가 실험 신뢰구간과 겹치면 그래프(조정 집합)가 이 효과에 대해서는 맞다는 증거이고,
x→y 직접 엣지가 있으면 실험 증거를 붙인다.
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
    """observed_ci: 관찰 데이터 추정치의 95% 신뢰구간. 두 구간이 겹치면 consistent."""
    e = experiment_effect(exp, x, y, **kw)
    consistent = observed_ci[0] <= e["ci"][1] and e["ci"][0] <= observed_ci[1]
    edge = next((d for d in hcg.active_edges() if d.key == (x, y)), None)
    score = 1.0 if consistent and e["ci"][0] * e["ci"][1] > 0 else 0.0  # 구간이 0을 포함하지 않고 관찰과 일치할 때만 엣지를 뒷받침한다
    if edge is not None and score:
        edge.evidence = [d for d in edge.evidence if d.method != "exp:rct"] + [Evidence(kind=EvidenceKind.EXPERIMENT, score=score, method="exp:rct", ref=f"n={e['n']}")]
        edge.update_confidence()
    return e | {"observed_ci": tuple(observed_ci), "consistent": consistent, "edge_evidence": bool(edge is not None and score)}
