"""검증 실행과 결과 반영. preliminary/03_dag_validation.md 7장(엣지 상태 전이, validation.json)을 구현한다."""
import json
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import pandas as pd
from scipy import stats

from ..graph.schema import HCG, EdgeStatus, Evidence, EvidenceKind
from .ci import CITester
from .falsify import edge_tests, lmc_tests, mechanism_fit, permutation_test
from .structure import check_structure

DIRECTION_KINDS = {EvidenceKind.TEXT, EvidenceKind.LLM_PRIOR, EvidenceKind.EXPERT, EvidenceKind.EXPERIMENT}
DATA_DIRECTION_METHODS = {"pc:vstruct", "pc:meek"}  # 데이터가 방향까지 정한 엣지(v-구조와 그로부터 유도된 방향)


@dataclass
class ValidationReport:
    graph_id: str
    n_samples: int
    alpha: float
    ci_method: str
    structure: dict
    lmc: pd.DataFrame
    permutation: dict
    edges: pd.DataFrame
    mechanisms: pd.DataFrame
    notes: list = field(default_factory=list)

    MIN_PERM_TESTS = 5

    @property
    def excess_violation_p(self) -> float:
        """위반 개수가 우연(검정마다 확률 α로 거짓 위반)으로 기대되는 수보다 많은지에 대한 이항 검정 p값. 작으면 그래프와 데이터가 어긋난다."""
        k, n = int(self.lmc.violated.sum()), len(self.lmc)
        return float(stats.binom.sf(k - 1, n, self.alpha)) if k else 1.0

    @property
    def strong_violations(self) -> int:
        """다중검정 보정(Bonferroni, p < α / 검정 수)을 해도 남는 위반 수. 하나라도 있으면 그래프가 데이터와 분명히 어긋난다."""
        return int((self.lmc.p < self.alpha / max(len(self.lmc), 1)).sum())

    @property
    def permutation_informative(self) -> bool:
        """LMC 검정이 너무 적은 작은 그래프에서는 순열 검정에 검정력이 없다."""
        return len(self.lmc) >= self.MIN_PERM_TESTS

    @property
    def verdict(self) -> str:
        """fail: 사이클·시간 순서 위반.
        pass: 위반 개수가 우연 수준이고(이항 검정 p ≥ 0.05), Bonferroni 보정 후에도 남는 강한 위반이 없고,
              순열 검정이 유효하면 무작위 그래프보다 유의하게 잘 맞는다(p < 0.05).
        그 밖은 warn.
        """
        s = self.structure
        if not s["is_dag"] or s["temporal_violations"]:
            return "fail"
        fits = self.excess_violation_p >= 0.05 and self.strong_violations == 0
        beats_random = self.permutation["p_value"] < 0.05 or not self.permutation_informative
        return "pass" if fits and beats_random else "warn"

    def missing_edge_candidates(self, k: int = 5) -> list:
        """가장 강하게 위반된 LMC 쌍: 그래프에 빠진 엣지나 교란 변수가 있을 가능성이 높은 곳."""
        v = self.lmc[self.lmc.violated].nsmallest(k, "p")
        return [[a, b, g, float(p)] for a, b, g, p in v[["node", "other", "given", "p"]].itertuples(index=False)]

    def summary(self) -> dict:
        return {
            "graph_id": self.graph_id, "verdict": self.verdict, "n_samples": self.n_samples, "alpha": self.alpha, "ci_method": self.ci_method,
            "is_dag": self.structure["is_dag"], "temporal_violations": len(self.structure["temporal_violations"]),
            "lmc_tests": len(self.lmc), "lmc_violation_rate": self.permutation["violation_rate"], "excess_violation_p": round(self.excess_violation_p, 4), "strong_violations": self.strong_violations,
            "permutation_p": self.permutation["p_value"], "permutation_informative": self.permutation_informative,
            "edges_tested": len(self.edges), "edges_unsupported": int((~self.edges.supported).sum()),
        }

    def to_dict(self) -> dict:
        return {"summary": self.summary(), "structure": self.structure, "missing_edge_candidates": self.missing_edge_candidates(),
                "edges": self.edges.to_dict("records"), "lmc": self.lmc.to_dict("records"), "mechanisms": self.mechanisms.to_dict("records"),
                "permutation": self.permutation, "notes": self.notes}

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return path


def validate(graph: HCG | nx.DiGraph, df: pd.DataFrame, time: dict | None = None, alpha: float = 0.05, n_perm: int = 100,
             seed: int = 0, method: str = "auto") -> ValidationReport:
    """L1 구조 + L2 데이터 반증을 한 번에 실행한다. graph는 HCG(반증되지 않은 엣지만 사용) 또는 networkx 그래프이다."""
    g = graph.to_networkx() if isinstance(graph, HCG) else graph
    gid = graph.graph_id if isinstance(graph, HCG) else "graph"
    structure = check_structure(g, time)
    ci = CITester(df[[c for c in df.columns if c in g]], method)
    dag = g if structure["is_dag"] else nx.DiGraph()
    notes = [] if structure["is_dag"] else ["사이클이 있어 L2 검정을 건너뛰었다"]
    return ValidationReport(gid, len(df), alpha, ci.method, structure, lmc_tests(dag, ci, alpha),
                            permutation_test(dag, ci, n_perm, alpha, seed),
                            edge_tests(dag, ci, alpha), mechanism_fit(dag, ci.df, ci.method == "g_test"), notes)


def apply_to_hcg(hcg: HCG, report: ValidationReport, before: dict | None = None) -> pd.DataFrame:
    """L2 결과를 HCG 엣지에 반영하고, 상태 변화표를 돌려준다.

    - 엣지 의존성 검정 결과를 data 증거(score = 1 - p)로 넣는다. 데이터만으로는 엣지를 반증하지 않는다(검정력 부족일 수 있다).
    - supported 이면서 데이터가 엣지를 뒷받침하고, 방향 근거(텍스트·LLM·전문가·실험, 또는 PC의 v-구조·Meek 규칙)가 있고,
      그래프 판정이 pass이면 verified로 올린다.
    """
    before = before or {e.key: e.status for e in hcg.edges}
    pvals = {(r.src, r.dst): r.p for r in report.edges.itertuples()}
    for e in hcg.active_edges():
        if e.key in pvals:
            e.evidence = [x for x in e.evidence if not x.method.startswith("ci:")]
            e.evidence.append(Evidence(kind=EvidenceKind.DATA, score=round(1 - pvals[e.key], 4), method=f"ci:{report.ci_method}", ref=f"n={report.n_samples}"))
            e.update_confidence()
        e.promote()
        verified = (e.status == EdgeStatus.SUPPORTED and pvals.get(e.key, 1.0) < report.alpha
                    and ({x.kind for x in e.evidence} & DIRECTION_KINDS or {x.method for x in e.evidence} & DATA_DIRECTION_METHODS)
                    and report.verdict == "pass")
        e.status = EdgeStatus.VERIFIED if verified else e.status
    return pd.DataFrame([{"edge": f"{e.src}→{e.dst}", "before": before[e.key].value, "after": e.status.value, "confidence": round(e.confidence, 3)}
                         for e in hcg.edges])


def validate_hcg(hcg: HCG, df: pd.DataFrame, time: dict | None = None, **kw) -> tuple[ValidationReport, pd.DataFrame]:
    """HCG 검증 전체 흐름: L1 정리(시간 순서 위반 → refuted, 남은 사이클은 가장 약한 엣지 refuted) → L2 검정 → 증거·상태 반영.

    원래 그래프의 L1 문제는 report.notes 와 report.structure['before_fix']에 남긴다.
    """
    before = {e.key: e.status for e in hcg.edges}
    raw = check_structure(hcg.to_networkx(), time)
    late = {tuple(e) for e in raw["temporal_violations"]}
    for e in hcg.active_edges():
        e.status = EdgeStatus.REFUTED if e.key in late else e.status
    broken = hcg.break_cycles()
    report = validate(hcg, df, time, **kw)
    report.structure["before_fix"] = raw
    report.notes += [f"시간 순서 위반으로 refuted: {u}→{v}" for u, v in late] + [f"사이클 해소로 refuted: {u}→{v}" for u, v in broken]
    return report, apply_to_hcg(hcg, report, before)
