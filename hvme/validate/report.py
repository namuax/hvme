"""Run validation and apply the results: implements edge status transitions and validation.json."""
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
DATA_DIRECTION_METHODS = {"pc:vstruct", "pc:meek"}  # edges whose direction the data determined (v-structures and directions derived from them)


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
        """Binomial-test p-value for whether the number of violations exceeds what chance (a false violation with probability α per test) would give. If small, the graph and data disagree."""
        k, n = int(self.lmc.violated.sum()), len(self.lmc)
        return float(stats.binom.sf(k - 1, n, self.alpha)) if k else 1.0

    @property
    def strong_violations(self) -> int:
        """Number of violations remaining after multiple-testing correction (Bonferroni, p < α / number of tests). If there is any, the graph clearly disagrees with the data."""
        return int((self.lmc.p < self.alpha / max(len(self.lmc), 1)).sum())

    @property
    def permutation_informative(self) -> bool:
        """In small graphs with too few LMC tests, the permutation test has no power."""
        return len(self.lmc) >= self.MIN_PERM_TESTS

    @property
    def verdict(self) -> str:
        """fail: cycle or time-order violation.
        pass: the violation count is at chance level (binomial test p ≥ 0.05), no strong violation survives Bonferroni correction,
              and, if the permutation test is valid, the graph fits significantly better than random graphs (p < 0.05).
        Otherwise warn.
        """
        s = self.structure
        if not s["is_dag"] or s["temporal_violations"]:
            return "fail"
        fits = self.excess_violation_p >= 0.05 and self.strong_violations == 0
        beats_random = self.permutation["p_value"] < 0.05 or not self.permutation_informative
        return "pass" if fits and beats_random else "warn"

    def missing_edge_candidates(self, k: int = 5) -> list:
        """The most strongly violated LMC pairs: where the graph most likely misses an edge or a confounder."""
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
    """Run L1 structure checks and L2 data falsification together. graph is an HCG (only non-refuted edges are used) or a networkx graph."""
    g = graph.to_networkx() if isinstance(graph, HCG) else graph
    gid = graph.graph_id if isinstance(graph, HCG) else "graph"
    structure = check_structure(g, time)
    ci = CITester(df[[c for c in df.columns if c in g]], method)
    dag = g if structure["is_dag"] else nx.DiGraph()
    notes = [] if structure["is_dag"] else ["graph has a cycle; L2 tests skipped"]
    return ValidationReport(gid, len(df), alpha, ci.method, structure, lmc_tests(dag, ci, alpha),
                            permutation_test(dag, ci, n_perm, alpha, seed),
                            edge_tests(dag, ci, alpha), mechanism_fit(dag, ci.df, ci.method == "g_test"), notes)


def apply_to_hcg(hcg: HCG, report: ValidationReport, before: dict | None = None) -> pd.DataFrame:
    """Apply L2 results to HCG edges and return a table of status changes.

    - Edge-dependence test results are added as data evidence (score = 1 - p). Data alone never refutes an edge (it may just lack power).
    - An edge that is supported, has data backing, and has directional evidence (text, LLM, expert, experiment, or PC v-structures and Meek rules)
      is promoted to verified if the graph verdict is pass.
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
    """Full HCG validation flow: L1 cleanup (time-order violations → refuted; remaining cycles refute the weakest edge) → L2 tests → apply evidence and status.

    L1 problems of the original graph are kept in report.notes and report.structure['before_fix'].
    """
    before = {e.key: e.status for e in hcg.edges}
    raw = check_structure(hcg.to_networkx(), time)
    late = {tuple(e) for e in raw["temporal_violations"]}
    for e in hcg.active_edges():
        e.status = EdgeStatus.REFUTED if e.key in late else e.status
    broken = hcg.break_cycles()
    report = validate(hcg, df, time, **kw)
    report.structure["before_fix"] = raw
    report.notes += [f"refuted for violating time order: {u}→{v}" for u, v in late] + [f"refuted to break a cycle: {u}→{v}" for u, v in broken]
    return report, apply_to_hcg(hcg, report, before)
