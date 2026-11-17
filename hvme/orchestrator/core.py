"""Orchestrator v0: query analysis -> graph acquisition -> validation -> identification -> estimation -> refutation -> sensitivity -> explanation.

graph_llm: provider used for graph construction (text extraction, canonicalization, verification, tabular prior)
llm:       general-purpose LLM used for query analysis and explanation summaries. If None, only rule-based analysis and template explanations are used.
"""
import re
from dataclasses import dataclass, replace

import pandas as pd

from ..graph.schema import HCG
from ..graph.tabular import tabular_to_hcg
from ..graph.text import link_nodes, merge_hcg, text_to_hcg
from ..validate import experiment_check, validate_hcg
from .analysis import Analysis, analyze
from .explain import check_narrative, facts, narrate, polish, to_markdown
from .query import KEYWORDS, parse
from .route import lm_verify, verify_markdown


@dataclass
class Answer:
    analysis: Analysis
    markdown: str
    narrative: str = ""  # LLM summary that passed the check (empty string if none)
    draft: str = ""  # raw summary written by the LLM
    issues: list | None = None  # summary check result. None means no LLM summary was used
    route: dict | None = None  # v1 query analysis (Rung, confidence, fast/lm)
    check: dict | None = None  # v1 Hvme-LM verification

    def to_dict(self) -> dict:
        a = self.analysis
        return facts(a) | {"route": self.route, "lm_check": None if self.check is None else {k: v for k, v in self.check.items() if k != "trace"}} | {"question": a.query.text, "kind": a.query.kind, "treatment": a.query.treatment, "outcome": a.query.outcome,
                           "strategy": a.estimand.strategy, "estimates": a.estimate, "parser": a.query.parser, "mediators": a.query.mediators, "expression": a.estimand.expression,
                           "adjustment": sorted(a.estimand.variables), "paths": a.paths, "backdoor_paths": a.backdoor,
                           "refutation": a.refutation.to_dict("records"), "sensitivity_table": a.sensitivity.to_dict("records"),
                           "assumptions": a.assumptions, "experiment": a.experiment, "narrative": self.narrative, "draft": self.draft, "issues": self.issues, "markdown": self.markdown}


def acquire(df: pd.DataFrame, docs: list[dict] | None = None, domain: str = "", graph_llm=None, column_aliases: dict | None = None,
            graph_id: str = "graph", table_kw: dict | None = None, text_kw: dict | None = None) -> tuple[HCG, dict]:
    """Graph acquisition: tabular data -> HCG, (if documents are given) text -> HCG, then merge the two graphs. Returns the pre-validation graph and intermediate results."""
    tab = tabular_to_hcg(df, f"{graph_id}-data", domain, provider=None, **{"n_boot": 10, "validate_kw": {"n_perm": 0}} | (table_kw or {}))
    if not docs:
        return tab.hcg, {"table": tab}
    txt = text_to_hcg(docs, f"{graph_id}-text", domain, graph_llm, **(text_kw or {}))
    link = link_nodes(txt.hcg, tab.hcg, column_aliases)
    return merge_hcg(txt.hcg, tab.hcg, link, graph_id), {"table": tab, "text": txt, "link": link}


class Orchestrator:
    def __init__(self, hcg: HCG, df: pd.DataFrame, llm=None, time: dict | None = None, experiments: dict | None = None,
                 n_perm: int = 100, n_boot: int = 200, n_boot_scm: int = 30, narrate: bool = True, router=None, verifier=None):
        """experiments: {randomized treatment variable: experiment data}. Questions about that treatment get an L4 experiment comparison.
        router(v1):   route.QueryRouter. For questions without a kind keyword, the kind is decided by Hvme-Fast Rung classification (Hvme-LM if low confidence).
        verifier(v1): Hvme-LM provider (HFProvider). For ate/ett/mediation questions, adds an independent LM solution and a comparison with the engine sign."""
        self.hcg, self.df, self.llm, self.experiments = hcg, df, llm, experiments or {}
        self.n_boot, self.n_boot_scm, self.use_narrate = n_boot, n_boot_scm, narrate
        self.router, self.verifier = router, verifier
        self.report, self.changes = validate_hcg(hcg, df, time, n_perm=n_perm)

    @classmethod
    def from_sources(cls, df: pd.DataFrame, docs=None, domain: str = "", graph_llm=None, column_aliases=None, graph_id: str = "graph",
                     llm=None, **kw) -> "Orchestrator":
        hcg, parts = acquire(df, docs, domain, graph_llm, column_aliases, graph_id)
        orch = cls(hcg, df, llm, **kw)
        orch.parts = parts
        return orch

    def ask(self, question: str) -> Answer:
        q = parse(question, self.hcg, self.llm)
        route = self.router.classify([question])[0] if self.router is not None else None
        keyword = any(re.search(pat, question, re.I) for _, pat in KEYWORDS)
        q = replace(q, kind=route["kind"], parser=f"{q.parser}+rung:{route['router']}") if route and not keyword and q.parser == "rules" else q
        a = analyze(q, self.hcg, self.df, self.n_boot, self.n_boot_scm)
        exp = self.experiments.get(q.treatment)
        a.experiment = experiment_check(self.hcg, exp, q.treatment, q.outcome, a.estimate["ate"]["ci"]) if exp is not None and q.outcome in exp else {}
        check = (lm_verify(self.verifier, self.hcg, self.df, q.treatment, q.outcome, a.estimate["ate"]["value"], a.estimate["ate"]["ci"])
                 if self.verifier is not None and q.kind in ("ate", "ett", "mediation") and a.estimand.strategy != "no_effect" else None)
        draft = narrate(a, self.llm) if self.llm is not None and self.use_narrate else ""
        text = polish(draft)
        issues = check_narrative(text, a) if draft else None
        text = "" if issues else text
        md = to_markdown(a, self.report.summary(), text) + ("\n" + verify_markdown(check) if check else "")
        return Answer(a, md, text, draft, issues, route, check)
