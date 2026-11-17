"""오케스트레이터 v0: 질의 분석 → 그래프 확보 → 검증 → 식별 → 추정 → 반박 → 민감도 → 설명.

graph_llm: 그래프 구축(텍스트 추출·정규화·검증, 정형 사전지식)에 쓰는 공급자
llm:       질의 분석과 설명 요약에 쓰는 범용 LLM. None이면 규칙 기반 분석과 템플릿 설명만 쓴다.
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
    narrative: str = ""  # 검사를 통과한 LLM 요약(없으면 빈 문자열)
    draft: str = ""  # LLM이 쓴 요약 원문
    issues: list | None = None  # 요약 검사 결과. None이면 LLM 요약을 쓰지 않았다
    route: dict | None = None  # v1 질의 분석(Rung, 확신도, fast/lm)
    check: dict | None = None  # v1 Humm-LM 검증

    def to_dict(self) -> dict:
        a = self.analysis
        return facts(a) | {"route": self.route, "lm_check": None if self.check is None else {k: v for k, v in self.check.items() if k != "trace"}} | {"question": a.query.text, "kind": a.query.kind, "treatment": a.query.treatment, "outcome": a.query.outcome,
                           "strategy": a.estimand.strategy, "estimates": a.estimate, "parser": a.query.parser, "mediators": a.query.mediators, "expression": a.estimand.expression,
                           "adjustment": sorted(a.estimand.variables), "paths": a.paths, "backdoor_paths": a.backdoor,
                           "refutation": a.refutation.to_dict("records"), "sensitivity_table": a.sensitivity.to_dict("records"),
                           "assumptions": a.assumptions, "experiment": a.experiment, "narrative": self.narrative, "draft": self.draft, "issues": self.issues, "markdown": self.markdown}


def acquire(df: pd.DataFrame, docs: list[dict] | None = None, domain: str = "", graph_llm=None, column_aliases: dict | None = None,
            graph_id: str = "graph", table_kw: dict | None = None, text_kw: dict | None = None) -> tuple[HCG, dict]:
    """그래프 확보: 정형 데이터 → HCG, (문서가 있으면) 텍스트 → HCG, 두 그래프 병합. 검증 전 그래프와 중간 결과를 돌려준다."""
    tab = tabular_to_hcg(df, f"{graph_id}-data", domain, provider=None, **{"n_boot": 10, "validate_kw": {"n_perm": 0}} | (table_kw or {}))
    if not docs:
        return tab.hcg, {"table": tab}
    txt = text_to_hcg(docs, f"{graph_id}-text", domain, graph_llm, **(text_kw or {}))
    link = link_nodes(txt.hcg, tab.hcg, column_aliases)
    return merge_hcg(txt.hcg, tab.hcg, link, graph_id), {"table": tab, "text": txt, "link": link}


class Orchestrator:
    def __init__(self, hcg: HCG, df: pd.DataFrame, llm=None, time: dict | None = None, experiments: dict | None = None,
                 n_perm: int = 100, n_boot: int = 200, n_boot_scm: int = 30, narrate: bool = True, router=None, verifier=None):
        """experiments: {무작위 배정한 처치 변수: 실험 데이터}. 그 처치에 대한 질문에는 L4 실험 대조를 붙인다.
        router(v1):   route.QueryRouter. 유형 키워드가 없는 질문의 유형을 Humm-Fast Rung 분류(저신뢰면 Humm-LM)로 정한다.
        verifier(v1): Humm-LM 공급자(HFProvider). ate·ett·mediation 질문에 LM 독립 풀이와 엔진 부호 대조를 붙인다."""
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
