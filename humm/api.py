"""경로 기반 Humm API. MCP 도구·노트북·스크립트가 같은 함수를 쓴다.

- 그래프는 data/graphs/<graph_id>.hcg.json 에 저장하고 graph_id로 주고받는다.
- 데이터는 CSV 경로, 문서는 JSONL({"id", "text"}) 경로나 .txt/.md 파일이 든 폴더로 받는다.
- 반환값은 모두 JSON으로 바꿀 수 있는 dict이다.
- llm: providers.from_spec 형식("hf:Qwen/Qwen3.5-9B@cuda:1", "anthropic:claude-sonnet-5-5", "replay:<경로>"). 기본값은 환경 변수 HUMM_LLM.
"""
import json
import os
import re
from pathlib import Path

import pandas as pd

from .engine import BinarySCM, estimate, identify, refute
from .graph.schema import HCG
from .graph.tabular import tabular_to_hcg
from .graph.text import link_nodes, merge_hcg, text_to_hcg
from .orchestrator import Orchestrator
from .orchestrator.analysis import fit_scm, latent_of
from .paths import GRAPHS, RESULTS, ROOT
from .providers.spec import from_spec
from .validate import validate_hcg

LLM_CACHE = RESULTS / "llm_cache" / "api.json"


def _path(graph_id: str) -> Path:
    return GRAPHS / f"{graph_id}.hcg.json"


def _load(graph_id: str) -> HCG:
    return HCG.load(_path(graph_id))


def _llm(spec: str | None):
    return from_spec(spec or os.environ.get("HUMM_LLM"), cache=LLM_CACHE)


def _done(llm) -> None:
    """기록용 공급자면 응답을 저장한다."""
    getattr(llm, "save", lambda: None)()


def read_docs(path: str | Path) -> list[dict]:
    path = Path(path)
    if path.is_dir():
        return [{"id": f.stem, "text": f.read_text(encoding="utf-8")} for f in sorted(path.iterdir()) if f.suffix in (".txt", ".md")]
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def graph_summary(hcg: HCG) -> dict:
    return {"graph_id": hcg.graph_id, "path": str(_path(hcg.graph_id)),
            "nodes": [{"id": n.id, "label": n.label, "aliases": n.aliases[:5]} for n in hcg.nodes],
            "edges": [{"src": e.src, "dst": e.dst, "sign": e.sign, "status": e.status.value, "confidence": round(e.confidence, 3),
                       "evidence": sorted({x.kind.value for x in e.evidence})} for e in hcg.edges]}


# ---------- 그래프 ----------
def graph_from_table(csv: str, graph_id: str, domain: str = "", alpha: float = 0.01, mode: str = "hybrid", llm: str | None = None) -> dict:
    """정형 데이터 → HCG (PC + 선택적 LLM 쌍별 사전지식 + L1·L2 검증)."""
    provider = _llm(llm)
    r = tabular_to_hcg(pd.read_csv(csv), graph_id, domain, provider=provider, alpha=alpha, mode=mode, n_boot=10, validate_kw={"n_perm": 50})
    _done(provider)
    r.hcg.save(_path(graph_id))
    return graph_summary(r.hcg) | {"verdict": r.report.verdict, "conflicts": r.conflicts}


def graph_from_text(docs: str, graph_id: str, domain: str = "", llm: str | None = None) -> dict:
    """문서 → HCG (인과 주장 추출 → 정규화 → 문서 간 집계 → 반사실 검증 증거). LLM이 필요하다."""
    provider = _llm(llm)
    r = text_to_hcg(read_docs(docs), graph_id, domain, provider)
    _done(provider)
    r.hcg.save(_path(graph_id))
    return graph_summary(r.hcg) | {"claims": len(r.claims)}


def graph_merge(text_graph: str, table_graph: str, graph_id: str, column_aliases: dict | None = None) -> dict:
    """텍스트 HCG를 정형 HCG에 병합한다. 노드는 이름·별칭 유사도로 잇고, 반대 방향 엣지는 방향 증거로 정리한다."""
    text, table = _load(text_graph), _load(table_graph)
    link = link_nodes(text, table, column_aliases)
    merged = merge_hcg(text, table, link, graph_id)
    merged.save(_path(graph_id))
    return graph_summary(merged) | {"link": link}


def graph_get(graph_id: str) -> dict:
    return graph_summary(_load(graph_id))


def graph_list() -> list[str]:
    return sorted(p.name.removesuffix(".hcg.json") for p in GRAPHS.glob("*.hcg.json"))


# ---------- 검증 ----------
def validate_run(graph_id: str, csv: str, n_perm: int = 100) -> dict:
    """L1 구조 + L2 데이터 반증. 엣지 상태(supported/verified/refuted)를 갱신해 저장한다."""
    hcg = _load(graph_id)
    report, changes = validate_hcg(hcg, pd.read_csv(csv), n_perm=n_perm)
    hcg.save(_path(graph_id))
    report.save(RESULTS / "validation" / f"{graph_id}.json")
    return {"summary": report.summary(), "notes": report.notes, "missing_edge_candidates": report.missing_edge_candidates(),
            "changes": changes.to_dict("records")}


# ---------- 인과 엔진 ----------
def engine_identify(graph_id: str, treatment: str, outcome: str, csv: str | None = None) -> dict:
    """식별(백도어 → 프론트도어 → IV). csv를 주면 데이터에 없는 노드를 관측되지 않은 변수로 본다."""
    g = _load(graph_id).to_networkx()
    lat = latent_of(g, pd.read_csv(csv, nrows=1)) if csv else []
    e = identify(g, treatment, outcome, lat)
    return {"strategy": e.strategy, "variables": sorted(e.variables), "expression": e.expression, "latent": lat}


def engine_estimate(graph_id: str, csv: str, treatment: str, outcome: str, n_boot: int = 200) -> dict:
    g, df = _load(graph_id).to_networkx(), pd.read_csv(csv)
    est = estimate(g, df, treatment, outcome, latent_of(g, df), n_boot=n_boot)
    return {"strategy": est.estimand.strategy, "variables": sorted(est.estimand.variables), "method": est.method,
            "value": est.value, "se": est.se, "ci": list(est.ci)}


def engine_refute(graph_id: str, csv: str, treatment: str, outcome: str) -> list[dict]:
    g, df = _load(graph_id).to_networkx(), pd.read_csv(csv)
    return refute(g, df, treatment, outcome, latent_of(g, df)).to_dict("records")


def engine_counterfactual(graph_id: str, csv: str, treatment: str, outcome: str, query: str = "ett", mediators: list | None = None) -> dict:
    """데이터로 적합한 이진 SCM에서 반사실 질의를 계산한다. query: ate, ett, nde, nie, pn, ps."""
    g, df = _load(graph_id).to_networkx(), pd.read_csv(csv)
    scm: BinarySCM = fit_scm(g, df, [treatment, outcome, *(mediators or [])])
    args = (treatment, outcome, mediators or []) if query in ("nde", "nie") else (treatment, outcome)
    return {"query": query, "value": getattr(scm, query)(*args), "exact": scm.exact, "nodes": sorted(scm.graph.nodes)}


# ---------- 오케스트레이터 ----------
def ask(question: str, graph_id: str, csv: str, experiment_csv: str | None = None, experiment_treatment: str | None = None,
        llm: str | None = None, n_perm: int = 100) -> dict:
    """질문 하나를 끝까지 처리한다: 질의 분석 → 검증 → 식별 → 추정 → 반박 → 민감도 → 근거·가정을 갖춘 답."""
    provider = _llm(llm)
    exps = {experiment_treatment: pd.read_csv(experiment_csv)} if experiment_csv else {}
    orch = Orchestrator(_load(graph_id), pd.read_csv(csv), provider, experiments=exps, n_perm=n_perm)
    out = orch.ask(question).to_dict()
    _done(provider)
    return out


# ---------- v1: Humm-Fast·Humm-LM (Step 11) ----------
# 모델 경로·장치는 환경 변수로 바꾼다. 처음 부를 때 한 번 올리고 프로세스 안에서 다시 쓴다.
FAST_PATH = os.environ.get("HUMM_FAST", str(ROOT / "models" / "humm-fast-4b-v2-story-distill"))
LM_PATH = os.environ.get("HUMM_LM", str(ROOT / "models" / "humm-lm-9b-sft-v217b"))
THRESHOLD = float(os.environ.get("HUMM_FAST_THRESHOLD", "0.8"))  # Fast 확신도(최대 보기 확률)가 이보다 낮으면 escalate
# Rung 분류는 Rung 문항으로 이어 학습한 라우터가 맡는다(CLadder Rung 0.515 → 0.967). 예/아니오 판정은 Corr2Cause가 떨어져 FAST_PATH로 둔다.
_ROUTER = ROOT / "models" / "humm-fast-4b-v2-router-ko"
ROUTER_PATH = os.environ.get("HUMM_ROUTER", str(_ROUTER) if _ROUTER.exists() else FAST_PATH)
ROUTER_THRESHOLD = float(os.environ.get("HUMM_ROUTER_THRESHOLD", "0.9"))  # 라우터 확신도가 이보다 낮으면 Humm-LM이 Rung을 고른다
_MODELS: dict = {}


def _fast():
    if "fast" not in _MODELS:
        from .fast.judge import FastJudge
        _MODELS["fast"] = FastJudge.load(FAST_PATH, device=os.environ.get("HUMM_FAST_DEVICE", "cuda:0"))
    return _MODELS["fast"]


def _router():
    if ROUTER_PATH == FAST_PATH:
        return _fast()
    if "router" not in _MODELS:
        from .fast.judge import FastJudge
        _MODELS["router"] = FastJudge.load(ROUTER_PATH, device=os.environ.get("HUMM_FAST_DEVICE", "cuda:0"))
    return _MODELS["router"]


def _lm():
    """HUMM_LM_URL(OpenAI 호환 서버, 예: vLLM http://127.0.0.1:8011/v1)이 있으면 원격으로, 없으면 HF로 올린다."""
    if "lm" not in _MODELS:
        if os.environ.get("HUMM_LM_URL"):
            from .providers.remote import RemoteProvider
            _MODELS["lm"] = RemoteProvider(os.environ["HUMM_LM_URL"], os.environ.get("HUMM_LM_NAME", "humm-lm"), tokenizer=LM_PATH)
        else:
            from .providers.hf import HFProvider
            _MODELS["lm"] = HFProvider(LM_PATH, device=os.environ.get("HUMM_LM_DEVICE", "cuda:0"), batch_size=4)
    return _MODELS["lm"]


def _fast_ask(context: str, question: str, kind: str, options: list[str]) -> list[float]:
    from .gym.formats import option_labels, render
    item = {"context": context, "question": question, "kind": kind, "options": options, "lang": "en"}
    return _fast().option_probs([render(item)], option_labels(item))[0]


def fast_noul(proposition: str, context: str = "") -> dict:
    """참/거짓 판단 한 번. p_yes가 threshold와 1 − threshold 사이면(확신이 낮으면) escalate."""
    p = _fast_ask(context, proposition, "noul", [])
    return {"p_yes": p[0], "confidence": max(p), "escalate": max(p) < THRESHOLD}


def fast_choice(question: str, options: list[str], context: str = "") -> dict:
    p = _fast_ask(context, question, "choice", options)
    k = max(range(len(p)), key=p.__getitem__)
    return {"choice": options[k], "probs": dict(zip(options, p)), "confidence": p[k], "escalate": p[k] < THRESHOLD}


def fast_score(question: str, levels: int = 5, context: str = "") -> dict:
    p = _fast_ask(context, question, "score", [f"level {k}" for k in range(1, levels + 1)])
    return {"score": sum((k + 1) * x for k, x in enumerate(p)), "dist": p, "confidence": max(p), "escalate": max(p) < THRESHOLD}


def classify_question(question: str, graph_id: str = "") -> dict:
    """Rung 분류(Fast, 저신뢰면 Humm-LM). graph_id를 주면 질문에 나온 그래프 변수로 처치·결과·매개도 정한다."""
    from .orchestrator.query import KEYWORDS, parse_rules
    from .orchestrator.route import QueryRouter
    r = QueryRouter(_router(), _lm(), ROUTER_THRESHOLD).classify([question])[0]
    out = {"rung": r["rung"], "kind": r["kind"], "confidence": r["confidence"], "router": r["router"], "treatment": None, "outcome": None, "mediators": []}
    if graph_id:
        q = parse_rules(question, _load(graph_id))
        kind = q.kind if any(re.search(p, question, re.I) for _, p in KEYWORDS) else r["kind"]
        out |= {"kind": kind, "treatment": q.treatment, "outcome": q.outcome, "mediators": q.mediators}
    return out


def lm_verify(graph_id: str, csv: str, treatment: str, outcome: str) -> dict:
    """Humm-LM(tool 모드)이 그래프 + 데이터로 적합한 확률표를 읽고 '처치가 결과를 올리는가'를 따로 풀어 엔진 ATE 부호와 비교한다."""
    from .orchestrator.route import lm_verify as verify
    hcg, df = _load(graph_id), pd.read_csv(csv)
    g = hcg.to_networkx()
    lat = latent_of(g, df)
    est = estimate(g, df, treatment, outcome, lat, n_boot=200, estimand=identify(g, treatment, outcome, lat))
    return verify(_lm(), hcg, df, treatment, outcome, est.value, tuple(map(float, est.ci)))


def ask_v1(question: str, graph_id: str, csv: str, experiment_csv: str | None = None, experiment_treatment: str | None = None,
           llm: str | None = None, n_perm: int = 100, verify: bool = True) -> dict:
    """ask와 같고, 질의 분석에 Fast Rung 분류(저신뢰면 Humm-LM)를, 검증에 Humm-LM 독립 풀이를 더한다."""
    from .orchestrator.route import QueryRouter
    provider = _llm(llm) if (llm or os.environ.get("HUMM_LLM")) else None
    exps = {experiment_treatment: pd.read_csv(experiment_csv)} if experiment_csv else {}
    orch = Orchestrator(_load(graph_id), pd.read_csv(csv), provider, experiments=exps, n_perm=n_perm,
                        router=QueryRouter(_router(), _lm(), ROUTER_THRESHOLD), verifier=_lm() if verify else None)
    out = orch.ask(question).to_dict()
    _done(provider)
    return out
