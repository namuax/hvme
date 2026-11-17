"""Path-based Hvme API. MCP tools, notebooks and scripts use the same functions.

- Graphs are stored in data/graphs/<graph_id>.hcg.json and passed around by graph_id.
- Data is given as a CSV path; documents as a JSONL ({"id", "text"}) path or a folder of .txt/.md files.
- All return values are JSON-serializable dicts.
- llm: providers.from_spec format ("hf:Qwen/Qwen3.5-9B@cuda:1", "anthropic:claude-sonnet-5-5", "replay:<path>"). Defaults to the HVME_LLM environment variable.
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
    return from_spec(spec or os.environ.get("HVME_LLM"), cache=LLM_CACHE)


def _done(llm) -> None:
    """If the provider is a recording one, save its responses."""
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


# ---------- graphs ----------
def graph_from_table(csv: str, graph_id: str, domain: str = "", alpha: float = 0.01, mode: str = "hybrid", llm: str | None = None) -> dict:
    """Tabular data -> HCG (PC + optional LLM pairwise prior + L1/L2 validation)."""
    provider = _llm(llm)
    r = tabular_to_hcg(pd.read_csv(csv), graph_id, domain, provider=provider, alpha=alpha, mode=mode, n_boot=10, validate_kw={"n_perm": 50})
    _done(provider)
    r.hcg.save(_path(graph_id))
    return graph_summary(r.hcg) | {"verdict": r.report.verdict, "conflicts": r.conflicts}


def graph_from_text(docs: str, graph_id: str, domain: str = "", llm: str | None = None) -> dict:
    """Documents -> HCG (causal claim extraction -> canonicalization -> cross-document aggregation -> counterfactual verification evidence). Requires an LLM."""
    provider = _llm(llm)
    r = text_to_hcg(read_docs(docs), graph_id, domain, provider)
    _done(provider)
    r.hcg.save(_path(graph_id))
    return graph_summary(r.hcg) | {"claims": len(r.claims)}


def graph_merge(text_graph: str, table_graph: str, graph_id: str, column_aliases: dict | None = None) -> dict:
    """Merge a text HCG into a tabular HCG. Nodes are linked by name/alias similarity; opposing edges are reconciled using direction evidence."""
    text, table = _load(text_graph), _load(table_graph)
    link = link_nodes(text, table, column_aliases)
    merged = merge_hcg(text, table, link, graph_id)
    merged.save(_path(graph_id))
    return graph_summary(merged) | {"link": link}


def graph_get(graph_id: str) -> dict:
    return graph_summary(_load(graph_id))


def graph_list() -> list[str]:
    return sorted(p.name.removesuffix(".hcg.json") for p in GRAPHS.glob("*.hcg.json"))


# ---------- validation ----------
def validate_run(graph_id: str, csv: str, n_perm: int = 100) -> dict:
    """L1 structure + L2 data falsification. Updates edge status (supported/verified/refuted) and saves."""
    hcg = _load(graph_id)
    report, changes = validate_hcg(hcg, pd.read_csv(csv), n_perm=n_perm)
    hcg.save(_path(graph_id))
    report.save(RESULTS / "validation" / f"{graph_id}.json")
    return {"summary": report.summary(), "notes": report.notes, "missing_edge_candidates": report.missing_edge_candidates(),
            "changes": changes.to_dict("records")}


# ---------- causal engine ----------
def engine_identify(graph_id: str, treatment: str, outcome: str, csv: str | None = None) -> dict:
    """Identification (backdoor -> frontdoor -> IV). If csv is given, nodes missing from the data are treated as unobserved."""
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
    """Compute a counterfactual query on a binary SCM fitted from the data. query: ate, ett, nde, nie, pn, ps."""
    g, df = _load(graph_id).to_networkx(), pd.read_csv(csv)
    scm: BinarySCM = fit_scm(g, df, [treatment, outcome, *(mediators or [])])
    args = (treatment, outcome, mediators or []) if query in ("nde", "nie") else (treatment, outcome)
    return {"query": query, "value": getattr(scm, query)(*args), "exact": scm.exact, "nodes": sorted(scm.graph.nodes)}


# ---------- orchestrator ----------
def ask(question: str, graph_id: str, csv: str, experiment_csv: str | None = None, experiment_treatment: str | None = None,
        llm: str | None = None, n_perm: int = 100) -> dict:
    """Handle one question end to end: query analysis -> validation -> identification -> estimation -> refutation -> sensitivity -> an answer with evidence and assumptions."""
    provider = _llm(llm)
    exps = {experiment_treatment: pd.read_csv(experiment_csv)} if experiment_csv else {}
    orch = Orchestrator(_load(graph_id), pd.read_csv(csv), provider, experiments=exps, n_perm=n_perm)
    out = orch.ask(question).to_dict()
    _done(provider)
    return out


# ---------- v1: Hvme-Fast and Hvme-LM ----------
# Model paths and devices are set through environment variables. Loaded once on first use and reused within the process.
FAST_PATH = os.environ.get("HVME_FAST", str(ROOT / "models" / "hvme-fast-4b-v2-story-distill"))
LM_PATH = os.environ.get("HVME_LM", str(ROOT / "models" / "hvme-lm-9b-sft-v217b"))
THRESHOLD = float(os.environ.get("HVME_FAST_THRESHOLD", "0.8"))  # escalate if Fast confidence (max option probability) is below this
# Rung classification is handled by a router further trained on Rung items (CLadder Rung 0.515 -> 0.967). Yes/no judgment stays on FAST_PATH because Corr2Cause regresses with the router.
_ROUTER = ROOT / "models" / "hvme-fast-4b-v2-router-ko"
ROUTER_PATH = os.environ.get("HVME_ROUTER", str(_ROUTER) if _ROUTER.exists() else FAST_PATH)
ROUTER_THRESHOLD = float(os.environ.get("HVME_ROUTER_THRESHOLD", "0.9"))  # if router confidence is below this, Hvme-LM picks the Rung
_MODELS: dict = {}


def _fast():
    if "fast" not in _MODELS:
        from .fast.judge import FastJudge
        _MODELS["fast"] = FastJudge.load(FAST_PATH, device=os.environ.get("HVME_FAST_DEVICE", "cuda:0"))
    return _MODELS["fast"]


def _router():
    if ROUTER_PATH == FAST_PATH:
        return _fast()
    if "router" not in _MODELS:
        from .fast.judge import FastJudge
        _MODELS["router"] = FastJudge.load(ROUTER_PATH, device=os.environ.get("HVME_FAST_DEVICE", "cuda:0"))
    return _MODELS["router"]


def _lm():
    """Load remotely if HVME_LM_URL (an OpenAI-compatible server, e.g. vLLM at http://127.0.0.1:8011/v1) is set, otherwise load with HF."""
    if "lm" not in _MODELS:
        if os.environ.get("HVME_LM_URL"):
            from .providers.remote import RemoteProvider
            _MODELS["lm"] = RemoteProvider(os.environ["HVME_LM_URL"], os.environ.get("HVME_LM_NAME", "hvme-lm"), tokenizer=LM_PATH)
        else:
            from .providers.hf import HFProvider
            _MODELS["lm"] = HFProvider(LM_PATH, device=os.environ.get("HVME_LM_DEVICE", "cuda:0"), batch_size=4)
    return _MODELS["lm"]


def _fast_ask(context: str, question: str, kind: str, options: list[str]) -> list[float]:
    from .gym.formats import option_labels, render
    item = {"context": context, "question": question, "kind": kind, "options": options, "lang": "en"}
    return _fast().option_probs([render(item)], option_labels(item))[0]


def fast_noul(proposition: str, context: str = "") -> dict:
    """A single true/false judgment. Escalate if p_yes is between 1 - threshold and threshold (low confidence)."""
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
    """Rung classification (Fast; Hvme-LM if low confidence). If graph_id is given, treatment, outcome and mediators are also decided from graph variables in the question."""
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
    """Hvme-LM (tool mode) reads the probability table fitted from the graph + data, solves 'does the treatment raise the outcome?' independently, and compares with the engine ATE sign."""
    from .orchestrator.route import lm_verify as verify
    hcg, df = _load(graph_id), pd.read_csv(csv)
    g = hcg.to_networkx()
    lat = latent_of(g, df)
    est = estimate(g, df, treatment, outcome, lat, n_boot=200, estimand=identify(g, treatment, outcome, lat))
    return verify(_lm(), hcg, df, treatment, outcome, est.value, tuple(map(float, est.ci)))


def ask_v1(question: str, graph_id: str, csv: str, experiment_csv: str | None = None, experiment_treatment: str | None = None,
           llm: str | None = None, n_perm: int = 100, verify: bool = True) -> dict:
    """Same as ask, plus Fast Rung classification (Hvme-LM if low confidence) for query analysis and an independent Hvme-LM solution for verification."""
    from .orchestrator.route import QueryRouter
    provider = _llm(llm) if (llm or os.environ.get("HVME_LLM")) else None
    exps = {experiment_treatment: pd.read_csv(experiment_csv)} if experiment_csv else {}
    orch = Orchestrator(_load(graph_id), pd.read_csv(csv), provider, experiments=exps, n_perm=n_perm,
                        router=QueryRouter(_router(), _lm(), ROUTER_THRESHOLD), verifier=_lm() if verify else None)
    out = orch.ask(question).to_dict()
    _done(provider)
    return out
