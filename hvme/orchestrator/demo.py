"""E2E demo scenarios. Generates observational data, experiment data and a document corpus from a ground-truth SCM.

- marketing: peak season (season) is a confounder that moves ad spend, discounts and sales together; part of the ad-spend effect goes through web visits.
- manufacturing (manufacturing quality): equipment age is a confounder that raises both preventive maintenance and defect rate. The effect of maintenance lowers the defect rate entirely through machine vibration, so
  the observed correlation can look weak or even reversed.
- it_ops (IT operations incident root-cause analysis): request traffic is a confounder that moves config changes, latency and outages together. Most of the config-change effect goes through the error rate.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from ..engine import BinarySCM
from ..graph.text import TEXT_VOCAB, Corpus, make_corpus

# Per domain: (cause, effect, sign, logit weight), node intercepts, questions, expected interpretation per question (kind, treatment, outcome, truth key), truth definitions, experiment treatment.
# Truth definition: ("ate", x, y) | ("nie"|"nde", x, y, mediator) | ("ett", x, y) | ("zero",)
SPECS = {
    "marketing": dict(
        edges=[("season", "ad_spend", "+", 1.6), ("season", "discount", "+", 1.2), ("season", "sales", "+", 1.4),
               ("ad_spend", "web_traffic", "+", 1.8), ("ad_spend", "sales", "+", 0.6), ("web_traffic", "sales", "+", 1.0),
               ("discount", "conversion", "+", 1.5), ("conversion", "sales", "+", 1.0), ("discount", "satisfaction", "+", 0.9),
               ("satisfaction", "churn", "-", -1.6)],
        bias={"season": -0.4, "ad_spend": -1.0, "discount": -0.8, "web_traffic": -1.0, "conversion": -0.9, "sales": -1.9, "satisfaction": -0.3, "churn": 0.2},
        questions=["광고비를 늘리면 매출이 오르는가?",
                   "광고비와 매출 사이의 상관관계는 인과 효과와 같은가?",
                   "광고비가 매출에 주는 효과 중 웹 방문을 거치는 간접 효과는 얼마인가?",
                   "광고비를 집행한 기간에 광고비를 집행하지 않았다면 매출은 얼마나 달라졌을까?",
                   "고객 이탈률이 광고비에 영향을 주는가?",
                   "할인 행사가 고객 이탈률을 낮추는가?"],
        expected=[("ate", "ad_spend", "sales", "ate"), ("association", "ad_spend", "sales", "ate"), ("mediation", "ad_spend", "sales", "nie"),
                  ("ett", "ad_spend", "sales", "ett"), ("ate", "churn", "ad_spend", "churn_on_ad"), ("ate", "discount", "churn", "discount_on_churn")],
        truth={"ate": ("ate", "ad_spend", "sales"), "nie": ("nie", "ad_spend", "sales", ["web_traffic"]), "nde": ("nde", "ad_spend", "sales", ["web_traffic"]),
               "ett": ("ett", "ad_spend", "sales"), "churn_on_ad": ("zero",), "discount_on_churn": ("ate", "discount", "churn")},
        treatment="ad_spend"),
    "manufacturing": dict(
        edges=[("equip_age", "maintenance", "+", 1.4), ("equip_age", "vibration", "+", 1.6), ("equip_age", "defect_rate", "+", 1.0),
               ("maintenance", "vibration", "-", -1.8), ("maintenance", "downtime", "+", 0.6), ("temperature", "defect_rate", "+", 1.0),
               ("vibration", "defect_rate", "+", 1.5), ("vibration", "downtime", "+", 1.2), ("defect_rate", "delivery_delay", "+", 1.0),
               ("downtime", "output", "-", -1.6), ("output", "delivery_delay", "-", -1.2)],
        bias={"equip_age": -0.2, "maintenance": -0.9, "temperature": -0.5, "vibration": -0.6, "defect_rate": -1.8, "downtime": -1.2, "output": 0.9,
              "delivery_delay": -0.6},
        questions=["예방 정비를 늘리면 불량률이 낮아지는가?",
                   "예방 정비와 불량률 사이의 상관관계는 인과 효과와 같은가?",
                   "예방 정비가 불량률에 주는 효과 중 설비 진동을 거치는 간접 효과는 얼마인가?",
                   "예방 정비를 한 설비에서 예방 정비를 하지 않았다면 불량률은 얼마나 달라졌을까?",
                   "납기 지연이 공정 온도에 영향을 주는가?",
                   "설비 진동이 생산량을 줄이는가?"],
        expected=[("ate", "maintenance", "defect_rate", "ate"), ("association", "maintenance", "defect_rate", "ate"),
                  ("mediation", "maintenance", "defect_rate", "nie"), ("ett", "maintenance", "defect_rate", "ett"),
                  ("ate", "delivery_delay", "temperature", "delay_on_temp"), ("ate", "vibration", "output", "vibration_on_output")],
        truth={"ate": ("ate", "maintenance", "defect_rate"), "nie": ("nie", "maintenance", "defect_rate", ["vibration"]),
               "nde": ("nde", "maintenance", "defect_rate", ["vibration"]), "ett": ("ett", "maintenance", "defect_rate"), "delay_on_temp": ("zero",),
               "vibration_on_output": ("ate", "vibration", "output")},
        treatment="maintenance"),
    "it_ops": dict(
        edges=[("traffic", "config_change", "+", 1.0), ("traffic", "cpu", "+", 1.6), ("traffic", "outage", "+", 0.6),
               ("deploy", "config_change", "+", 1.2), ("deploy", "memory_leak", "+", 1.0), ("config_change", "error_rate", "+", 1.4),
               ("config_change", "outage", "+", 0.4), ("memory_leak", "latency", "+", 2.0), ("cpu", "latency", "+", 1.2),
               ("latency", "error_rate", "+", 1.6), ("error_rate", "outage", "+", 1.6)],
        bias={"deploy": -0.3, "config_change": -1.0, "traffic": -0.2, "cpu": -0.9, "memory_leak": -0.8, "latency": -1.5, "error_rate": -1.8, "outage": -2.4},
        questions=["설정 변경을 늘리면 서비스 장애가 늘어나는가?",
                   "설정 변경과 서비스 장애 사이의 상관관계는 인과 효과와 같은가?",
                   "설정 변경이 서비스 장애에 주는 효과 중 오류율을 거치는 간접 효과는 얼마인가?",
                   "설정 변경을 한 기간에 설정 변경을 하지 않았다면 서비스 장애는 얼마나 달라졌을까?",
                   "서비스 장애가 배포 빈도에 영향을 주는가?",
                   "메모리 누수가 오류율을 높이는가?"],
        expected=[("ate", "config_change", "outage", "ate"), ("association", "config_change", "outage", "ate"), ("mediation", "config_change", "outage", "nie"),
                  ("ett", "config_change", "outage", "ett"), ("ate", "outage", "deploy", "outage_on_deploy"), ("ate", "memory_leak", "error_rate", "leak_on_error")],
        truth={"ate": ("ate", "config_change", "outage"), "nie": ("nie", "config_change", "outage", ["error_rate"]),
               "nde": ("nde", "config_change", "outage", ["error_rate"]), "ett": ("ett", "config_change", "outage"), "outage_on_deploy": ("zero",),
               "leak_on_error": ("ate", "memory_leak", "error_rate")},
        treatment="config_change"),
}
EDGES, BIAS, QUESTIONS, EXPECTED = (SPECS["marketing"][k] for k in ("edges", "bias", "questions", "expected"))  # v0 names (notebook 08)


@dataclass
class Scenario:
    graph: nx.DiGraph
    scm: BinarySCM
    data: pd.DataFrame
    experiment: pd.DataFrame  # A/B test with the experiment treatment (spec["treatment"]) randomly assigned
    corpus: Corpus
    column_aliases: dict
    domain: str = "marketing"
    spec: dict = field(default_factory=lambda: SPECS["marketing"])

    @property
    def questions(self) -> list[str]:
        return self.spec["questions"]

    @property
    def expected(self) -> list[tuple]:
        return self.spec["expected"]

    def save(self, folder: str | Path) -> dict:
        """Save as files for MCP/API use: data.csv, experiment.csv, docs.jsonl, aliases.json."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        files = {k: folder / f for k, f in [("csv", "data.csv"), ("experiment_csv", "experiment.csv"), ("docs", "docs.jsonl"), ("aliases", "aliases.json")]}
        self.data.to_csv(files["csv"], index=False)
        self.experiment.to_csv(files["experiment_csv"], index=False)
        files["docs"].write_text("\n".join(json.dumps({"id": d["id"], "text": d["text"]}, ensure_ascii=False) for d in self.corpus.docs), encoding="utf-8")
        files["aliases"].write_text(json.dumps(self.column_aliases, ensure_ascii=False, indent=1), encoding="utf-8")
        return {k: str(v) for k, v in files.items()}

    def truth(self) -> dict:
        """Value per truth key (computed exactly from the ground-truth SCM)."""
        s = self.scm
        calc = {"ate": lambda x, y: s.ate(x, y), "ett": lambda x, y: s.ett(x, y), "nie": lambda x, y, m: s.nie(x, y, m), "nde": lambda x, y, m: s.nde(x, y, m)}
        return {k: 0.0 if d[0] == "zero" else calc[d[0]](*d[1:]) for k, d in self.spec["truth"].items()}


def logistic_cpt(g: nx.DiGraph, weights: dict, bias: dict) -> dict:
    """P(v=1 | parents) = sigmoid(bias + sum of w * parent). Parents are in name order; the i-th parent is bit i (BinarySCM convention)."""
    cpt = {}
    for v in g.nodes:
        pa = sorted(g.predecessors(v))
        bits = np.array([[(k >> i) & 1 for i in range(len(pa))] for k in range(2 ** len(pa))]).reshape(2 ** len(pa), len(pa))
        z = bias[v] + bits @ np.array([weights[(p, v)] for p in pa]).reshape(len(pa))
        cpt[v] = (1 / (1 + np.exp(-z))).round(3)
    return cpt


def scenario(domain: str, n: int = 8000, n_exp: int = 3000, lang: str = "ko", n_docs: int = 12, seed: int = 0) -> Scenario:
    spec = SPECS[domain]
    g = nx.DiGraph()
    g.add_nodes_from(v[0] for v in TEXT_VOCAB[domain])
    g.add_edges_from((a, b, {"sign": s}) for a, b, s, _ in spec["edges"])
    scm = BinarySCM(g, logistic_cpt(g, {(a, b): w for a, b, _, w in spec["edges"]}, spec["bias"]))
    data = scm.sample(n, seed=seed + 1)[sorted(g.nodes)]
    rng = np.random.default_rng(seed + 2)
    t = spec["treatment"]
    exp = pd.concat([scm.sample(n_exp // 2, {t: v}, seed=seed + 3 + v) for v in (0, 1)]).sample(frac=1, random_state=int(rng.integers(1 << 31)))
    corpus = make_corpus(domain, lang, n_docs=n_docs, seed=seed, graph=g)
    aliases = {v[0]: v[1] + v[2] for v in TEXT_VOCAB[domain]}
    return Scenario(g, scm, data, exp[sorted(g.nodes)].reset_index(drop=True), corpus, aliases, domain, spec)


def marketing(n: int = 8000, n_exp: int = 3000, lang: str = "ko", n_docs: int = 12, seed: int = 0) -> Scenario:
    return scenario("marketing", n, n_exp, lang, n_docs, seed)


def scorecard(answers, scenario: Scenario) -> pd.DataFrame:
    """Per-question interpretation accuracy, difference between the headline number and the truth, whether the interval covers it, and a summary of sensitivity and experiment comparison."""
    truth = scenario.truth()
    rows = []
    for ans, (kind, x, y, key) in zip(answers, scenario.expected):
        a = ans.analysis
        m = a.estimate[a.main]
        s = a.sensitivity_summary
        rows.append({"question": a.query.text, "parser": a.query.parser, "parsed_ok": (a.query.kind, a.query.treatment, a.query.outcome) == (kind, x, y),
                     "kind": a.query.kind, "strategy": a.estimand.strategy, "value": round(m["value"], 4), "ci_low": round(m["ci"][0], 4),
                     "ci_high": round(m["ci"][1], 4), "truth": round(truth[key], 4), "abs_error": round(abs(m["value"] - truth[key]), 4),
                     "covered": m["ci"][0] - 1e-9 <= truth[key] <= m["ci"][1] + 1e-9, "alternatives": s["scenarios"], "flipped": s["flipped"],
                     "vanished": s["vanished"], "experiment_ok": a.experiment.get("consistent"),
                     "narrative_ok": None if ans.issues is None else not ans.issues})
    return pd.DataFrame(rows)
