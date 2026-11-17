import sys
import nbformat as nbf

cells = [
"""md:# 06. 정형 데이터 → 인과 그래프 — PC + LLM 사전지식

표 데이터에서 인과 그래프(HCG)를 만드는 파이프라인이다(`02_proposal.md` 4.3).

| 단계 | 내용 | 모듈 |
|---|---|---|
| S1 | 변수 프로파일(유형·범주·결측) | `profile` |
| S2 | LLM 사전지식: **order** 전략은 변수 전체의 인과 순서 계층을 한 번에 묻고, **pairwise** 전략(기본값)은 PC 골격의 인접 쌍마다 방향을 묻는다 | `llm_order`, `llm_pairwise` |
| S3 | PC 알고리즘(조건부 독립 → 골격 → v-구조 → Meek 규칙) + 배경지식 | `pc` |
| S4 | order 전략에서 그래도 남는 방향 미정 엣지만 쌍별로 묻는다 | `llm_pairwise` |
| S5 | 부트스트랩 엣지 빈도 → HCG 증거 → Step 4 검증 | `tabular_to_hcg` |

LLM 사전지식을 넣는 방식은 두 가지이다.
- **soft**: 데이터가 정한 방향(v-구조, Meek 규칙)을 먼저 확정하고, 남은 방향 미정 엣지에만 사전지식을 쓴다.
- **hard**: 사전지식을 데이터보다 먼저 적용한다.

이 노트북의 질문은 "LLM 사전지식이 얼마나 정확해야 도움이 되고, 어느 방식이 안전한가"이다. 1~3장은 CPU, 4~5장은 GPU에서 실행한다.""",
"""import json

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from humm.bench.metrics import graph_report
from humm.graph.tabular import NETWORKS, NoisyOracle, compare, describe_variables, pc, profile, tabular_to_hcg, to_dag
from humm.paths import RESULTS
from humm.validate import sample_bnlearn

NAMES = ["asia", "sachs", "insurance", "alarm"]
data = {n: sample_bnlearn(n, 5000, seed=0) for n in NAMES}""",
"""md:## 1. PC 알고리즘 한 단계씩 (Asia)

PC는 조건부 독립 검정으로 엣지를 지워 골격을 만들고, v-구조(`a → c ← b`)로 일부 방향을 정한 뒤 Meek 규칙으로 방향을 전파한다. 끝까지 방향이 안 정해진 엣지는 데이터만으로는 구별할 수 없는(마르코프 동치) 부분이다.""",
"""true, asia = data["asia"]
print(describe_variables(profile(asia), NETWORKS["asia"]["descriptions"]))
g = pc(asia, alpha=0.01)
print("\\n방향 엣지:", sorted(g.directed))
print("방향 미정:", sorted(tuple(sorted(e)) for e in g.undirected))
print("방향을 정한 근거:", g.oriented_by)
print("지표:", graph_report(true, to_dag(g)))""",
"""md:## 2. 사전지식 품질 실험 (가상 LLM)

`NoisyOracle`은 참 그래프를 알지만 일부러 틀리는 가상 LLM이다. 오류율 e는 인과 순서에서 자리를 바꾸는 변수의 비율이고, 쌍별 질의에서 틀린 방향을 고를 확률이다. e = 0이면 완벽한 사전지식이다.""",
"""rows = []
for n in NAMES:
    true, df = data[n]
    provs = {"none": None} | {f"e={e}": NoisyOracle(true, e, seed=1) for e in (0.0, 0.1, 0.2, 0.4)}
    rows.append(compare(df, true, provs, NETWORKS[n]["domain"], NETWORKS[n]["descriptions"], modes=("soft", "hard"), priors=("order", "pairwise")).assign(network=n))
oracle_res = pd.concat(rows, ignore_index=True)
oracle_res.pivot_table(index=["prior", "strategy", "mode"], columns="network", values="shd", sort=False)""",
"""oracle_res.pivot_table(index=["prior", "strategy", "mode"], columns="network", values="sid", sort=False)""",
"""fig, axes = plt.subplots(1, 4, figsize=(15, 3.2))
for ax, n in zip(axes, NAMES):
    d = oracle_res[oracle_res.network == n]
    base = d[d.prior == "none"].sid.iloc[0]
    for (strategy, mode), (color, ls) in {("order", "soft"): ("#2e86de", "--"), ("order", "hard"): ("#c0392b", "--"),
                                         ("pairwise", "soft"): ("#2e86de", "-"), ("pairwise", "hard"): ("#c0392b", "-")}.items():
        m = d[(d.strategy == strategy) & (d["mode"] == mode)]
        ax.plot([float(p[2:]) for p in m.prior], m.sid, "o", color=color, ls=ls, label=f"{strategy}·{mode}")
    ax.axhline(base, color="#7f8c8d", ls="--", label="통계만")
    ax.set_title(n)
    ax.set_xlabel("사전지식 오류율 e")
axes[0].set_ylabel("SID (낮을수록 좋음)")
axes[0].legend()
plt.tight_layout()""",
"""md:읽는 법은 다음과 같다.

- **hard**: 사전지식이 정확하면(e = 0) 가장 좋다. 오류율이 10~20%만 돼도 큰 그래프(insurance, alarm)에서 통계만 쓴 경우보다 나빠진다.
- **soft**: 정확한 사전지식에서 얻는 이득은 작지만, 사전지식이 틀려도 통계만 쓴 경우보다 크게 나빠지지 않는다.
- **전략**: 오류가 있을 때 pairwise(실선)가 order(점선)보다 덜 망가진다. 다만 가상 LLM의 오류 방식(순서 자리 바꿈 vs 쌍별 방향 뒤집기)이 달라 직접 비교에는 한계가 있다.
- **결론**: 출처를 모르는 LLM 사전지식에는 soft를, 전문가가 확인한 순서에는 hard를 쓴다. 실제 LLM의 오류율이 어디쯤인지가 4장의 질문이다.""",
"""md:## 3. 파이프라인 결과 읽기 (sachs, 가상 LLM e = 0.1)

`tabular_to_hcg`는 그래프와 함께 엣지별 증거, 검증 리포트, 그리고 **충돌 목록**(데이터가 정한 방향과 LLM 순서가 어긋나는 엣지)을 돌려준다. 데이터와 LLM 순서가 일치하는 엣지는 두 출처의 증거를 갖게 되어 verified까지 오를 수 있다. 충돌 엣지는 L3(지식) 검토 대상이다.""",
"""true, sachs = data["sachs"]
r = tabular_to_hcg(sachs, "sachs", NETWORKS["sachs"]["domain"], NETWORKS["sachs"]["descriptions"], NoisyOracle(true, 0.1, seed=1), mode="soft", prior="order", n_boot=20)
print(graph_report(true, r.dag))
print(r.report.summary()["verdict"], "| 강한 위반", r.report.summary()["strong_violations"], "| 충돌", r.conflicts)
chk = r.changes.assign(true_edge=[tuple(e.split("→")) in true.edges for e in r.changes.edge],
                      conflict=[tuple(e.split("→")) in {(u, v) for u, v, _ in r.conflicts} for e in r.changes.edge])
chk""",
"""md:상태별·충돌 여부별로 참 엣지 비율을 보면, 두 출처(데이터와 LLM 순서)가 일치한 엣지와 어긋난 엣지의 정확도 차이가 드러난다. 충돌 목록은 사람이나 L3 검증이 먼저 볼 엣지 목록으로 쓸 수 있다.""",
"""chk.groupby(["after", "conflict"]).true_edge.agg(n="size", true_rate="mean")""",
"""md:## 4. (GPU) 실제 LLM 사전지식 — Qwen3.5

두 모델로 두 전략을 모두 잰다. 먼저 사전지식이 참 그래프와 얼마나 맞는지 본다. order는 순서 정확도, pairwise는 참 엣지 쌍에서 확신한 답의 정확도이다. 그다음 PC와 결합한 결과를 2장의 가상 LLM 곡선 위에 놓아 본다. asia·sachs는 변수 설명을 주고, insurance·alarm은 변수 이름만 준다.

> 2차 실행(2026-10-25) 결과: pairwise 쌍별 정확도는 9B 기준 sachs 0.88·insurance 0.86·alarm 0.67이고, pairwise·hard가 sachs SID 49→16, insurance 456→352로 크게 좋아졌지만 alarm은 55→219로 나빠졌다. order 전략은 변수 20개가 넘으면 4096토큰에서도 잘린다. 이번 실행부터 LLM 응답을 `results/llm_cache/`에 기록한다.
>
> 1차 실행(2026-10-25)에서 order 전략은 4B가 본문 추론 중 잘려 JSON을 못 냈고(sachs·alarm), 9B는 변수를 알파벳순으로 한 계층에 몰았다(alarm·insurance). 토큰 한도를 4096으로 올리고 계층 수 하한·계층 크기 상한을 지시하도록 고쳤으며, pairwise 전략을 추가했다.""",
"""import gc

import torch

from humm.providers.cache import CachedProvider
from humm.providers.hf import HFProvider

MODELS = {
    "Qwen/Qwen3.5-4B": "cuda:0",
    "Qwen/Qwen3.5-9B": "cuda:1",
}
llm_rows, orders = [], {}
for model_id, device in MODELS.items():
    lm = CachedProvider(HFProvider(model_id, device=device, batch_size=16), RESULTS / "llm_cache" / f"{model_id.split('/')[-1]}.json")
    for n in NAMES:
        true, df = data[n]
        res = compare(df, true, {model_id: lm}, NETWORKS[n]["domain"], NETWORKS[n]["descriptions"], modes=("soft", "hard"), priors=("order", "pairwise"))
        o = tabular_to_hcg(df, n, NETWORKS[n]["domain"], NETWORKS[n]["descriptions"], lm, prior="order", n_boot=0, validate_kw={"n_perm": 0})
        orders[f"{model_id}|{n}"] = {"tiers": o.knowledge.tiers, "raw": o.prior_raw}
        llm_rows.append(res.assign(network=n))
    lm.save()  # 응답 기록: Mac에서 ReplayProvider로 GPU 없이 재생한다
    del lm
    gc.collect()
    torch.cuda.empty_cache()

llm_res = pd.concat(llm_rows, ignore_index=True)
llm_res.to_csv(RESULTS / "tabular_llm_prior.csv", index=False)
(RESULTS / "tabular_llm_orders.json").write_text(json.dumps(orders, ensure_ascii=False, indent=2))
soft = llm_res[llm_res["mode"] == "soft"].set_index(["prior", "network"])
order_q = soft[soft.strategy == "order"][["order_accuracy", "order_tiers"]]
pair_q = soft[soft.strategy == "pairwise"][["pairs_on_true_edges", "correct", "wrong", "unsure", "pair_accuracy"]]
order_q.join(pair_q).round(2)""",
"""summary = pd.concat([oracle_res[oracle_res.prior == "none"], llm_res], ignore_index=True)
summary.pivot_table(index=["prior", "strategy", "mode"], columns="network", values=["shd", "sid"], sort=False)""",
"""md:## 5. (GPU) 실제 LLM으로 만든 HCG (sachs)

sachs에서 쌍별 정확도가 더 높은 모델의 pairwise 사전지식(혼합 모드 기본값)으로 HCG를 만들고, 검증 리포트와 HCG를 저장한다.""",
"""best = llm_res[(llm_res.network == "sachs") & (llm_res.strategy == "pairwise")].sort_values("pair_accuracy").prior.iloc[-1]
lm = CachedProvider(HFProvider(best, device=MODELS[best], batch_size=16), RESULTS / "llm_cache" / f"{best.split('/')[-1]}.json")
true, sachs = data["sachs"]
r = tabular_to_hcg(sachs, "sachs", NETWORKS["sachs"]["domain"], NETWORKS["sachs"]["descriptions"], lm, mode="hybrid", prior="pairwise", n_boot=20)
r.report.save(RESULTS / "validation" / "sachs-llm.json")
r.hcg.save(RESULTS / "validation" / "sachs-llm.hcg.json")
print(best, graph_report(true, r.dag), "| 충돌", r.conflicts)
lm.save()
r.changes.assign(true_edge=[tuple(e.split("→")) in true.edges for e in r.changes.edge])""",
"""md:## 6. (CPU) 혼합 모드 — 기록한 응답 재생

4장에서 기록한 LLM 응답(`results/llm_cache/`)을 재생해 GPU 없이 실험한다. 먼저 LLM의 확신도가 실제 정답률과 맞는지(보정) 보고, 데이터가 정한 방향과 LLM 방향 중 어느 쪽이 맞는지 네트워크별로 비교한다.""",
"""import numpy as np

from humm.graph.tabular import DEV, HELDOUT, llm_pairwise
from humm.providers.cache import ReplayProvider

replay = {m: ReplayProvider(RESULTS / "llm_cache" / f"Qwen3.5-{m}.json", name=m) for m in ("4B", "9B")}
rows = []
for n in DEV:
    true, df = data[n]
    g0 = pc(df, 0.01)
    adj = sorted({tuple(sorted(e)) for e in g0.undirected} | {tuple(sorted(e)) for e in g0.directed})
    pws = {m: llm_pairwise(p, adj, NETWORKS[n]["domain"], NETWORKS[n]["descriptions"]).set_index(["a", "b"]).p_ab for m, p in replay.items()}
    for a, b in adj:
        if true.has_edge(a, b) or true.has_edge(b, a):
            rows.append({"network": n, "truth": int(true.has_edge(a, b)), "4B": pws["4B"][(a, b)], "9B": pws["9B"][(a, b)],
                         "data_dir": 1 if (a, b) in g0.directed else (0 if (b, a) in g0.directed else np.nan)})
cal = pd.DataFrame(rows)
conf, ok = np.maximum(cal["9B"], 1 - cal["9B"]), (cal["9B"] >= 0.5) == (cal.truth == 1)
pd.DataFrame({"n": ok.groupby(pd.cut(conf, [0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0001], right=False), observed=False).size(),
              "9B 정답률": ok.groupby(pd.cut(conf, [0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0001], right=False), observed=False).mean().round(2)}).T""",
"""d = cal.dropna(subset=["data_dir"])
d.groupby("network").apply(lambda x: pd.Series({"데이터가 방향을 정한 참 엣지": len(x), "데이터 정확도": ((x.data_dir == 1) == (x.truth == 1)).mean(),
                                                "9B 정확도": ((x["9B"] >= 0.5) == (x.truth == 1)).mean()}), include_groups=False).round(2)""",
"""md:데이터와 LLM 중 누가 맞는지는 네트워크마다 다르다(alarm은 데이터, sachs는 LLM). 그래서 **혼합 모드**를 쓴다. 모든 모델이 같은 방향을 τ 이상 확신한 쌍만 데이터보다 우선하고, 나머지는 soft로 쓴다. τ와 모델 조합을 바꿔 가며 SID를 비교한다.""",
"""providers = {"none": None, "9B": replay["9B"], "4B+9B": [replay["4B"], replay["9B"]]}
grid = pd.concat([compare(data[n][1], data[n][0], providers, NETWORKS[n]["domain"], NETWORKS[n]["descriptions"],
                          modes=("soft", "hard"), taus=(0.8, 0.9, 0.95, 0.99)).assign(network=n) for n in DEV], ignore_index=True)
grid.to_csv(RESULTS / "tabular_hybrid_grid.csv", index=False)
t = grid.pivot_table(index=["prior", "mode"], columns="network", values="sid", sort=False)
t.assign(평균_상대SID=(t / t.loc[("none", "-")]).mean(axis=1).round(3))""",
"""md:**기본값**: 혼합 모드, 공급자 하나면 τ = 0.99, 둘 이상이면 τ = 0.95. 이 설정은 개발용 네 네트워크 모두에서 통계만보다 나쁘지 않았다. 다만 τ를 같은 네트워크에서 골랐으므로, 7장에서 보류 네트워크로 다시 확인한다.""",
"""md:## 7. (GPU) 보류 네트워크 검증

τ를 고를 때 쓰지 않은 네 네트워크(child, hepar2, win95pts, hailfinder)에서 기본값이 통계만보다 나은지 확인한다. 변수 설명은 주지 않고 이름만 준다. 먼저 두 모델로 쌍별 응답을 기록한 뒤, 비교는 재생으로 한다.""",
"""held = {n: sample_bnlearn(n, 5000, seed=0) for n in HELDOUT}
for model_id, device in MODELS.items():
    lm = CachedProvider(HFProvider(model_id, device=device, batch_size=16), RESULTS / "llm_cache" / f"{model_id.split('/')[-1]}.json")
    for n in HELDOUT:
        tabular_to_hcg(*held[n][::-1], NETWORKS[n]["domain"], provider=lm, mode="soft", n_boot=0, validate_kw={"n_perm": 0}) if False else \
            tabular_to_hcg(held[n][1], n, NETWORKS[n]["domain"], provider=lm, mode="soft", n_boot=0, validate_kw={"n_perm": 0})
    lm.save()
    del lm
    gc.collect()
    torch.cuda.empty_cache()

replay = {m: ReplayProvider(RESULTS / "llm_cache" / f"Qwen3.5-{m}.json", name=m) for m in ("4B", "9B")}
providers = {"none": None, "9B": replay["9B"], "4B+9B": [replay["4B"], replay["9B"]]}
heldout = pd.concat([compare(held[n][1], held[n][0], providers, NETWORKS[n]["domain"], modes=("soft", "hard", "hybrid")).assign(network=n)
                     for n in HELDOUT], ignore_index=True)
heldout.to_csv(RESULTS / "tabular_heldout.csv", index=False)
t = heldout.pivot_table(index=["prior", "mode"], columns="network", values="sid", sort=False)
t.assign(평균_상대SID=(t / t.loc[("none", "-")]).mean(axis=1).round(3))""",
"""heldout[heldout["mode"] == "soft"].set_index(["prior", "network"])[["pairs_on_true_edges", "correct", "wrong", "unsure", "pair_accuracy"]].round(2)""",
"""md:## Mac으로 옮길 파일

실행한 노트북은 원본과 다른 이름으로 저장하고, `notebooks/`의 원본은 출력 없이 둔다.

| 파일 | 필수 여부 | 비고 |
|---|---|---|
| 이 노트북 | 필수 | `runs/<날짜>-gpu/` |
| `results/tabular_llm_prior.csv` | 필수 | 4장: 모델·네트워크·전략별 사전지식 정확도와 SHD·SID |
| `results/tabular_llm_orders.json` | 필수 | 4장: LLM이 답한 인과 순서와 원문 응답 |
| `results/validation/sachs-llm.json` | 필수 | 5장: 검증 리포트 |
| `results/validation/sachs-llm.hcg.json` | 필수 | 5장: 실제 LLM 사전지식으로 만든 HCG |
| `results/llm_cache/Qwen3.5-4B.json`, `Qwen3.5-9B.json` | 필수 | LLM 응답 기록(개발·보류 네트워크). Mac에서 GPU 없이 재생한다 |
| `results/tabular_hybrid_grid.csv` | 필수 | 6장: 혼합 모드 τ 탐색 |
| `results/tabular_heldout.csv` | 필수 | 7장: 보류 네트워크 검증 |""",
]

n = nbf.v4.new_notebook()
n.cells = [nbf.v4.new_markdown_cell(c[3:]) if c.startswith("md:") else nbf.v4.new_code_cell(c) for c in cells]
n.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(n, sys.argv[1])
