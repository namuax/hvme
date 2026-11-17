import sys
import nbformat as nbf

cells = [
"""md:# 05. DAG 검증 — 그래프를 데이터로 반증하기

인과 그래프는 원리상 "증명"할 수 없다. 데이터·지식·실험과 모순되지 않는지 반증을 시도하고, 거기서 살아남은 그래프만 쓸 수 있다. `humm.validate`는 `preliminary/03_dag_validation.md`의 6층 검증 중 데이터만으로 가능한 세 층을 구현한다.

| 층 | 내용 | 상태 |
|---|---|---|
| L1 구조 | 사이클, 자기 루프, 고립 노드, 시간 순서 위반 | 구현 |
| L2 데이터 반증 | 국소 마르코프 조건(LMC) 검정, 순열 반증 검정, 엣지 의존성 검정, 메커니즘 적합도 | 구현 |
| L3 지식 | 텍스트 증거, LLM 반사실 검증, 전문가 검토 | Step 6·8 |
| L4 실험 | 개입 데이터와 예측 효과 비교 | 이후 |
| L5 결론 강건성 | 대안 그래프 효과 범위, 반박 테스트 | `humm.engine.refute` 일부 |
| L6 벤치마크 | 정답 그래프 대비 SHD·SID | 구현 (탐색 알고리즘은 Step 5) |

노트북 03에서 본 것처럼, 교란 변수를 빠뜨린 그래프는 추정치 반박 테스트를 모두 통과한다. 그래프 자체의 오류는 이 노트북의 L2가 잡아야 한다.""",
"""import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from humm.engine import BinarySCM, LinearSCM
from humm.graph import HCG, Evidence, EvidenceKind
from humm.paths import RESULTS
from humm.validate import benchmark, fisher_z, g_test, sample_bnlearn, validate, validate_hcg""",
"""md:## 1. 조건부 독립 검정

L2의 기본 단위는 조건부 독립 검정이다. 이산 데이터는 G-검정(층별 우도비), 연속 데이터는 Fisher-z(편상관)를 쓴다. 사슬 `a → b → c`에서 a와 c는 의존하지만, b를 조건으로 주면 독립이 된다. p값이 크면 "독립과 모순되지 않는다"는 뜻이다.""",
"""CHAIN = nx.DiGraph([("a", "b"), ("b", "c")])
cont = LinearSCM(CHAIN, {("a", "b"): 1.0, ("b", "c"): 1.0}, {v: 1.0 for v in CHAIN}).sample(3000, seed=0)
disc = BinarySCM(CHAIN, {"a": np.array([0.5]), "b": np.array([0.2, 0.8]), "c": np.array([0.2, 0.8])}).sample(5000, seed=0)
pd.DataFrame({"Fisher-z (연속)": [fisher_z(cont, "a", "c"), fisher_z(cont, "a", "c", ("b",))],
              "G-검정 (이산)": [g_test(disc, "a", "c"), g_test(disc, "a", "c", ("b",))]},
             index=["a ⫫ c", "a ⫫ c | b"]).map(lambda p: f"{p:.2e}")""",
"""md:## 2. 참 그래프와 훼손된 그래프

bnlearn Asia 네트워크에서 5,000개 표본을 뽑고, 참 그래프와 일부러 훼손한 그래프를 검증한다.

- **LMC 위반 비율**: 그래프가 함의하는 조건부 독립 중 데이터가 기각한 비율. 참 그래프라면 유의수준(5%) 근처여야 한다.
- **순열 p값**: 노드 이름을 무작위로 바꾼 그래프 100개와 비교해, 위반이 그만큼 적은 그래프의 비율. 작을수록 그래프가 데이터에 대해 정보를 가진다.
- **판정**: pass는 위반 개수가 우연 수준(이항 검정 p ≥ 0.05)이고, 다중검정 보정 후에도 남는 강한 위반이 없고, 순열 p < 0.05일 때이다. LMC 검정이 5개 미만인 작은 그래프는 순열 검정의 검정력이 없어 이 조건을 빼고 판정한다.""",
"""true, asia = sample_bnlearn("asia", 5000, seed=0)


def edit(g, remove=(), add=()):
    h = g.copy()
    h.remove_edges_from(remove)
    h.add_edges_from(add)
    return h


cases = {
    "참 그래프": true,
    "either→dysp 누락": edit(true, remove=[("either", "dysp")]),
    "smoke→bronc 누락": edit(true, remove=[("smoke", "bronc")]),
    "lung→either 반전": edit(true, remove=[("lung", "either")], add=[("either", "lung")]),
    "smoke→lung 반전": edit(true, remove=[("smoke", "lung")], add=[("lung", "smoke")]),
    "asia→xray 추가": edit(true, add=[("asia", "xray")]),
}
reports = {k: validate(g, asia, n_perm=100) for k, g in cases.items()}
pd.DataFrame([{"그래프": k, **{c: r.summary()[c] for c in ["verdict", "lmc_tests", "lmc_violation_rate", "excess_violation_p", "strong_violations", "permutation_p", "edges_unsupported"]},
               "빠진 엣지 후보 1순위": r.missing_edge_candidates(1)[0][:2] if r.missing_edge_candidates(1) else "-"}
              for k, r in reports.items()]).round(3)""",
"""md:읽는 법은 다음과 같다.

- **누락**: 엣지를 빼면 위반 비율이 크게 오르고, 빠진 엣지 후보 1순위가 실제로 뺀 쌍을 가리킨다.
- **lung→either 반전**: v-구조(lung → either ← tub)가 바뀌므로 데이터가 잡아낸다.
- **smoke→lung 반전**: 같은 마르코프 동치류 안의 변화라 관측 데이터로는 구별되지 않는다(pass). 이 방향은 L3 지식이나 L4 실험으로 정해야 한다.
- **asia→xray 추가**: 그래프 전체 판정은 pass이지만, 엣지 의존성 검정이 이 엣지를 "데이터가 뒷받침하지 않음"으로 표시한다.""",
"""fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True)
for ax, k in zip(axes, ["참 그래프", "either→dysp 누락"]):
    perm = reports[k].permutation
    ax.hist(perm["perm_rates"], bins=20, color="#9aa5b1")
    ax.axvline(perm["violation_rate"], color="#c0392b", lw=2, label=f"주어진 그래프 {perm['violation_rate']:.3f}")
    ax.set_title(f"{k} (순열 p = {perm['p_value']:.2f})")
    ax.set_xlabel("LMC 위반 비율")
    ax.legend()
axes[0].set_ylabel("무작위 순열 그래프 수")
plt.tight_layout()""",
"""md:## 3. HCG 검증 흐름

실제 파이프라인에서는 텍스트·LLM·전문가가 만든 HCG를 데이터로 검증한다. `validate_hcg`는 다음 순서로 동작한다.

1. **L1 정리**: 시간 순서를 어긴 엣지를 refuted로 바꾸고, 남은 사이클은 가장 약한 엣지를 refuted로 바꿔 해소한다.
2. **L2 검정**: 남은 DAG로 LMC·순열·엣지 검정을 한다.
3. **반영**: 엣지마다 데이터 증거(score = 1 − p)를 붙이고 상태를 바꾼다. verified가 되려면 서로 다른 종류의 증거 2개 이상, 데이터 뒷받침, 방향 근거(텍스트·LLM·전문가·실험), 그래프 판정 pass가 모두 필요하다.

마케팅 예시에서 LLM이 "매출이 늘면 광고를 늘린다"는 역방향 엣지를 낮은 신뢰도로 제안했다고 하자. 시간 순서(계절 → 광고 → 방문 → 매출)와 어긋나므로 L1에서 반증된다.""",
"""SALES = nx.DiGraph([("season", "ad"), ("ad", "visits"), ("visits", "sales"), ("season", "sales")])
scm = BinarySCM(SALES, {"season": np.array([0.4]), "ad": np.array([0.2, 0.7]), "visits": np.array([0.2, 0.75]),
                        "sales": np.array([0.15, 0.5, 0.45, 0.85])})
data = scm.sample(6000, seed=3)

h = HCG(graph_id="sales-demo")
for nid, label in [("season", "계절"), ("ad", "광고비"), ("visits", "방문"), ("sales", "매출")]:
    h.add_node(nid, label=label)
h.add_edge("season", "ad", evidence=[Evidence(kind=EvidenceKind.EXPERT, score=0.9, ref="마케팅팀 인터뷰")])
h.add_edge("ad", "visits", evidence=[Evidence(kind=EvidenceKind.TEXT, score=0.8, ref="report_07.pdf#p3")])
h.add_edge("visits", "sales", evidence=[Evidence(kind=EvidenceKind.TEXT, score=0.7), Evidence(kind=EvidenceKind.LLM_PRIOR, score=0.9)])
h.add_edge("season", "sales")
h.add_edge("sales", "ad", evidence=[Evidence(kind=EvidenceKind.LLM_PRIOR, score=0.3)])

report, changes = validate_hcg(h, data, time={"season": 0, "ad": 1, "visits": 2, "sales": 3}, n_perm=100)
print(report.summary()["verdict"], "|", report.notes)
changes""",
"""md:이 그래프는 LMC 검정이 2개뿐이라 순열 검정은 판정에 쓰지 않았다(검정력 부족). 시간 순서·데이터·방향 근거를 모두 갖춘 엣지는 verified가 된다. season→sales는 데이터 증거만 있고 방향 근거가 없어서 hypothesis에 머문다. 데이터는 "두 변수가 관련 있다"는 것만 말해 줄 뿐, 방향은 알려 주지 않기 때문이다. 검증 결과는 `validation.json`으로 남긴다.""",
"""path = report.save(RESULTS / "validation" / "sales-demo.json")
h.save(RESULTS / "validation" / "sales-demo.hcg.json")
print(path)
report.mechanisms.round(3)""",
"""md:## 4. L6 벤치마크

파이프라인을 바꿀 때마다 같은 bnlearn 네트워크에서 SHD·SID를 재는 회귀 테스트이다. 인과 탐색 알고리즘은 Step 5에서 만들므로, 여기서는 두 극단(빈 그래프, 정답 그래프)으로 지표의 범위만 확인한다.""",
"""names = ("asia", "sachs", "insurance", "alarm")
truth = {n: sample_bnlearn(n, 100)[0] for n in names}
empty = benchmark(lambda df: nx.DiGraph(), names=names, n=2000).assign(method="빈 그래프")
oracle = pd.concat([benchmark(lambda df, g=truth[n]: g.copy(), names=(n,), n=2000) for n in names]).assign(method="정답 그래프")
pd.concat([empty, oracle])[["method", "network", "nodes", "edges", "shd", "sid", "f1_directed"]]""",
]

n = nbf.v4.new_notebook()
n.cells = [nbf.v4.new_markdown_cell(c[3:]) if c.startswith("md:") else nbf.v4.new_code_cell(c) for c in cells]
n.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(n, sys.argv[1])
