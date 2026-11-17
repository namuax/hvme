import sys
import nbformat as nbf

cells = [
"""md:# 03. 기호 인과 엔진 — 식별 · 추정 · 반박 · 반사실

`humm.engine`은 Humm 전체의 **정답 기준**이다. Humm-Gym이 만드는 문제의 정답, Humm-LM 강화학습의 보상, DAG 검증의 판정이 모두 이 엔진의 계산에 기대고 있다. 그래서 LLM이 아니라 그래프 이론과 확률 계산만으로 동작하게 만들었다.

| 모듈 | 하는 일 |
|---|---|
| `dsep` | d-분리 판정, 국소 마르코프 조건 |
| `identify` | 백도어(일반화 조정 기준·O-set) → 프론트도어 → 도구변수 순으로 식별 |
| `scm` | 이진·선형 구조적 인과 모형. 관측·개입·반사실 값을 정확히 계산 |
| `estimate` | 식별된 추정식을 데이터로 계산(이진은 층화, 연속은 선형 회귀), 부트스트랩 신뢰구간 |
| `refute` | 가짜 처치, 무작위 공통 원인, 부분표본 반박 테스트 |

이 노트북은 GPU 없이 실행된다.""",
"""import logging

import networkx as nx
import numpy as np
import pandas as pd

from humm.engine import (BinarySCM, LinearSCM, adjustment_sets, d_separated, estimate, identify, is_valid_adjustment,
                         optimal_adjustment, refute)

logging.disable(logging.WARNING)""",
"""md:## 1. d-분리: 세 가지 기본 구조

인과 그래프의 모든 조건부 독립은 사슬(chain), 분기(fork), 충돌(collider) 세 구조의 조합으로 결정된다. 사슬과 분기는 가운데 변수를 조건으로 주면 막히고, 충돌은 반대로 조건을 주면 열린다.""",
"""structures = {
    "사슬 a→b→c": nx.DiGraph([("a", "b"), ("b", "c")]),
    "분기 a←b→c": nx.DiGraph([("b", "a"), ("b", "c")]),
    "충돌 a→b←c": nx.DiGraph([("a", "b"), ("c", "b")]),
}
pd.DataFrame({name: {"a ⫫ c": d_separated(g, "a", "c"), "a ⫫ c | b": d_separated(g, "a", "c", "b")}
              for name, g in structures.items()}).T""",
"""md:## 2. 식별: 백도어 · 프론트도어 · 도구변수

`identify`는 관측 변수만으로 개입 효과 P(y | do(x))를 계산할 수 있는지 판정하고, 가능하면 추정식을 돌려준다. `latent`로 지정한 노드는 관측되지 않는다고 본다.

- **교란**: Z가 X와 Y의 공통 원인이다 → 백도어 조정
- **흡연-타르-폐암**: 흡연(S)과 폐암(C)에 미관측 공통 원인 U가 있지만 타르(T)를 거치는 경로만 있다 → 프론트도어
- **도구변수**: Z는 X에만 영향을 준다 → IV
- **식별 불가**: 미관측 교란만 있고 다른 길이 없다""",
"""CONFOUNDED = nx.DiGraph([("Z", "X"), ("Z", "Y"), ("X", "Y"), ("W", "Y")])
SMOKING = nx.DiGraph([("U", "S"), ("U", "C"), ("S", "T"), ("T", "C")])
IV = nx.DiGraph([("Z", "X"), ("U", "X"), ("U", "Y"), ("X", "Y")])
HIDDEN = nx.DiGraph([("U", "X"), ("U", "Y"), ("X", "Y")])

cases = [("교란", CONFOUNDED, "X", "Y", set()), ("흡연-타르-폐암", SMOKING, "S", "C", {"U"}),
         ("도구변수", IV, "X", "Y", {"U"}), ("식별 불가", HIDDEN, "X", "Y", {"U"}), ("역방향 질의", CONFOUNDED, "Y", "X", set())]
pd.DataFrame([{"사례": n, "전략": (e := identify(g, x, y, lat)).strategy, "변수": sorted(e.variables), "추정식": e.expression}
              for n, g, x, y, lat in cases])""",
"""md:### 조정 집합 고르기

유효한 조정 집합은 여러 개일 수 있다. `adjustment_sets`는 최소 집합을 작은 것부터 찾고, `optimal_adjustment`는 추정 분산이 가장 작은 **O-set**(Henckel et al., 2022)을 준다. 교란 예제에서 W는 X와 무관하지만 Y의 원인이므로, 조정에 넣으면 편향은 없고 분산만 줄어든다.

M-bias 그래프는 "관련 있어 보이는 변수는 다 통제하라"는 직관이 틀리는 대표 사례이다. M은 충돌 변수라서 M만 통제하면 막혀 있던 경로가 열린다.""",
"""M_BIAS = nx.DiGraph([("A", "X"), ("A", "M"), ("B", "M"), ("B", "Y"), ("X", "Y")])
print("교란 — 최소 집합:", adjustment_sets(CONFOUNDED, "X", "Y"), "| O-set:", optimal_adjustment(CONFOUNDED, "X", "Y"))
pd.Series({str(sorted(z)): is_valid_adjustment(M_BIAS, "X", "Y", z) for z in [set(), {"M"}, {"M", "A"}, {"M", "B"}]}, name="M-bias 유효")""",
"""md:## 3. SCM으로 정답 만들기

`BinarySCM`은 각 변수를 `v = 1[u_v < p_v(부모)]`로 정의한다. 외생 잡음 u를 고정한 채 개입만 바꾼 "세계"들을 비교하므로 반사실 값까지 계산할 수 있다. 잡음 공간을 문턱값으로 나눈 칸 수가 20만 개 이하이면 **정확한 값**을, 넘으면 몬테카를로 근사값을 쓴다.

흡연 예제에서 단순 비교(흡연자와 비흡연자의 폐암 비율 차)와 진짜 인과 효과(ATE)를 비교한다.""",
"""scm = BinarySCM.random(SMOKING, seed=1)
obs = scm.prob("C", given={"S": 1}) - scm.prob("C", given={"S": 0})
print(f"정확 계산: {scm.exact}")
print(f"단순 비교 P(C|S=1) - P(C|S=0) = {obs:+.4f}")
print(f"인과 효과 P(C|do(S=1)) - P(C|do(S=0)) = {scm.ate('S', 'C'):+.4f}")""",
"""md:### 반사실 질의 (Rung 3)

CLadder의 반사실 질의 유형을 그대로 계산한다. 매개 그래프 `X → M → Y`, `X → Y`(Z는 교란)에서 직접효과와 간접효과를 분리하고, 필요·충분 확률(PN·PS)을 구한다. PN·PS는 처치가 결과를 높이는 방향일 때 의미가 있으므로 ATE가 양수인 모형을 골랐다.""",
"""MED = nx.DiGraph([("Z", "X"), ("X", "M"), ("M", "Y"), ("X", "Y"), ("Z", "Y")])
med = BinarySCM.random(MED, seed=18)
pd.Series({
    "ATE": med.ate("X", "Y"),
    "CDE (M=1)": med.cde("X", "Y", {"M": 1}),
    "ETT": med.ett("X", "Y"),
    "NDE": med.nde("X", "Y", ["M"]),
    "NIE": med.nie("X", "Y", ["M"]),
    "PN": med.pn("X", "Y"),
    "PS": med.ps("X", "Y"),
}, name="값").round(4)""",
"""md:## 4. 데이터로 추정하고 참값과 비교

SCM에서 표본을 뽑아 미관측 변수를 지운 뒤, 그래프만 보고 효과를 추정한다. 추정치의 95% 신뢰구간이 SCM의 참값을 담으면 엔진의 식별과 추정이 맞다는 뜻이다. 이진 데이터는 층화, 연속 데이터는 선형 회귀를 쓴다.""",
"""rows = []
for name, g, x, y, lat, seed in [("교란", CONFOUNDED, "X", "Y", set(), 3), ("흡연-타르-폐암", SMOKING, "S", "C", {"U"}, 1)]:
    s = BinarySCM.random(g, seed=seed)
    est = estimate(g, s.sample(100_000, seed=seed + 1).drop(columns=list(lat)), x, y, latent=lat)
    rows.append({"모형": "이진", "사례": name, "방법": est.method, "참값": s.ate(x, y), "추정": est.value, "CI 하한": est.ci[0], "CI 상한": est.ci[1]})
for name, g, x, y, lat, seed in [("교란", CONFOUNDED, "X", "Y", set(), 9), ("흡연-타르-폐암", SMOKING, "S", "C", {"U"}, 7), ("도구변수", IV, "X", "Y", {"U"}, 5)]:
    l = LinearSCM.random(g, seed=seed)
    est = estimate(g, l.sample(100_000, seed=seed + 1).drop(columns=list(lat)), x, y, latent=lat)
    rows.append({"모형": "선형", "사례": name, "방법": est.method, "참값": l.total_effect(x, y), "추정": est.value, "CI 하한": est.ci[0], "CI 상한": est.ci[1]})
pd.DataFrame(rows).round(4)""",
"""md:## 5. 반박 테스트

추정치가 맞는 모형 위에서 나왔다면 세 테스트를 모두 통과해야 한다. 판정 기준은 기대값과의 차이가 부트스트랩 표준오차의 2배 이내인지이다.""",
"""s = BinarySCM.random(CONFOUNDED, seed=3)
df = s.sample(50_000, seed=4)
refute(CONFOUNDED, df, "X", "Y").round(4)""",
"""md:교란 변수 Z를 빠뜨린 잘못된 그래프로 추정하면 편향이 생긴다. 이런 오류는 반박 테스트로 잡히지 않는다. 반박 테스트는 주어진 그래프 위에서의 추정 절차를 점검할 뿐이고, 그래프 자체의 오류는 Step 4의 DAG 검증(L2 데이터 반증)이 잡아야 한다.""",
"""WRONG = nx.DiGraph([("X", "Y"), ("W", "Y")])
WRONG.add_node("Z")
print(f"참값 {s.ate('X', 'Y'):+.4f} | 올바른 그래프 {estimate(CONFOUNDED, df, 'X', 'Y', n_boot=0).value:+.4f} | Z 누락 그래프 {estimate(WRONG, df, 'X', 'Y', n_boot=0).value:+.4f}")
refute(WRONG, df, "X", "Y").round(4)""",
]

n = nbf.v4.new_notebook()
n.cells = [nbf.v4.new_markdown_cell(c[3:]) if c.startswith("md:") else nbf.v4.new_code_cell(c) for c in cells]
n.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(n, sys.argv[1])
