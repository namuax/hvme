---
name: humm-identify-estimate
description: 처치 효과·개입 효과(ATE)를 계산할 때 쓴다. "광고비를 늘리면 매출이 얼마나 오르나?", "이 정책의 효과는?", "X가 Y에 미치는 영향" 같은 Rung 2 질문에 식별 가능성 판정 → 추정 → 95% 신뢰구간을 낸다.
---

## 언제 쓰나

- `humm-causal-question`이 Rung 2(`kind="ate"`)로 분류한 질문에 쓴다.
- 검증된 그래프(`humm-validate-dag` 통과)와 0/1 이진 열 CSV가 있을 때 쓴다.

## 절차

1. 가정 명시: 처치(1/0의 뜻)·결과·모집단을 적는다.
2. 그래프 확인: `humm.graph_get(graph_id)`로 처치→결과 경로와 처치로 들어오는 화살표(교란 경로)를 확인한다.
3. 식별: `humm.engine_identify(graph_id, treatment, outcome, csv)`를 부른다. `csv`를 주면 데이터에 없는 노드를 관측되지 않은 변수로 본다.
4. `strategy`가 `not_identified`이면 추정을 멈추고 "식별 불가 + 필요한 데이터"로 답한다(어떤 변수를 측정하면 백도어·프론트도어·IV가 열리는지 적는다).
5. 추정: `humm.engine_estimate(graph_id, csv, treatment, outcome, n_boot=200)`을 부른다.
6. 엔드투엔드 답이 필요하면 `humm.ask(question, graph_id, csv)`를 한 번 불러 추정·반박·민감도·가정을 함께 받는다.
7. 독립 확인: `humm.lm_verify(graph_id, csv, treatment, outcome)`로 Hume-LM의 답과 엔진 부호가 맞는지 본다.

## 흄 원칙 체크리스트

- [ ] 상관만으로 인과를 단정하지 않는다.
- [ ] 숨은 교란을 항상 점검한다.
- [ ] 반사실은 가정 위에서만 말한다.
- [ ] 조정 집합에 처치의 결과(매개·충돌 변수)가 들어가지 않았는지 확인한다.
- [ ] 구간이 0을 포함하면 "효과가 있다/없다"를 단정하지 않는다.

## 결과 해석

- `engine_identify`: `strategy`(backdoor/frontdoor/iv/no_effect/not_identified), `variables`(조정·매개·도구 변수), `expression`(추정식), `latent`(관측되지 않은 노드)이다.
- `engine_estimate`: `value`(ATE, 처치 1 대 0의 결과=1 확률 차이), `se`, `ci`(부트스트랩 95% 구간), `method`, `strategy`이다.
- `ask` 결과: `estimates.ate.value`·`estimates.ate.ci`, `adjustment`, `expression`, `paths`(근거 경로), `backdoor_paths`, `refutation`, `sensitivity_table`, `graph_sensitivity`, `assumptions`, `answer`이다.
- `lm_verify`: `agree`가 false이면 엔진과 Hume-LM이 부호에서 갈렸으므로 그래프·인자를 다시 본다. `trace`에 LM의 도구 호출 과정이 있다.
- `strategy="no_effect"`는 그래프에 인과 경로가 없다는 뜻이며, 결론이 그래프 구조에 전적으로 기댄다.

## 한계 고지

- `assumptions`(조정 집합이 모든 교란 경로를 막음, 양수성, SUTVA, 검증되지 않은 엣지)를 반드시 함께 말한다.
- 신뢰구간과 `graph_sensitivity`(부호가 뒤집힌 대안 수, 숨은 교란에서 식별이 깨지는지)를 함께 말한다.
- 식별 불가면 숫자를 내지 않고 "식별 불가 + 필요한 데이터"로 답한다.
- `lm_verify.agree=false`이면 그 사실을 숨기지 않는다.

## 다른 Skill로 넘기기

- 결과 검증 → `humm-refute`
- 반사실·매개 질문으로 바뀌면 → `humm-counterfactual`
- 보고 → `humm-explain`
