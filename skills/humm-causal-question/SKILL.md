---
name: humm-causal-question
description: 사용자가 "왜 ~했나?", "~하면 어떻게 돼?", "~가 ~의 원인인가?", "만약 ~했다면?" 같은 인과 질문을 할 때 가장 먼저 쓰는 진입 Skill이다. 질문을 Rung 1/2/3으로 분류하고 처리 경로를 고르며, 상관을 인과로 단정하지 않도록 막는다.
---

## 언제 쓰나

- 질문에 원인·효과·개입·반사실이 들어 있을 때 쓴다("광고비를 늘리면 매출이 오르나?", "이탈이 늘어난 건 가격 때문인가?").
- 다른 humm Skill을 고르기 전에 질문 유형이 불분명할 때 쓴다.
- 숫자 답이 필요 없는 대량 예/아니오 판단은 이 Skill이 아니라 `humm-fast-judge`로 보낸다.

## 절차

1. 가정 명시: 질문에서 처치(원인 후보)·결과·조건을 사용자 말로 다시 적고, 데이터(CSV)와 그래프(graph_id)가 있는지 확인한다.
2. 그래프 확인: `humm.graph_list()`로 저장된 그래프를 찾는다. 없으면 `humm-graph-from-table` 또는 `humm-graph-from-text`로 먼저 만든다. 있으면 `humm.graph_get(graph_id)`로 질문의 변수가 노드에 있는지 본다.
3. 분류: `humm.classify_question(question, graph_id)`를 부른다. `router`가 `"lm"`이거나 `confidence`가 낮으면 분류를 그대로 믿지 않고 `treatment`·`outcome`을 사용자에게 확인한다.
4. Rung별 경로를 고른다.
   - Rung 1(`kind="association"`): `humm.ask(question, graph_id, csv)`로 관찰 차이와 교란 조정 효과를 함께 받아 비교한다.
   - Rung 2(`kind="ate"`): `humm-identify-estimate`로 넘긴다.
   - Rung 3(`kind="ett"`·`"mediation"`, PN·PS 같은 "그때 ~했다면"): `humm-counterfactual`로 넘긴다.
   - 한 번에 끝까지 처리할 때는 `humm.ask_v1(question, graph_id, csv)`를 쓴다. 유형 키워드가 없는 질문의 유형을 Rung 분류로 정하고, 답에 `route`(분류)와 `lm_check`(Humm-LM 독립 풀이와 엔진 부호 대조)를 붙인다.
5. 데이터가 없으면 수치를 만들지 않고, 그래프 경로만으로 정성 답(경로 유무·부호)을 하거나 필요한 데이터를 말한다.

## 흄 원칙 체크리스트

- [ ] 상관만으로 인과를 단정하지 않는다.
- [ ] 숨은 교란을 항상 점검한다.
- [ ] 반사실은 가정 위에서만 말한다.
- [ ] 질문 문장 자체가 인과를 전제하면("X 때문에 Y가 올랐는데") 그 전제를 확인할 대상으로 되돌려 놓는다.
- [ ] 처치와 결과의 방향이 질문 문장과 맞는지 확인한다(`classify_question`의 `treatment`·`outcome`).

## 결과 해석

- `classify_question` 결과의 `rung`(1/2/3)·`kind`(ate/association/mediation/ett)가 경로를 정한다.
- `mediators`가 비어 있지 않으면 매개 질문이므로 Rung 3으로 본다.
- `router="fast"`는 Humm-Fast 분류가 확신한다는 뜻이고, `"lm"`은 확신이 낮아 상위 경로로 올렸다는 뜻이다.
- Rung 1 경로의 `ask` 결과에서는 `estimates.association`(관찰 차이)과 `estimates.ate`(조정 효과)의 차이를 교란 경로 몫으로 읽는다(`backdoor_paths` 참고).

## 한계 고지

- 분류는 추정이며 `confidence`를 함께 말한다.
- Rung 1 질문에 "인과다"라고 답하지 않고, 조정 효과는 그래프 가정 위에서만 성립한다고 말한다.
- 질문 변수를 그래프에서 찾지 못하면 "식별 불가 + 필요한 데이터(해당 변수의 열, 관련 문서)"로 답한다.

## 다른 Skill로 넘기기

- Rung 2 → `humm-identify-estimate`
- Rung 3 → `humm-counterfactual`
- 그래프가 없거나 오래됨 → `humm-graph-from-table` / `humm-graph-from-text` → `humm-validate-dag`
- 최종 보고 → `humm-explain`
