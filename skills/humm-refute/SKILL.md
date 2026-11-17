---
name: humm-refute
description: 추정한 인과 효과가 믿을 만한지 검증할 때 쓴다. "이 결과 믿어도 돼?", "효과 검증해줘", "반박 테스트", "민감도 분석" 같은 요청에 placebo·랜덤 공통원인·부분집합 반박과 그래프 민감도(L5)를 확인한다.
---

## 언제 쓰나

- `humm-identify-estimate`나 `humm-counterfactual`에서 숫자를 얻은 직후, 보고 전에 쓴다.
- 사용자가 결과의 견고성을 물을 때 쓴다.

## 절차

1. 가정 명시: 어떤 추정(처치·결과·식별 전략)을 검증하는지 적는다.
2. 그래프 확인: `humm.graph_get(graph_id)`로 검증 대상 그래프가 추정에 쓴 것과 같은지 확인한다.
3. 반박: `humm.engine_refute(graph_id, csv, treatment, outcome)`을 부른다.
4. 그래프 민감도: `humm.ask(question, graph_id, csv)`의 `sensitivity_table`과 `graph_sensitivity`를 본다(대안 그래프: 엣지 삭제·반전·복원, 숨은 교란 `_U` 추가).
5. 독립 확인: `humm.lm_verify(graph_id, csv, treatment, outcome)`의 `agree`를 본다.
6. 실패하면 오케스트레이터 ②로 돌아가 교란 변수를 추가한다(`humm-graph-from-table` → `humm-validate-dag`).

## 흄 원칙 체크리스트

- [ ] 상관만으로 인과를 단정하지 않는다.
- [ ] 숨은 교란을 항상 점검한다.
- [ ] 반사실은 가정 위에서만 말한다.
- [ ] 반박 통과를 "인과가 증명되었다"로 바꿔 말하지 않는다.
- [ ] 숨은 교란 시나리오에서 식별이 깨지는지(`breaks_with_hidden_confounder`) 확인한다.

## 결과 해석

- `engine_refute` 각 행: `test`(placebo_treatment / random_common_cause / data_subset), `expected`, `new_value`, `estimate`, `tolerance`, `passed`이다.
  - placebo_treatment: 처치를 섞으면 0에 가까워야 한다.
  - random_common_cause: 난수 공통 원인을 넣어도 원래 값과 같아야 한다.
  - data_subset: 80% 부분표본에서도 같아야 한다.
- `sensitivity_table` 행: `scenario`, `strategy`, `variables`, `value`이며, `strategy="not_identified"`는 그 대안에서 식별이 깨진다는 뜻이다.
- `graph_sensitivity`: `alternative_graphs`, `sign_flipped`, `effect_vanished`, `breaks_with_hidden_confounder`이다. `sign_flipped`가 0이 아니면 결론이 특정 엣지에 기댄다.

## 한계 고지

- 반박 테스트는 추정 절차의 오류를 잡을 뿐 그래프 오류나 숨은 교란을 배제하지 못한다고 말한다.
- 부호가 뒤집히는 대안 그래프가 있으면 그 시나리오 이름을 함께 말한다.
- 실패한 테스트는 숨기지 않고 결론의 신뢰도를 낮춘다.

## 다른 Skill로 넘기기

- 통과 → `humm-explain`
- 실패 → `humm-graph-from-table` / `humm-validate-dag`
