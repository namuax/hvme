---
name: humm-fast-judge
description: 대량의 예/아니오·선택·척도 판단을 빠르게 처리할 때 쓴다. "이 엣지들 방향 맞는지 다 판단해줘", "보기 중 하나 골라줘", "1~5점으로 매겨줘" 같은 요청에 Hume-Fast(fast_noul·fast_choice·fast_score)를 부르고, 저신뢰 결과는 상위 경로로 올린다.
---

## 언제 쓰나

- 쌍별 엣지 판정, 반사실 문장 검증, 질의 분류처럼 같은 형식의 판단이 많을 때 쓴다.
- 수치 효과·반사실 계산이 필요한 질문에는 쓰지 않는다(그 경우 `humm-identify-estimate`·`humm-counterfactual`).

## 절차

1. 가정 명시: 판단할 명제·보기·척도의 뜻을 적고, `context`에 필요한 배경(그래프 요약·도메인)을 넣는다.
2. 그래프 확인: 그래프에 관한 판단이면 `humm.graph_get(graph_id)`에서 관련 엣지·노드를 `context`로 옮긴다.
3. 도구 호출을 고른다.
   - 예/아니오: `humm.fast_noul(proposition, context="")`
   - 선택: `humm.fast_choice(question, options=[...], context="")`
   - 척도: `humm.fast_score(question, levels=5, context="")`
4. 상향: `escalate`가 true인 결과는 쓰지 않는다. 수치 계산이 필요하면 `humm.ask`로, 그래프·데이터 위 판단이면 `humm.lm_verify(graph_id, csv, treatment, outcome)`로 올리고, 둘 다 안 되면 사용자에게 판단 불가로 알린다.
5. 상향 비율을 집계해 사용자에게 함께 보고한다.

## 흄 원칙 체크리스트

- [ ] 상관만으로 인과를 단정하지 않는다.
- [ ] 숨은 교란을 항상 점검한다.
- [ ] 반사실은 가정 위에서만 말한다.
- [ ] `escalate=true` 결과를 최종 답에 섞지 않는다.
- [ ] Fast 판단은 데이터 검증 증거가 아니므로 엣지 상태를 `verified`로 올리는 근거로 쓰지 않는다.

## 결과 해석

- `fast_noul`: `p_yes`(예일 확률), `confidence`, `escalate`이다.
- `fast_choice`: `choice`(고른 보기), `probs`(보기별 확률), `confidence`, `escalate`이다.
- `fast_score`: `score`(대표 점수), `dist`(점수별 확률), `confidence`, `escalate`이다.
- `confidence`는 보정된 값이므로 확률처럼 읽되, 외부 분포 문항에서는 보정이 덜 맞을 수 있다.

## 한계 고지

- Hume-Fast는 단일 패스 분류기이며 계산을 하지 않는다고 말한다.
- 판단마다 `confidence`를 함께 보고하고, 상향한 문항 수를 밝힌다.
- 학습 분포와 다른 문형(외부 셋)에서는 정확도가 수 pt 흔들릴 수 있다고 알린다.

## 다른 Skill로 넘기기

- 상향된 인과 질문 → `humm-causal-question` 또는 `humm-identify-estimate`
- 엣지 판정을 그래프에 반영 → `humm-validate-dag`
