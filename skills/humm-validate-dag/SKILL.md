---
name: humm-validate-dag
description: 구축한 인과 그래프를 분석에 쓰기 전에 데이터로 검증할 때 쓴다. "그래프 맞는지 확인해줘", "DAG 검증", "이 구조가 데이터와 맞나?" 같은 요청에 L1 구조·L2 조건부 독립 반증을 실행하고 엣지 상태를 갱신하며 validation.json을 남긴다.
---

## 언제 쓰나

- `humm-graph-from-table`·`humm-graph-from-text`·`graph_merge` 직후, 추정·반사실 전에 항상 쓴다.
- 데이터가 바뀌었거나 그래프를 손으로 고친 뒤 다시 쓴다.

## 절차

1. 가정 명시: 검증에 쓸 데이터가 그래프를 만든 데이터와 같은 모집단인지 적는다.
2. 그래프 확인: `humm.graph_get(graph_id)`로 노드가 CSV 열과 맞는지 본다(열이 없는 노드는 검정에서 빠진다).
3. 검증: `humm.validate_run(graph_id, csv, n_perm=100)`을 부른다. 결과는 그래프에 저장되고 `results/validation/<graph_id>.json`에 남는다.
4. 위반 처리: `missing_edge_candidates`를 사용자에게 보여 주고 빠진 엣지나 교란 변수를 추가할지 정한다. 고치면 그래프를 다시 만들고 3을 반복한다.
5. 실험 데이터가 있으면 `humm.ask(..., experiment_csv, experiment_treatment)`에서 L4 실험 대조가 붙는다고 안내한다.

## 흄 원칙 체크리스트

- [ ] 상관만으로 인과를 단정하지 않는다.
- [ ] 숨은 교란을 항상 점검한다.
- [ ] 반사실은 가정 위에서만 말한다.
- [ ] 검증 통과는 "반증되지 않았다"는 뜻이지 그래프가 참이라는 증명이 아님을 확인한다.
- [ ] `refuted`로 바뀐 엣지가 질문의 처치→결과 경로에 있는지 본다.

## 결과 해석

- `summary.verdict`: `pass`(위반이 우연 수준이고 강한 위반 없음, 무작위 그래프보다 잘 맞음), `warn`, `fail`(사이클·시간 순서 위반)이다.
- `summary.is_dag`, `temporal_violations`: L1 구조 검사이다.
- `summary.lmc_tests`, `lmc_violation_rate`, `excess_violation_p`, `strong_violations`: L2 국소 마르코프 조건 검정이다.
- `summary.permutation_p`, `permutation_informative`: 무작위 그래프 대비 적합도이며, 검정 수가 적으면 정보가 없다.
- `summary.edges_tested`, `edges_unsupported`: 엣지별 데이터 지지 여부이다.
- `missing_edge_candidates`: `[node, other, given, p]` 목록으로, 그래프가 독립이라 했지만 데이터가 의존을 보이는 쌍이다.
- `changes`: 엣지 상태 변경 기록이다.
- `notes`: 검증 중 남긴 주의 사항이다.

## 한계 고지

- L2 검정은 관측된 열 사이의 독립 관계만 보며, 같은 독립 구조를 갖는 다른 그래프(동치류)는 구별하지 못한다고 말한다.
- 표본이 작으면 위반을 놓칠 수 있다고 `n_samples`와 함께 말한다.
- `verdict`가 `warn`·`fail`이면 이후 모든 결론에 그 사실을 붙인다.

## 다른 Skill로 넘기기

- 통과 → `humm-identify-estimate` 또는 `humm-counterfactual`
- 위반 다수 → `humm-graph-from-table`(교란 변수·엣지 추가 후 재구축)
- 보고 → `humm-explain`
