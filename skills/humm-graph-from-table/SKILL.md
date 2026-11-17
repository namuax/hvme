---
name: humm-graph-from-table
description: CSV·DB 테이블에서 인과 그래프(HCG)를 만들 때 쓴다. "이 데이터로 인과 구조 만들어줘", "변수들 사이 인과 관계 찾아줘", "CSV 인과 분석" 같은 요청에 PC 탐색·LLM 쌍별 사전지식·데이터 검증을 거친 그래프를 저장한다.
---

## 언제 쓰나

- 정형 데이터(CSV)가 있고 저장된 그래프(`graph_list`)가 없거나 새로 만들어야 할 때 쓴다.
- 인과 질문에 답하기 전 그래프 확보 단계(오케스트레이터 ②)에 쓴다.

## 절차

1. 가정 명시: 열 이름·의미·측정 시점을 사용자에게 확인하고, 분석 단위(고객·일·지역)를 적는다. 엔진의 추정·반사실은 0/1 이진 열을 전제로 하므로 연속·범주형 열은 이진화가 필요한지 묻는다.
2. 그래프 구축: `humm.graph_from_table(csv, graph_id, domain="<분야 설명>", alpha=0.01, mode="hybrid", llm="")`를 부른다. `mode="hybrid"`는 PC 탐색에 LLM 쌍별 사전지식을 섞는다. `llm`이 비면 환경 변수 `HUMM_LLM`을 쓴다.
3. 그래프 확인: 반환된 `edges`를 사용자에게 보여 주고, 도메인 상식과 어긋나는 방향(예: 결과가 원인을 가리킴)을 짚는다. 다시 보려면 `humm.graph_get(graph_id)`를 쓴다.
4. 문서가 있으면 `humm-graph-from-text`로 텍스트 그래프를 만든 뒤 `humm.graph_merge(text_graph, table_graph, graph_id, column_aliases)`로 합친다.
5. 쓰기 전에 반드시 `humm-validate-dag`로 넘긴다.

## 흄 원칙 체크리스트

- [ ] 상관만으로 인과를 단정하지 않는다.
- [ ] 숨은 교란을 항상 점검한다.
- [ ] 반사실은 가정 위에서만 말한다.
- [ ] PC가 찾은 엣지는 조건부 독립 패턴이며, 방향은 사전지식·시간 순서에 기댄다는 점을 확인한다.
- [ ] 데이터에 없는 중요한 변수(측정 안 된 교란 후보)를 사용자에게 묻는다.

## 결과 해석

- `nodes`: 변수 id·`label`·`aliases`이다.
- `edges[]`: `src`→`dst`, `sign`, `status`(hypothesis/supported/verified/refuted), `confidence`, `evidence`(data/llm_prior/text/expert/experiment)이다.
- `evidence`가 `llm_prior`뿐인 엣지는 데이터 근거가 약한 방향 판정으로 본다.
- `verdict`: 구축 시 L1·L2 검증 판정(pass/warn/fail)이다.
- `conflicts`: 데이터와 LLM 사전지식이 충돌한 엣지 목록이며, 사용자 확인 대상이다.

## 한계 고지

- 관측 데이터만으로는 마르코프 동치류 안의 방향을 구별할 수 없고, 방향은 사전지식 가정에 기댄다고 말한다.
- `conflicts`와 `verdict`가 pass가 아니면 그 사실을 먼저 알린다.
- 측정되지 않은 교란은 그래프에 나타나지 않으므로 이후 결론이 그 가정에 기댄다고 말한다.

## 다른 Skill로 넘기기

- 다음 단계 → `humm-validate-dag`
- 문서 근거 추가 → `humm-graph-from-text`
- 엣지 방향 대량 확인 → `humm-fast-judge`(`fast_noul`)
