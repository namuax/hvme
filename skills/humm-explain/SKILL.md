---
name: humm-explain
description: 인과 분석 결과를 사용자에게 보고할 때 쓴다. "결과 정리해줘", "보고서로 써줘", "왜 그런 결론인지 설명해줘" 같은 요청에 ask의 마크다운에서 근거 경로·가정·신뢰도·그래프 민감도(L5)를 빠짐없이 담은 리포트를 만든다.
---

## 언제 쓰나

- 다른 humm Skill로 숫자를 얻은 뒤 최종 답을 쓸 때 항상 쓴다.
- 이미 쓴 답에 근거·가정이 빠졌는지 점검할 때 쓴다.

## 절차

1. 가정 명시: 질문·그래프(graph_id)·데이터(csv)·검증 판정을 첫머리에 적는다.
2. 그래프 확인: `humm.graph_get(graph_id)`로 근거 경로 엣지의 `status`·`confidence`를 다시 확인한다.
3. 결과 받기: `humm.ask(question, graph_id, csv)`를 부르고 `markdown`을 받는다. 실험 데이터가 있으면 `experiment_csv`·`experiment_treatment`를 넣어 L4 대조를 붙인다.
4. 필수 항목 점검: `markdown`에 아래 네 가지가 모두 있는지 확인하고, 하나라도 빠지면 리포트를 내지 않고 해당 도구를 다시 부른다.
   - 근거 경로(`### 근거 경로`, 필드 `paths`)
   - 가정(`### 가정`, 필드 `assumptions`)
   - 신뢰도(`estimates`의 `ci`, 엣지 `confidence`, 검증 `verdict`, `refutation`의 `passed`)
   - 그래프 민감도(`### 그래프 민감도 (L5)`, 필드 `sensitivity_table`·`graph_sensitivity`)
5. 요약 문단: `narrative`가 비어 있지 않으면 쓰고, `issues`가 있으면 LLM 요약을 버리고 `answer`와 표만 쓴다. 숫자는 결과에 있는 값만 옮긴다.
6. 그림이 필요하면 `paths`의 엣지로 근거 경로를 텍스트 도식(`A → B → C`)으로 그린다.

## 흄 원칙 체크리스트

- [ ] 상관만으로 인과를 단정하지 않는다.
- [ ] 숨은 교란을 항상 점검한다.
- [ ] 반사실은 가정 위에서만 말한다.
- [ ] 리포트의 모든 숫자가 도구 출력에 있는 값인지 확인한다.
- [ ] 95% 구간의 양 끝을 효과 크기로 해석하지 않는다.
- [ ] `verified`가 아닌 엣지에 기댄 결론이면 그 엣지를 이름으로 적는다.

## 결과 해석

- `answer`: 엔진 결과로 만든 한 줄 답이며 리포트 첫 문장으로 쓴다.
- `kind`·`identification`·`strategy`·`expression`·`adjustment`: 질문 유형과 식별 방법이다.
- `estimates`: `{이름: {"value", "ci"}}`이며 `ate`, `association`, `ett`, `nde`, `nie`, `ate_scm`이 유형에 따라 있다.
- `paths`: 경로마다 엣지 목록(`edge`, `status`, `confidence`, `sign`, `evidence`, `quote`)이다.
- `backdoor_paths`: 조정으로 막은 교란 경로 후보이다.
- `refutation`: 반박 테스트 행(`test`, `expected`, `new_value`, `passed`)이다.
- `graph_sensitivity`·`sensitivity_table`: 대안 그래프 수, 부호 뒤집힘, 효과 소멸, 숨은 교란에서 식별 붕괴 여부이다.
- `experiment`: L4 실험 대조 결과(`consistent` 포함)이며 실험 데이터를 준 경우에만 있다.
- `parser`: 질의 분석 방식(`rules`, `rules+<llm>`, v1의 `rules+rung:fast`·`rules+rung:lm`)이다.
- `route`·`lm_check`(`ask_v1`만): Rung 분류(확신도, fast/lm)와 Humm-LM 독립 풀이(`lm_answer`, `agree`)이다. `agree`가 false면 결론을 내리기 전에 그래프와 가정을 다시 보라고 적고, null이면 판단 보류라고 적는다. markdown에는 "Humm-LM 검증" 절로 들어 있다.

## 한계 고지

- `assumptions` 목록을 번호 그대로 싣는다.
- 신뢰구간, 검증 판정(`pass`/`warn`/`fail`), 반박 실패 수, 부호가 뒤집힌 대안 그래프 수를 반드시 적는다.
- 식별 불가면 "식별 불가 + 필요한 데이터(측정해야 할 변수·실험)"를 답으로 쓴다.
- 반사실·매개 수치는 이진 SCM 가정 위의 값이라고 적는다.

## 다른 Skill로 넘기기

- 근거가 부족하면 → `humm-refute` 또는 `humm-validate-dag`
- 새 질문 → `humm-causal-question`
