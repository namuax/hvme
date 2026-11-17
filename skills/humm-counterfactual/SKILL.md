---
name: humm-counterfactual
description: 사용자가 "만약 그때 ~했다면?", "~하지 않았다면 결과가 달랐을까?", "효과 중 얼마가 ~를 거쳐 왔나?" 같은 Rung 3 질문에 쓴다. 데이터로 적합한 이진 SCM에서 ETT·NDE·NIE·PN·PS를 계산하고 가정과 함께 서술한다.
---

## 언제 쓰나

- `humm-causal-question`이 Rung 3(`kind="ett"`·`"mediation"`)으로 분류했거나 필요성·충분성 확률(PN·PS)을 물을 때 쓴다.
- 처치·결과·매개 변수가 모두 0/1 이진 열로 데이터에 있을 때만 쓴다.

## 절차

1. 가정 명시: 사실 세계(실제로 일어난 일)와 반사실 세계(바꾼 처치)를 문장으로 나눠 적는다.
2. 그래프 확인: `humm.graph_get(graph_id)`로 처치·결과·매개의 조상이 데이터에 있는지 보고, `humm.engine_identify(graph_id, treatment, outcome, csv)`로 개입 효과가 식별되는지 먼저 확인한다.
3. 반사실 계산: `humm.engine_counterfactual(graph_id, csv, treatment, outcome, query="ett", mediators=None)`을 부른다. `query`는 `ate`, `ett`, `nde`, `nie`, `pn`, `ps` 중 하나이고, `nde`·`nie`는 `mediators=[...]`가 필요하다.
4. 구간·민감도가 필요하면 `humm.ask(question, graph_id, csv)`를 불러 `estimates`(ett 또는 nde·nie, ate_scm)와 `sensitivity_table`을 받는다.
5. 식별이 깨지면(`strategy="not_identified"`) 반사실 숫자를 내지 않는다.

## 흄 원칙 체크리스트

- [ ] 상관만으로 인과를 단정하지 않는다.
- [ ] 숨은 교란을 항상 점검한다.
- [ ] 반사실은 가정 위에서만 말한다.
- [ ] 반사실 값이 SCM 메커니즘 가정(문턱값·단조성, 노드별 독립 잡음)에 기댄다는 문장을 답에 넣는다.
- [ ] 매개 분해에서는 매개–결과 사이에 처치의 영향을 받는 교란이 없다는 가정을 확인한다.
- [ ] ETT 부호는 "실제 − 반사실"인지, 사용자 질문의 방향과 맞는지 확인한다.

## 결과 해석

- `engine_counterfactual`: `query`, `value`, `exact`(true면 정확 계산, false면 몬테카를로 근사), `nodes`(SCM에 쓴 노드)이다.
- `ask` 결과: `kind`가 `ett`이면 `estimates.ett`, `mediation`이면 `estimates.nde`·`estimates.nie`, 공통으로 `estimates.ate`·`estimates.ate_scm`이 있고 각 항목은 `value`·`ci`이다. `mediators`, `assumptions`, `graph_sensitivity`도 함께 읽는다.
- `pn`·`ps`는 `ask`가 다루지 않으므로 `engine_counterfactual`로만 구하며 구간이 없다.

## 한계 고지

- 반사실은 관측 데이터만으로 식별되지 않는 양이며, 적합한 이진 SCM 가정 위에서만 성립한다고 반드시 말한다.
- 엔진은 이진 변수만 지원하므로 연속 변수 질문은 이진화 기준을 밝히거나 "식별 불가 + 필요한 데이터"로 답한다.
- `exact=false`이면 근사값이라고 말한다.
- 숨은 교란 대안에서 매개·ETT 식별이 깨지는지(`graph_sensitivity.breaks_with_hidden_confounder`) 말한다.

## 다른 Skill로 넘기기

- 결과 검증 → `humm-refute`
- 보고 → `humm-explain`
- 개입 효과만 필요하면 → `humm-identify-estimate`
