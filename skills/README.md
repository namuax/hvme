# Hume-Skills

Hume 인과 도구(humm-mcp)를 에이전트가 올바른 순서와 원칙으로 쓰게 하는 Skill 모음이다.
모든 Skill은 "가정 명시 → 그래프 확인 → 도구 호출 → 결과 해석 → 한계 고지" 골격을 따르고, 흄 원칙(상관만으로 인과를 단정하지 않는다, 숨은 교란을 항상 점검한다, 반사실은 가정 위에서만 말한다)을 체크리스트로 담는다.

## Skill 목록

| 이름 | 언제 쓰나 | 주요 도구 |
|---|---|---|
| `humm-causal-question` | "왜?", "~하면 어떻게 돼?" 류 인과 질문의 진입점 | `classify_question`, `graph_list`, `graph_get`, `ask` |
| `humm-graph-from-table` | CSV·DB 테이블로 인과 그래프를 만들 때 | `graph_from_table`, `graph_get`, `graph_merge` |
| `humm-graph-from-text` | 보고서·논문·로그에서 인과 구조를 뽑을 때 | `graph_from_text`, `graph_merge` |
| `humm-validate-dag` | 그래프를 분석에 쓰기 전 | `validate_run`, `graph_get` |
| `humm-identify-estimate` | 개입 효과(ATE) 계산 | `engine_identify`, `engine_estimate`, `ask`, `lm_verify` |
| `humm-refute` | 결과 검증·민감도 분석 | `engine_refute`, `ask`, `lm_verify` |
| `humm-fast-judge` | 대량의 예/아니오·선택·척도 판단 | `fast_noul`, `fast_choice`, `fast_score` |
| `humm-counterfactual` | "만약 그때 ~했다면?", 매개 효과 | `engine_counterfactual`, `engine_identify`, `ask` |
| `humm-explain` | 최종 결과 보고 | `ask`(`markdown`), `graph_get` |

기본 흐름은 `humm-causal-question` → (`humm-graph-from-table` / `humm-graph-from-text`) → `humm-validate-dag` → `humm-identify-estimate` 또는 `humm-counterfactual` → `humm-refute` → `humm-explain`이다.

## Claude Code에서 쓰기

1. MCP 서버를 설치한다.

```bash
pip install -e ".[mcp]"
```

2. Skill을 복사하거나 심볼릭 링크로 건다. 사용자 전역은 `~/.claude/skills/`, 프로젝트 한정은 `<프로젝트>/.claude/skills/`이다.

```bash
mkdir -p ~/.claude/skills
ln -s "$PWD"/skills/hume-* ~/.claude/skills/        # 또는 cp -r skills/hume-* ~/.claude/skills/
```

3. MCP 서버를 등록한다(`humm/mcp/server.py` 설명과 같다). 서버 이름은 `humm`이어야 Skill 본문의 `humm.<도구>` 표기와 맞는다.

```json
{"mcpServers": {"humm": {"command": "python", "args": ["-m", "humm.mcp"], "cwd": "<humm 저장소>", "env": {"HUMM_LLM": "anthropic:claude-sonnet-5-5"}}}}
```

프로젝트 `.mcp.json`에 넣거나 `claude mcp add`로 등록한다. Claude Code는 각 SKILL.md의 `description`을 보고 알맞은 Skill을 고른다.

## 사내 에이전트에서 쓰기

Claude가 아닌 사내 에이전트도 같은 SKILL.md 본문을 시스템 프롬프트의 한 절로 넣고 같은 MCP 서버(`python -m humm.mcp`, stdio)를 붙이면 된다.
Skill을 고르는 기능이 없으면 `humm-causal-question`과 `humm-explain` 본문을 항상 넣고, 나머지는 질문 유형에 따라 붙인다.
