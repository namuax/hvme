import sys
import nbformat as nbf

cells = [
"""md:# 08. 얇은 엔드투엔드 — MCP 도구 + 오케스트레이터 v0

지금까지 만든 그래프 구축·검증·인과 엔진을 질문 하나의 흐름으로 잇는다(`05_development_guide.md` Step 7).

| 단계 | 내용 | 모듈 |
|---|---|---|
| 질의 분석 | 질문 → 유형(ATE·상관 대 인과·매개·ETT)·처치·결과·매개 변수. 규칙 기반, 범용 LLM이 있으면 JSON 질의 | `orchestrator.query` |
| 그래프 확보 | 정형 데이터 → HCG, 문서 → HCG, 병합 | `orchestrator.acquire` |
| 검증 | L1 구조·L2 데이터 반증, (실험 데이터가 있으면) L4 실험 대조 | `validate_hcg`, `experiment_check` |
| 식별·추정 | 백도어 → 프론트도어 → IV, 부트스트랩 구간. Rung 3은 데이터로 적합한 이진 SCM | `engine`, `BinarySCM.fit` |
| 반박 | 위약 처치·무작위 공통 원인·부분 표본 | `refute` |
| 그래프 민감도(L5) | 엣지 하나씩 지우기·뒤집기·반증 엣지 복원·숨은 교란 추가 → 결론이 바뀌는가 | `orchestrator.sensitivity` |
| 설명 | 답·근거 경로(엣지 상태·증거·인용문)·가정 목록. LLM은 요약 문단만 쓴다 | `orchestrator.explain` |

같은 기능을 `humm.api`(경로 기반 함수)와 `humm.mcp`(MCP 서버)로 노출한다. 1~3장과 5·6장은 CPU, 4장은 GPU에서 실행한다.""",
"""import pandas as pd
from IPython.display import Markdown, display

from humm.graph.text import TextOracle
from humm.orchestrator import Orchestrator
from humm.orchestrator.demo import QUESTIONS, marketing, scorecard
from humm.paths import DATA, GRAPHS, RESULTS

pd.set_option("display.max_colwidth", 60)
s = marketing()
files = s.save(DATA / "demo" / "marketing")
files""",
"""md:## 1. 데모 시나리오 (마케팅)

정답 SCM에서 관찰 데이터 8,000행, 광고비를 무작위 배정한 A/B 테스트 3,000행, 문서 12개를 만든다. 성수기가 광고비·할인·매출을 함께 움직이는 **교란 변수**이고, 광고비 효과의 일부는 **웹 방문을 거친다**. 질문 6개의 정답은 정답 SCM에서 정확히 계산한다.""",
"""print("정답 엣지:", [(a, b, d["sign"]) for a, b, d in s.graph.edges(data=True)])
print("질문:", *QUESTIONS, sep="\\n- ")
pd.Series(s.truth()).round(4)""",
"""md:## 2. (CPU) 가상 LLM으로 끝까지

텍스트 추출은 가상 LLM(오류율 0.1)이 맡고, 질의 분석과 설명은 규칙·템플릿만 쓴다. 그래프 확보 → 검증은 한 번 하고, 질문마다 식별 → 추정 → 반박 → 민감도 → 설명을 돈다.""",
"""orch = Orchestrator.from_sources(s.data, s.corpus.docs, "marketing", TextOracle(s.corpus, error=0.1), s.column_aliases, "marketing-demo",
                                 experiments={"ad_spend": s.experiment})
print("검증 판정:", orch.report.verdict, "| 정답 엣지와 같은가:", {e.key for e in orch.hcg.active_edges()} == set(s.graph.edges))
orch.changes""",
"""answers = [orch.ask(q) for q in QUESTIONS]
card = scorecard(answers, s)
card""",
"""md:`parsed_ok`는 질문 해석이 기대(유형·처치·결과)와 같은지, `covered`는 정답이 95% 구간 안에 있는지이다. `alternatives`는 민감도 분석의 대안 그래프 수, `flipped`·`vanished`는 결론의 부호가 뒤집히거나 효과가 0이 된 대안 수이다.

## 3. 답변 전체

질문마다 답·근거 경로·반박·그래프 민감도·가정이 자동으로 나온다(완료 기준).""",
"""for a in answers:
    display(Markdown(a.markdown + "\\n\\n---"))""",
"""md:## 4. (GPU) Qwen3.5로 그래프 구축 + 질의 분석 + 설명

같은 흐름에서 가상 LLM 자리를 실제 모델로 바꾼다. 모델 하나가 텍스트 추출·정규화·반사실 검증(그래프 확보), 질의 분석, 설명 요약 문단을 모두 맡는다. 응답은 `results/llm_cache/`에 기록되므로 Mac에서 재생할 수 있다.

- 질의 분석: 변수는 질문 문자열에서 찾은 표현(규칙)으로 정하고, LLM은 변수를 못 찾았을 때와 유형 키워드가 없을 때만 쓴다.
- 요약 문단: 근거에 없는 숫자·'~다'로 끝나지 않는 문장·내부 이름(변수 id, `_U`)이 있으면 답에 싣지 않는다(`narrative_ok`). 원문과 검사 결과는 `results/e2e/<모델>_narratives.json`에 남긴다.

> 1차 실행(2026-10-25): 두 모델 모두 그래프를 정확히 복원했고(정답 엣지 10개, 판정 pass) 수치는 가상 LLM 결과와 같았다. 다만 LLM 질의 분석은 '고객 이탈률이 광고비에 영향을 주는가'의 방향을 두 모델 모두 뒤집었고(4B는 ETT도 ATE로 분류), 요약 문단 12개 중 11개가 합쇼체·내부 이름·근거에 없는 숫자 문제로 검사를 통과하지 못했다. 위 두 규칙은 이 결과로 정했다.
>
> 2차 실행(2026-10-25): 질의 해석은 두 모델 모두 6/6이 되었다. 요약은 방향·내부 이름 문제가 사라졌지만, 지시해도 대부분 합쇼체로 써서 검사 통과가 4B 1/6, 9B 0/6이었다. 합쇼체를 '~다'체로 바꾸는 후처리(`explain.to_plain`)를 넣어 같은 기록을 재생하면 4B 5/6, 9B 6/6이다(`reports/2026-10-25_e2e.md` 5장).""",
"""gpu:import gc
import json

import torch

from humm.providers.cache import CachedProvider
from humm.providers.hf import HFProvider

MODELS = {
    "Qwen/Qwen3.5-4B": "cuda:0",
    "Qwen/Qwen3.5-9B": "cuda:1",
}
(RESULTS / "e2e").mkdir(parents=True, exist_ok=True)
cards = []
for model_id, device in MODELS.items():
    name = model_id.split("/")[-1]
    lm = CachedProvider(HFProvider(model_id, device=device, batch_size=8), RESULTS / "llm_cache" / f"{name}.json")
    o = Orchestrator.from_sources(s.data, s.corpus.docs, "marketing", lm, s.column_aliases, f"marketing-demo-{name}", llm=lm,
                                  experiments={"ad_spend": s.experiment})
    ans = [o.ask(q) for q in QUESTIONS]
    cards.append(scorecard(ans, s).assign(model=name, verdict=o.report.verdict,
                                          graph_exact={e.key for e in o.hcg.active_edges()} == set(s.graph.edges)))
    (RESULTS / "e2e" / f"{name}.md").write_text("\\n\\n---\\n\\n".join(a.markdown for a in ans), encoding="utf-8")
    (RESULTS / "e2e" / f"{name}_narratives.json").write_text(json.dumps([{"question": a.analysis.query.text, "draft": a.draft, "issues": a.issues} for a in ans],
                                                                        ensure_ascii=False, indent=1), encoding="utf-8")
    o.hcg.save(RESULTS / "validation" / f"marketing-demo-{name}.hcg.json")
    lm.save()
    del lm, o
    gc.collect()
    torch.cuda.empty_cache()

e2e = pd.concat(cards)
e2e.to_csv(RESULTS / "e2e_qwen35.csv", index=False)
e2e.groupby("model").agg(parsed_ok=("parsed_ok", "mean"), narrative_ok=("narrative_ok", "mean"), abs_error=("abs_error", "mean"),
                         covered=("covered", "mean"), verdict=("verdict", "first"), graph_exact=("graph_exact", "first"))""",
"""gpu:pd.DataFrame([{"model": m, **d} for m in ("Qwen3.5-4B", "Qwen3.5-9B")
              for d in json.loads((RESULTS / "e2e" / f"{m}_narratives.json").read_text(encoding="utf-8"))])""",
"""gpu:display(Markdown((RESULTS / "e2e" / "Qwen3.5-9B.md").read_text(encoding="utf-8").split("\\n\\n---\\n\\n")[2]))""",
"""md:## 5. (CPU) 기록 재생

GPU에서 기록한 9B 응답을 재생해 4장과 같은 결과가 나오는지 확인한다. GPU 실행 전에는 기록이 없어 `KeyError`가 난다.""",
"""replay:from humm.providers.cache import ReplayProvider

rp = ReplayProvider(RESULTS / "llm_cache" / "Qwen3.5-9B.json", name="Qwen3.5-9B")
o = Orchestrator.from_sources(s.data, s.corpus.docs, "marketing", rp, s.column_aliases, "marketing-demo-Qwen3.5-9B", llm=rp,
                              experiments={"ad_spend": s.experiment})
replayed = scorecard([o.ask(q) for q in QUESTIONS], s)
gpu = pd.read_csv(RESULTS / "e2e_qwen35.csv").query("model == 'Qwen3.5-9B'").reset_index(drop=True)
print("수치가 GPU 결과와 같은가:", (replayed.value.round(4).values == gpu.value.round(4).values).all())
print("요약 검사 통과 — GPU 기록:", int(gpu.narrative_ok.sum()), "/ 재생:", int(replayed.narrative_ok.sum()))  # 문체 정리 등 후처리를 바꾸면 재생 쪽만 달라진다
replayed""",
"""md:## 6. MCP 서버

`humm.mcp`는 `humm.api` 함수를 MCP 도구로 노출한다. Claude Desktop·Claude Code에 아래처럼 등록하면, 에이전트가 도구를 불러 그래프를 만들고 질문에 답한다(`pip install -e ".[mcp]"`).

```json
{"mcpServers": {"humm": {"command": "python", "args": ["-m", "humm.mcp"], "cwd": "<humm 저장소 경로>",
                         "env": {"HUMM_LLM": "hf:Qwen/Qwen3.5-9B@cuda:1"}}}}
```

| MCP 도구 | 제안서 이름 | 내용 |
|---|---|---|
| `graph_from_table`, `graph_from_text`, `graph_merge`, `graph_get`, `graph_list` | `humm.graph.*` | 그래프 구축·병합·조회 |
| `validate_run` | `humm.validate.run` | L1·L2 검증, 엣지 상태 갱신 |
| `engine_identify`, `engine_estimate`, `engine_refute`, `engine_counterfactual` | `humm.engine.*` | 식별·추정·반박·반사실 |
| `ask` | (오케스트레이터) | 질문 하나를 끝까지 처리 |

아래는 같은 함수를 노트북에서 바로 부르는 예이다(LLM 없이, 2장 그래프 사용).""",
"""from humm import api
from humm.mcp.server import mcp

orch.hcg.save(GRAPHS / "marketing-demo.hcg.json")
print("MCP 도구:", [t.name for t in await mcp.list_tools()])
print(api.engine_identify("marketing-demo", "ad_spend", "sales", files["csv"]))
print(api.engine_counterfactual("marketing-demo", files["csv"], "ad_spend", "sales", "nie", ["web_traffic"]))
print(api.ask(QUESTIONS[0], "marketing-demo", files["csv"], files["experiment_csv"], "ad_spend")["answer"])""",
"""md:## Mac으로 옮길 파일

실행한 노트북은 원본과 다른 이름으로 저장하고, `notebooks/`의 원본은 출력 없이 둔다.

| 파일 | 필수 여부 | 비고 |
|---|---|---|
| 이 노트북 | 필수 | `runs/<날짜>-gpu/` |
| `results/e2e_qwen35.csv` | 필수 | 4장: 모델별 질문 채점표 |
| `results/e2e/Qwen3.5-4B.md`, `Qwen3.5-9B.md` | 필수 | 4장: 질문별 전체 답변(검사를 통과한 LLM 요약 문단 포함) |
| `results/e2e/Qwen3.5-4B_narratives.json`, `Qwen3.5-9B_narratives.json` | 필수 | 4장: 요약 문단 원문과 검사 결과 |
| `results/llm_cache/Qwen3.5-4B.json`, `Qwen3.5-9B.json` | 필수 | LLM 응답 기록(07장 기록에 추가됨). 5장 재생에 쓴다 |
| `results/validation/marketing-demo-Qwen3.5-4B.hcg.json`, `marketing-demo-Qwen3.5-9B.hcg.json` | 선택 | 모델별 검증된 HCG |

`data/demo/`, `data/graphs/marketing-demo.hcg.json`은 노트북이 다시 만들므로 옮기지 않아도 된다.""",
]

n = nbf.v4.new_notebook()
out = []
for c in cells:
    tag = next((t for t in ("md", "gpu", "replay") if c.startswith(t + ":")), None)
    body = c[len(tag) + 1:] if tag else c
    cell = nbf.v4.new_markdown_cell(body) if tag == "md" else nbf.v4.new_code_cell(body)
    cell.metadata["tags"] = [tag] if tag in ("gpu", "replay") else []
    out.append(cell)
n.cells = out
n.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(n, sys.argv[1])
