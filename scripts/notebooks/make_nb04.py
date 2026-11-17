import sys
import nbformat as nbf

cells = [
"""md:# 04. Hume-Gym — 검증 가능한 인과 추론 문제 생성기

Hume-Gym은 Hume-LM과 Hume-Fast의 학습·평가 데이터를 만든다. 문제마다 무작위 인과 그래프와 SCM을 만들고, 정답은 `humm.engine`(노트북 03)이 계산한다. 정답을 기호적으로 확인할 수 있기 때문에 SFT 데이터 선별(rejection sampling)과 GRPO 보상에 그대로 쓸 수 있다.

| 질의 유형 | Rung | 형식 | 내용 |
|---|---|---|---|
| marginal · correlation | 1 | Noul | 주변 확률, 관측 조건부 차이 |
| dsep · corr2cause | 1 | Noul | d-분리, 통계적 관계만 보고 인과 가설 판정(마르코프 동치류 전수 계산) |
| ate · cde · backadj | 2 | Noul | 개입 효과, 통제 직접효과, 조정 집합 판정 |
| identify | 2 | Choice | 백도어·프론트도어·IV·식별 불가·효과 없음 |
| ate_score | 2 | Score | 개입 효과의 방향과 크기 5단계 |
| ett · nde · nie | 3 | Noul | 처치군 효과, 자연 직접·간접효과 |

설계 원칙은 다음과 같다.
- **정답 비율 50:50**: 질문 방향(높아지는가/낮아지는가)을 골라 맞춘다.
- **애매한 문제 제거**: 효과가 판정 기준에서 5%p 이상 떨어진 SCM만 쓴다.
- **상식 단축 방지**: 변수 이름을 무작위로 배치해, 상식으로 짐작하지 못하고 주어진 그래프와 확률로만 풀게 한다.
- **분할 분리**: 학습은 마케팅·제조·의료 어휘, 테스트는 IT 운영 어휘와 무의미 이름을 쓴다.
- **문항 비중**: 기준선에서 약했던 nie·nde·ate·반사실 문항을 많이 넣는다.""",
"""import pandas as pd

from humm.gym import GENERATORS, audit, build, generate, render, summary, target_text
from humm.paths import GYM

pd.set_option("display.max_colwidth", 120)""",
"""md:## 1. 문항 예시

같은 질의 유형을 한국어와 영어로 하나씩 본다. `render(item, "direct")`는 답 토큰 하나만 받는 형식(Hume-Fast), `render(item, "cot")`는 단계별 풀이를 요구하는 형식(Hume-LM)이다.""",
"""sample = generate(240, "train", seed=42)
ko_nie = next(d for d in sample if d["query_type"] == "nie" and d["lang"] == "ko")
print(render(ko_nie, "cot"))
print("\\n정답:", target_text(ko_nie), "| 참값 NIE =", ko_nie["value"])""",
"""en_identify = next(d for d in sample if d["query_type"] == "identify" and d["lang"] == "en")
print(render(en_identify, "direct"))
print("\\n정답:", target_text(en_identify), "|", en_identify["meta"]["strategy"])""",
"""ko_c2c = next(d for d in sample if d["query_type"] == "corr2cause" and d["lang"] == "ko")
print(render(ko_c2c, "direct"))
print("\\n정답:", target_text(ko_c2c), "| 동치류 크기:", ko_c2c["meta"]["mec_size"])""",
"""md:## 2. 정답 재검증

문항에 남긴 그래프·확률표·메타 정보만으로 정답을 처음부터 다시 계산해 저장된 정답과 대조한다. 불일치 목록이 비어 있어야 한다.""",
"""bad = audit(sample)
print("불일치:", len(bad))
summary(sample).round(2)""",
"""md:## 3. 데이터셋 생성

`build`는 train → dev → test 순서로 만들고, 앞 분할에 나온 문항은 뒤 분할에서 뺀다. 결과는 `data/gym/{train,dev,test}.jsonl`에 저장된다. 10만 문항 생성에 CPU로 약 7분 걸린다.""",
"""SIZES = {"train": 100_000, "dev": 2_000, "test": 5_000}
stats = build(SIZES, seed=0)
stats.pivot_table(index="query_type", columns="split", values="n", aggfunc="sum")""",
"""from humm.gym import read_jsonl

test = read_jsonl(GYM / "test.jsonl")
print("test 재검증 불일치:", len(audit(test)))
pd.DataFrame(test).groupby(["domain", "lang"]).size().unstack()""",
"""md:## 4. (GPU) Qwen3.5 기준선 — Hume-Gym test

학습 전에 두 모델이 Hume-Gym에서 몇 점인지 잰다. direct는 test 전체, cot는 시간이 오래 걸리므로 질의 유형별로 균형을 맞춘 600문항만 쓴다. 이 점수가 Step 8(Hume-Fast)·Step 9(Hume-LM)의 출발점이다.""",
"""import gc

import torch

from humm.bench import runner as R
from humm.gym import eval_cot, eval_direct, eval_summary
from humm.providers.hf import HFProvider

MODELS = {
    "Qwen/Qwen3.5-4B": "cuda:0",
    "Qwen/Qwen3.5-9B": "cuda:1",
}
BATCH = 16
cot_items = pd.DataFrame(test).groupby("query_type").head(50).to_dict("records")
len(test), len(cot_items)""",
"""results = []
for model_id, device in MODELS.items():
    lm = HFProvider(model_id, device=device, batch_size=BATCH)
    results += [eval_direct(lm, test), eval_cot(lm, cot_items, max_new_tokens=4096)]
    del lm
    gc.collect()
    torch.cuda.empty_cache()

res = pd.concat(results, ignore_index=True)
R.save(res.drop(columns=["context", "probs"]), "gym_baseline_qwen35")""",
"""eval_summary(res, by=("rung",)).round(3)""",
"""eval_summary(res).pivot_table(index="query_type", columns=["model", "mode"], values="acc").round(3)""",
]

n = nbf.v4.new_notebook()
n.cells = [nbf.v4.new_markdown_cell(c[3:]) if c.startswith("md:") else nbf.v4.new_code_cell(c) for c in cells]
n.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(n, sys.argv[1])
