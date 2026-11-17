import sys
import nbformat as nbf

cells = [
"""md:# 07. 텍스트 → 인과 그래프 — 추출 · 정규화 · 집계 · 반사실 검증

보고서·문서 같은 비정형 텍스트에서 인과 그래프(HCG)를 만들고, 정형 데이터 그래프와 합친다(`02_proposal.md` 4.2).

| 단계 | 내용 | 모듈 |
|---|---|---|
| T1 | 문서를 문장 경계에서 조각으로 나눈다 | `chunk` |
| T2·T3 | 조각마다 LLM이 관계 주장(인과·상관·무효과, 증감, 불확실 여부)을 JSON으로 뽑는다 | `extract_claims` |
| T5 | 같은 변수의 여러 표현을 정규 이름으로 묶는다 | `canonicalize` |
| T6 | 문서 단위로 주장을 세어 엣지 후보와 점수를 만든다. 반대 방향·무효과 주장은 반대 증거이다 | `aggregate` |
| T7 | "원인이 바뀌면 결과가 바뀌는가"를 정방향·역방향으로 묻고, 확률을 LLM 증거로 붙인다(거르지는 않는다) | `verify_edges` |
| 병합 | 텍스트 HCG를 정형 HCG에 붙이고 데이터로 검증한다 | `merge_hcg`, `validate_hcg` |

정답 그래프가 있어야 성능을 잴 수 있으므로, **합성 코퍼스**로 평가한다. 정답 DAG에서 문서를 만들되, 동의어·여러 문형·상관 진술(방해)·무효과 진술·불확실 표현·틀린 주장을 담은 문서를 섞는다. 1~4장과 6장은 CPU(가상 LLM 또는 기록 재생), 5장은 GPU에서 실행한다.""",
"""import networkx as nx
import pandas as pd

from humm.engine import BinarySCM
from humm.graph.tabular import tabular_to_hcg
from humm.graph.text import TEXT_VOCAB, TextOracle, evaluate, link_nodes, make_corpus, merge_hcg, text_to_hcg
from humm.paths import RESULTS
from humm.validate import validate_hcg

pd.set_option("display.max_colwidth", 80)
DOMAINS, LANGS = ["marketing", "manufacturing", "it_ops"], ["ko", "en"]
corpora = {(d, l): make_corpus(d, l, seed=1) for d in DOMAINS for l in LANGS}""",
"""md:## 1. 합성 코퍼스

도메인 3개 × 언어 2개, 코퍼스마다 문서 12개이다. 정답 엣지 9개가 각각 문서 3개에 등장하고, 문서의 15%는 틀린 주장(뒤집힌 방향, 없는 엣지)을 담는다. 정답 엣지는 도메인의 그럴듯한 인과 순서를 따른다.""",
"""c = corpora[("marketing", "ko")]
print("정답 엣지:", [(a, b, d["sign"]) for a, b, d in c.graph.edges(data=True)])
print("\\n문서 예시:", c.docs[0]["text"])
pd.DataFrame([{"문장": s, **{k: n[k] for k in ("cause", "effect", "relation", "polarity", "hedged")}, "틀린 주장": n.get("wrong", False)}
              for s, n in list(c.annotations.items())[:12]])""",
"""md:## 2. 파이프라인 한 단계씩 (가상 LLM, 오류율 0.1)

`TextOracle`은 정답을 알지만 일부러 틀리는 가상 LLM이다. 추출에서 주장을 빠뜨리거나 방향을 뒤집고, 정규화에서 표현을 잘못 떼어 놓고, 반사실 검증에서 답을 뒤집는다.""",
"""r = text_to_hcg(c.docs, "marketing-ko", "marketing", TextOracle(c, 0.1, seed=3))
print("추출된 주장:", len(r.claims), "| 정규 변수:", len(set(r.mapping.values())))
r.claims[["doc_id", "cause", "effect", "relation", "polarity", "hedged"]].head(8)""",
"""pd.Series(r.mapping, name="정규 이름").to_frame().sort_values("정규 이름")""",
"""r.candidates[["src", "dst", "support", "contra", "correlational", "score", "sign"]].merge(r.edges[["src", "dst", "p_fwd", "p_rev"]], how="left", on=["src", "dst"])""",
"""md:집계 점수 하한(기본 0.55)은 서로 다른 문서 두 곳 이상이 같은 방향을 말한 엣지만 남긴다. 틀린 문서 하나가 주장한 엣지는 여기서 걸러진다. 반사실 검증 확률(p_fwd)은 엣지를 거르지 않고 LLM 증거로 붙는다. 최종 엣지는 텍스트 증거와 LLM 증거, 두 종류를 가지므로 supported가 된다.

> 1차 GPU 실행(2026-10-25)에서 반사실 검증으로 엣지를 거르면 9B의 F1이 0.82에서 0.51로 떨어졌다. 참 후보와 거짓 후보의 p_fwd 평균이 0.48과 0.51로 구별되지 않았다. LLM의 반사실 판단은 텍스트가 무엇을 말하는지가 아니라 세상 상식을 반영하기 때문이다. 그래서 거르는 기준을 문서 간 집계 점수로 바꿨다(`reports/2026-10-25_text_llm.md`).""",
"""print(evaluate(list(zip(r.edges.src, r.edges.dst)), r.mapping, c))
[(e.key, e.sign, e.status.value, round(e.confidence, 2)) for e in r.hcg.edges]""",
"""md:## 3. 오류율 실험 (가상 LLM)

추출·정규화의 오류율을 바꿔 가며 여섯 코퍼스 평균 성능을 본다. 집계 점수 하한을 0.4(문서 하나로도 통과)와 0.55(문서 둘 이상)로 비교한다. 가상 LLM은 반사실 검증에서 정답을 알고 답하므로, 검증으로 거르는 설정은 여기서는 비교하지 않는다.""",
"""rows = []
for (d, l), corp in corpora.items():
    for err in (0.0, 0.1, 0.2, 0.3):
        for ms in (0.4, 0.55):
            res = text_to_hcg(corp.docs, f"{d}-{l}", d, TextOracle(corp, err, seed=3), min_score=ms)
            rows.append({"domain": d, "lang": l, "error": err, "min_score": ms, **evaluate(list(zip(res.edges.src, res.edges.dst)), res.mapping, corp)})
oracle_res = pd.DataFrame(rows)
oracle_res.groupby(["error", "min_score"])[["precision", "recall", "f1", "reversed", "group_purity", "var_split"]].mean().round(3)""",
"""md:## 4. (GPU) 실제 LLM — Qwen3.5

두 모델로 여섯 코퍼스를 처리하고 정답 그래프와 비교한다. 이전 설정(점수 하한 0.4 + 검증 임계 0.5로 거르기)과 새 기본값(점수 하한 0.55, 검증은 증거로만)을 함께 잰다. 응답은 `results/llm_cache/`에 기록되므로, Mac에서 GPU 없이 재생해 기준값을 바꿔 가며 다시 실험할 수 있다.

> 재생 분석(2026-10-25, 9B): 새 기본값은 정밀도 0.94, 재현율 0.87, F1 0.90이고 방향을 뒤집은 엣지가 없었다. 이전 설정은 F1 0.51이었다.""",
"""import gc

import torch

from humm.providers.cache import CachedProvider
from humm.providers.hf import HFProvider

MODELS = {
    "Qwen/Qwen3.5-4B": "cuda:0",
    "Qwen/Qwen3.5-9B": "cuda:1",
}
llm_rows = []
for model_id, device in MODELS.items():
    lm = CachedProvider(HFProvider(model_id, device=device, batch_size=8), RESULTS / "llm_cache" / f"{model_id.split('/')[-1]}.json")
    for (d, l), corp in corpora.items():
        for setting, kw in {"old: score≥0.4 + verify filter": {"min_score": 0.4, "verify_threshold": 0.5}, "default: score≥0.55": {}}.items():
            res = text_to_hcg(corp.docs, f"{d}-{l}", d, lm, **kw)
            llm_rows.append({"model": model_id, "domain": d, "lang": l, "setting": setting, "claims": len(res.claims),
                             **evaluate(list(zip(res.edges.src, res.edges.dst)), res.mapping, corp)})
    lm.save()
    del lm
    gc.collect()
    torch.cuda.empty_cache()

llm_res = pd.DataFrame(llm_rows)
llm_res.to_csv(RESULTS / "text_llm.csv", index=False)
llm_res.groupby(["model", "setting"])[["claims", "precision", "recall", "f1", "reversed", "group_purity", "unmapped_phrases"]].mean().round(3)""",
"""llm_res.pivot_table(index=["model", "setting"], columns=["lang", "domain"], values="f1").round(2)""",
"""md:## 5. (CPU) 텍스트 + 데이터 병합 (기록 재생)

마케팅 코퍼스의 정답 DAG로 표 데이터를 만들고(PC만, LLM 없이), 9B가 만든 텍스트 HCG와 합친다. 두 그래프의 노드는 별칭 사전으로 잇는다. 반대 방향 엣지가 함께 있으면 **방향 증거**(텍스트·LLM·전문가, PC의 v-구조·Meek 규칙)가 강한 쪽을 남긴다. PC가 임의로 정한 방향은 방향 증거로 치지 않는다.""",
"""from humm.providers.cache import ReplayProvider

c = corpora[("marketing", "ko")]
df = BinarySCM.random(c.graph, seed=5).sample(8000, seed=1)[sorted(c.graph.nodes)]
tab = tabular_to_hcg(df, "marketing-data", n_boot=10, validate_kw={"n_perm": 50})
txt = text_to_hcg(c.docs, "marketing-text", "marketing", ReplayProvider(RESULTS / "llm_cache" / "Qwen3.5-9B.json", name="9B"))
link = link_nodes(txt.hcg, tab.hcg, {v[0]: v[1] + v[2] for v in TEXT_VOCAB["marketing"]})
merged = merge_hcg(txt.hcg, tab.hcg, link, "marketing-merged")
report, changes = validate_hcg(merged, df, n_perm=50)
print("연결:", link)
print("판정:", report.summary()["verdict"], "| 데이터만 썼을 때 엣지:", len(tab.hcg.active_edges()), "| 병합 후 활성 엣지:", len(merged.active_edges()))
changes.assign(true_edge=[tuple(e.split("→")) in set(c.graph.edges) for e in changes.edge])""",
"""report.save(RESULTS / "validation" / "marketing-merged.json")
merged.save(RESULTS / "validation" / "marketing-merged.hcg.json")
changes.groupby("after").edge.count()""",
"""md:## Mac으로 옮길 파일

실행한 노트북은 원본과 다른 이름으로 저장하고, `notebooks/`의 원본은 출력 없이 둔다.

| 파일 | 필수 여부 | 비고 |
|---|---|---|
| 이 노트북 | 필수 | `runs/<날짜>-gpu/` |
| `results/text_llm.csv` | 필수 | 4장: 모델·코퍼스별 엣지 F1과 정규화 품질 |
| `results/llm_cache/Qwen3.5-4B.json`, `Qwen3.5-9B.json` | 필수 | LLM 응답 기록(06장 기록에 추가됨). Mac에서 재생해 기준값 실험에 쓴다 |
| `results/validation/marketing-merged.json`, `marketing-merged.hcg.json` | 필수 | 5장: 병합 그래프 검증 리포트와 HCG |""",
]

n = nbf.v4.new_notebook()
n.cells = [nbf.v4.new_markdown_cell(c[3:]) if c.startswith("md:") else nbf.v4.new_code_cell(c) for c in cells]
n.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(n, sys.argv[1])
