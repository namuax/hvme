"""Analysis result -> explanation (Markdown). All numbers and verdicts are taken verbatim from the engine results; the LLM optionally writes only a summary paragraph."""
import json
import re

from ..gym.verbalize import josa
from .analysis import Analysis

KIND_KO = {"ate": "개입 효과(ATE)", "association": "상관 대 인과", "mediation": "매개(직접·간접) 효과", "ett": "처치군 반사실 효과(ETT)"}
RUNG = {"ate": 2, "association": 2, "mediation": 3, "ett": 3}  # correlation-vs-causation questions are Rung 2 (same rule as CLadder backadj)
STRATEGY_KO = {"backdoor": "백도어 조정", "frontdoor": "프론트도어", "iv": "도구 변수", "no_effect": "인과 경로 없음", "not_identified": "식별 불가"}

NARRATE_PROMPT = """아래 인과 분석 결과만 사용해서 질문에 대한 답을 한국어 3문장 안팎으로 쉽게 풀어 쓴다.

규칙:
- 첫 문장은 '답'을 쉬운 말로 바꿔 쓴다. 숫자는 결과에 있는 값만 그대로 옮긴다.
- 95% 구간은 "95% 구간 a ~ b"로만 쓴다. 구간의 양 끝을 효과 크기로 해석하지 않는다.
- 결과에 없는 사실·원인·권고를 덧붙이지 않는다. 변수는 결과에 적힌 한국어 이름으로만 부른다.
- 모든 문장은 '~다'로 끝낸다('~습니다'·'~요' 금지).

질문: {question}
결과(JSON):
{facts}"""

def _f(v: float) -> str:
    return "NaN" if v != v else f"{v:+.3f}"


def _ci(d: dict) -> str:
    return f"{_f(d['value'])} (95% 구간 {_f(d['ci'][0])} ~ {_f(d['ci'][1])})"


def verdict_line(a: Analysis) -> str:
    q, v = a.query, a.estimate
    x, y = (a.names.get(k, k) for k in (q.treatment, q.outcome))
    if a.estimand.strategy == "not_identified":
        return f"현재 그래프에서는 {josa(x, '이')} {y}에 주는 효과를 관측 데이터만으로 식별할 수 없다."
    if a.estimand.strategy == "no_effect":
        return f"그래프에 {x}에서 {josa(y, '로')} 가는 인과 경로가 없으므로 {josa(x, '을')} 바꿔도 {josa(y, '은')} 달라지지 않는다(효과 0)."
    main = v[a.main]
    sig = main["ci"][0] * main["ci"][1] > 0
    word = ("높인다" if main["value"] > 0 else "낮춘다") if sig else "뚜렷하게 바꾼다고 말할 수 없다"
    return {"ate": lambda: f"{josa(x, '을')} 1로 바꾸면 {y}=1일 확률이 {_ci(main)}만큼 달라진다. 즉 {josa(x, '은')} {josa(y, '을')} {word}.",
            "association": lambda: f"관찰된 차이(상관)는 {_ci(v['association'])}이고, 교란을 조정한 인과 효과는 {_ci(v['ate'])}이다. "
                           f"차이 {_f(v['association']['value'] - v['ate']['value'])}는 교란 경로에서 온 몫이다.",
            "mediation": lambda: f"총효과 {_ci(v['ate'])} 가운데 {josa(', '.join(a.names.get(m, m) for m in a.query.mediators) or '매개 변수', '을')} 거치는 자연 간접 효과(NIE)는 {_ci(v['nie'])}, "
                         f"자연 직접 효과(NDE)는 {_ci(v['nde'])}이다.",
            "ett": lambda: f"실제로 {x}=1이었던 단위에서 실제 결과와 '{josa(x, '이')} 0이었다면'의 반사실 결과를 비교하면, {y}=1일 확률 차이가 평균 {_ci(v['ett'])}이다"
                   f"(모집단 전체 효과 ATE는 {_ci(v['ate'])})."}[q.kind]()


def to_markdown(a: Analysis, validation: dict | None = None, narrative: str = "") -> str:
    q, e, s = a.query, a.estimand, a.sensitivity_summary
    lines = [f"## 질문: {q.text}", "", f"**답**: {verdict_line(a)}", ""]
    lines += [narrative, ""] if narrative else []
    lines += [f"- 유형: {KIND_KO[q.kind]} (Rung {RUNG[q.kind]}), 처치 `{q.treatment}` → 결과 `{q.outcome}`"
              + (f", 매개 `{', '.join(q.mediators)}`" if q.mediators else "") + f" — 질의 분석: {q.parser}",
              f"- 식별: {STRATEGY_KO[e.strategy]} — `{e.expression}`"]
    lines += [f"- 그래프 검증(L1·L2): **{validation['verdict']}** (LMC 검정 {validation['lmc_tests']}개, 강한 위반 {validation['strong_violations']}개)"] if validation else []
    lines += [f"- 실험 대조(L4): 실험 효과 {_ci(a.experiment)}, 관찰 추정 구간과 {'일치' if a.experiment['consistent'] else '불일치'}"] if a.experiment else []
    lines += ["", "### 근거 경로"]
    lines += [" → ".join([p[0]["edge"].split("→")[0]] + [f"{d['edge'].split('→')[1]}" for d in p]) + "  \n  "
              + "; ".join(f"{d['edge']} [{d['status']}, {d['confidence']:.2f}, {'+'.join(d['evidence'])}]" + (f" “{d['quote'][:60]}”" if d["quote"] else "") for d in p)
              for p in a.paths] or ["(인과 경로 없음)"]
    lines += ["", f"교란 경로(백도어): {'; '.join(' – '.join(p) for p in a.backdoor) or '없음'}"]
    if len(a.refutation):
        lines += ["", "### 반박 테스트", "| 테스트 | 기대값 | 새 값 | 통과 |", "|---|---|---|---|"]
        lines += [f"| {r.test} | {r.expected:+.3f} | {r.new_value:+.3f} | {'예' if r.passed else '아니오'} |" for r in a.refutation.itertuples()]
    lines += ["", "### 그래프 민감도 (L5)",
              f"대안 그래프 {s['scenarios']}개 중 {s['identified']}개에서 식별되었다. 대표 수치 범위 {_f(s['min'])} ~ {_f(s['max'])}, "
              f"원래 95% 구간 안 {s['within_ci']}개, 부호가 뒤집힌 대안 {s['flipped']}개, 효과가 0이 된 대안 {s['vanished']}개."]
    lines += [f"식별이 깨지는 경우: {', '.join(s['not_identified'])}."] if s["not_identified"] else []
    lines += ["", "| 시나리오 | 전략 | 변수 | 값 |", "|---|---|---|---|"]
    lines += [f"| {r.scenario} | {r.strategy} | {r.variables} | {_f(r.value)} |" for r in a.sensitivity.itertuples()]
    lines += ["", "### 가정"] + [f"{i}. {t}" for i, t in enumerate(a.assumptions, 1)]
    return "\n".join(lines)


def facts(a: Analysis) -> dict:
    """Facts passed to the LLM summary. Internal names (variable ids, hidden confounder _U) are replaced by display names."""
    s = a.sensitivity_summary
    return {"kind": KIND_KO[a.query.kind], "treatment": a.names.get(a.query.treatment, a.query.treatment),
            "outcome": a.names.get(a.query.outcome, a.query.outcome), "identification": STRATEGY_KO[a.estimand.strategy], "answer": verdict_line(a),
            "estimates": {k: {"value": round(v["value"], 3), "ci_95": [round(c, 3) for c in v["ci"]]} for k, v in a.estimate.items()},
            "graph_sensitivity": {"alternative_graphs": s["scenarios"], "sign_flipped": s["flipped"], "effect_vanished": s["vanished"],
                                  "breaks_with_hidden_confounder": any(x.startswith("숨은 교란") for x in s["not_identified"])},
            "key_assumption": _named(a.assumptions[1] if len(a.assumptions) > 1 else a.assumptions[0], a.names)}


def _named(text: str, names: dict) -> str:
    """Replace variable ids in a sentence with display names."""
    for k in sorted(names, key=len, reverse=True):
        text = re.sub(rf"(?<![\w]){re.escape(k)}(?![\w])", names[k], text)
    return text


def _numbers(text: str) -> set:
    return {round(abs(float(x)), 3) for x in re.findall(r"\d+\.\d+", text)}


ADJ_STEMS = {"있", "없", "같", "많", "적", "높", "낮", "작", "좋", "크", "다르", "아니", "빠르", "느리", "쉽", "어렵", "가능하", "유의하", "필요하", "동일하", "중요하"}


def _syllable(ch: str):
    code = ord(ch) - 0xAC00
    return (code // 588, code % 588 // 28, code % 28) if 0 <= code < 11172 else None


def _compose(cho: int, jung: int, jong: int) -> str:
    return chr(0xAC00 + cho * 588 + jung * 28 + jong)


def _plain_word(word: str) -> str:
    """Convert one formal-polite (합쇼체) word to plain '~다' style: 입니다 -> 이다, adjective stem -> stem + 다, verb stem -> ㄴ다/는다."""
    if word.endswith("입니다"):
        return word[:-3] + "이다"
    if word.endswith("습니다"):
        stem = word[:-3]
        tense = (_syllable(stem[-1]) or (0, 0, 0))[2] == 20  # ㅆ final consonant (았/었/했/겠): 나타났습니다 -> 나타났다
        return stem + ("다" if tense or any(stem.endswith(a) for a in ADJ_STEMS) else "는다")
    last = _syllable(word[-3]) if len(word) >= 3 and word.endswith("니다") else None
    if last and last[2] == 17:  # ㅂ final consonant + 니다: 증가합니다 -> 증가한다, 큽니다 -> 크다
        stem = word[:-3] + _compose(last[0], last[1], 0)
        return stem + "다" if any(stem.endswith(a) for a in ADJ_STEMS) else word[:-3] + _compose(last[0], last[1], 4) + "다"
    return word


def to_plain(text: str) -> str:
    """Convert formal-polite sentence endings in the LLM summary to plain '~다' style (Qwen3.5 often writes formal-polite even when told not to). '않습니다' is matched to the preceding word."""
    def fix(m):
        before, word = m.group(1) or "", m.group(2)
        adj_neg = word.endswith("않습니다") and any(before.rstrip().removesuffix("지").endswith(a) for a in ADJ_STEMS)
        return before + (word[:-4] + "않다" if adj_neg else _plain_word(word))
    return re.sub(r"(?<!\S)(\S+\s)?([^\s.!?]+니다)(?=[.!?\s]|$)", fix, text)


def _plain(sentence: str) -> bool:
    """Is it a plain '~다' declarative sentence? '~습니다', '~ㅂ니다' and '~요' are not."""
    end = sentence.rstrip(".!?\"' )")
    return end.endswith("다") and not end.endswith("니다")


def check_narrative(text: str, a: Analysis) -> list[str]:
    """Check a summary paragraph. Returns a list of problems if it has numbers not in the evidence, formal-polite/해요 sentences, or internal names (an empty list means it passes).
    Semantic errors (e.g. reading an interval as an effect) are not caught."""
    allowed = _numbers(json.dumps(facts(a), ensure_ascii=False)) | _numbers(verdict_line(a))
    sentences = [x.strip() for x in re.split(r"(?<=[.!?])\s+", text.strip()) if x.strip()]
    ids = {k for k, v in a.names.items() if k != v and re.search(rf"\b{re.escape(k)}\b", text)}
    return ([f"근거에 없는 숫자 {sorted(_numbers(text) - allowed)}"] if _numbers(text) - allowed else []) \
        + ([f"'~다'로 끝나지 않는 문장 {len(bad)}개"] if (bad := [x for x in sentences if not _plain(x)]) else []) \
        + ([f"내부 이름 {sorted(ids | ({'_U'} & set(re.findall(r'_U', text))))}"] if ids or "_U" in text else [])


def narrate(a: Analysis, llm) -> str:
    prompt = NARRATE_PROMPT.format(question=a.query.text, facts=json.dumps(facts(a), ensure_ascii=False, indent=1))
    return llm.complete([prompt], max_new_tokens=512)[0].strip()


def polish(draft: str) -> str:
    """Raw summary -> style-normalized version. The check is run on the normalized version."""
    return to_plain(draft)
