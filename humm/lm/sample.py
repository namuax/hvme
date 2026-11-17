"""교사 궤적 rejection sampling: 교사 모델이 같은 문항을 여러 번 풀게 하고, 엔진 정답과 맞으며 반복 루프가 없는 궤적만 남긴다."""
import re

import pandas as pd

from ..gym.formats import render, reward
from .data import looping
from .tools import TOOLS, respond


def sample_cot(provider, items: list[dict], n: int = 2, temperature: float = 0.7, top_p: float = 0.95, repetition_penalty: float = 1.05,
               max_new_tokens: int = 3072) -> pd.DataFrame:
    """provider: humm.providers.hf.HFProvider. 문항마다 n개를 샘플링한다."""
    rows = []
    for i in range(0, len(items), provider.batch_size):
        batch = items[i:i + provider.batch_size]
        enc = provider._encode([render(d, "cot") for d in batch])
        gen = provider.model.generate(**enc, do_sample=True, temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty,
                                      max_new_tokens=max_new_tokens, num_return_sequences=n)
        outs = provider.tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        rows += [{"id": d["id"], "task": d["query_type"], "k": k, "output": outs[j * n + k], "reward": reward(d, outs[j * n + k]),
                  "chars": len(outs[j * n + k]), "loop": looping(outs[j * n + k])} for j, d in enumerate(batch) for k in range(n)]
    return pd.DataFrame(rows)


def accepted(samples: pd.DataFrame, items: list[dict], max_chars: int = 6000, per_item: int = 1) -> list[dict]:
    """정답·루프 없음·길이 상한을 통과한 궤적을 문항당 per_item개까지 SFT 예제로 바꾼다(짧은 것부터)."""
    ok = samples[(samples.reward == 1) & ~samples.loop & (samples.chars <= max_chars)].sort_values(["id", "chars"]).groupby("id").head(per_item)
    by_id = {d["id"]: d for d in items}
    return [{"id": r.id, "prompt": render(by_id[r.id], "cot"), "response": re.sub(r"\s+$", "", r.output), "task": r.task, "source": "teacher"}
            for r in ok.itertuples()]


# ---------- 도구 실행 생성 (SFT v2 평가·교사 샘플링) ----------
TOOL_TURN = "<|im_end|>\n<|im_start|>user\n<tool_response>\n{result}\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def generate_tools(provider, prompts: list[str], tools: bool = True, max_rounds: int = 3, max_new_tokens: int = 1536, do_sample: bool = False,
                   temperature: float = 0.7, top_p: float = 0.95, prefixes: list[str] | None = None) -> list[str]:
    """도구 호출을 실행하며 생성한다. '</tool_call>'에서 멈추면 causal_engine 결과를 학습 때와 같은 템플릿 조각으로 붙이고 이어 쓴다.

    돌려주는 문자열은 어시스턴트 출력과 <tool_response> 블록을 이어 붙인 것이다(humm.lm.trajectories.transcript와 같은 모양).
    prefixes: 프롬프트마다 이미 끝난 호출·결과(같은 모양의 문자열, trajectories.error_prefix). 그 뒤부터 생성하고 돌려주는 문자열은 앞부분을 포함한다.
    호출은 max_rounds번까지 실행하고, 마지막 결과 뒤에 답만 쓰는 차례를 한 번 더 준다(v2.15: 전에는 세 번째 호출의 값을 받고 답 없이 끝났다).
    """
    tok = provider.tok
    ctx = [tok.apply_chat_template([{"role": "user", "content": p}], tools=TOOLS if tools else None, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False) for p in prompts]
    outs, active = list(prefixes or [""] * len(prompts)), list(range(len(prompts)))
    for j, p in enumerate(outs):
        turns = split_turns(p)
        ctx[j] += "".join(a["content"] + TOOL_TURN.format(result=r["content"]) for a, r in zip(turns[::2], turns[1::2]))
    kw = dict(do_sample=True, temperature=temperature, top_p=top_p) if do_sample else dict(do_sample=False)
    for r in range(max_rounds + tools):
        nxt = []
        for i in range(0, len(active), provider.batch_size):
            idx = active[i:i + provider.batch_size]
            if hasattr(provider, "generate_raw"):  # 원격 서버(providers.remote, Step 12)
                texts = provider.generate_raw([ctx[j] for j in idx], max_new_tokens, ["</tool_call>"], do_sample, temperature, top_p)
            else:
                enc = tok([ctx[j] for j in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(provider.model.device)
                gen = provider.model.generate(**enc, max_new_tokens=max_new_tokens, stop_strings=["</tool_call>"], tokenizer=tok, **kw)
                texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for j, text in zip(idx, texts):
                if "</tool_call>" in text and tools and r < max_rounds:
                    text = text[:text.rindex("</tool_call>") + len("</tool_call>")]
                    result = respond(text, outs[j])
                    ctx[j] += text + TOOL_TURN.format(result=result)
                    outs[j] += text + f"\n<tool_response>\n{result}\n</tool_response>\n"
                    nxt.append(j)
                else:
                    outs[j] += text
        active = nxt
        if not active:
            break
    return outs


def sample_tools(provider, items: list[dict], n: int = 2, mode: str = "tool", **kw) -> pd.DataFrame:
    """교사 rejection sampling(도구 모드 또는 cot 모드). 문항마다 n개."""
    rep = [d for d in items for _ in range(n)]
    outs = generate_tools(provider, [render(d, mode) for d in rep], tools=mode == "tool", do_sample=True, **kw)
    return pd.DataFrame([{"id": d["id"], "task": d["query_type"], "mode": mode, "output": o, "reward": reward(d, o), "chars": len(o), "loop": looping(o),
                          "calls": o.count("<tool_response>")} for d, o in zip(rep, outs)])


def accepted_messages(samples: pd.DataFrame, items: list[dict], max_chars: int = 6000, per_item: int = 1) -> list[dict]:
    """채택된 샘플을 messages 형식의 SFT 예제로 바꾼다. 도구 모드 궤적은 호출·결과를 턴으로 다시 나눈다."""
    ok = samples[(samples.reward == 1) & ~samples.loop & (samples.chars <= max_chars)].sort_values(["id", "chars"]).groupby("id").head(per_item)
    by_id = {d["id"]: d for d in items}
    return [{"id": r.id, "messages": [{"role": "user", "content": render(by_id[r.id], r.mode)}] + split_turns(r.output), "tools": r.mode == "tool",
             "task": r.task, "source": "teacher"} for r in ok.itertuples()]


def split_turns(text: str) -> list[dict]:
    """generate_tools 출력 → assistant/tool 메시지. 도구 호출은 원문 그대로 어시스턴트 내용에 남긴다(템플릿 렌더링과 같은 모양)."""
    msgs = []
    for part in re.split(r"\n<tool_response>\n(.*?)\n</tool_response>\n", text, flags=re.S):
        msgs.append({"role": "tool" if len(msgs) % 2 else "assistant", "content": part.strip() if len(msgs) % 2 else re.sub(r"\s+$", "", part)})
    return msgs
