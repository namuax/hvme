"""Teacher-trajectory rejection sampling: have the teacher model solve each item several times and keep only trajectories that match the engine answer and contain no repetition loop."""
import re

import pandas as pd

from ..gym.formats import render, reward
from .data import looping
from .tools import TOOLS, respond


def sample_cot(provider, items: list[dict], n: int = 2, temperature: float = 0.7, top_p: float = 0.95, repetition_penalty: float = 1.05,
               max_new_tokens: int = 3072) -> pd.DataFrame:
    """provider: humm.providers.hf.HFProvider. Samples n completions per item."""
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
    """Turn trajectories that pass the correctness, no-loop and length-cap checks into SFT examples, up to per_item per item (shortest first)."""
    ok = samples[(samples.reward == 1) & ~samples.loop & (samples.chars <= max_chars)].sort_values(["id", "chars"]).groupby("id").head(per_item)
    by_id = {d["id"]: d for d in items}
    return [{"id": r.id, "prompt": render(by_id[r.id], "cot"), "response": re.sub(r"\s+$", "", r.output), "task": r.task, "source": "teacher"}
            for r in ok.itertuples()]


# ---------- Generation with tool execution (evaluation and teacher sampling) ----------
TOOL_TURN = "<|im_end|>\n<|im_start|>user\n<tool_response>\n{result}\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def generate_tools(provider, prompts: list[str], tools: bool = True, max_rounds: int = 3, max_new_tokens: int = 1536, do_sample: bool = False,
                   temperature: float = 0.7, top_p: float = 0.95, prefixes: list[str] | None = None) -> list[str]:
    """Generate while executing tool calls. Generation stops at '</tool_call>'; the causal_engine result is then appended as the same template fragment used in training and generation continues.

    The returned string is the assistant output concatenated with the <tool_response> blocks (same shape as humm.lm.trajectories.transcript).
    prefixes: per prompt, calls and results already completed (a string of the same shape, trajectories.error_prefix). Generation continues after it, and the returned string includes the prefix.
    Up to max_rounds calls are executed, then one more turn is given after the last result to write only the answer (otherwise generation could end right after the final tool result with no answer).
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
            if hasattr(provider, "generate_raw"):  # remote server (providers.remote)
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
    """Teacher rejection sampling (tool mode or cot mode). n samples per item."""
    rep = [d for d in items for _ in range(n)]
    outs = generate_tools(provider, [render(d, mode) for d in rep], tools=mode == "tool", do_sample=True, **kw)
    return pd.DataFrame([{"id": d["id"], "task": d["query_type"], "mode": mode, "output": o, "reward": reward(d, o), "chars": len(o), "loop": looping(o),
                          "calls": o.count("<tool_response>")} for d, o in zip(rep, outs)])


def accepted_messages(samples: pd.DataFrame, items: list[dict], max_chars: int = 6000, per_item: int = 1) -> list[dict]:
    """Convert accepted samples into messages-format SFT examples. For tool-mode trajectories, calls and results are split back into turns."""
    ok = samples[(samples.reward == 1) & ~samples.loop & (samples.chars <= max_chars)].sort_values(["id", "chars"]).groupby("id").head(per_item)
    by_id = {d["id"]: d for d in items}
    return [{"id": r.id, "messages": [{"role": "user", "content": render(by_id[r.id], r.mode)}] + split_turns(r.output), "tools": r.mode == "tool",
             "task": r.task, "source": "teacher"} for r in ok.itertuples()]


def split_turns(text: str) -> list[dict]:
    """generate_tools output -> assistant/tool messages. Tool calls stay verbatim in the assistant content (same shape as template rendering)."""
    msgs = []
    for part in re.split(r"\n<tool_response>\n(.*?)\n</tool_response>\n", text, flags=re.S):
        msgs.append({"role": "tool" if len(msgs) % 2 else "assistant", "content": part.strip() if len(msgs) % 2 else re.sub(r"\s+$", "", part)})
    return msgs
