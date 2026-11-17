"""SFT example construction (no torch needed): query-type weighted sampling, solution-template examples, repetition-loop detection."""
import re

import numpy as np
import pandas as pd

from ..gym.formats import render
from .solutions import solve
from .tools import parse_call

# Oversample structure-reasoning types and nie, where the 4B model is weak
WEIGHTS = {"corr2cause": 2.0, "identify": 2.0, "backadj": 2.0, "nie": 1.5, "dsep": 1.2}
# Oversample counterfactual types (ett, det_cf), which did not transfer to CLadder in the first round, and structure-reasoning types
WEIGHTS_V3 = {"corr2cause": 1.5, "identify": 1.5, "backadj": 1.5, "det_cf": 1.5, "ett": 1.3, "nie": 1.3, "dsep": 1.2}
# Raise the structure-reasoning types that regressed in the previous mix to about 1,350 examples per type
WEIGHTS_V31 = {"corr2cause": 2.0, "identify": 2.0, "backadj": 2.0, "dsep": 2.0, "det_cf": 1.3, "ett": 1.2, "nie": 1.2}
# GRPO: the types the SFT model still gets wrong on CLadder and Gym (ett, nde, nie, backadj, corr2cause, marginal) carry the reward signal
WEIGHTS_GRPO = {"ett": 2.0, "nde": 2.0, "nie": 2.0, "backadj": 2.0, "corr2cause": 2.0, "marginal": 1.5, "correlation": 0.7, "cde": 0.5, "det_cf": 0.5, "ate_score": 0.5}


def pick(items: list[dict], n: int, weights: dict = WEIGHTS, seed: int = 0) -> list[dict]:
    """Draw n items according to per-query-type weights (random within a type)."""
    rng = np.random.default_rng(seed)
    by = pd.Series(range(len(items))).groupby([d["query_type"] for d in items]).apply(list).to_dict()
    w = pd.Series({t: weights.get(t, 1.0) for t in by})
    quota = (w / w.sum() * n).round().astype(int)
    idx = [i for t, k in quota.items() for i in rng.permutation(by[t])[: min(k, len(by[t]))]]
    return [items[i] for i in rng.permutation(idx)]


def template_examples(items: list[dict]) -> list[dict]:
    return [{"id": d["id"], "prompt": render(d, "cot"), "response": solve(d), "task": d["query_type"], "source": "template"} for d in items]


def looping(text: str, min_repeat: int = 4) -> bool:
    """A line (10+ characters) appearing min_repeat or more times counts as a repetition loop. Tool-call markup lines ('</parameter>' etc.) are not counted."""
    lines = [s.strip() for s in text.splitlines() if len(s.strip()) >= 10 and not s.strip().startswith("<")]
    return any(lines.count(s) >= min_repeat for s in set(lines))


def error_repeats(out: str, start: int = 0) -> int:
    """Number of times a call that got an error is re-sent with the same arguments in the very next call. start: the first few calls (the given prefix) are not counted as re-sends."""
    calls, resps = re.findall(r"<tool_call>(.*?)</tool_call>", out, re.S), re.findall(r"<tool_response>\n(.*?)</tool_response>", out, re.S)
    return sum(r.startswith("error") and a.strip() == b.strip() for i, (a, b, r) in enumerate(zip(calls, calls[1:], resps)) if i + 1 >= start)


STRUCT = ("dsep", "backadj", "identify", "corr2cause")


def _canon(line: str) -> frozenset:
    """Set of pieces of a line split on commas, "and" and colons. Lines differing only in the order of listed paths, edges or variables count as equal."""
    return frozenset(c.strip(" .") for c in re.split(r",\s*|\s+and\s+|:\s+", " ".join(line.split())) if c.strip(" ."))


def process(d: dict, out: str) -> float:
    """Process score for structure-type solutions: fraction of the engine solution's key lines after step 3 (excluding option picking and Final) that appear in the output."""
    if "call" in d["meta"]:  # corr2cause: the tool makes the verdict, so check that the arguments (hypothesis kind, variables, relation list) were transcribed correctly
        c, ref = parse_call(out) or {}, d["meta"]["call"]
        return (c.get("hypothesis") == ref["hypothesis"] and [c.get("a"), c.get("b")] == [ref["a"], ref["b"]]) / 2 + \
            (sorted(map(str, c.get("relations") or [])) == sorted(ref["relations"])) / 2
    ref = solve(d).splitlines()
    i = next(k for k, l in enumerate(ref) if l.startswith("3)"))
    keys = [_canon(l) for l in ref[i:] if l.strip() and not l.startswith("Final") and not re.match(r"\s*(해당하는 보기|Matching option)", l)]
    lines = {_canon(l) for l in out.splitlines()}
    return sum(k in lines for k in keys) / len(keys)
