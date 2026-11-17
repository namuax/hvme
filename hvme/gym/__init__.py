"""Hvme-Gym: Synthetic SCM, query and answer generator plus verifier. Answers are computed by hvme.engine."""
from .dataset import MIX, build, generate, read_jsonl, summary, write_jsonl
from .check import audit
from .evaluate import eval_cot, eval_direct, summarize as eval_summary
from .formats import option_labels, parse_final, render, reward, target_index, target_text
from .queries import GENERATORS, Item

__all__ = ["audit", "eval_cot", "eval_direct", "eval_summary", "MIX", "build", "generate", "read_jsonl", "summary", "write_jsonl", "option_labels", "parse_final", "render", "reward",
           "target_index", "target_text", "GENERATORS", "Item"]
