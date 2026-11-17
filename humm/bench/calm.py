"""CaLM-lite (45 English tasks, excluding open-ended CEG-O): uses the original repository's (data/bench/calm_repo) prompt-generation and scoring functions as is.

Both the result of applying the original function to the whole response (correct) and to the final answer only (correct_tail) are recorded.
The original scorer reads the first yes/no or option string in the response, so for models that write the solution first, an intermediate sentence gets read as the answer.
In tool mode the variable-state yes/no appears first, so the original scorer almost always reads a state name. The final answer is what follows the last 'Final answer:', or else the last paragraph.
"""
import random
import re
import sys
from pathlib import Path

from .datasets import BENCH

REPO = BENCH / "calm_repo"
sys.path.insert(0, str(REPO / "calm"))

from data_processing.generate_questions import generate_question_list  # noqa: E402
from data_processing.task_hiearchy import task_hiearchy_dict  # noqa: E402
from evaluation.core_metrics import initialize_core_metric_evaluation_components, task_to_accuracy_module_map  # noqa: E402

TASKS = [t for t in task_hiearchy_dict if t.endswith("_EN") and task_to_accuracy_module_map[t] != "open-ended"]


def questions(task: str, style: str = "basic", n: int | None = None, seed: int = 0) -> list[dict]:
    """Items of one task (including the original prompt 'question'). If n is given, only n items with a fixed seed."""
    items = generate_question_list(dataset_path=REPO / "calm_lite_dataset" / f"{task_hiearchy_dict[task]}{task}.json", prompt_style=style)
    return items if n is None else sorted(random.Random(seed).sample(items, n), key=lambda d: d["index"])


def correct(task: str, item: dict, response: str, style: str = "basic") -> bool:
    """Score one item with the original scoring function."""
    gt, pred, acc = initialize_core_metric_evaluation_components(task)
    return acc([gt(item)], [pred(response, item, style, task.split("-")[0])]) == 1


def tail(response: str) -> str:
    """Final answer: text after the last 'Final answer:', or else the paragraph after the last blank line."""
    i = response.lower().rfind("final answer:")
    return response[i + len("final answer:"):].strip() if i >= 0 else response.strip().split("\n\n")[-1]


GIVEN = re.compile(r"direct effect on|probability|causes|->", re.I)


def needs_tools(question: str) -> bool:
    """Routing rule: tools are given only to problems that state causal relations (a graph) or probabilities. Textual and commonsense causality (COPA, e-CARE, ECI, CRASS, etc.) is solved without tools.
    Previously, tool mode forced a graph onto commonsense causality and scored poorly (CRASS 80 → 17)."""
    return bool(GIVEN.search(question))
