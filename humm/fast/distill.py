"""Humm-Fast v2: System 2 → 1 distillation.

Humm-LM (9B, tool mode) solves each item N times, and the empirical distribution of verified answers becomes Fast's target distribution.
- Source items: Gym item contexts rewritten by an LLM into natural prose. Rewriting does not guarantee the generator label still holds (dropped numbers, altered relations),
  so instead of the label we use the answer the slow path (Humm-LM + engine) derives from that text. Texts that drop or invent numbers are discarded (numbers_kept).
- Large-surface source (story): the whole yes/no item (context + question) is rewritten as a story from another field. Variable and state names and the question wording change,
  so question polarity may flip; the generator label is kept for diagnostics only (meta.orig_label) and the target uses only the slow path's answer.
- Verified: there is a final answer, no repetition loop, and, if a tool was called, the last engine result is not an error. Engine values are symbolic computations, so only answers passing this check are counted.
- Target: frequency of verified answers + alpha smoothing. Items with no verified solution are dropped.
"""
import re

import numpy as np

from ..gym.formats import option_labels, parse_final, render
from ..lm.data import looping

PARAPHRASE = ("Rewrite the following description of a hypothetical world as a natural passage, as a person would explain it in a short report or story. "
              "Keep every variable name, every state name, every causal relation and every number exactly as given, and do not add, drop or change any fact. "
              "Do not answer any question and do not add conclusions. Output only the rewritten passage.\n\n{context}")
STORY = ("Below is a yes/no causal reasoning problem. Rewrite the whole problem so that it is set in a different real-world field of your choice "
         "(for example medicine, farming, schooling, economics, engineering, ecology or sports). Give every variable a new meaningful name and give its states "
         "fitting names, and write the conditions as a natural passage. Keep the same causal relations, every number exactly as given, and a question that asks "
         "about the same quantity and still has a yes or no answer. Do not solve the problem and do not add facts or numbers. "
         "Output exactly two parts in this form:\nPassage: <the rewritten conditions>\nQuestion: <the rewritten question>\n\n{context}\n\n{question}")
# story2: rotate field and text format per item to widen the surface, and make the model write the question only once (80% of story rejections wrote the question several different ways)
FIELDS = ("medicine", "farming", "schooling", "economics", "manufacturing", "ecology", "sports", "public health", "software operations", "retail",
          "transportation", "energy")
FORMS = ("a short news report", "a lab notebook entry", "a textbook exercise", "a consultant's memo", "a field study summary")
STORY2 = ("Below is a yes/no causal reasoning problem. Rewrite the whole problem as {form} set in {field}. Give every variable a new meaningful name "
          "and give its states fitting names, and write the conditions in the voice of that kind of text. Keep the same causal relations, every number exactly as given, "
          "and a question that asks about the same quantity and still has a yes or no answer. Do not solve the problem and do not add facts or numbers. "
          "Write the question only once and stop right after it. Output exactly two parts in this form:\nPassage: <the rewritten conditions>\nQuestion: <the rewritten question>"
          "\n\n{context}\n\n{question}")
STORY_PARTS = re.compile(r"Passage:\s*(.+?)\s*Question:", re.S)
NUMBER = re.compile(r"\d+(?:\.\d+)?")


def paraphrase_prompt(item: dict) -> str:
    return PARAPHRASE.format(context=item["context"])


def story_prompt(item: dict) -> str:
    return STORY.format(context=item["context"], question=item["question"])


def story2_prompt(item: dict, i: int) -> str:
    """Field and format of the i-th item (FIELDS × FORMS cycled by i)."""
    return STORY2.format(field=FIELDS[i % len(FIELDS)], form=FORMS[i // len(FIELDS) % len(FORMS)], context=item["context"], question=item["question"])


def storied(item: dict, text: str, counts: bool = True) -> dict | None:
    """Item rebuilt from story output (Passage/Question). With counts=False, integers up to 12 without % (variable counts such as 'five') are excluded from the number comparison. None if either part is missing, the question was written several different ways, or the number set of context + question differs from the original."""
    m, qs = STORY_PARTS.search(text), {q.strip() for q in re.findall(r"Question:[ \t]*(.+)", text)}
    ctx, q = (m.group(1).strip(), next(iter(qs))) if m and len(qs) == 1 else ("", "")
    ok = ctx and q and numbers_kept(f"{item['context']} {item['question']}", f"{ctx} {q}", counts)
    return item | {"id": f"{item['id']}-s", "context": ctx, "question": q,
                   "meta": item.get("meta", {}) | {"orig_context": item["context"], "orig_question": item["question"], "orig_label": item["label"]}} if ok else None


def numbers_kept(src: str, text: str, counts: bool = True) -> bool:
    """Whether the set of numbers (probabilities, values) in the rewritten text equals the original's. Neither dropped nor invented numbers (e.g. a nonexistent probability table) are allowed; repeat counts are ignored.
    With counts=False, integers up to 12 not followed by % are ignored (variable counts written as words)."""
    nums = lambda t: {m.group() for m in NUMBER.finditer(t) if counts or "." in m.group() or t[m.end():m.end() + 1] == "%" or int(m.group()) > 12}
    return nums(src) == nums(text)


def paraphrased(item: dict, text: str) -> dict | None:
    """Item rebuilt with the rewritten context. None if numbers were dropped. Question, options and label stay as in the original item (the label is for comparison)."""
    text = text.strip()
    return item | {"id": f"{item['id']}-p", "context": text, "meta": item.get("meta", {}) | {"orig_context": item["context"]}} \
        if text and numbers_kept(item["context"], text) else None


def verified(output: str) -> str | None:
    """Final answer (parse_final) of a solution that passed verification, else None."""
    calls = re.findall(r"<tool_response>\n(.*?)\n</tool_response>", output, flags=re.S)
    ok = not looping(output) and not (calls and calls[-1].startswith("error"))
    return parse_final(output) if ok else None


def answer_index(item: dict, answer: str) -> int | None:
    """parse_final answer → position within option_labels."""
    keys = {"noul": ["yes", "no"], "choice": [x.upper() for x in option_labels(item)], "score": option_labels(item)}[item["kind"]]
    return keys.index(answer) if answer in keys else None


def target_from(item: dict, answers: list[str | None], alpha: float = 0.1) -> list[float] | None:
    """Distribution of verified-answer frequencies + alpha smoothing. None if there is no usable answer."""
    idx = [i for i in (answer_index(item, a) for a in answers if a) if i is not None]
    if not idx:
        return None
    c = np.bincount(idx, minlength=len(option_labels(item))) + alpha
    return (c / c.sum()).tolist()


def distill_items(items: list[dict], samples: list[dict], alpha: float = 0.1) -> list[dict]:
    """samples: humm.lm.jobs sample output rows ({id, output, ...}). Attaches the distillation target and verification statistics to Fast training items (from_gym shape)."""
    by_id = {}
    for s in samples:
        by_id.setdefault(s["id"], []).append(verified(s["output"]))
    out = []
    for d in items:
        answers = by_id.get(d["id"], [])
        t = target_from(d, answers, alpha)
        if t is not None:
            out.append({"id": d["id"], "prompt": render(d, "direct"), "labels": option_labels(d), "target": t, "kind": d["kind"], "task": d["query_type"],
                        "lang": d["lang"], "source": "distill", "n_samples": len(answers), "n_verified": sum(a is not None for a in answers)})
    return out
