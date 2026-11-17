import sys
import nbformat as nbf

cells = [
"""md:# 04. Humm-Gym — A Generator of Verifiable Causal Reasoning Problems

Humm-Gym produces training and evaluation data for Humm-LM and Humm-Fast. Each problem gets a random causal graph and SCM, and the answer is computed by `humm.engine` (notebook 03). Because answers can be checked symbolically, they can be used directly for SFT data selection (rejection sampling) and as GRPO rewards.

| Query type | Rung | Format | Content |
|---|---|---|---|
| marginal · correlation | 1 | Noul | Marginal probability, observational conditional difference |
| dsep · corr2cause | 1 | Noul | d-separation; judging a causal hypothesis from statistical relations alone (exhaustive Markov equivalence class computation) |
| ate · cde · backadj | 2 | Noul | Interventional effect, controlled direct effect, adjustment set validity |
| identify | 2 | Choice | Backdoor, front-door, IV, not identifiable, no effect |
| ate_score | 2 | Score | Direction and magnitude of the interventional effect on a 5-level scale |
| ett · nde · nie | 3 | Noul | Effect on the treated, natural direct and indirect effects |

The design principles are as follows.
- **50:50 answer balance**: the question direction (does it increase / decrease) is chosen to balance the answers.
- **No ambiguous problems**: only SCMs whose effect is at least 5 percentage points away from the decision threshold are used.
- **No common-sense shortcuts**: variable names are assigned at random, so the problem can be solved only from the given graph and probabilities, not from common sense.
- **Separate splits**: training uses marketing, manufacturing and medical vocabulary; test uses IT operations vocabulary and meaningless names.
- **Item mix**: nie, nde, ate and counterfactual items, which were weak at baseline, are over-represented.""",
"""import pandas as pd

from humm.gym import GENERATORS, audit, build, generate, render, summary, target_text
from humm.paths import GYM

pd.set_option("display.max_colwidth", 120)""",
"""md:## 1. Example items

We look at one query type each in Korean and English. `render(item, "direct")` produces the format that expects a single answer token (Humm-Fast), and `render(item, "cot")` produces the format that asks for a step-by-step solution (Humm-LM).""",
"""sample = generate(240, "train", seed=42)
ko_nie = next(d for d in sample if d["query_type"] == "nie" and d["lang"] == "ko")
print(render(ko_nie, "cot"))
print("\\nAnswer:", target_text(ko_nie), "| true NIE =", ko_nie["value"])""",
"""en_identify = next(d for d in sample if d["query_type"] == "identify" and d["lang"] == "en")
print(render(en_identify, "direct"))
print("\\nAnswer:", target_text(en_identify), "|", en_identify["meta"]["strategy"])""",
"""ko_c2c = next(d for d in sample if d["query_type"] == "corr2cause" and d["lang"] == "ko")
print(render(ko_c2c, "direct"))
print("\\nAnswer:", target_text(ko_c2c), "| equivalence class size:", ko_c2c["meta"]["mec_size"])""",
"""md:## 2. Re-verifying the answers

Using only the graph, probability tables and metadata stored with each item, we recompute the answer from scratch and compare it with the stored one. The list of mismatches should be empty.""",
"""bad = audit(sample)
print("Mismatches:", len(bad))
summary(sample).round(2)""",
"""md:## 3. Building the dataset

`build` creates the splits in the order train, dev, test, and removes items that appeared in an earlier split from the later ones. The result is saved to `data/gym/{train,dev,test}.jsonl`. Generating 100,000 items takes about 7 minutes on CPU.""",
"""SIZES = {"train": 100_000, "dev": 2_000, "test": 5_000}
stats = build(SIZES, seed=0)
stats.pivot_table(index="query_type", columns="split", values="n", aggfunc="sum")""",
"""from humm.gym import read_jsonl

test = read_jsonl(GYM / "test.jsonl")
print("Mismatches on test re-verification:", len(audit(test)))
pd.DataFrame(test).groupby(["domain", "lang"]).size().unstack()""",
"""md:## 4. (GPU) Qwen3.5 baseline — Humm-Gym test

Before training, we measure how the two models score on Humm-Gym. The direct mode uses the whole test set; cot is slow, so it uses only 600 items balanced across query types. These scores are the starting point for training Humm-Fast and Humm-LM.""",
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
