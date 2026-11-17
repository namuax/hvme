"""Simulated LLM for text: knows the ground truth of the synthetic corpus but errs at the given error rate. Lets you develop and experiment with the pipeline without a GPU.

- Extraction: returns the ground-truth claim for each sentence in the prompt. Omits it with probability error, swaps cause and effect with probability error/2, and misreads correlation claims as causal.
- Canonicalization: groups according to the ground-truth alias dictionary, but with probability error splits one expression off on its own.
- Counterfactual verification: if the cause is an ancestor of the effect in the true graph, answers 'yes' (0.9), flipped with probability error.
"""
import json
import re

import networkx as nx
import numpy as np

from .canonical import norm

VERIFY = re.compile(r"if (.+?) were changed while everything else stayed the same, would (.+?) change", re.S)


class TextOracle:
    supports_logits = True

    def __init__(self, corpora, error: float = 0.0, seed: int = 0):
        corpora = corpora if isinstance(corpora, (list, tuple)) else [corpora]
        self.notes = {s: n for c in corpora for s, n in c.annotations.items()}
        self.alias = {norm(s): i for c in corpora for s, i in c.aliases.items()}
        self.graph = nx.compose_all([c.graph for c in corpora])
        self.error, self.rng = error, np.random.default_rng(seed)
        self.name = f"text-oracle(error={error})"

    def _extract(self, prompt):
        text = prompt.split("Text:\n", 1)[-1]
        claims = []
        for s, n in self.notes.items():
            if s in text and self.rng.random() >= self.error:
                c = {"cause": n["cause_text"], "effect": n["effect_text"], "relation": n["relation"], "polarity": n["polarity"], "hedged": n["hedged"], "quote": s}
                c |= {"cause": c["effect"], "effect": c["cause"]} if self.rng.random() < self.error / 2 else {}
                c |= {"relation": "causal"} if c["relation"] == "correlational" and self.rng.random() < self.error / 2 else {}
                claims.append(c)
        return f"```json\n{json.dumps({'claims': claims}, ensure_ascii=False)}\n```"

    def _canon(self, prompt):
        phrases = [line[2:] for line in prompt.split("Phrases:\n", 1)[-1].splitlines() if line.startswith("- ")]
        groups = {}
        for p in phrases:
            key = self.alias.get(norm(p), p) if self.rng.random() >= self.error else f"{p}-split"
            groups.setdefault(key, []).append(p)
        return f"```json\n{json.dumps({'groups': [{'name': m[0], 'members': m} for m in groups.values()]}, ensure_ascii=False)}\n```"

    def complete(self, prompts, max_new_tokens=1024):
        return [self._extract(p) if "Extract every statement" in p else self._canon(p) for p in prompts]

    def option_probs(self, prompts, labels):
        out = []
        for p in prompts:
            a, b = (self.alias.get(norm(x)) for x in VERIFY.search(p).groups())
            truth = a is not None and b is not None and b in nx.descendants(self.graph, a)
            yes = truth != (self.rng.random() < self.error)
            out.append([0.9, 0.1] if yes else [0.1, 0.9])
        return out
