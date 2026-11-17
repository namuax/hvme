"""Simulated LLM: a provider that knows the true graph but is deliberately wrong at a given rate. Lets you study how LLM prior quality affects results without a GPU.

- Order query: swaps the positions of two random variables in the true topological order, error x n_variables times, and answers with that.
- Pairwise query: gets the true direction (ancestor relation) right with probability 1 - error, and returns `confidence` as the probability.
"""
import json
import re

import networkx as nx
import numpy as np

PAIR = re.compile(r"A\. (.+?) causes (.+?)\n")


class NoisyOracle:
    supports_logits = True

    def __init__(self, true: nx.DiGraph, error: float = 0.0, confidence: float = 0.85, seed: int = 0):
        self.true, self.error, self.confidence = true, error, confidence
        self.rng = np.random.default_rng(seed)
        self.name = f"oracle(error={error})"

    def complete(self, prompts, max_new_tokens=1024):
        order = list(nx.topological_sort(self.true))
        for _ in range(int(round(self.error * len(order)))):
            i, j = self.rng.choice(len(order), 2, replace=False)
            order[i], order[j] = order[j], order[i]
        return [f"```json\n{json.dumps({'tiers': [[v] for v in order]})}\n```" for _ in prompts]

    def option_probs(self, prompts, labels):
        out = []
        for p in prompts:
            a, b = PAIR.search(p).groups()
            truth = 1.0 if b in nx.descendants(self.true, a) else (0.0 if a in nx.descendants(self.true, b) else 0.5)
            right = self.rng.random() >= self.error
            p_ab = self.confidence if (truth > 0.5) == right else 1 - self.confidence
            out.append([p_ab, 1 - p_ab] if truth != 0.5 else [0.5, 0.5])
        return out
