"""가상 LLM: 참 그래프를 알고 있지만 일부러 틀리는 공급자. LLM 사전지식의 품질이 결과에 주는 영향을 GPU 없이 실험한다.

- 순서 질의: 참 위상 순서에서 error × 변수 수 만큼 무작위로 두 변수의 자리를 바꿔 답한다.
- 쌍별 질의: 참 방향(조상 관계)을 확률 1 - error 로 맞히고, confidence 만큼의 확률을 준다.
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
