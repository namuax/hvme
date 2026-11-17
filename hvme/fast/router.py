"""Router: Hvme-Fast answers first, and only low-confidence items are escalated to a slow provider (a general LLM).

It follows the provider interface, so it can be passed as the provider to tabular_to_hcg and text_to_hcg. Generation (complete) is handled by the slow provider.
Confidence is the maximum option probability. Items below threshold are escalated. Call counts accumulate in n_fast and n_slow.
"""
import numpy as np


class Routed:
    supports_logits = True

    def __init__(self, fast, slow, threshold: float = 0.8):
        self.fast, self.slow, self.threshold = fast, slow, threshold
        self.name = f"{fast.name}→{slow.name}@{threshold}"
        self.n_fast = self.n_slow = self.n_complete = 0

    def option_probs(self, prompts, labels):
        p = np.array(self.fast.option_probs(prompts, labels), float).reshape(len(prompts), len(labels))
        low = np.flatnonzero(p.max(1) < self.threshold)
        p[low] = np.array(self.slow.option_probs([prompts[i] for i in low], labels)).reshape(len(low), len(labels)) if len(low) else p[low]
        self.n_fast, self.n_slow = self.n_fast + len(prompts), self.n_slow + len(low)
        return p.tolist()

    def complete(self, prompts, max_new_tokens=1024):
        self.n_complete += len(prompts)
        return self.slow.complete(prompts, max_new_tokens)

    @property
    def escalation_rate(self) -> float:
        return self.n_slow / max(self.n_fast, 1)


def routing_curve(fast_probs: np.ndarray, slow_probs: np.ndarray, target: np.ndarray, thresholds=np.linspace(0.5, 1.0, 11)) -> list[dict]:
    """From fast and slow probabilities on the same items, compute the escalation rate and final accuracy per threshold (without re-running anything)."""
    rows = []
    for t in thresholds:
        up = fast_probs.max(1) < t
        final = np.where(up[:, None], slow_probs, fast_probs)
        rows.append({"threshold": round(float(t), 3), "escalation": float(up.mean()), "acc": float((final.argmax(1) == target).mean()),
                     "fast_acc_kept": float((fast_probs[~up].argmax(1) == target[~up]).mean()) if (~up).any() else float("nan")})
    return rows
