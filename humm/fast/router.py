"""라우터: Humm-Fast가 먼저 답하고, 확신이 낮은 문항만 느린 공급자(범용 LLM)로 올린다 (04_humm_fast.md 5장).

공급자 인터페이스를 따르므로 tabular_to_hcg·text_to_hcg에 provider로 넘기면 된다. 생성(complete)은 느린 공급자가 맡는다.
확신도는 최대 보기 확률이다. threshold보다 낮으면 올린다. 호출 수는 n_fast·n_slow에 쌓인다.
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
    """같은 문항에 대한 fast·slow 확률로, 임계값별 상향 비율과 최종 정확도를 계산한다(실제로 다시 돌리지 않는다)."""
    rows = []
    for t in thresholds:
        up = fast_probs.max(1) < t
        final = np.where(up[:, None], slow_probs, fast_probs)
        rows.append({"threshold": round(float(t), 3), "escalation": float(up.mean()), "acc": float((final.argmax(1) == target).mean()),
                     "fast_acc_kept": float((fast_probs[~up].argmax(1) == target[~up]).mean()) if (~up).any() else float("nan")})
    return rows
