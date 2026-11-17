"""Humm-Fast 추론기: 한 번의 순전파로 답 토큰(보기) 위의 확률을 낸다 (04_humm_fast.md 2장).

공급자 인터페이스(option_probs)를 따르므로 Step 5 쌍별 방향 질의·Step 6 L3 검증·Gym 평가에 그대로 끼운다.
보정 온도는 보기 종류(noul·choice·score)별로 둔다. 학습한 모델은 save()로 가중치·토크나이저·온도를 함께 저장한다.
"""
import json
from pathlib import Path

import numpy as np
import torch
from scipy.special import softmax

NOUL = {("Yes", "No"), ("예", "아니오")}
KNOWLEDGE_PREFIX = "You are an expert in"  # PAIR_PROMPT(쌍별 방향)·VERIFY_PROMPT(반사실 검증): 도메인 지식 판단


def kind_of(labels) -> str:
    labels = tuple(labels)
    return "noul" if labels in NOUL else "score" if all(x.isdigit() for x in labels) else "choice"


def temp_key(prompt: str, labels) -> str:
    """보정 온도 키. 도메인 지식 판단과 계산형 판단(Gym 등)은 과신 정도가 달라 온도를 따로 둔다
    (2026-10-25 GPU: 한 온도를 함께 쓰면 쌍별 방향 ECE 0.17, 따로 맞추면 0.08. reports/2026-10-25_humm_fast.md)."""
    return f"knowledge:{kind_of(labels)}" if prompt.startswith(KNOWLEDGE_PREFIX) else kind_of(labels)


def confidence(p: np.ndarray) -> float:
    """1 − 정규화 엔트로피 (04_humm_fast.md)."""
    p = np.clip(np.asarray(p, float), 1e-12, 1)
    return float(1 + (p * np.log(p)).sum() / np.log(len(p)))


class FastJudge:
    supports_logits = True

    def __init__(self, model, tok, name: str = "humm-fast", temperature: dict | None = None, batch_size: int = 16, max_length: int = 4096):
        self.model, self.tok, self.name = model, tok, name
        self.temperature = temperature or {}
        self.batch_size, self.max_length = batch_size, max_length

    @classmethod
    def load(cls, path_or_id: str, device: str = "cuda:0", dtype=torch.bfloat16, **kw) -> "FastJudge":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(path_or_id, padding_side="left")
        tok.pad_token = tok.pad_token or tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(path_or_id, dtype=dtype, device_map=device).eval()
        meta = Path(path_or_id) / "humm_fast.json"
        temp = json.loads(meta.read_text())["temperature"] if meta.exists() else {}
        return cls(model, tok, Path(path_or_id).name if meta.exists() else path_or_id, temp, **kw)

    def save_temperature(self, path: str | Path) -> Path:
        """가중치는 그대로 두고 humm_fast.json의 온도만 바꾼다."""
        meta = Path(path) / "humm_fast.json"
        meta.write_text(json.dumps(json.loads(meta.read_text()) | {"temperature": self.temperature}, ensure_ascii=False, indent=1))
        return meta

    def save(self, path: str | Path, extra: dict | None = None) -> Path:
        path = Path(path)
        self.model.save_pretrained(path)
        self.tok.save_pretrained(path)
        (path / "humm_fast.json").write_text(json.dumps({"temperature": self.temperature} | (extra or {}), ensure_ascii=False, indent=1))
        return path

    # ---------- 입력 ----------
    def encode(self, prompts: list[str]):
        msgs = [[{"role": "user", "content": p}] for p in prompts]
        text = [self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False) for m in msgs]
        enc = self.tok(text, return_tensors="pt", padding=True, add_special_tokens=False, truncation=True, max_length=self.max_length)
        return enc.to(self.model.device)

    def label_ids(self, labels) -> list[int]:
        """답 토큰의 첫 토큰 id. 서로 겹치면 확률을 나눌 수 없으므로 막는다."""
        ids = [self.tok.encode(x, add_special_tokens=False)[0] for x in labels]
        assert len(set(ids)) == len(ids), f"답 토큰의 첫 토큰이 겹친다: {labels}"
        return ids

    def last_logits(self, enc) -> torch.Tensor:
        """마지막 위치의 전체 어휘 로짓 [배치, 어휘]. logits_to_keep=1로 다른 위치의 로짓은 만들지 않는다."""
        return self.model(**enc, logits_to_keep=1).logits[:, -1].float()

    # ---------- 추론 ----------
    @torch.inference_mode()
    def logits(self, prompts: list[str], labels) -> np.ndarray:
        """보정 전 보기 로짓 [문항, 보기]."""
        ids = self.label_ids(labels)
        out = [self.last_logits(self.encode(prompts[i:i + self.batch_size]))[:, ids].cpu().numpy() for i in range(0, len(prompts), self.batch_size)]
        return np.concatenate(out) if out else np.zeros((0, len(ids)))

    def temperature_for(self, prompt: str, labels) -> float:
        """과제별 온도가 없으면 보기 종류 온도, 그것도 없으면 1."""
        return self.temperature.get(temp_key(prompt, labels), self.temperature.get(kind_of(labels), 1.0))

    def option_probs(self, prompts: list[str], labels) -> list[list[float]]:
        t = np.array([[self.temperature_for(p, labels)] for p in prompts]).reshape(len(prompts), 1)
        return softmax(self.logits(prompts, labels) / t, axis=1).tolist()

    def noul(self, prompt: str, labels=("Yes", "No")) -> float:
        return self.option_probs([prompt], list(labels))[0][0]

    def choice(self, prompt: str, options: dict) -> dict:
        """options = {"A": "X→Y", "B": "Y→X", ...}"""
        p = np.array(self.option_probs([prompt], list(options))[0])
        return {"choice": list(options.values())[int(p.argmax())], "probs": dict(zip(options.values(), p.tolist())), "confidence": confidence(p)}

    def score(self, prompt: str, levels: int = 5) -> dict:
        p = np.array(self.option_probs([prompt], [str(k) for k in range(1, levels + 1)])[0])
        return {"score": float((p * np.arange(1, levels + 1)).sum()), "dist": p.tolist(), "confidence": confidence(p)}
