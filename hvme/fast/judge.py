"""Hvme-Fast inference: one forward pass yields probabilities over the answer tokens (options).

It follows the provider interface (option_probs), so it plugs directly into pairwise direction queries, L3 verification and Gym evaluation.
Calibration temperatures are kept per option kind (noul, choice, score). A trained model is saved with save(), which stores weights, tokenizer and temperatures together.
"""
import json
from pathlib import Path

import numpy as np
import torch
from scipy.special import softmax

NOUL = {("Yes", "No"), ("예", "아니오")}
KNOWLEDGE_PREFIX = "You are an expert in"  # PAIR_PROMPT (pairwise direction) and VERIFY_PROMPT (counterfactual verification): domain-knowledge judgments


def kind_of(labels) -> str:
    labels = tuple(labels)
    return "noul" if labels in NOUL else "score" if all(x.isdigit() for x in labels) else "choice"


def temp_key(prompt: str, labels) -> str:
    """Calibration temperature key. Domain-knowledge judgments and computational judgments (Gym etc.) differ in overconfidence, so they get separate temperatures
    (measured: a shared temperature gives pairwise-direction ECE 0.17, separate ones 0.08)."""
    return f"knowledge:{kind_of(labels)}" if prompt.startswith(KNOWLEDGE_PREFIX) else kind_of(labels)


def confidence(p: np.ndarray) -> float:
    """1 − normalized entropy."""
    p = np.clip(np.asarray(p, float), 1e-12, 1)
    return float(1 + (p * np.log(p)).sum() / np.log(len(p)))


def _find_meta(path_or_id: str) -> Path | None:
    """hvme_fast.json next to the weights: a local folder, or a Hugging Face repo id such as namuax/hvme-fast-4b."""
    local = Path(path_or_id)
    if local.exists():
        meta = local / "hvme_fast.json"
        return meta if meta.exists() else None
    try:
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(path_or_id, "hvme_fast.json"))
    except Exception:
        return None


class FastJudge:
    supports_logits = True

    def __init__(self, model, tok, name: str = "hvme-fast", temperature: dict | None = None, batch_size: int = 16, max_length: int = 4096):
        self.model, self.tok, self.name = model, tok, name
        self.temperature = temperature or {}
        self.batch_size, self.max_length = batch_size, max_length

    @classmethod
    def load(cls, path_or_id: str, device: str = "cuda:0", dtype=torch.bfloat16, **kw) -> "FastJudge":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(path_or_id, padding_side="left")
        tok.pad_token = tok.pad_token or tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(path_or_id, dtype=dtype, device_map=device).eval()
        meta = _find_meta(path_or_id)
        temp = json.loads(meta.read_text())["temperature"] if meta else {}
        return cls(model, tok, Path(path_or_id).name if meta else path_or_id, temp, **kw)

    def save_temperature(self, path: str | Path) -> Path:
        """Change only the temperatures in hvme_fast.json, leaving the weights as is."""
        meta = Path(path) / "hvme_fast.json"
        meta.write_text(json.dumps(json.loads(meta.read_text()) | {"temperature": self.temperature}, ensure_ascii=False, indent=1))
        return meta

    def save(self, path: str | Path, extra: dict | None = None) -> Path:
        path = Path(path)
        self.model.save_pretrained(path)
        self.tok.save_pretrained(path)
        (path / "hvme_fast.json").write_text(json.dumps({"temperature": self.temperature} | (extra or {}), ensure_ascii=False, indent=1))
        return path

    # ---------- Input ----------
    def encode(self, prompts: list[str]):
        msgs = [[{"role": "user", "content": p}] for p in prompts]
        text = [self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False) for m in msgs]
        enc = self.tok(text, return_tensors="pt", padding=True, add_special_tokens=False, truncation=True, max_length=self.max_length)
        return enc.to(self.model.device)

    def label_ids(self, labels) -> list[int]:
        """First-token id of each answer token. Overlaps are rejected because probability could not be split."""
        ids = [self.tok.encode(x, add_special_tokens=False)[0] for x in labels]
        assert len(set(ids)) == len(ids), f"first tokens of the answer tokens overlap: {labels}"
        return ids

    def last_logits(self, enc) -> torch.Tensor:
        """Full-vocabulary logits at the last position [batch, vocab]. logits_to_keep=1 avoids computing logits at other positions."""
        return self.model(**enc, logits_to_keep=1).logits[:, -1].float()

    # ---------- Inference ----------
    @torch.inference_mode()
    def logits(self, prompts: list[str], labels) -> np.ndarray:
        """Uncalibrated option logits [item, option]."""
        ids = self.label_ids(labels)
        out = [self.last_logits(self.encode(prompts[i:i + self.batch_size]))[:, ids].cpu().numpy() for i in range(0, len(prompts), self.batch_size)]
        return np.concatenate(out) if out else np.zeros((0, len(ids)))

    def temperature_for(self, prompt: str, labels) -> float:
        """Per-task temperature if present, else the option-kind temperature, else 1."""
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
