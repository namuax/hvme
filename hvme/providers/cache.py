"""LLM response record/replay. Record once on a GPU, then replay the same prompts without a GPU to repeat experiments (threshold tuning, etc.)."""
import hashlib
import json
from pathlib import Path


def _key(kind: str, prompt: str, extra) -> str:
    return hashlib.sha1(json.dumps([kind, prompt, extra], ensure_ascii=False).encode()).hexdigest()


class CachedProvider:
    """Wraps another provider and records its responses. A repeated prompt is answered from the record. save() writes it to JSON."""

    def __init__(self, provider, path: str | Path):
        self.inner, self.path = provider, Path(path)
        self.name = provider.name
        self.supports_logits = getattr(provider, "supports_logits", False)
        self.store = json.loads(self.path.read_text()) if self.path.exists() else {}

    def complete(self, prompts, max_new_tokens=1024):
        keys = [_key("complete", p, max_new_tokens) for p in prompts]
        todo = [p for p, k in zip(prompts, keys) if k not in self.store]
        self.store |= dict(zip([_key("complete", p, max_new_tokens) for p in todo], self.inner.complete(todo, max_new_tokens) if todo else []))
        return [self.store[k] for k in keys]

    def option_probs(self, prompts, labels):
        keys = [_key("option_probs", p, labels) for p in prompts]
        todo = [p for p, k in zip(prompts, keys) if k not in self.store]
        self.store |= dict(zip([_key("option_probs", p, labels) for p in todo], self.inner.option_probs(todo, labels) if todo else []))
        return [self.store[k] for k in keys]

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.store, ensure_ascii=False))
        return self.path


class ReplayProvider:
    """Returns recorded responses only. A prompt missing from the record raises KeyError (a new GPU run is needed)."""

    def __init__(self, path: str | Path, name: str | None = None, supports_logits: bool = True):
        self.store = json.loads(Path(path).read_text())
        self.name = name or Path(path).stem
        self.supports_logits = supports_logits

    def complete(self, prompts, max_new_tokens=1024):
        return [self.store[_key("complete", p, max_new_tokens)] for p in prompts]

    def option_probs(self, prompts, labels):
        return [self.store[_key("option_probs", p, labels)] for p in prompts]
