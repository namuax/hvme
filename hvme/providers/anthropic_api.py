"""Anthropic API provider. The API does not return token logits, so it is used only as a generation (complete) baseline."""
from concurrent.futures import ThreadPoolExecutor

import anthropic


class AnthropicProvider:
    supports_logits = False

    def __init__(self, model: str = "claude-sonnet-5-5", api_key: str | None = None, workers: int = 8):
        """If api_key is not given, the ANTHROPIC_API_KEY environment variable is used."""
        self.name = model
        self.workers = workers
        self.client = anthropic.Anthropic(api_key=api_key)

    def _one(self, prompt, max_new_tokens):
        msg = self.client.messages.create(model=self.name, max_tokens=max_new_tokens, messages=[{"role": "user", "content": prompt}])
        return "".join(b.text for b in msg.content if b.type == "text")  # skip thinking blocks; collect only the answer text

    def complete(self, prompts, max_new_tokens=1024):
        with ThreadPoolExecutor(self.workers) as ex:
            return list(ex.map(lambda p: self._one(p, max_new_tokens), prompts))

    def option_probs(self, prompts, labels):
        raise NotImplementedError("The Anthropic API does not provide logits. Use complete + answer parsing.")
