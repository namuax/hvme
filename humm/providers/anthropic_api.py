"""Anthropic API 공급자. API는 토큰 로짓을 주지 않으므로 생성 방식(complete) 기준선에만 쓴다."""
from concurrent.futures import ThreadPoolExecutor

import anthropic


class AnthropicProvider:
    supports_logits = False

    def __init__(self, model: str = "claude-sonnet-5-5", api_key: str | None = None, workers: int = 8):
        """api_key를 주지 않으면 환경변수 ANTHROPIC_API_KEY를 쓴다."""
        self.name = model
        self.workers = workers
        self.client = anthropic.Anthropic(api_key=api_key)

    def _one(self, prompt, max_new_tokens):
        msg = self.client.messages.create(model=self.name, max_tokens=max_new_tokens, messages=[{"role": "user", "content": prompt}])
        return "".join(b.text for b in msg.content if b.type == "text")  # thinking 블록은 건너뛰고 답변 텍스트만 모은다

    def complete(self, prompts, max_new_tokens=1024):
        with ThreadPoolExecutor(self.workers) as ex:
            return list(ex.map(lambda p: self._one(p, max_new_tokens), prompts))

    def option_probs(self, prompts, labels):
        raise NotImplementedError("Anthropic API는 로짓을 제공하지 않는다. complete + 답 파싱 방식을 쓴다.")
