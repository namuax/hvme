"""Provider that calls Humm-LM through an OpenAI-compatible server (vLLM etc.) for on-premises serving.

The chat template is applied with the local tokenizer and the raw text is sent to /v1/completions. This way the tool loop of humm.lm.sample.generate_tools
(stop at </tool_call>, append the causal_engine result, continue) runs on the same strings as with an HF model.

    vllm serve models/humm-lm-9b-sft-v214 --served-model-name humm-lm-9b --port 8011
    lm = RemoteProvider("http://127.0.0.1:8011/v1", "humm-lm-9b", tokenizer="models/humm-lm-9b-sft-v214")
"""
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from transformers import AutoTokenizer


class RemoteProvider:
    supports_logits = False

    def __init__(self, base_url: str, model: str, tokenizer: str, batch_size: int = 16, timeout: int = 600):
        self.base_url, self.model_name, self.batch_size, self.timeout = base_url.rstrip("/"), model, batch_size, timeout
        self.name = f"remote:{model}"
        self.tok = AutoTokenizer.from_pretrained(tokenizer)

    def _post(self, prompt: str, max_new_tokens: int, stop: list[str], sampling: dict) -> str:
        body = {"model": self.model_name, "prompt": prompt, "max_tokens": max_new_tokens, "stop": stop, "include_stop_str_in_output": True,
                "skip_special_tokens": True} | sampling
        req = urllib.request.Request(f"{self.base_url}/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())["choices"][0]["text"]

    def generate_raw(self, texts: list[str], max_new_tokens: int, stop: list[str], do_sample: bool = False, temperature: float = 0.7,
                     top_p: float = 0.95) -> list[str]:
        """Chat-templated strings -> continuations. Requests are sent concurrently so the server can batch them."""
        sampling = {"temperature": temperature, "top_p": top_p} if do_sample else {"temperature": 0.0}
        with ThreadPoolExecutor(self.batch_size) as ex:
            return list(ex.map(lambda t: self._post(t, max_new_tokens, stop, sampling), texts))

    def complete(self, prompts, max_new_tokens=1024):
        ctx = [self.tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for p in prompts]
        return self.generate_raw(ctx, max_new_tokens, [])
