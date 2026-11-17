from typing import Protocol


class Provider(Protocol):
    """LLM interface that all of Hvme depends on.

    - complete: list of prompts -> list of generated texts (CoT, generative evaluation, graph extraction)
    - option_probs: list of prompts x option labels -> per-option probabilities (Noul/Choice/Score, Hvme-Fast)
    """

    name: str

    def complete(self, prompts: list[str], max_new_tokens: int = 1024) -> list[str]: ...

    def option_probs(self, prompts: list[str], labels: list[str]) -> list[list[float]]: ...
