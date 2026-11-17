from typing import Protocol


class Provider(Protocol):
    """Humm 전체가 의존하는 LLM 인터페이스.

    - complete: 프롬프트 목록 → 생성 텍스트 목록 (CoT·생성형 평가, 그래프 추출)
    - option_probs: 프롬프트 목록 × 보기 라벨 → 보기별 확률 목록 (Noul·Choice·Score, Humm-Fast)
    """

    name: str

    def complete(self, prompts: list[str], max_new_tokens: int = 1024) -> list[str]: ...

    def option_probs(self, prompts: list[str], labels: list[str]) -> list[list[float]]: ...
