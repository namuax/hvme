"""LLM 공급자 추상화. 모든 공급자는 complete(생성)와 option_probs(보기 토큰 확률)를 같은 모양으로 제공한다."""
from .base import Provider

__all__ = ["Provider"]
