"""LLM provider abstraction. Every provider exposes complete (generation) and option_probs (option-token probabilities) with the same shape."""
from .base import Provider

__all__ = ["Provider"]
