"""Humm-LM: solution-template trajectories, teacher rejection sampling, SFT, GRPO (humm.lm.grpo).

Only solutions is imported eagerly, so trajectories can be built without torch.
"""
from .solutions import solve

__all__ = ["solve"]
