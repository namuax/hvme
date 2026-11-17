"""Config string -> provider. The MCP server and notebooks select the LLM with the same format.

  hf:Qwen/Qwen3.5-9B@cuda:1        local HF model (cuda:0 if the device is omitted)
  anthropic:claude-sonnet-5-5      Anthropic API (key from the ANTHROPIC_API_KEY environment variable)
  replay:results/llm_cache/Qwen3.5-9B.json   replay recorded responses (no GPU)
Giving a cache path records responses (CachedProvider). replay does not record.
"""
from pathlib import Path

from ..paths import ROOT
from .cache import CachedProvider, ReplayProvider


def _hf(arg: str):
    from .hf import HFProvider
    model, _, device = arg.partition("@")
    return HFProvider(model, device=device or "cuda:0")


def _anthropic(arg: str):
    from .anthropic_api import AnthropicProvider
    return AnthropicProvider(model=arg or "claude-sonnet-5-5")


def _replay(arg: str):
    path = Path(arg) if Path(arg).is_absolute() else ROOT / arg
    return ReplayProvider(path, name=path.stem)


KINDS = {"hf": _hf, "anthropic": _anthropic, "replay": _replay}


def from_spec(spec: str | None, cache: str | Path | None = None):
    """None if spec is empty (rules and data only, no LLM)."""
    if not spec:
        return None
    kind, _, arg = spec.partition(":")
    provider = KINDS[kind](arg)
    return CachedProvider(provider, cache) if cache and kind != "replay" else provider
