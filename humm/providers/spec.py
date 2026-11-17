"""설정 문자열 → 공급자. MCP 서버·노트북이 같은 형식으로 LLM을 고른다.

  hf:Qwen/Qwen3.5-9B@cuda:1        로컬 HF 모델 (장치 생략 시 cuda:0)
  anthropic:claude-sonnet-5-5      Anthropic API (키는 ANTHROPIC_API_KEY 환경 변수)
  replay:results/llm_cache/Qwen3.5-9B.json   기록된 응답 재생 (GPU 없이)
cache 경로를 주면 응답을 기록한다(CachedProvider). replay는 기록하지 않는다.
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
    """spec이 비어 있으면 None(LLM 없이 규칙·데이터만 쓴다)."""
    if not spec:
        return None
    kind, _, arg = spec.partition(":")
    provider = KINDS[kind](arg)
    return CachedProvider(provider, cache) if cache and kind != "replay" else provider
