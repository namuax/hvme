"""의존성 없는 최소 LoRA. 기반 가중치는 고정(bf16)하고, 선형층마다 저랭크 보정 B·A(fp32)만 학습한다.

GPU를 다른 작업과 나눠 쓸 때 전체 미세조정(fp32 가중치·기울기·Adam 상태 약 64GB, 4B 기준) 대신 쓴다.
학습 뒤 merge_lora로 보정을 가중치에 합치면 일반 모델로 저장·불러오기할 수 있다.
"""
import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 16, alpha: float = 32.0):
        super().__init__()
        self.base, self.scale = base, alpha / r
        base.weight.requires_grad_(False)
        self.lora_a = nn.Parameter(torch.empty(r, base.in_features, device=base.weight.device))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x):
        delta = (x.to(self.lora_a.dtype) @ self.lora_a.T) @ self.lora_b.T
        return self.base(x) + (delta * self.scale).to(x.dtype)

    def merged(self) -> nn.Linear:
        w = self.base.weight
        w.data += (self.scale * self.lora_b @ self.lora_a).to(w.dtype)
        return self.base


def add_lora(model: nn.Module, r: int = 16, alpha: float = 32.0, exclude=("lm_head",)) -> int:
    """exclude에 든 이름을 뺀 모든 nn.Linear를 LoRALinear로 바꾸고 나머지 가중치를 고정한다. 학습 파라미터 수를 돌려준다."""
    for p in model.parameters():
        p.requires_grad_(False)
    targets = [(name, m) for name, m in model.named_modules() if isinstance(m, nn.Linear) and not any(e in name for e in exclude)]
    for name, m in targets:
        parent, _, child = name.rpartition(".")
        setattr(model.get_submodule(parent) if parent else model, child, LoRALinear(m, r, alpha))
    getattr(model, "enable_input_require_grads", lambda: None)()  # gradient checkpointing에서 고정된 임베딩 뒤로 기울기가 흐르게 한다
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def merge_lora(model: nn.Module) -> nn.Module:
    """LoRALinear를 보정이 합쳐진 nn.Linear로 되돌린다."""
    targets = [(name, m) for name, m in model.named_modules() if isinstance(m, LoRALinear)]
    for name, m in targets:
        parent, _, child = name.rpartition(".")
        setattr(model.get_submodule(parent) if parent else model, child, m.merged())
    return model
