"""Minimal dependency-free LoRA. Base weights are frozen (bf16); only a low-rank correction B·A (fp32) is trained per linear layer.

Use it instead of full fine-tuning (fp32 weights, gradients and Adam state, about 64GB for 4B) when sharing the GPU with other jobs.
After training, merge_lora folds the correction into the weights so the model can be saved and loaded as a plain model.
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
    """Replace every nn.Linear except names in exclude with LoRALinear and freeze the remaining weights. Returns the number of trainable parameters."""
    for p in model.parameters():
        p.requires_grad_(False)
    targets = [(name, m) for name, m in model.named_modules() if isinstance(m, nn.Linear) and not any(e in name for e in exclude)]
    for name, m in targets:
        parent, _, child = name.rpartition(".")
        setattr(model.get_submodule(parent) if parent else model, child, LoRALinear(m, r, alpha))
    getattr(model, "enable_input_require_grads", lambda: None)()  # let gradients flow past the frozen embeddings under gradient checkpointing
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def merge_lora(model: nn.Module) -> nn.Module:
    """Turn LoRALinear back into an nn.Linear with the correction merged in."""
    targets = [(name, m) for name, m in model.named_modules() if isinstance(m, LoRALinear)]
    for name, m in targets:
        parent, _, child = name.rpartition(".")
        setattr(model.get_submodule(parent) if parent else model, child, m.merged())
    return model
