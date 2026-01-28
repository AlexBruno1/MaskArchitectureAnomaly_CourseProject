# ---------------------------------------------------------------
# LoRA utilities for lightweight fine-tuning of linear layers.
# ---------------------------------------------------------------

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_LORA_TARGET_MODULES = (
    "qkv",
    "proj",
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
)


class LoRALinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Linear,
        r: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank r must be > 0")

        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.r = r
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.r
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.weight = nn.Parameter(base_layer.weight.detach().clone())
        if base_layer.bias is not None:
            self.bias = nn.Parameter(base_layer.bias.detach().clone())
        else:
            self.register_parameter("bias", None)

        self.lora_A = nn.Parameter(torch.empty(self.r, self.in_features))
        self.lora_B = nn.Parameter(torch.empty(self.out_features, self.r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = F.linear(x, self.weight, self.bias)
        if not self.merged:
            lora_out = self.lora_dropout(x)
            lora_out = torch.matmul(lora_out, self.lora_A.t())
            lora_out = torch.matmul(lora_out, self.lora_B.t())
            result = result + lora_out * self.scaling
        return result


def apply_lora(
    model: nn.Module,
    target_modules: Iterable[str],
    r: int,
    alpha: float,
    dropout: float,
) -> int:
    replaced = 0
    target_modules = tuple(target_modules)

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        module_name = name.split(".")[-1]
        if module_name not in target_modules:
            continue

        parent = model
        parts = name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        child_name = parts[-1]

        if isinstance(getattr(parent, child_name), LoRALinear):
            continue

        setattr(
            parent,
            child_name,
            LoRALinear(module, r=r, alpha=alpha, dropout=dropout),
        )
        replaced += 1

    return replaced


def mark_only_lora_as_trainable(
    model: nn.Module,
    train_bias: str = "none",
    trainable_modules: Optional[Sequence[str]] = None,
) -> int:
    if train_bias not in ("none", "lora_only", "all"):
        raise ValueError("train_bias must be one of: none, lora_only, all")

    for param in model.parameters():
        param.requires_grad = False

    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_A.requires_grad = True
            module.lora_B.requires_grad = True
            if train_bias in ("lora_only", "all") and module.bias is not None:
                module.bias.requires_grad = True

    if train_bias == "all":
        for name, param in model.named_parameters():
            if name.endswith(".bias"):
                param.requires_grad = True

    if trainable_modules:
        for name, param in model.named_parameters():
            if any(name.startswith(prefix) for prefix in trainable_modules):
                param.requires_grad = True

    return sum(p.numel() for p in model.parameters() if p.requires_grad)
