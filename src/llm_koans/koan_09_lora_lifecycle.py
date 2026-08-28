"""Koan 09: LoRA adapter lifecycle."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .common import TODO
from .koan_08_finetuning import TinyBaseTextClassifier


class LoRALinear(nn.Module):
    """A frozen linear layer plus a trainable low-rank adapter."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float = 1.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")

        self.weight = nn.Parameter(base.weight.detach().clone(), requires_grad=False)
        if base.bias is None:
            self.bias = None
        else:
            self.bias = nn.Parameter(base.bias.detach().clone(), requires_grad=False)

        self.rank = rank
        self.alpha = alpha
        self.A = nn.Parameter(
            torch.randn(base.in_features, rank) / math.sqrt(base.in_features)
        )
        self.B = nn.Parameter(torch.zeros(rank, base.out_features))

    def forward(self, x: Tensor) -> Tensor:
        """Return frozen base projection plus scaled low-rank adapter output."""
        return F.linear(x, self.weight, self.bias) + (x @ self.A @ self.B) * (
            self.alpha / self.rank
        )


def add_lora_classifier_adapter(
    model: TinyBaseTextClassifier,
    rank: int,
    alpha: float = 1.0,
) -> TinyBaseTextClassifier:
    """Replace the classifier head with a LoRA-adapted frozen classifier.

    The base embedding and encoder should also be frozen. After this function,
    only the LoRA adapter matrices should require gradients.
    """
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.classifier = LoRALinear(model.classifier, rank, alpha)
    return model


def lora_adapter_state(layer: LoRALinear) -> dict[str, Tensor]:
    """Return the small adapter-only artifact you would save after LoRA tuning."""
    return {"A": layer.A.detach().clone(), "B": layer.B.detach().clone()}


def load_lora_adapter_state(
    layer: LoRALinear, adapter_state: dict[str, Tensor]
) -> LoRALinear:
    """Load a saved adapter artifact into an existing LoRA layer."""
    with torch.no_grad():
        layer.A.copy_(adapter_state["A"])
        layer.B.copy_(adapter_state["B"])
    return layer


def merge_lora_linear(layer: LoRALinear) -> nn.Linear:
    """Merge a LoRA layer into one ordinary Linear layer for simpler deployment."""
    merged = nn.Linear(
        layer.weight.shape[1],
        layer.weight.shape[0],
        bias=layer.bias is not None,
        device=layer.weight.device,
        dtype=layer.weight.dtype,
    )
    with torch.no_grad():
        merged.weight.copy_(layer.weight + (layer.A @ layer.B * (layer.alpha / layer.rank)).T)
        if layer.bias is not None:
            merged.bias.copy_(layer.bias)
    return merged
