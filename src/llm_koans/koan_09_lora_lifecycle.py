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
        TODO("Compute F.linear(x, frozen weight/bias) + (x @ A @ B) * (alpha / rank).")


def add_lora_classifier_adapter(
    model: TinyBaseTextClassifier,
    rank: int,
    alpha: float = 1.0,
) -> TinyBaseTextClassifier:
    """Replace the classifier head with a LoRA-adapted frozen classifier.

    The base embedding and encoder should also be frozen. After this function,
    only the LoRA adapter matrices should require gradients.
    """
    TODO(
        "Freeze embedding/encoder, replace classifier with LoRALinear(model.classifier, rank, alpha), return model."
    )


def lora_adapter_state(layer: LoRALinear) -> dict[str, Tensor]:
    """Return the small adapter-only artifact you would save after LoRA tuning."""
    TODO("Return detached clones for A and B only, not the frozen base weight/bias.")


def load_lora_adapter_state(
    layer: LoRALinear, adapter_state: dict[str, Tensor]
) -> LoRALinear:
    """Load a saved adapter artifact into an existing LoRA layer."""
    TODO(
        "Copy adapter_state['A'] and adapter_state['B'] into layer.A and layer.B without tracking gradients, then return layer."
    )


def merge_lora_linear(layer: LoRALinear) -> nn.Linear:
    """Merge a LoRA layer into one ordinary Linear layer for simpler deployment."""
    TODO(
        "Create nn.Linear, set weight to frozen weight + transposed scaled LoRA delta, copy bias, and return it."
    )
