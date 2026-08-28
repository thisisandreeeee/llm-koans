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
        TODO(
            "Combine the frozen base projection with the rank-limited adapter contribution at its configured strength."
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
    TODO(
        "Install the adapter on the classifier while ensuring it is the model's only trainable component."
    )


def lora_adapter_state(layer: LoRALinear) -> dict[str, Tensor]:
    """Return the small adapter-only artifact you would save after LoRA tuning."""
    TODO("Save only the trainable adapter artifact as independent tensors.")


def load_lora_adapter_state(
    layer: LoRALinear, adapter_state: dict[str, Tensor]
) -> LoRALinear:
    """Load a saved adapter artifact into an existing LoRA layer."""
    TODO(
        "Restore the saved adapter tensors without adding the load operation to autograd."
    )


def merge_lora_linear(layer: LoRALinear) -> nn.Linear:
    """Merge a LoRA layer into one ordinary Linear layer for simpler deployment."""
    TODO(
        "Fold the adapter's effective update into an ordinary linear layer with equivalent output."
    )
