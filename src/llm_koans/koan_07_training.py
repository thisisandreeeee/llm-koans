"""Koan 07: training updates."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .common import TODO


class TinyAttentionClassifier(nn.Module):
    """A tiny model used by the training koans.

    It is intentionally simple:
        embeddings -> one multi-head self-attention layer -> mean pooling -> classifier
    """

    def __init__(self, vocab_size: int, d_model: int, num_heads: int, num_classes: int):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, token_ids: Tensor) -> Tensor:
        X = self.embedding(token_ids)
        attended, _ = self.attn(X, X, X, need_weights=False)
        pooled = attended.mean(dim=1)
        return self.classifier(pooled)


def train_one_step(model: nn.Module, optimizer: torch.optim.Optimizer, token_ids: Tensor, labels: Tensor) -> Tensor:
    """Run one supervised training step and return the loss tensor.

    The expected training loop is:
        optimizer.zero_grad()
        logits = model(token_ids)
        loss = cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
    """
    TODO("Implement one standard PyTorch training step with cross-entropy loss.")


def parameter_delta_norm(before: dict[str, Tensor], after: dict[str, Tensor]) -> float:
    """Return the total L2 norm of parameter changes between two state dict snapshots."""
    TODO("Sum squared parameter differences across matching keys, then return sqrt(total).")
