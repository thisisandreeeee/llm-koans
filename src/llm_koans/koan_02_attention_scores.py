"""Koan 02: attention scores and weights."""

from __future__ import annotations

from torch import Tensor

from .common import TODO


def attention_scores(Q: Tensor, K: Tensor) -> Tensor:
    """Compute raw attention scores for a single sequence or batch of heads.

    For 2D tensors:
        Q: (T_query, d_k), K: (T_key, d_k)
        return: (T_query, T_key)

    For higher-rank tensors, the same rule applies to the last two dims:
        Q: (..., T_query, d_k), K: (..., T_key, d_k)
        return: (..., T_query, T_key)
    """
    TODO("Compute Q @ K transposed over its last two dimensions.")


def scaled_scores(scores: Tensor, d_k: int) -> Tensor:
    """Scale attention scores by sqrt(d_k)."""
    TODO("Divide scores by math.sqrt(d_k).")


def softmax_last_dim(scores: Tensor) -> Tensor:
    """Apply softmax over the last dimension.

    In attention, the last dimension is usually the key-token dimension.
    Each row becomes weights over all keys.
    """
    TODO("Use torch.softmax(..., dim=-1).")


def attention_weights(Q: Tensor, K: Tensor) -> Tensor:
    """Compute scaled-softmax attention weights from Q and K."""
    TODO("Combine attention_scores, scaled_scores, and softmax_last_dim.")
