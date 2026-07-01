"""Koan 02: attention scores and weights."""

from __future__ import annotations

from torch import Tensor
import torch
import math

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
    return Q @ K.transpose(-1, -2)


def scaled_scores(scores: Tensor, d_k: int) -> Tensor:
    """Scale attention scores by sqrt(d_k)."""

    return scores / math.sqrt(d_k)


def softmax_last_dim(scores: Tensor) -> Tensor:
    """Apply softmax over the last dimension.

    In attention, the last dimension is usually the key-token dimension.
    Each row becomes weights over all keys.
    """
    return torch.softmax(scores, dim=-1)


def attention_weights(Q: Tensor, K: Tensor) -> Tensor:
    """Compute scaled-softmax attention weights from Q and K."""
    return torch.softmax(Q @ K.transpose(-1, -2) / math.sqrt(K.shape[1]), dim=-1)
