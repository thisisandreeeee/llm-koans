"""Koan 01: attention scores and weights."""

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
    TODO(
        "Align the feature axes so every query is compared with every key."
    )


def scaled_scores(scores: Tensor, d_k: int) -> Tensor:
    """Scale attention scores by sqrt(d_k)."""
    TODO("Use the key width to keep dot-product magnitudes well behaved.")


def softmax_last_dim(scores: Tensor) -> Tensor:
    """Apply softmax over the last dimension.

    In attention, the last dimension is usually the key-token dimension.
    Each row becomes weights over all keys.
    """
    TODO("Turn each query's key scores into a probability distribution.")


def attention_weights(Q: Tensor, K: Tensor) -> Tensor:
    """Compute scaled-softmax attention weights from Q and K."""
    TODO("Build attention weights from the three transformations above.")
