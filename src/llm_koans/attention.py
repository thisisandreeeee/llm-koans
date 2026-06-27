"""Koans 02-04: attention scores, context vectors, and multi-head attention."""

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


def context_from_weights(weights: Tensor, V: Tensor) -> Tensor:
    """Use attention weights to collect value information.

    weights: (..., T_query, T_key)
    V:       (..., T_key, d_v)
    output:  (..., T_query, d_v)
    """
    TODO("The context vector is weights @ V.")


def self_attention_for_one_query(query: Tensor, keys: Tensor, values: Tensor) -> tuple[Tensor, Tensor]:
    """Compute attention for one query vector over all keys and values.

    query:  (d_k,)
    keys:   (T, d_k)
    values: (T, d_v)

    Returns:
        context: (d_v,)
        weights: (T,)
    """
    TODO("Compute scores query @ keys.T, softmax them, then use weights @ values.")


def single_head_self_attention(X: Tensor, W_q: Tensor, W_k: Tensor, W_v: Tensor) -> tuple[Tensor, Tensor]:
    """Single-head self-attention for one sequence.

    X:   (T, d_model)
    W_q: (d_k, d_model)
    W_k: (d_k, d_model)
    W_v: (d_v, d_model)

    Returns:
        context: (T, d_v)
        weights: (T, T)
    """
    TODO("Project X to Q/K/V, compute weights, then compute context.")


def split_heads(X: Tensor, num_heads: int) -> Tensor:
    """Split a batched sequence into attention heads.

    X:      (B, T, D)
    return: (B, H, T, Dh), where Dh = D // H

    The operation should not change the data order; it only reshapes and permutes.
    """
    TODO("Reshape to (B, T, H, Dh), then permute to (B, H, T, Dh).")


def combine_heads(X: Tensor) -> Tensor:
    """Combine attention heads back into the model dimension.

    X:      (B, H, T, Dh)
    return: (B, T, H * Dh)
    """
    TODO("Permute to (B, T, H, Dh), make contiguous, then view/reshape.")


def multi_head_self_attention(
    X: Tensor,
    W_q: Tensor,
    W_k: Tensor,
    W_v: Tensor,
    W_o: Tensor,
    num_heads: int,
    mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Multi-head self-attention using explicit projection matrices.

    X:   (B, T, D)
    W_q: (D, D)
    W_k: (D, D)
    W_v: (D, D)
    W_o: (D, D)

    This repo uses row-vector projections for batched model code:
        Q = X @ W_q

    Returns:
        output:  (B, T, D)
        weights: (B, H, T, T)
    """
    TODO("Project, split heads, attend, combine heads, then apply W_o.")
