"""Koan 04: multi-head attention."""

from __future__ import annotations

from torch import Tensor
import torch
import math

from .common import TODO


def split_heads(X: Tensor, num_heads: int) -> Tensor:
    """Split a batched sequence into attention heads.

    X:      (B, T, D)
    return: (B, H, T, Dh), where Dh = D // H

    The operation should not change the data order; it only reshapes and permutes.
    """
    B, T, D = X.shape
    Dh = D // num_heads
    return X.reshape(B, T, num_heads, Dh).transpose(1, 2)


def combine_heads(X: Tensor) -> Tensor:
    """Combine attention heads back into the model dimension.

    X:      (B, H, T, Dh)
    return: (B, T, H * Dh)
    """
    B, H, T, Dh = X.shape
    return X.transpose(1, 2).reshape(B, T, H * Dh)


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
    Q = X @ W_q  # (B, T, D)
    K = X @ W_k  # (B, T, D)
    Q = split_heads(Q, num_heads)  # (B, H, T, Dh)
    K = split_heads(K, num_heads)  # (B, H, T, Dh)
    scores = Q @ K.transpose(-1, -2) / math.sqrt(K.shape[-1])  # (B, H, T, T)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)  # (B, H, T, T)
    V = X @ W_v  # (B, T, D)
    V = split_heads(V, num_heads)  # (B, H, T, Dh)
    context = weights @ V  # (B, H, T, Dh)
    context = combine_heads(context)  # (B, T, D)
    output = context @ W_o  # (B, T, D)
    return output, weights
