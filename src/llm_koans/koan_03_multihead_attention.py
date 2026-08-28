"""Koan 03: multi-head attention."""

from __future__ import annotations

import torch
from torch import Tensor

from .common import TODO


def split_heads(X: Tensor, num_heads: int) -> Tensor:
    """Split a batched sequence into attention heads.

    X:      (B, T, D)
    return: (B, H, T, Dh), where Dh = D // H

    The operation should not change the data order; it only reshapes and permutes.
    """
    B, T, D = X.shape
    return X.reshape(B, T, num_heads, D // num_heads).permute(0, 2, 1, 3)


def combine_heads(X: Tensor) -> Tensor:
    """Combine attention heads back into the model dimension.

    X:      (B, H, T, Dh)
    return: (B, T, H * Dh)
    """
    B, H, T, Dh = X.shape
    return X.permute(0, 2, 1, 3).contiguous().reshape(B, T, H * Dh)


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
    Q = split_heads(X @ W_q, num_heads)
    K = split_heads(X @ W_k, num_heads)
    V = split_heads(X @ W_v, num_heads)
    scores = Q @ K.transpose(-2, -1) / Q.shape[-1] ** 0.5
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return combine_heads(weights @ V) @ W_o, weights
