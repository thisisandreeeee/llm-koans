"""Koan 03: multi-head attention."""

from __future__ import annotations

from torch import Tensor

from .common import TODO


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
