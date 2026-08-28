"""Koan 02: self-attention and context vectors."""

from __future__ import annotations

from torch import Tensor

from .common import TODO


def context_from_weights(weights: Tensor, V: Tensor) -> Tensor:
    """Use attention weights to collect value information.

    weights: (..., T_query, T_key)
    V:       (..., T_key, d_v)
    output:  (..., T_query, d_v)
    """
    TODO("The context vector is weights @ V.")


def self_attention_for_one_query(
    query: Tensor, keys: Tensor, values: Tensor
) -> tuple[Tensor, Tensor]:
    """Compute attention for one query vector over all keys and values.

    query:  (d_k,)
    keys:   (T, d_k)
    values: (T, d_v)

    Returns:
        context: (d_v,)
        weights: (T,)
    """
    TODO("Compute scores query @ keys.T, softmax them, then use weights @ values.")


def single_head_self_attention(
    X: Tensor, W_q: Tensor, W_k: Tensor, W_v: Tensor
) -> tuple[Tensor, Tensor]:
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
