"""Koan 03: self-attention and context vectors."""

from __future__ import annotations

from torch import Tensor
import torch
import math

from .common import TODO


def context_from_weights(weights: Tensor, V: Tensor) -> Tensor:
    """Use attention weights to collect value information.

    weights: (..., T_query, T_key)
    V:       (..., T_key, d_v)
    output:  (..., T_query, d_v)
    """
    return weights @ V


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
    weights = torch.softmax(query @ keys.T / math.sqrt(query.shape[0]), dim=-1)
    context = weights @ values
    return context, weights


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
    Q = X @ W_q.T  # (T, d_k)
    K = X @ W_k.T  # (T, d_k)
    scores = Q @ K.T / math.sqrt(K.shape[1])  # (T, T)
    weights = torch.softmax(scores, dim=-1)  # (T, T)
    V = X @ W_v.T  # (T, d_v)
    context = weights @ V
    return context, weights
