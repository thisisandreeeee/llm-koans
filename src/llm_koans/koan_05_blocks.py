"""Koan 05: encoder and decoder blocks."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from .common import TODO
from .koan_03_multihead_attention import multi_head_self_attention
from .koan_04_masks import causal_mask


def position_wise_ffn(
    X: Tensor, W1: Tensor, b1: Tensor, W2: Tensor, b2: Tensor
) -> Tensor:
    """Apply the same two-layer feed-forward network to every token.

    X:  (..., D)
    W1: (D, D_ff)
    b1: (D_ff,)
    W2: (D_ff, D)
    b2: (D,)

    Intuition: attention lets tokens talk; FFN lets each token process itself.
    """
    return torch.relu(X @ W1 + b1) @ W2 + b2


@dataclass
class EncoderBlockWeights:
    W_q: Tensor
    W_k: Tensor
    W_v: Tensor
    W_o: Tensor
    W1: Tensor
    b1: Tensor
    W2: Tensor
    b2: Tensor


@dataclass
class DecoderBlockWeights:
    self_W_q: Tensor
    self_W_k: Tensor
    self_W_v: Tensor
    self_W_o: Tensor
    cross_W_q: Tensor
    cross_W_k: Tensor
    cross_W_v: Tensor
    cross_W_o: Tensor
    W1: Tensor
    b1: Tensor
    W2: Tensor
    b2: Tensor


def split_heads(X: Tensor, num_heads: int) -> Tensor:
    B, T, D = X.shape
    Dh = D // num_heads
    return X.reshape(B, T, num_heads, Dh).transpose(1, 2)


def combine_heads(X: Tensor) -> Tensor:
    B, H, T, Dh = X.shape
    return X.transpose(1, 2).reshape(B, T, H * Dh)


def encoder_block_forward(
    X: Tensor, weights: EncoderBlockWeights, num_heads: int
) -> Tensor:
    """A minimal post-norm encoder block.

    The block is:
        X1 = layer_norm(X + multi_head_self_attention(X))
        X2 = layer_norm(X1 + position_wise_ffn(X1))

    This uses functional layer_norm without trainable gamma/beta to keep the koan focused.
    """
    attn, _ = multi_head_self_attention(X, weights.W_q, weights.W_k, weights.W_v, weights.W_o, num_heads)
    X1 = F.layer_norm(X + attn, (X.shape[-1],))
    ffn = position_wise_ffn(X1, weights.W1, weights.b1, weights.W2, weights.b2)
    return F.layer_norm(X1 + ffn, (X.shape[-1],))


def cross_attention(
    decoder_states: Tensor,
    encoder_states: Tensor,
    W_q: Tensor,
    W_k: Tensor,
    W_v: Tensor,
    W_o: Tensor,
    num_heads: int,
) -> tuple[Tensor, Tensor]:
    """Cross-attention: queries from decoder, keys/values from encoder.

    decoder_states: (B, T_dec, D)
    encoder_states: (B, T_enc, D)

    Returns:
        output:  (B, T_dec, D)
        weights: (B, H, T_dec, T_enc)
    """
    Q = split_heads(decoder_states @ W_q, num_heads)
    K = split_heads(encoder_states @ W_k, num_heads)
    V = split_heads(encoder_states @ W_v, num_heads)
    scores = Q @ K.transpose(-2, -1) / Q.shape[-1] ** 0.5
    weights = torch.softmax(scores, dim=-1)
    return combine_heads(weights @ V) @ W_o, weights


def decoder_block_forward(
    Y: Tensor,
    encoder_states: Tensor,
    weights: DecoderBlockWeights,
    num_heads: int,
) -> Tensor:
    """A minimal post-norm decoder block.

    The block is:
        Y1 = layer_norm(Y + masked_self_attention(Y))
        Y2 = layer_norm(Y1 + cross_attention(Y1, encoder_states))
        Y3 = layer_norm(Y2 + position_wise_ffn(Y2))
    """
    self_attn, _ = multi_head_self_attention(
        Y,
        weights.self_W_q,
        weights.self_W_k,
        weights.self_W_v,
        weights.self_W_o,
        num_heads,
        mask=causal_mask(Y.shape[1], Y.device),
    )
    Y1 = F.layer_norm(Y + self_attn, (Y.shape[-1],))
    cross_attn, _ = cross_attention(
        Y1,
        encoder_states,
        weights.cross_W_q,
        weights.cross_W_k,
        weights.cross_W_v,
        weights.cross_W_o,
        num_heads,
    )
    Y2 = F.layer_norm(Y1 + cross_attn, (Y.shape[-1],))
    ffn = position_wise_ffn(Y2, weights.W1, weights.b1, weights.W2, weights.b2)
    return F.layer_norm(Y2 + ffn, (Y.shape[-1],))
