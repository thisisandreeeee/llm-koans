"""Koan 05: encoder and decoder blocks."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from .common import TODO


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
    TODO("Compute relu(X @ W1 + b1) @ W2 + b2.")


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
    TODO("Implement self-attention, FFN, and residual+layernorm after each.")


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
    TODO("Project Q from decoder states; project K/V from encoder states; then attend.")


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
    TODO(
        "Implement masked self-attention, cross-attention, FFN, with residual+layernorm after each."
    )
