"""Koan 06: encoder and decoder blocks."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from .common import TODO


def position_wise_ffn(X: Tensor, W1: Tensor, b1: Tensor, W2: Tensor, b2: Tensor) -> Tensor:
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


@dataclass
class MoEEncoderBlockWeights:
    """Weights for an encoder block whose FFN is replaced by top-1 MoE."""

    W_q: Tensor
    W_k: Tensor
    W_v: Tensor
    W_o: Tensor
    router_W: Tensor
    router_b: Tensor
    expert_W1: Tensor
    expert_b1: Tensor
    expert_W2: Tensor
    expert_b2: Tensor


def encoder_block_forward(X: Tensor, weights: EncoderBlockWeights, num_heads: int) -> Tensor:
    """A minimal post-norm encoder block.

    The block is:
        X1 = layer_norm(X + multi_head_self_attention(X))
        X2 = layer_norm(X1 + position_wise_ffn(X1))

    This uses functional layer_norm without trainable gamma/beta to keep the koan focused.
    """
    TODO("Implement attention -> residual+layernorm -> FFN -> residual+layernorm.")


def expert_router_logits(X: Tensor, router_W: Tensor, router_b: Tensor) -> Tensor:
    """Score each token for each expert.

    X:        (B, T, D)
    router_W: (D, E)
    router_b: (E,)
    return:   (B, T, E)

    Intuition: attention decides which tokens to read from; the router decides
    which token-local FFN expert should process each token after attention.
    """
    TODO("Compute X @ router_W + router_b to produce one score per expert.")


def top1_expert_routing(router_logits: Tensor) -> tuple[Tensor, Tensor]:
    """Choose the highest-probability expert for every token.

    router_logits: (B, T, E)

    Returns:
        expert_indices: (B, T) integer expert id for each token
        expert_gates:   (B, T) softmax probability assigned to that chosen expert

    This is the tiny version of Switch Transformer routing: one token goes to
    one expert, and the chosen expert output is scaled by the router confidence.
    """
    TODO("Softmax over experts, take argmax, then gather the selected probability.")


def routed_expert_ffn(
    X: Tensor,
    expert_indices: Tensor,
    expert_W1: Tensor,
    expert_b1: Tensor,
    expert_W2: Tensor,
    expert_b2: Tensor,
) -> Tensor:
    """Apply the selected expert FFN to each token.

    X:              (B, T, D)
    expert_indices: (B, T)
    expert_W1:      (E, D, D_ff)
    expert_b1:      (E, D_ff)
    expert_W2:      (E, D_ff, D)
    expert_b2:      (E, D)
    return:          (B, T, D)

    Each expert is a normal two-layer position-wise FFN. The MoE difference is
    that tokens can choose different FFN parameters through the router.
    """
    TODO("For each expert id, run relu(X @ W1 + b1) @ W2 + b2 on only its tokens.")


def moe_encoder_block_forward(
    X: Tensor,
    weights: MoEEncoderBlockWeights,
    num_heads: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """A minimal encoder block with a top-1 routed MoE FFN.

    The block is:
        X1 = layer_norm(X + multi_head_self_attention(X))
        logits = router(X1)
        expert_ids, gates = top1(logits)
        X2 = layer_norm(X1 + gates * routed_expert_ffn(X1, expert_ids))

    Returns:
        output:         (B, T, D)
        expert_indices: (B, T)
        expert_gates:   (B, T)
    """
    TODO("Replace the dense FFN in an encoder block with top-1 routed expert FFNs.")


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
    TODO("Implement masked self-attention, cross-attention, FFN, with residual+layernorm after each.")
