"""Koan 06: mixture-of-experts feed-forward layers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from .common import TODO
from .koan_03_multihead_attention import multi_head_self_attention


@dataclass
class MoEEncoderBlockWeights:
    """Weights for a minimal encoder block whose FFN is replaced by top-1 MoE."""

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


def expert_router_logits(X: Tensor, router_W: Tensor, router_b: Tensor) -> Tensor:
    """Score each token for each expert.

    X:        (B, T, D)
    router_W: (D, E)
    router_b: (E,)
    return:   (B, T, E)

    Intuition: attention decides which tokens to read from; the router decides
    which token-local FFN expert should process each token after attention.
    """
    return X @ router_W + router_b


def top1_expert_routing(router_logits: Tensor) -> tuple[Tensor, Tensor]:
    """Choose the highest-probability expert for every token.

    router_logits: (B, T, E)

    Returns:
        expert_indices: (B, T) integer expert id for each token
        expert_gates:   (B, T) softmax probability assigned to that chosen expert

    This is the tiny version of Switch Transformer routing: one token goes to
    one expert, and the chosen expert output is scaled by the router confidence.
    """
    probabilities = torch.softmax(router_logits, dim=-1)
    expert_indices = probabilities.argmax(dim=-1)
    expert_gates = probabilities.gather(-1, expert_indices.unsqueeze(-1)).squeeze(-1)
    return expert_indices, expert_gates


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
    output = torch.empty_like(X)
    for expert_id in range(expert_W1.shape[0]):
        mask = expert_indices == expert_id
        if mask.any():
            tokens = X[mask]
            output[mask] = (
                torch.relu(tokens @ expert_W1[expert_id] + expert_b1[expert_id])
                @ expert_W2[expert_id]
                + expert_b2[expert_id]
            )
    return output


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
    attn, _ = multi_head_self_attention(X, weights.W_q, weights.W_k, weights.W_v, weights.W_o, num_heads)
    X1 = F.layer_norm(X + attn, (X.shape[-1],))
    router_logits = expert_router_logits(X1, weights.router_W, weights.router_b)
    expert_indices, gates = top1_expert_routing(router_logits)
    expert_output = routed_expert_ffn(
        X1,
        expert_indices,
        weights.expert_W1,
        weights.expert_b1,
        weights.expert_W2,
        weights.expert_b2,
    )
    output = F.layer_norm(X1 + expert_output * gates.unsqueeze(-1), (X.shape[-1],))
    return output, expert_indices, gates
