"""Koan 04: masks and decoder attention."""

from __future__ import annotations

import torch
from torch import Tensor

from .common import TODO


def causal_mask(seq_len: int, device: torch.device | None = None) -> Tensor:
    """Return a boolean mask that blocks future tokens.

    Shape: (seq_len, seq_len)

    True means "mask this out".

    Example for seq_len=4:
        [[False, True,  True,  True ],
         [False, False, True,  True ],
         [False, False, False, True ],
         [False, False, False, False]]
    """
    TODO(
        "Construct a boolean grid where each position blocks only tokens to its right."
    )


def apply_attention_mask(scores: Tensor, mask: Tensor | None) -> Tensor:
    """Apply an attention mask by setting masked scores to -inf.

    scores can be (T, T), (B, H, T, T), or cross-attention-shaped.
    mask should broadcast over scores.
    """
    TODO(
        "Make blocked scores contribute no probability; treat a missing mask as a no-op."
    )


def masked_attention_weights(Q: Tensor, K: Tensor, mask: Tensor | None) -> Tensor:
    """Scaled-softmax attention weights with an optional mask."""
    TODO(
        "Produce normalized attention weights while giving blocked positions no probability."
    )
