"""LLM koans public API.

The exercises are split across numbered koan modules so the intended learning
order is visible in the file tree. Tests import this module as
`llm_koans.koans` for a stable learner-facing API.
"""

from __future__ import annotations

from .common import KoanIncomplete, TODO
from .koan_01_shapes import dot_product, name_shape, project_sequence, project_token
from .koan_02_attention_scores import (
    attention_scores,
    attention_weights,
    scaled_scores,
    softmax_last_dim,
)
from .koan_03_self_attention import (
    context_from_weights,
    self_attention_for_one_query,
    single_head_self_attention,
)
from .koan_04_multihead_attention import combine_heads, multi_head_self_attention, split_heads
from .koan_05_masks import apply_attention_mask, causal_mask, masked_attention_weights
from .koan_06_blocks import (
    DecoderBlockWeights,
    EncoderBlockWeights,
    cross_attention,
    decoder_block_forward,
    encoder_block_forward,
    position_wise_ffn,
)
from .koan_07_training import TinyAttentionClassifier, parameter_delta_norm, train_one_step

__all__ = [
    "DecoderBlockWeights",
    "EncoderBlockWeights",
    "KoanIncomplete",
    "TODO",
    "TinyAttentionClassifier",
    "apply_attention_mask",
    "attention_scores",
    "attention_weights",
    "causal_mask",
    "combine_heads",
    "context_from_weights",
    "cross_attention",
    "decoder_block_forward",
    "dot_product",
    "encoder_block_forward",
    "masked_attention_weights",
    "multi_head_self_attention",
    "name_shape",
    "parameter_delta_norm",
    "position_wise_ffn",
    "project_sequence",
    "project_token",
    "scaled_scores",
    "self_attention_for_one_query",
    "single_head_self_attention",
    "softmax_last_dim",
    "split_heads",
    "train_one_step",
]
