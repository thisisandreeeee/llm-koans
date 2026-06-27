"""LLM koans public API.

The exercises are split across focused modules so each learning topic stays
small. Tests import this module as `llm_koans.koans` for a stable learner-facing
API.
"""

from __future__ import annotations

from .attention import (
    attention_scores,
    attention_weights,
    combine_heads,
    context_from_weights,
    multi_head_self_attention,
    scaled_scores,
    self_attention_for_one_query,
    single_head_self_attention,
    softmax_last_dim,
    split_heads,
)
from .blocks import (
    DecoderBlockWeights,
    EncoderBlockWeights,
    cross_attention,
    decoder_block_forward,
    encoder_block_forward,
    position_wise_ffn,
)
from .common import KoanIncomplete, TODO
from .deployment import (
    GenerationRequest,
    InferenceBackend,
    create_generation_app,
    estimate_tokens,
    pack_micro_batch,
    select_backend,
    should_retry_error,
    validate_generation_budget,
)
from .masks import apply_attention_mask, causal_mask, masked_attention_weights
from .shapes import dot_product, name_shape, project_sequence, project_token
from .training import TinyAttentionClassifier, parameter_delta_norm, train_one_step

__all__ = [
    "DecoderBlockWeights",
    "EncoderBlockWeights",
    "GenerationRequest",
    "InferenceBackend",
    "KoanIncomplete",
    "TODO",
    "TinyAttentionClassifier",
    "apply_attention_mask",
    "attention_scores",
    "attention_weights",
    "causal_mask",
    "combine_heads",
    "context_from_weights",
    "create_generation_app",
    "cross_attention",
    "decoder_block_forward",
    "dot_product",
    "encoder_block_forward",
    "estimate_tokens",
    "masked_attention_weights",
    "multi_head_self_attention",
    "name_shape",
    "pack_micro_batch",
    "parameter_delta_norm",
    "position_wise_ffn",
    "project_sequence",
    "project_token",
    "scaled_scores",
    "select_backend",
    "self_attention_for_one_query",
    "should_retry_error",
    "single_head_self_attention",
    "softmax_last_dim",
    "split_heads",
    "train_one_step",
    "validate_generation_budget",
]
