"""LLM koans public API.

The exercises are split across numbered koan modules so the intended learning
order is visible in the file tree. Tests import this module as
`llm_koans.koans` for a stable learner-facing API.
"""

from __future__ import annotations

from .common import KoanIncomplete, TODO
from .koan_00_matmul_intuition import (
    batched_linear_projection,
    batch_specific_linear_projection,
    matmul_matrix_matrix,
    matmul_matrix_vector,
    matmul_vector_dot,
    pairwise_dot_products,
    split_batch_and_matrix_dims,
)
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
from .koan_07_training import (
    TinyEncoder,
    TinyEncoderDecoder,
    TinyTransformer,
    parameter_delta_norm,
    train_one_step,
)
from .koan_08_finetuning import (
    TinyBaseTextClassifier,
    TinyCausalLM,
    assistant_only_labels,
    encode_chat_messages,
    freeze_base_for_classifier_tuning,
    sft_step,
    supervised_finetune_step,
)
from .koan_09_lora_lifecycle import (
    LoRALinear,
    add_lora_classifier_adapter,
    load_lora_adapter_state,
    lora_adapter_state,
    merge_lora_linear,
)
from .koan_10_distillation import (
    blended_distillation_loss,
    distillation_kl_loss,
    distillation_step,
)
from .koan_11_dpo import dpo_loss, dpo_step, sequence_logprobs
from .koan_12_eval_gated import accept_candidate_if_improves, classification_accuracy

__all__ = [
    "DecoderBlockWeights",
    "EncoderBlockWeights",
    "KoanIncomplete",
    "LoRALinear",
    "TODO",
    "TinyEncoder",
    "TinyEncoderDecoder",
    "TinyTransformer",
    "TinyBaseTextClassifier",
    "TinyCausalLM",
    "accept_candidate_if_improves",
    "add_lora_classifier_adapter",
    "apply_attention_mask",
    "assistant_only_labels",
    "attention_scores",
    "attention_weights",
    "batched_linear_projection",
    "batch_specific_linear_projection",
    "blended_distillation_loss",
    "causal_mask",
    "classification_accuracy",
    "combine_heads",
    "context_from_weights",
    "cross_attention",
    "decoder_block_forward",
    "distillation_kl_loss",
    "distillation_step",
    "dot_product",
    "dpo_loss",
    "dpo_step",
    "encode_chat_messages",
    "encoder_block_forward",
    "freeze_base_for_classifier_tuning",
    "load_lora_adapter_state",
    "lora_adapter_state",
    "masked_attention_weights",
    "matmul_matrix_matrix",
    "matmul_matrix_vector",
    "matmul_vector_dot",
    "merge_lora_linear",
    "multi_head_self_attention",
    "name_shape",
    "pairwise_dot_products",
    "parameter_delta_norm",
    "position_wise_ffn",
    "project_sequence",
    "project_token",
    "scaled_scores",
    "self_attention_for_one_query",
    "sequence_logprobs",
    "sft_step",
    "single_head_self_attention",
    "softmax_last_dim",
    "split_batch_and_matrix_dims",
    "split_heads",
    "supervised_finetune_step",
    "train_one_step",
]
