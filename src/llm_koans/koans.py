"""LLM koans public API.

The exercises are split across numbered koan modules so the intended learning
order is visible in the file tree. Tests import this module as
`llm_koans.koans` for a stable learner-facing API.
"""

from __future__ import annotations

from .common import KoanIncomplete, TODO
from .koan_00_shapes_and_projections import (
    dot_product,
    name_shape,
    project_sequence,
    project_token,
    split_batch_and_matrix_dims,
)
from .koan_01_attention_scores import (
    attention_scores,
    attention_weights,
    scaled_scores,
    softmax_last_dim,
)
from .koan_02_self_attention import (
    context_from_weights,
    self_attention_for_one_query,
    single_head_self_attention,
)
from .koan_03_multihead_attention import (
    combine_heads,
    multi_head_self_attention,
    split_heads,
)
from .koan_04_masks import apply_attention_mask, causal_mask, masked_attention_weights
from .koan_05_blocks import (
    DecoderBlockWeights,
    EncoderBlockWeights,
    cross_attention,
    decoder_block_forward,
    encoder_block_forward,
    position_wise_ffn,
)
from .koan_06_moe import (
    MoEEncoderBlockWeights,
    expert_router_logits,
    moe_encoder_block_forward,
    routed_expert_ffn,
    top1_expert_routing,
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

__all__ = [
    "DecoderBlockWeights",
    "EncoderBlockWeights",
    "KoanIncomplete",
    "LoRALinear",
    "MoEEncoderBlockWeights",
    "TODO",
    "TinyEncoder",
    "TinyEncoderDecoder",
    "TinyTransformer",
    "TinyBaseTextClassifier",
    "TinyCausalLM",
    "add_lora_classifier_adapter",
    "apply_attention_mask",
    "assistant_only_labels",
    "attention_scores",
    "attention_weights",
    "blended_distillation_loss",
    "causal_mask",
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
    "expert_router_logits",
    "freeze_base_for_classifier_tuning",
    "load_lora_adapter_state",
    "lora_adapter_state",
    "masked_attention_weights",
    "merge_lora_linear",
    "moe_encoder_block_forward",
    "multi_head_self_attention",
    "name_shape",
    "parameter_delta_norm",
    "position_wise_ffn",
    "project_sequence",
    "project_token",
    "routed_expert_ffn",
    "scaled_scores",
    "self_attention_for_one_query",
    "sequence_logprobs",
    "sft_step",
    "single_head_self_attention",
    "softmax_last_dim",
    "split_batch_and_matrix_dims",
    "split_heads",
    "supervised_finetune_step",
    "top1_expert_routing",
    "train_one_step",
]
