# LLM Koan Learning Path

The path moves from essential tensor operations through attention and
Transformer blocks, then into model training and adaptation.

## 00. Shapes and projections

You will implement:

- `split_batch_and_matrix_dims`
- `name_shape`
- `dot_product`
- `project_token`
- `project_sequence`

Main idea:

```text
last two dims = matrix
earlier dims = batch
matmul transforms; dot compares
```

This koan keeps only the PyTorch mechanics needed by the attention exercises.
Tokens stay in rows, so projecting a sequence uses `X @ W.T`.

## 01. Attention scores

You will implement:

- `attention_scores`
- `scaled_scores`
- `softmax_last_dim`
- `attention_weights`

Main idea:

```text
Q @ K.T gives every query-token vs. every key-token score.
softmax turns those raw scores into weights.
```

## 02. Self-attention

You will implement:

- `context_from_weights`
- `single_head_self_attention`
- `self_attention_for_one_query`

Main idea:

```text
weights @ V = gathered context
```

The context vector is the token after gathering information from relevant
tokens.

## 03. Multi-head attention

You will implement:

- `split_heads`
- `combine_heads`
- `multi_head_self_attention`

Main idea:

```text
(B, T, D) -> (B, H, T, Dh)
multiple heads = multiple learned views of token relationships
```

## 04. Masks and decoder attention

You will implement:

- `causal_mask`
- `apply_attention_mask`
- `masked_attention_weights`

Main idea: decoder self-attention must not look into the future.

## 05. Encoder and decoder blocks

You will implement:

- `position_wise_ffn`
- `encoder_block_forward`
- `cross_attention`
- `decoder_block_forward`

Main ideas:

```text
attention = tokens talk
FFN       = each token processes itself
encoder   = read
decoder   = write
```

## 06. Mixture-of-experts blocks

You will implement:

- `expert_router_logits`
- `top1_expert_routing`
- `routed_expert_ffn`
- `moe_encoder_block_forward`

Main idea:

```text
dense FFN -> router + many FFN experts
each token chooses one expert
attention structure stays the same
```

MoE follows the dense block exercise because it changes the block's
position-wise FFN, not its attention mechanism.

## 07. Assembling and training Transformer variants

You will implement:

- `TinyTransformer.forward` — GPT-style causal LM
- `TinyEncoder.forward` — BERT-style bidirectional encoder
- `TinyEncoderDecoder.forward` — T5-style encoder-decoder
- `train_one_step`
- `parameter_delta_norm`

Main idea:

```text
causal mask       -> decoder-style language model
no causal mask    -> bidirectional encoder
encoder + decoder -> sequence-to-sequence model
```

The earlier koans build Transformer mechanics by hand. This koan composes
PyTorch `nn` building blocks into trainable model variants.

## 08. SFT data and supervised fine-tuning

You will implement:

- `encode_chat_messages`
- `assistant_only_labels`
- `sft_step`
- `freeze_base_for_classifier_tuning`
- `supervised_finetune_step`

Main idea:

```text
chat messages -> template tokens -> assistant-only labels -> SFT loss
base model -> frozen base + trainable head
```

The koan focuses on the most common fine-tuning failure mode: applying loss to
the wrong tokens or updating the wrong parameters.

## 09. LoRA adapter lifecycle

You will implement:

- `LoRALinear.forward`
- `add_lora_classifier_adapter`
- `lora_adapter_state`
- `load_lora_adapter_state`
- `merge_lora_linear`

Main idea:

```text
frozen base output + low-rank adapter output = adapted model
save A and B only; optionally merge for deployment
```

## 10. Distillation

You will implement:

- `distillation_kl_loss`
- `blended_distillation_loss`
- `distillation_step`

Main idea:

```text
teacher logits -> softened distribution -> student update
```

The student learns from both hard labels and the teacher's distribution over
alternatives while the teacher remains frozen.

## 11. DPO preference fine-tuning

You will implement:

- `sequence_logprobs`
- `dpo_loss`
- `dpo_step`

Main idea:

```text
policy chosen-vs-rejected gap > reference chosen-vs-rejected gap
```

Completion masks keep prompt likelihood out of the preference signal. The
policy updates while the reference model remains frozen.
