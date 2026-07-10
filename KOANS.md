# LLM Koan Learning Path

The path builds from basic tensor operations to attention, Transformer blocks, and practical fine-tuning workflows.

## 00. PyTorch matmul intuition

You will implement:

- `split_batch_and_matrix_dims`
- `matmul_vector_dot`
- `matmul_matrix_vector`
- `matmul_matrix_matrix`
- `batched_linear_projection`
- `batch_specific_linear_projection`
- `pairwise_dot_products`

Main idea:

```text
last two dims = matrix
earlier dims = batch
shared inner dim disappears
```

This is the shape rule behind projections, batched projections, and attention
scores. Learn this first and `Q @ K.transpose(-2, -1)` becomes much less spooky.

## 01. Shapes and projections

You will implement:

- `dot_product`
- `project_token`
- `project_sequence`
- `name_shape`

Main idea:

```text
matmul transforms
 dot compares
```

A projection like `q = W_q @ x` creates a specialised view of a token. A dot product like `q · k` creates one relevance score.

## 02. Attention scores

You will implement:

- `attention_scores`
- `scaled_scores`
- `softmax_last_dim`
- `attention_weights`

Main idea:

```text
Q @ K.T gives every query-token vs every key-token score.
softmax turns those raw scores into weights.
```

## 03. Self-attention

You will implement:

- `context_from_weights`
- `single_head_self_attention`
- `self_attention_for_one_query`

Main idea:

```text
weights @ V = gathered context
```

The context vector is the token after it has gathered information from relevant tokens.

## 04. Multi-head attention

You will implement:

- `split_heads`
- `combine_heads`
- `multi_head_self_attention`

Main idea:

```text
multiple heads = multiple learned ways to look at relationships
```

The common shape is:

```text
(B, T, D) -> (B, H, T, Dh)
```

## 05. Masks and decoder attention

You will implement:

- `causal_mask`
- `apply_attention_mask`
- `masked_attention_weights`

Main idea:

Decoder self-attention must not look into the future.

## 06. Encoder and decoder blocks

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

## 07. Assembling and training transformer variants

You will implement:

- `TinyTransformer.forward` — GPT‑style causal LM
- `TinyEncoder.forward` — BERT‑style bidirectional encoder
- `TinyEncoderDecoder.forward` — T5‑style encoder‑decoder with cross‑attention
- `train_one_step` — standard next‑token prediction loop
- `parameter_delta_norm` — measure how much parameters moved

Main idea:

```text
Exercise A:  nn.Embedding + position → causal mask → nn.TransformerEncoder → LM head
Exercise B:  same blocks, no mask → mean pool → classifier
Exercise C:  encoder (no mask) + decoder (causal + cross‑attention) → LM head
```

Koans 00–06 built attention mechanics by hand. This koan shows the
practitioner taxonomy: three architectures that cover virtually every
transformer in production, all using the same `nn` building blocks. If you
understand what changes between them (the mask, the pooling, the
cross‑attention target), you can reason about any transformer variant.

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

You start with the most common fine-tuning failure mode: wrong training signal. The koan makes prompt tokens disappear from the loss, then shows a small supervised head fine-tune where only the intended parameters move.

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
adapter artifact = A and B only
merge for deployment = base weight + adapter delta
```

This koan turns LoRA from a buzzword into a concrete module lifecycle: train only the adapter, save only the adapter, load it onto a fresh base, and optionally merge it into a normal linear layer.

## 10. Distillation fine-tuning

You will implement:

- `distillation_kl_loss`
- `blended_distillation_loss`
- `distillation_step`

Main idea:

```text
teacher logits -> softened distribution -> student update
```

The student learns from the teacher's probability distribution, not only from hard labels. The koan verifies that the student changes while the teacher stays frozen.

## 11. DPO preference fine-tuning

You will implement:

- `sequence_logprobs`
- `dpo_loss`
- `dpo_step`

Main idea:

```text
policy chosen-vs-rejected gap > reference chosen-vs-rejected gap
```

This koan uses prompt+completion sequences and masks the prompt so preference tuning is driven by completion tokens. The policy updates; the reference model stays frozen.

## 12. Evaluation-gated fine-tuning

You will implement:

- `classification_accuracy`
- `accept_candidate_if_improves`

Main idea:

```text
fine-tune candidate -> validation gate -> keep or roll back
```

A fine-tune is not successful because training ran. It is successful only if it clears a task-specific validation gate without regressing the baseline.

## 13. Mixture-of-Experts transformer blocks

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

This koan builds on the block mechanics from Koan 06. MoE is not "more attention" — it replaces the position-wise FFN sublayer after attention. The router scores each token against experts, picks the top expert, runs that token through the selected FFN, and scales the result by the router gate.

## 14. Tool calling for a function-calling chatbot

You will implement:

- `make_tool_schema`
- `parse_tool_arguments`
- `execute_tool_call`
- `run_tool_calling_chat`

Main idea:

```text
schema tells the model what exists
assistant tool_call asks your app to run it
tool message gives the result back
assistant final answer uses that result
```

This koan strips tool calling down to the runtime loop. The model does not execute tools. It emits a structured request. Your chatbot parses JSON arguments, dispatches to a registered Python function, appends a `role="tool"` result with the original `tool_call_id`, and calls the model again for final text.
