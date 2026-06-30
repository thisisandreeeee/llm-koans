# LLM Koan Learning Path

The path builds from basic tensor operations to attention, Transformer blocks, and the training step that updates model parameters.

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

## 07. Training updates

You will implement:

- `train_one_step`
- `parameter_delta_norm`

Main idea:

Q, K, V vectors are temporary. The matrices that create them are trained.
