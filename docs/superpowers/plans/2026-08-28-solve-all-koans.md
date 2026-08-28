# Solve All LLM Koans Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace every focused-koan placeholder with the smallest implementation that satisfies the repository tests.

**Architecture:** Keep the existing public API and module boundaries. Implement tensor mechanics with PyTorch operations, compose the supplied model components directly, and use the existing tests as the failing-first contract.

**Tech Stack:** Python 3.12 via `uv run`, PyTorch, pytest.

**Spec:** `KOANS.md` and the matching tests under `tests/`.

## Global Constraints

- Preserve all existing function signatures and public exports.
- Use the repository’s documented tensor shape conventions.
- Do not add dependencies or unrelated refactors.
- Run focused tests after each koan and `uv run pytest` before the PR.

---

### Task 1: Shapes, projections, and attention scores

**Files:**
- Modify: `src/llm_koans/koan_00_shapes_and_projections.py`
- Modify: `src/llm_koans/koan_01_attention_scores.py`
- Test: `tests/test_00_shapes_and_projections.py`, `tests/test_01_attention_scores.py`

**Interfaces:**
- `split_batch_and_matrix_dims`, `name_shape`, `dot_product`, `project_token`, and `project_sequence` remain the public shape/projection helpers.
- `attention_scores`, `scaled_scores`, `softmax_last_dim`, and `attention_weights` return the documented tensor shapes.

- [ ] **Step 1: Confirm the existing tests are red**

Run: `uv run pytest tests/test_00_shapes_and_projections.py tests/test_01_attention_scores.py -q`
Expected: failures come from `KoanIncomplete` placeholders.

- [ ] **Step 2: Implement the minimal tensor operations**

```python
return shape[:-2], shape[-2:]
return dict(zip(names, x.shape))
return (a * b).sum()
return W @ x
return X @ W.T
return Q @ K.transpose(-2, -1)
return scores / d_k**0.5
return torch.softmax(scores, dim=-1)
return softmax_last_dim(scaled_scores(attention_scores(Q, K), Q.shape[-1]))
```

- [ ] **Step 3: Run the focused tests**

Run: `uv run pytest tests/test_00_shapes_and_projections.py tests/test_01_attention_scores.py -q`
Expected: all tests pass.

### Task 2: Single-head self-attention

**Files:**
- Modify: `src/llm_koans/koan_02_self_attention.py`
- Test: `tests/test_02_self_attention.py`

**Interfaces:**
- Context functions preserve leading batch dimensions and return `(context, weights)` for one-query and sequence APIs.

- [ ] **Step 1: Implement weighted context and the vectorized formula**

```python
return weights @ V
scores = query @ keys.T
weights = torch.softmax(scores / scores.shape[-1]**0.5, dim=-1)
return context_from_weights(weights, values), weights
Q, K, V = X @ W_q.T, X @ W_k.T, X @ W_v.T
scores = attention_scores(Q, K)
weights = softmax_last_dim(scores / Q.shape[-1]**0.5)
return context_from_weights(weights, V), weights
```

- [ ] **Step 2: Run `uv run pytest tests/test_02_self_attention.py -q` and confirm all tests pass.**

### Task 3: Multi-head attention

**Files:**
- Modify: `src/llm_koans/koan_03_multihead_attention.py`
- Test: `tests/test_03_multihead_attention.py`

**Interfaces:**
- `split_heads` maps `(B, T, D)` to `(B, H, T, D//H)`; `combine_heads` reverses it.
- `multi_head_self_attention` applies row-vector projections, per-head scaled attention, an optional broadcastable mask, and `W_o`.

- [ ] **Step 1: Implement reshape/permutation and the reference formula**

```python
B, T, D = X.shape
return X.reshape(B, T, num_heads, D // num_heads).permute(0, 2, 1, 3)
B, H, T, Dh = X.shape
return X.permute(0, 2, 1, 3).contiguous().reshape(B, T, H * Dh)
Q = split_heads(X @ W_q, num_heads)
K = split_heads(X @ W_k, num_heads)
V = split_heads(X @ W_v, num_heads)
scores = Q @ K.transpose(-2, -1) / Q.shape[-1]**0.5
if mask is not None:
    scores = scores.masked_fill(mask, float("-inf"))
weights = torch.softmax(scores, dim=-1)
return combine_heads(weights @ V) @ W_o, weights
```

- [ ] **Step 2: Run `uv run pytest tests/test_03_multihead_attention.py -q` and confirm all tests pass.**

### Task 4: Masks and dense encoder/decoder blocks

**Files:**
- Modify: `src/llm_koans/koan_04_masks.py`
- Modify: `src/llm_koans/koan_05_blocks.py`
- Test: `tests/test_04_masks_and_decoder_attention.py`, `tests/test_05_encoder_decoder_blocks.py`

**Interfaces:**
- Masks use `True` for blocked positions and broadcast over score tensors.
- Block functions retain post-norm residual ordering and the existing dataclasses.

- [ ] **Step 1: Implement masks and FFN**

```python
return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
return scores if mask is None else scores.masked_fill(mask, float("-inf"))
scores = attention_scores(Q, K) / Q.shape[-1]**0.5
return torch.softmax(apply_attention_mask(scores, mask), dim=-1)
return torch.relu(X @ W1 + b1) @ W2 + b2
```

- [ ] **Step 2: Implement encoder, cross-attention, and decoder composition using the existing `split_heads`/`combine_heads` helpers and `torch.nn.functional.layer_norm`.**

```python
attn, _ = multi_head_self_attention(X, weights.W_q, weights.W_k, weights.W_v, weights.W_o, num_heads)
X1 = F.layer_norm(X + attn, (X.shape[-1],))
return F.layer_norm(X1 + position_wise_ffn(X1, weights.W1, weights.b1, weights.W2, weights.b2), (X.shape[-1],))
```

Use separate decoder queries and encoder keys/values for cross-attention, then apply causal self-attention, cross-attention, and FFN in the documented order.

- [ ] **Step 3: Run `uv run pytest tests/test_04_masks_and_decoder_attention.py tests/test_05_encoder_decoder_blocks.py -q` and confirm all tests pass.**

### Task 5: Mixture-of-experts routing

**Files:**
- Modify: `src/llm_koans/koan_06_moe.py`
- Test: `tests/test_06_moe.py`

**Interfaces:**
- Routing returns `(B, T)` expert indices and gates; routed FFN returns `(B, T, D)`.

- [ ] **Step 1: Implement routing and per-expert FFNs**

```python
return X @ router_W + router_b
probabilities = torch.softmax(router_logits, dim=-1)
indices = probabilities.argmax(dim=-1)
gates = probabilities.gather(-1, indices.unsqueeze(-1)).squeeze(-1)
```

For each expert, select `X[expert_indices == expert_id]`, apply `relu(tokens @ expert_W1[expert_id] + expert_b1[expert_id]) @ expert_W2[expert_id] + expert_b2[expert_id]`, and write the results into an `empty_like(X)` output. Compose attention, layer norm, router, gated expert output, and final layer norm as documented.

- [ ] **Step 2: Run `uv run pytest tests/test_06_moe.py -q` and confirm all tests pass.**

### Task 6: Transformer assembly and training helpers

**Files:**
- Modify: `src/llm_koans/koan_07_training.py`
- Test: `tests/test_07_training_updates.py`

**Interfaces:**
- The three model `forward` methods preserve the supplied output shapes and causal/bidirectional behavior.
- `train_one_step` returns a detached scalar loss; `parameter_delta_norm` returns a float.

- [ ] **Step 1: Implement position-aware model forwards**

```python
positions = torch.arange(token_ids.shape[1], device=token_ids.device)
x = self.token_embedding(token_ids) + self.position_embedding(positions)
return self.lm_head(self.transformer(x, mask=_causal_mask(token_ids.shape[1], token_ids.device)))
```

Use the same position pattern without a mask for `TinyEncoder`, mean-pool its transformer output, and classify. For `TinyEncoderDecoder`, encode source without a mask, decode target with `_causal_mask`, and project decoder states with `lm_head`.

- [ ] **Step 2: Implement one-step training and norm aggregation**

```python
optimizer.zero_grad()
logits = model(token_ids)
loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), token_ids[:, 1:].reshape(-1))
loss.backward()
optimizer.step()
return loss.detach()
return sum((before[name] - after[name]).pow(2).sum() for name in before).sqrt().item()
```

- [ ] **Step 3: Run `uv run pytest tests/test_07_training_updates.py -q` and confirm all tests pass.**

### Task 7: SFT and classifier fine-tuning

**Files:**
- Modify: `src/llm_koans/koan_08_finetuning.py`
- Test: `tests/test_08_finetuning.py`

**Interfaces:**
- Chat encoding emits role token, whitespace-separated content tokens, and EOS for each message.
- SFT labels mask prompts with `ignore_index`; tuning helpers control `requires_grad` exactly as documented.

- [ ] **Step 1: Implement encoding and assistant-only labels**

```python
ids = []
for message in messages:
    ids.append(vocab[f"<{message['role']}>" ])
    ids.extend(vocab[token] for token in message["content"].split())
    ids.append(vocab[eos_token])
return torch.tensor(ids, dtype=torch.long)
labels = torch.full_like(input_ids, ignore_index)
assistant = False
for i, token in enumerate(input_ids.tolist()):
    if token == assistant_token_id:
        assistant = True
    elif assistant:
        labels[i] = token
        if token == eos_token_id:
            assistant = False
return labels
```

- [ ] **Step 2: Implement shifted SFT loss and classifier-only training**

Use `F.cross_entropy(logits[:, :-1].reshape(-1, V), labels[:, 1:].reshape(-1), ignore_index=ignore_index)`, then zero gradients, backpropagate, step, and detach. Set all base parameters false and classifier parameters true; classifier fine-tuning uses ordinary cross-entropy on `model(token_ids)`.

- [ ] **Step 3: Run `uv run pytest tests/test_08_finetuning.py -q` and confirm all tests pass.**

### Task 8: LoRA lifecycle

**Files:**
- Modify: `src/llm_koans/koan_09_lora_lifecycle.py`
- Test: `tests/test_09_lora_lifecycle.py`

**Interfaces:**
- LoRA output is the frozen linear output plus `(x @ A @ B) * (alpha / rank)`.
- Adapter save/load handles only independent `A` and `B` tensors; merging returns an ordinary equivalent `nn.Linear`.

- [ ] **Step 1: Implement adapter forward and installation**

```python
return F.linear(x, self.weight, self.bias) + (x @ self.A @ self.B) * (self.alpha / self.rank)
for parameter in model.parameters():
    parameter.requires_grad = False
model.classifier = LoRALinear(model.classifier, rank, alpha)
model.classifier.A.requires_grad = True
model.classifier.B.requires_grad = True
return model
```

- [ ] **Step 2: Implement artifact lifecycle**

```python
return {"A": layer.A.detach().clone(), "B": layer.B.detach().clone()}
with torch.no_grad():
    layer.A.copy_(adapter_state["A"])
    layer.B.copy_(adapter_state["B"])
return layer
merged = nn.Linear(layer.weight.shape[1], layer.weight.shape[0], bias=layer.bias is not None)
with torch.no_grad():
    merged.weight.copy_(layer.weight + (layer.A @ layer.B * (layer.alpha / layer.rank)).T)
    if layer.bias is not None:
        merged.bias.copy_(layer.bias)
return merged
```

- [ ] **Step 3: Run `uv run pytest tests/test_09_lora_lifecycle.py -q` and confirm all tests pass.**

### Task 9: Distillation

**Files:**
- Modify: `src/llm_koans/koan_10_distillation.py`
- Test: `tests/test_10_distillation.py`

**Interfaces:**
- KL uses softened teacher probabilities and student log-probabilities, scaled by `temperature**2`.
- The teacher is evaluated under `torch.no_grad()` and only the student optimizer steps.

- [ ] **Step 1: Implement losses**

```python
teacher_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1)
student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
return (teacher_log_probs.exp() * (teacher_log_probs - student_log_probs)).sum(dim=-1).mean() * temperature**2
return alpha * distillation_kl_loss(student_logits, teacher_logits, temperature) + (1 - alpha) * F.cross_entropy(student_logits, hard_labels)
```

- [ ] **Step 2: Implement the student-only update and run `uv run pytest tests/test_10_distillation.py -q`.**

### Task 10: DPO preference tuning

**Files:**
- Modify: `src/llm_koans/koan_11_dpo.py`
- Test: `tests/test_11_dpo.py`

**Interfaces:**
- Sequence scores gather target log-probabilities, shift to next-token predictions, and sum only masked completion positions.
- DPO compares policy and reference chosen-vs-rejected gaps; reference evaluation is gradient-free.

- [ ] **Step 1: Implement sequence scoring and DPO loss**

```python
token_logps = F.log_softmax(logits, dim=-1).gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
token_logps = token_logps[:, 1:]
if mask is not None:
    token_logps = token_logps.masked_fill(~mask[:, 1:], 0)
return token_logps.sum(dim=-1)
return -F.logsigmoid(beta * ((policy_chosen_logp - policy_rejected_logp) - (reference_chosen_logp - reference_rejected_logp))).mean()
```

- [ ] **Step 2: Implement policy/reference forward scoring, optimizer step, and run `uv run pytest tests/test_11_dpo.py -q`.**

### Task 11: Full verification and PR

**Files:**
- Verify all modified source and plan files.

- [ ] **Step 1: Run the complete suite**

Run: `uv run pytest`
Expected: 52 passed, 0 failed.

- [ ] **Step 2: Inspect the diff and verify no TODO placeholders remain in focused source modules.**

Run: `git diff --check` and `rg -n 'TODO\(' src/llm_koans`
Expected: no whitespace errors; only intentional public `TODO` helper references if any remain.

- [ ] **Step 3: Commit the model answers**

```bash
git add src/llm_koans docs/superpowers/plans/2026-08-28-solve-all-koans.md
git commit -m "feat: add model answers for all LLM koans"
```

- [ ] **Step 4: Push the branch and create a PR without merging**

```bash
git push -u origin ref/model-answers
gh pr create --base main --head ref/model-answers --title "ref: model answers for all LLM koans" --body "Implements every koan placeholder with minimal PyTorch solutions.\n\nTests: uv run pytest (52 passed)."
```
