"""LLM koans.

Your task: replace each TODO with working PyTorch code.
Run `pytest` after each small change.

This repo starts with the attention and Transformer mechanics that make LLMs
work, then widens into the practical deployment concerns you meet when turning
a model into a production service.

Conventions used in this file
-----------------------------
Single sequence examples:
    X: (T, d_model)      # one token per row
    W: (d_out, d_model)  # projection matrix
    X @ W.T -> (T, d_out)

Batched Transformer examples:
    X:      (B, T, D)
    heads:  (B, H, T, Dh)
    scores: (B, H, T_query, T_key)
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import NoReturn

import torch
import torch.nn.functional as F
from fastapi import FastAPI
from torch import Tensor, nn


class KoanIncomplete(NotImplementedError):
    """Raised by unfinished koans."""


def TODO(message: str = "Replace this TODO with your implementation.") -> NoReturn:
    raise KoanIncomplete(message)


# ---------------------------------------------------------------------------
# 01. Shapes and projections
# ---------------------------------------------------------------------------


def name_shape(x: Tensor, names: tuple[str, ...]) -> dict[str, int]:
    """Return a mapping from dimension name to size.

    Example:
        X.shape == (6, 16), names == ("tokens", "features")
        -> {"tokens": 6, "features": 16}
    """
    TODO("Create a dict mapping each provided dimension name to the matching x.shape value.")


def dot_product(a: Tensor, b: Tensor) -> Tensor:
    """Return the dot product between two 1D vectors.

    Intuition: a dot product is a single alignment score.
    """
    TODO("Compute a · b. Hint: multiply elementwise, then sum; or use torch.dot.")


def project_token(W: Tensor, x: Tensor) -> Tensor:
    """Project one token vector.

    W has shape (d_out, d_in), x has shape (d_in,), output is (d_out,).
    This is the basic q = W_q @ x idea.
    """
    TODO("Use matrix-vector multiplication to transform x with W.")


def project_sequence(X: Tensor, W: Tensor) -> Tensor:
    """Project every token in a sequence.

    X has shape (T, d_in), W has shape (d_out, d_in), output is (T, d_out).

    Keep tokens as rows. That makes the projection X @ W.T.
    """
    TODO("Project all token rows. Hint: X @ W.T")


# ---------------------------------------------------------------------------
# 02. Attention scores and weights
# ---------------------------------------------------------------------------


def attention_scores(Q: Tensor, K: Tensor) -> Tensor:
    """Compute raw attention scores for a single sequence or batch of heads.

    For 2D tensors:
        Q: (T_query, d_k), K: (T_key, d_k)
        return: (T_query, T_key)

    For higher-rank tensors, the same rule applies to the last two dims:
        Q: (..., T_query, d_k), K: (..., T_key, d_k)
        return: (..., T_query, T_key)
    """
    TODO("Compute Q @ K transposed over its last two dimensions.")


def scaled_scores(scores: Tensor, d_k: int) -> Tensor:
    """Scale attention scores by sqrt(d_k)."""
    TODO("Divide scores by math.sqrt(d_k).")


def softmax_last_dim(scores: Tensor) -> Tensor:
    """Apply softmax over the last dimension.

    In attention, the last dimension is usually the key-token dimension.
    Each row becomes weights over all keys.
    """
    TODO("Use torch.softmax(..., dim=-1).")


def attention_weights(Q: Tensor, K: Tensor) -> Tensor:
    """Compute scaled-softmax attention weights from Q and K."""
    TODO("Combine attention_scores, scaled_scores, and softmax_last_dim.")


# ---------------------------------------------------------------------------
# 03. Self-attention and context vectors
# ---------------------------------------------------------------------------


def context_from_weights(weights: Tensor, V: Tensor) -> Tensor:
    """Use attention weights to collect value information.

    weights: (..., T_query, T_key)
    V:       (..., T_key, d_v)
    output:  (..., T_query, d_v)
    """
    TODO("The context vector is weights @ V.")


def self_attention_for_one_query(query: Tensor, keys: Tensor, values: Tensor) -> tuple[Tensor, Tensor]:
    """Compute attention for one query vector over all keys and values.

    query:  (d_k,)
    keys:   (T, d_k)
    values: (T, d_v)

    Returns:
        context: (d_v,)
        weights: (T,)
    """
    TODO("Compute scores query @ keys.T, softmax them, then use weights @ values.")


def single_head_self_attention(X: Tensor, W_q: Tensor, W_k: Tensor, W_v: Tensor) -> tuple[Tensor, Tensor]:
    """Single-head self-attention for one sequence.

    X:   (T, d_model)
    W_q: (d_k, d_model)
    W_k: (d_k, d_model)
    W_v: (d_v, d_model)

    Returns:
        context: (T, d_v)
        weights: (T, T)
    """
    TODO("Project X to Q/K/V, compute weights, then compute context.")


# ---------------------------------------------------------------------------
# 04. Multi-head attention
# ---------------------------------------------------------------------------


def split_heads(X: Tensor, num_heads: int) -> Tensor:
    """Split a batched sequence into attention heads.

    X:      (B, T, D)
    return: (B, H, T, Dh), where Dh = D // H

    The operation should not change the data order; it only reshapes and permutes.
    """
    TODO("Reshape to (B, T, H, Dh), then permute to (B, H, T, Dh).")


def combine_heads(X: Tensor) -> Tensor:
    """Combine attention heads back into the model dimension.

    X:      (B, H, T, Dh)
    return: (B, T, H * Dh)
    """
    TODO("Permute to (B, T, H, Dh), make contiguous, then view/reshape.")


def multi_head_self_attention(
    X: Tensor,
    W_q: Tensor,
    W_k: Tensor,
    W_v: Tensor,
    W_o: Tensor,
    num_heads: int,
    mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Multi-head self-attention using explicit projection matrices.

    X:   (B, T, D)
    W_q: (D, D)
    W_k: (D, D)
    W_v: (D, D)
    W_o: (D, D)

    This repo uses row-vector projections for batched model code:
        Q = X @ W_q

    Returns:
        output:  (B, T, D)
        weights: (B, H, T, T)
    """
    TODO("Project, split heads, attend, combine heads, then apply W_o.")


# ---------------------------------------------------------------------------
# 05. Masks and decoder attention
# ---------------------------------------------------------------------------


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
    TODO("Use torch.triu with diagonal=1 to create an upper-triangular boolean mask.")


def apply_attention_mask(scores: Tensor, mask: Tensor | None) -> Tensor:
    """Apply an attention mask by setting masked scores to -inf.

    scores can be (T, T), (B, H, T, T), or cross-attention-shaped.
    mask should broadcast over scores.
    """
    TODO("Use scores.masked_fill(mask, float('-inf')), but return scores unchanged if mask is None.")


def masked_attention_weights(Q: Tensor, K: Tensor, mask: Tensor | None) -> Tensor:
    """Scaled-softmax attention weights with an optional mask."""
    TODO("Compute scores, apply mask, then softmax over the last dimension.")


# ---------------------------------------------------------------------------
# 06. Encoder and decoder blocks
# ---------------------------------------------------------------------------


def position_wise_ffn(X: Tensor, W1: Tensor, b1: Tensor, W2: Tensor, b2: Tensor) -> Tensor:
    """Apply the same two-layer feed-forward network to every token.

    X:  (..., D)
    W1: (D, D_ff)
    b1: (D_ff,)
    W2: (D_ff, D)
    b2: (D,)

    Intuition: attention lets tokens talk; FFN lets each token process itself.
    """
    TODO("Compute relu(X @ W1 + b1) @ W2 + b2.")


@dataclass
class EncoderBlockWeights:
    W_q: Tensor
    W_k: Tensor
    W_v: Tensor
    W_o: Tensor
    W1: Tensor
    b1: Tensor
    W2: Tensor
    b2: Tensor


@dataclass
class DecoderBlockWeights:
    self_W_q: Tensor
    self_W_k: Tensor
    self_W_v: Tensor
    self_W_o: Tensor
    cross_W_q: Tensor
    cross_W_k: Tensor
    cross_W_v: Tensor
    cross_W_o: Tensor
    W1: Tensor
    b1: Tensor
    W2: Tensor
    b2: Tensor


def encoder_block_forward(X: Tensor, weights: EncoderBlockWeights, num_heads: int) -> Tensor:
    """A minimal post-norm encoder block.

    The block is:
        X1 = layer_norm(X + multi_head_self_attention(X))
        X2 = layer_norm(X1 + position_wise_ffn(X1))

    This uses functional layer_norm without trainable gamma/beta to keep the koan focused.
    """
    TODO("Implement attention -> residual+layernorm -> FFN -> residual+layernorm.")


def cross_attention(
    decoder_states: Tensor,
    encoder_states: Tensor,
    W_q: Tensor,
    W_k: Tensor,
    W_v: Tensor,
    W_o: Tensor,
    num_heads: int,
) -> tuple[Tensor, Tensor]:
    """Cross-attention: queries from decoder, keys/values from encoder.

    decoder_states: (B, T_dec, D)
    encoder_states: (B, T_enc, D)

    Returns:
        output:  (B, T_dec, D)
        weights: (B, H, T_dec, T_enc)
    """
    TODO("Project Q from decoder states; project K/V from encoder states; then attend.")


def decoder_block_forward(
    Y: Tensor,
    encoder_states: Tensor,
    weights: DecoderBlockWeights,
    num_heads: int,
) -> Tensor:
    """A minimal post-norm decoder block.

    The block is:
        Y1 = layer_norm(Y + masked_self_attention(Y))
        Y2 = layer_norm(Y1 + cross_attention(Y1, encoder_states))
        Y3 = layer_norm(Y2 + position_wise_ffn(Y2))
    """
    TODO("Implement masked self-attention, cross-attention, FFN, with residual+layernorm after each.")


# ---------------------------------------------------------------------------
# 07. Training updates
# ---------------------------------------------------------------------------


class TinyAttentionClassifier(nn.Module):
    """A tiny model used by the training koans.

    It is intentionally simple:
        embeddings -> one multi-head self-attention layer -> mean pooling -> classifier
    """

    def __init__(self, vocab_size: int, d_model: int, num_heads: int, num_classes: int):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, token_ids: Tensor) -> Tensor:
        X = self.embedding(token_ids)
        attended, _ = self.attn(X, X, X, need_weights=False)
        pooled = attended.mean(dim=1)
        return self.classifier(pooled)


def train_one_step(model: nn.Module, optimizer: torch.optim.Optimizer, token_ids: Tensor, labels: Tensor) -> Tensor:
    """Run one supervised training step and return the loss tensor.

    The expected training loop is:
        optimizer.zero_grad()
        logits = model(token_ids)
        loss = cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
    """
    TODO("Implement one standard PyTorch training step with cross-entropy loss.")


def parameter_delta_norm(before: dict[str, Tensor], after: dict[str, Tensor]) -> float:
    """Return the total L2 norm of parameter changes between two state dict snapshots."""
    TODO("Sum squared parameter differences across matching keys, then return sqrt(total).")


# ---------------------------------------------------------------------------
# 08. LLM deployment: from toy serving to production pressure
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GenerationRequest:
    """A minimal generation request used by deployment koans.

    Real systems usually include more fields, but these three show up almost
    everywhere: the prompt, a generation budget, and a sampling temperature.
    """

    prompt: str
    max_new_tokens: int
    temperature: float = 0.7


@dataclass(frozen=True)
class InferenceBackend:
    """A deployable model endpoint with capacity metadata.

    `max_context_tokens` protects a single request from exceeding the model's
    context window. `max_batch_tokens` protects throughput-oriented batching.
    """

    name: str
    max_context_tokens: int
    max_batch_tokens: int
    healthy: bool = True


def estimate_tokens(text: str) -> int:
    """Estimate tokens for capacity planning.

    Production systems should use the model tokenizer. This koan intentionally
    starts with the common rough heuristic: about four characters per token,
    rounded up, with blank strings costing zero.
    """
    TODO("Return 0 for blank text; otherwise return ceil(len(text) / 4).")


def validate_generation_budget(
    prompt: str,
    max_new_tokens: int,
    context_window: int,
    reserved_tokens: int = 0,
) -> dict[str, int | bool]:
    """Report whether a generation request fits inside a context window.

    The total budget is prompt tokens + requested output tokens + reserved
    tokens. Reserved tokens cover system prompts, tool schemas, routing headers,
    or safety wrappers that are easy to forget in prototypes.
    """
    TODO("Return prompt_tokens, requested_tokens, reserved_tokens, total_tokens, fits, and overflow_tokens.")


def create_generation_app(
    generate_text: Callable[[str, int, float], str],
    model_name: str,
) -> FastAPI:
    """Create a tiny FastAPI app for LLM text generation.

    Implement two endpoints:
        GET  /health   -> {"status": "ok", "model": model_name}
        POST /generate -> {"model": model_name, "text": generated_text}

    Reject empty prompts with HTTP 400 before calling the model. This is the
    bridge from "I can run inference in a notebook" to "I can serve it safely".
    """
    TODO("Build a FastAPI app with /health and /generate endpoints.")


def select_backend(
    prompt: str,
    max_new_tokens: int,
    backends: Sequence[InferenceBackend],
    reserved_tokens: int = 0,
) -> InferenceBackend:
    """Choose the smallest healthy backend that can fit the request.

    This models a common production move: route tiny requests to cheaper
    capacity, skip unhealthy backends, and escalate large context-window jobs to
    bigger serving pools.
    """
    TODO("Filter to healthy backends that fit the token budget, then return the one with the smallest max_context_tokens.")


def pack_micro_batch(
    requests: Sequence[GenerationRequest],
    max_batch_tokens: int,
) -> list[GenerationRequest]:
    """Greedily pack generation requests under a token budget.

    Dynamic batching improves throughput, but production systems must bound the
    combined prompt + output budget so one batch does not blow up latency or
    memory. Preserve input order and stop before the first request that would
    exceed the budget.
    """
    TODO("Accumulate request token costs until the next request would exceed max_batch_tokens.")


def should_retry_error(status_code: int | None, error_message: str) -> bool:
    """Return whether a failed generation call should be retried.

    Retry transient failures such as rate limits, overloaded/model-loading
    servers, and timeouts. Do not retry permanent caller problems like bad
    prompts, auth failures, or context-window overflow.
    """
    TODO("Retry 429/5xx/timeouts; do not retry 4xx auth/bad-request or context-length errors.")
