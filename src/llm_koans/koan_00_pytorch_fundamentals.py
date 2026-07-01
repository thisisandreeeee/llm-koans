"""Koan 00: PyTorch fundamentals — shapes, ops, and the training loop.

These exercises build the core tensor-operation intuition that makes transformer
code (and deep learning code in general) feel like a first language, not a
foreign one.  We cover:

    Part 1 — Naming axes: the (B, T, D) mental model
    Part 2 — Matmul at every scale: vector, matrix, batched
    Part 3 — Linear layers act on the last dimension
    Part 4 — Transpose brings axes together (the attention shape move)
    Part 5 — Softmax over choices
    Part 6 — Reshape vs. transpose (the multi‑head split/merge dance)
    Part 7 — CrossEntropyLoss convention (the annoying dim‑1 quirk)
    Part 8 — The training loop in five essential lines

Nearly every transformer operation lives in one of these patterns.  When you
get lost in a sea of dims, return to the core mantra:

    Hidden states:     (B, T, D)
    Attention scores:  (B, H, T, T)
    Attention output:  (B, T, D)
    LM logits:         (B, T, V)
    CrossEntropy wants: (B, V, T)  or  (B*T, V)

And the shape‑preserving intuition: transformer blocks go (B,T,D) → (B,T,D).
Only the final lm_head flips the last dim: (B,T,D) → (B,T,V).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .common import TODO

Shape = tuple[int, ...]

# ═══════════════════════════════════════════════════════════════════════════════
# Part 1 — Naming axes
# ═══════════════════════════════════════════════════════════════════════════════


def name_axes(shape: Shape, names: tuple[str, ...]) -> dict[str, int]:
    """Map each dimension name to its size.

    This is rule zero of transformer tensor work: always name your axes.
    The core vocabulary is:

        B = batch      — which example?
        T = time       — which token position?
        D = depth      — feature vector for that token
        H = heads      — which attention head?
        V = vocab      — vocabulary size

    Example:
        shape=(2, 10, 64), names=("B", "T", "D")  →  {"B": 2, "T": 10, "D": 64}
    """
    TODO("Zip names with shape sizes into a dict.")


# ═══════════════════════════════════════════════════════════════════════════════
# Part 2 — Matmul at every scale
# ═══════════════════════════════════════════════════════════════════════════════


def split_batch_and_matrix_dims(shape: Shape) -> tuple[Shape, Shape]:
    """Split a 2D-or-larger shape into batch dims and matrix dims.

    PyTorch matmul treats the last two dims as the matrix and every earlier
    dimension as independent batch.  This is the key rule for understanding
    every @ in transformer code.

    Examples:
        (T, D)        →  batch=(),        matrix=(T, D)
        (B, T, D)     →  batch=(B,),      matrix=(T, D)
        (B, H, T, D)  →  batch=(B, H),    matrix=(T, D)
    """
    TODO("Return (shape[:-2], shape[-2:]) for tensors with at least two dimensions.")


def matmul_vector_dot(a: Tensor, b: Tensor) -> Tensor:
    """Multiply two 1D vectors with matmul.

    1D @ 1D consumes both vector dimensions and returns a scalar dot product.
    Result shape is torch.Size([]) — a single number.
    """
    TODO("Use the @ operator or torch.matmul for the vector dot product.")


def matmul_matrix_vector(matrix: Tensor, vector: Tensor) -> Tensor:
    """Multiply a (rows, features) matrix by a (features,) vector.

    The shared feature dimension is contracted away.  Result is (rows,).
    This is the core of "project one token": W @ x.
    """
    TODO("Use matrix @ vector.")


def matmul_matrix_matrix(left: Tensor, right: Tensor) -> Tensor:
    """Multiply two 2D matrices.

    left:  (rows, shared)    — each row is a query vector
    right: (shared, cols)    — each column is a key vector
    result: (rows, cols)     — every query compared to every key

    Only the shared middle dimension disappears.
    """
    TODO("Use left @ right. The inner dimensions must match.")


# ═══════════════════════════════════════════════════════════════════════════════
# Part 3 — Linear layers act on the last dimension
# ═══════════════════════════════════════════════════════════════════════════════


def batched_linear_projection(tokens: Tensor, weight: Tensor) -> Tensor:
    """Apply the same projection matrix to every token in a batch.

    tokens: (B, T, D)    — a batch of token sequences
    weight: (D, E)       — one projection matrix
    result: (B, T, E)    — every token independently projected

    This is the most important rule of transformer code: nn.Linear(D, V)
    does NOT mix batches or token positions.  It transforms each token
    vector independently:

        for every batch b:
            for every token t:
                convert x[b, t, :] from size D to size E

    So lm_head(hidden_states) means: "for every token position, convert
    the hidden vector into vocabulary logits."
    """
    TODO("Use tokens @ weight. The D dimension is consumed, E is produced.")


def batch_specific_linear_projection(tokens: Tensor, weights: Tensor) -> Tensor:
    """Apply a different projection matrix to each batch item.

    tokens:  (B, T, D)    — a batch of token sequences
    weights: (B, D, E)    — one projection matrix per batch item
    result:  (B, T, E)

    The leading B dimensions match, so PyTorch performs one matrix multiply
    per batch item.  All earlier dims that match are treated as batch.
    """
    TODO("Use tokens @ weights. Matching leading dims are batch dims.")


# ═══════════════════════════════════════════════════════════════════════════════
# Part 4 — Transpose brings axes together
# ═══════════════════════════════════════════════════════════════════════════════


def pairwise_dot_products(left: Tensor, right: Tensor) -> Tensor:
    """Compute every row-vector dot product between two tensors.

    left:  (..., N, D)    — N query vectors of size D
    right: (..., M, D)    — M key vectors of size D
    result: (..., N, M)   — one score per (query, key) pair

    This is the core shape move behind Q @ K.transpose(-2, -1) in attention:

        Q:            (B, T, D)
        K transposed: (B, D, T)
        result:       (B, T, T)   ← "each query token vs. each key token"

    Rule: transpose when the dimension you want to dot-product over is not
    in the right place.  In attention, you dot-product over D, not over T.
    """
    TODO("Transpose the last two dims of right, then matmul: left @ right.transpose(-2, -1).")


# ═══════════════════════════════════════════════════════════════════════════════
# Part 5 — Softmax over choices
# ═══════════════════════════════════════════════════════════════════════════════


def softmax_over_choices(scores: Tensor) -> Tensor:
    """Apply softmax along the last dimension of attention scores.

    Attention scores are (B, T, T) — for each query token, you have scores
    over all key tokens.  Softmax should happen over the key/choice dimension:

        attention_weights = softmax(scores, dim=-1)

    Meaning: for each query token, distribute 100% attention across the key
    tokens.  The rule is:

        softmax over last dim = choose which tokens to attend to

    In multi-head attention the scores are (B, H, T, T) and the rule is the
    same: softmax over dim=-1, the key‑token axis.
    """
    TODO("Apply torch.softmax along dim=-1.")


# ═══════════════════════════════════════════════════════════════════════════════
# Part 6 — Reshape vs. transpose (the multi‑head split/merge)
# ═══════════════════════════════════════════════════════════════════════════════


def split_heads_for_attention(x: Tensor, num_heads: int) -> Tensor:
    """Split the feature dimension into heads and rearrange for attention.

    x: (B, T, D)  where D = num_heads * d_head

    Steps:
    1. reshape:  (B, T, D) → (B, T, H, d_head)   — split D
    2. transpose: (B, T, H, d_head) → (B, H, T, d_head)  — put H before T

    Why the transpose?  Because each head does its own attention
    independently.  With shape (B, H, T, d_head), the attention matmul

        Q @ K.transpose(-2, -1)

    naturally gives (B, H, T, T) — one attention matrix per head.

    Rule:
        reshape = combine or split dimensions
        transpose = swap which dimension is in which position

    Never use view() when you need to transpose — view requires contiguous
    memory; reshape handles that for you.
    """
    B, T, D = x.shape
    d_head = D // num_heads
    TODO("Reshape to (B, T, H, d_head), then transpose to (B, H, T, d_head).")


def merge_heads_after_attention(x: Tensor) -> Tensor:
    """Reverse the head split: merge (B, H, T, d_head) back to (B, T, D).

    Steps:
    1. transpose: (B, H, T, d_head) → (B, T, H, d_head)  — put H back beside D
    2. reshape:   (B, T, H, d_head) → (B, T, D)           — merge H * d_head
    """
    TODO("Transpose back, then reshape to merge heads into D.")


# ═══════════════════════════════════════════════════════════════════════════════
# Part 7 — CrossEntropyLoss convention
# ═══════════════════════════════════════════════════════════════════════════════


def prepare_for_cross_entropy(
    logits: Tensor,
    targets: Tensor,
) -> tuple[Tensor, Tensor]:
    """Reshape transformer logits and targets for nn.CrossEntropyLoss.

    Transformer logits usually come out as (B, T, V) — for each batch item
    and token position, predict a vocab distribution.

    But nn.CrossEntropyLoss expects classes at dimension 1: (N, C, ...).
    The cleanest fix is to flatten everything except vocab:

        logits  = logits.reshape(B * T, V)   →  (N, V)
        targets = targets.reshape(B * T)      →  (N,)

    This treats every token prediction as one independent classification
    example — which is exactly what language modelling is.

    Input:
        logits:  (B, T, V)     — raw model output
        targets: (B, T)        — ground‑truth token ids

    Return:
        (logits_2d, targets_1d)  ready for nn.CrossEntropyLoss()
    """
    TODO("Flatten logits to (B*T, V), targets to (B*T).")


# ═══════════════════════════════════════════════════════════════════════════════
# Part 8 — The training loop
# ═══════════════════════════════════════════════════════════════════════════════


def training_step(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
) -> float:
    """Run one training step and return the loss value (as a plain float).

    The five essential lines that every PyTorch training loop contains:

    1. optimizer.zero_grad()    — clear stale gradients from the last step
    2. output = model(x)        — forward pass (computes the graph)
    3. loss = loss_fn(output, y) — compare predictions to ground truth
    4. loss.backward()          — back‑propagate (fills .grad on every param)
    5. optimizer.step()         — update parameters with their gradients

    Without zero_grad, gradients accumulate across steps (rarely what you
    want).  Without backward, nothing learns.  Without step, you computed
    gradients for nothing.  These five lines are the heartbeat of deep
    learning in PyTorch.

    Return float(loss.item()), not the tensor — this keeps the value
    detached from the computation graph and prevents memory leaks.
    """
    TODO("Write the five-line training loop. Return float(loss.item()).")
