"""Koan 00: PyTorch matmul intuition.

These exercises build the small shape rules that make attention code feel less
mysterious. For tensors with two or more dimensions, PyTorch matmul treats the
last two dimensions as the matrix and every earlier dimension as batch.
"""

from __future__ import annotations

from torch import Tensor

from .common import TODO

Shape = tuple[int, ...]


def split_batch_and_matrix_dims(shape: Shape) -> tuple[Shape, Shape]:
    """Split a 2D-or-larger shape into batch dims and matrix dims.

    Examples:
        (T, D) -> batch=(), matrix=(T, D)
        (B, T, D) -> batch=(B,), matrix=(T, D)
        (B, H, T, D) -> batch=(B, H), matrix=(T, D)

    Intuition: for matmul, PyTorch looks at the last two dims as rows/cols.
    Everything before them is a stack of independent matrix multiplies.
    """
    TODO("Return (shape[:-2], shape[-2:]) for tensors with at least two dimensions.")


def matmul_vector_dot(a: Tensor, b: Tensor) -> Tensor:
    """Multiply two 1D vectors with matmul.

    1D @ 1D consumes both vector dimensions and returns one scalar dot product.
    """
    TODO("Use the @ operator or torch.matmul for the vector dot product.")


def matmul_matrix_vector(matrix: Tensor, vector: Tensor) -> Tensor:
    """Multiply a matrix by a vector.

    matrix has shape (rows, features), vector has shape (features,), and the
    result has shape (rows,). The shared feature dimension is contracted away.
    """
    TODO("Use matrix @ vector.")


def matmul_matrix_matrix(left: Tensor, right: Tensor) -> Tensor:
    """Multiply two 2D matrices.

    left has shape (rows, shared), right has shape (shared, cols), and the
    result has shape (rows, cols). Only the shared middle dimension disappears.
    """
    TODO("Use left @ right. The inner dimensions must match.")


def batched_linear_projection(tokens: Tensor, weight: Tensor) -> Tensor:
    """Apply the same projection matrix to every token in a batch.

    tokens has shape (B, T, D), weight has shape (D, E), and the result has
    shape (B, T, E). PyTorch treats the final two dims of tokens as a (T, D)
    matrix and broadcasts the 2D weight across the batch.
    """
    TODO("Use tokens @ weight. The D dimension is consumed and E is produced.")


def batch_specific_linear_projection(tokens: Tensor, weights: Tensor) -> Tensor:
    """Apply a different projection matrix to each batch item.

    tokens has shape (B, T, D), weights has shape (B, D, E), and the result has
    shape (B, T, E). The leading B dimensions match, so PyTorch performs one
    matrix multiply per batch item.
    """
    TODO("Use tokens @ weights. Matching leading dims are batch dims.")


def pairwise_dot_products(left: Tensor, right: Tensor) -> Tensor:
    """Compute every row-vector dot product between two tensors.

    left has shape (..., N, D), right has shape (..., M, D), and the result has
    shape (..., N, M). This is the core shape move behind Q @ K.transpose(-2, -1)
    in attention.
    """
    TODO("Transpose the last two dims of right, then matmul: left @ right.transpose(-2, -1).")
