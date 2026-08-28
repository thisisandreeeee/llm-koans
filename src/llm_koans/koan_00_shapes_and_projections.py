"""Koan 00: tensor shapes and projections.

Learn the minimum PyTorch matrix mechanics needed for attention: distinguish
batch axes from matrix axes, name tensor dimensions, compare vectors with a dot
product, and project one token or a whole sequence.
"""

from __future__ import annotations

from torch import Tensor

from .common import TODO


def split_batch_and_matrix_dims(
    shape: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Separate batch axes from the final two matrix axes.

    PyTorch matmul treats the final two axes as a matrix and every earlier axis
    as batch: ``(..., rows, shared) @ (..., shared, cols)``.
    """
    TODO("Return (shape[:-2], shape[-2:]).")


def name_shape(x: Tensor, names: tuple[str, ...]) -> dict[str, int]:
    """Return a mapping from dimension name to size.

    Example:
        X.shape == (6, 16), names == ("tokens", "features")
        -> {"tokens": 6, "features": 16}
    """
    TODO(
        "Create a dict mapping each provided dimension name to the matching x.shape value."
    )


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
