"""Koan 01: shapes and projections."""

from __future__ import annotations

from torch import Tensor

from .common import TODO


def name_shape(x: Tensor, names: tuple[str, ...]) -> dict[str, int]:
    """Return a mapping from dimension name to size.

    Example:
        X.shape == (6, 16), names == ("tokens", "features")
        -> {"tokens": 6, "features": 16}
    """
    return {id: n for n, id in zip(x.shape, names)}


def dot_product(a: Tensor, b: Tensor) -> Tensor:
    """Return the dot product between two 1D vectors.

    Intuition: a dot product is a single alignment score.
    """
    return a @ b


def project_token(W: Tensor, x: Tensor) -> Tensor:
    """Project one token vector.

    W has shape (d_out, d_in), x has shape (d_in,), output is (d_out,).
    This is the basic q = W_q @ x idea.
    """
    return x @ W.T


def project_sequence(X: Tensor, W: Tensor) -> Tensor:
    """Project every token in a sequence.

    X has shape (T, d_in), W has shape (d_out, d_in), output is (T, d_out).

    Keep tokens as rows. That makes the projection X @ W.T.
    """
    return X @ W.T
