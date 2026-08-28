"""Koan 00 tests: tensor shapes and projections."""

import torch

from llm_koans import koans as K


def test_split_batch_and_matrix_dims_separates_batch_axes_from_matrix_axes():
    assert K.split_batch_and_matrix_dims((5, 7)) == ((), (5, 7))
    assert K.split_batch_and_matrix_dims((2, 5, 7)) == ((2,), (5, 7))
    assert K.split_batch_and_matrix_dims((2, 4, 5, 7)) == ((2, 4), (5, 7))


def test_name_shape_teaches_named_dimensions():
    X = torch.zeros(6, 16)
    assert K.name_shape(X, ("tokens", "features")) == {"tokens": 6, "features": 16}


def test_dot_product_is_a_single_alignment_score():
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])
    assert K.dot_product(a, b).shape == torch.Size([])
    assert torch.allclose(K.dot_product(a, b), torch.tensor(32.0))


def test_project_token_is_matrix_vector_matmul():
    W = torch.tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])
    x = torch.tensor([10.0, 20.0, 30.0])
    expected = torch.tensor([140.0, 320.0])
    assert torch.allclose(K.project_token(W, x), expected)


def test_project_sequence_keeps_tokens_as_rows():
    X = torch.tensor([
        [1.0, 2.0, 3.0],
        [10.0, 20.0, 30.0],
    ])
    W = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 1.0],
    ])
    expected = torch.tensor([
        [1.0, 5.0],
        [10.0, 50.0],
    ])
    assert torch.allclose(K.project_sequence(X, W), expected)
