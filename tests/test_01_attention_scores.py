"""Koan 01 tests: scaled attention scores."""

import math

import torch

from llm_koans import koans as K


def test_attention_scores_are_all_query_key_dot_products():
    Q = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    Kmat = torch.tensor([
        [2.0, 0.0],
        [0.0, 3.0],
        [4.0, 5.0],
    ])
    expected = torch.tensor([
        [2.0, 0.0, 4.0],
        [0.0, 3.0, 5.0],
    ])
    assert torch.allclose(K.attention_scores(Q, Kmat), expected)


def test_attention_scores_work_for_batched_heads():
    torch.manual_seed(0)
    Q = torch.randn(2, 3, 4, 5)  # B, H, Tq, D
    Kmat = torch.randn(2, 3, 7, 5)  # B, H, Tk, D
    actual = K.attention_scores(Q, Kmat)
    expected = torch.matmul(Q, Kmat.transpose(-2, -1))
    assert actual.shape == (2, 3, 4, 7)
    assert torch.allclose(actual, expected)


def test_scaled_scores_divide_by_sqrt_dk():
    scores = torch.tensor([[2.0, 4.0]])
    assert torch.allclose(K.scaled_scores(scores, d_k=4), torch.tensor([[1.0, 2.0]]))


def test_softmax_last_dim_produces_positive_rows_summing_to_one():
    scores = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]])
    weights = K.softmax_last_dim(scores)
    assert torch.all(weights > 0)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2))
    assert torch.argmax(weights[0]).item() == 2


def test_attention_weights_are_scaled_softmax_scores():
    Q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    Kmat = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    actual = K.attention_weights(Q, Kmat)
    expected = torch.softmax(torch.eye(2) / math.sqrt(2), dim=-1)
    assert torch.allclose(actual, expected)
