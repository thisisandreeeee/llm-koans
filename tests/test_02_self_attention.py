"""Koan 02 tests: single-head self-attention."""

import math

import torch

from llm_koans import koans as K


def test_context_from_weights_is_weighted_sum_of_values():
    weights = torch.tensor([
        [0.75, 0.25],
        [0.10, 0.90],
    ])
    V = torch.tensor([
        [10.0, 0.0, 1.0],
        [0.0, 20.0, 3.0],
    ])
    expected = torch.tensor([
        [7.5, 5.0, 1.5],
        [1.0, 18.0, 2.8],
    ])
    assert torch.allclose(K.context_from_weights(weights, V), expected)


def test_self_attention_for_one_query_matches_manual_formula():
    query = torch.tensor([1.0, 2.0])
    keys = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ])
    values = torch.tensor([
        [10.0, 0.0],
        [0.0, 20.0],
        [5.0, 5.0],
    ])

    scores = query @ keys.T
    weights = torch.softmax(scores / math.sqrt(2), dim=-1)
    expected_context = weights @ values

    context, actual_weights = K.self_attention_for_one_query(query, keys, values)
    assert actual_weights.shape == (3,)
    assert context.shape == (2,)
    assert torch.allclose(actual_weights, weights)
    assert torch.allclose(context, expected_context)


def test_single_head_self_attention_matches_vectorised_formula():
    torch.manual_seed(123)
    X = torch.randn(4, 6)
    W_q = torch.randn(3, 6)
    W_k = torch.randn(3, 6)
    W_v = torch.randn(5, 6)

    Q = X @ W_q.T
    Kmat = X @ W_k.T
    V = X @ W_v.T
    scores = Q @ Kmat.T
    expected_weights = torch.softmax(scores / math.sqrt(3), dim=-1)
    expected_context = expected_weights @ V

    context, weights = K.single_head_self_attention(X, W_q, W_k, W_v)
    assert context.shape == (4, 5)
    assert weights.shape == (4, 4)
    assert torch.allclose(weights, expected_weights)
    assert torch.allclose(context, expected_context)
