"""Koan 03 tests: multi-head self-attention."""

import math

import torch

from llm_koans import koans as K


def reference_split_heads(X, num_heads):
    B, T, D = X.shape
    Dh = D // num_heads
    return X.reshape(B, T, num_heads, Dh).permute(0, 2, 1, 3)


def reference_combine_heads(X):
    B, H, T, Dh = X.shape
    return X.permute(0, 2, 1, 3).contiguous().reshape(B, T, H * Dh)


def reference_mha(X, W_q, W_k, W_v, W_o, num_heads, mask=None):
    Q = reference_split_heads(X @ W_q, num_heads)
    Kmat = reference_split_heads(X @ W_k, num_heads)
    V = reference_split_heads(X @ W_v, num_heads)
    Dh = Q.shape[-1]
    scores = torch.matmul(Q, Kmat.transpose(-2, -1)) / math.sqrt(Dh)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    context = torch.matmul(weights, V)
    output = reference_combine_heads(context) @ W_o
    return output, weights


def test_split_heads_names_the_head_axis_explicitly():
    X = torch.arange(2 * 3 * 8, dtype=torch.float32).reshape(2, 3, 8)
    heads = K.split_heads(X, num_heads=4)
    expected = reference_split_heads(X, 4)
    assert heads.shape == (2, 4, 3, 2)
    assert torch.equal(heads, expected)


def test_combine_heads_inverts_split_heads():
    X = torch.randn(2, 5, 12)
    heads = K.split_heads(X, num_heads=3)
    combined = K.combine_heads(heads)
    assert combined.shape == X.shape
    assert torch.allclose(combined, X)


def test_multi_head_self_attention_matches_reference_formula():
    torch.manual_seed(7)
    B, T, D, H = 2, 4, 8, 2
    X = torch.randn(B, T, D)
    W_q = torch.randn(D, D)
    W_k = torch.randn(D, D)
    W_v = torch.randn(D, D)
    W_o = torch.randn(D, D)

    expected_output, expected_weights = reference_mha(X, W_q, W_k, W_v, W_o, H)
    output, weights = K.multi_head_self_attention(X, W_q, W_k, W_v, W_o, H)

    assert output.shape == (B, T, D)
    assert weights.shape == (B, H, T, T)
    assert torch.allclose(weights, expected_weights)
    assert torch.allclose(output, expected_output)


def test_multi_head_self_attention_accepts_a_causal_mask():
    torch.manual_seed(8)
    B, T, D, H = 1, 5, 8, 2
    X = torch.randn(B, T, D)
    W_q = torch.randn(D, D)
    W_k = torch.randn(D, D)
    W_v = torch.randn(D, D)
    W_o = torch.randn(D, D)
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)

    _, weights = K.multi_head_self_attention(X, W_q, W_k, W_v, W_o, H, mask=mask)
    assert torch.all(weights[..., mask] == 0)
