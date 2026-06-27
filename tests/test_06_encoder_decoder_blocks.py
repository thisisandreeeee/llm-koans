import math

import torch
import torch.nn.functional as F

from attention_koans import koans as K


def ref_split(X, H):
    B, T, D = X.shape
    Dh = D // H
    return X.reshape(B, T, H, Dh).permute(0, 2, 1, 3)


def ref_combine(X):
    B, H, T, Dh = X.shape
    return X.permute(0, 2, 1, 3).contiguous().reshape(B, T, H * Dh)


def ref_mha(X, Wq, Wk, Wv, Wo, H, mask=None):
    Q = ref_split(X @ Wq, H)
    Kmat = ref_split(X @ Wk, H)
    V = ref_split(X @ Wv, H)
    scores = torch.matmul(Q, Kmat.transpose(-2, -1)) / math.sqrt(Q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return ref_combine(torch.matmul(weights, V)) @ Wo


def ref_cross(dec, enc, Wq, Wk, Wv, Wo, H):
    Q = ref_split(dec @ Wq, H)
    Kmat = ref_split(enc @ Wk, H)
    V = ref_split(enc @ Wv, H)
    scores = torch.matmul(Q, Kmat.transpose(-2, -1)) / math.sqrt(Q.shape[-1])
    weights = torch.softmax(scores, dim=-1)
    return ref_combine(torch.matmul(weights, V)) @ Wo, weights


def test_position_wise_ffn_applies_same_network_to_every_token():
    torch.manual_seed(10)
    X = torch.randn(2, 3, 4)
    W1 = torch.randn(4, 9)
    b1 = torch.randn(9)
    W2 = torch.randn(9, 4)
    b2 = torch.randn(4)
    actual = K.position_wise_ffn(X, W1, b1, W2, b2)
    expected = torch.relu(X @ W1 + b1) @ W2 + b2
    assert actual.shape == X.shape
    assert torch.allclose(actual, expected)


def test_encoder_block_forward_matches_reference_formula():
    torch.manual_seed(11)
    B, T, D, H = 2, 4, 8, 2
    X = torch.randn(B, T, D)
    weights = K.EncoderBlockWeights(
        W_q=torch.randn(D, D),
        W_k=torch.randn(D, D),
        W_v=torch.randn(D, D),
        W_o=torch.randn(D, D),
        W1=torch.randn(D, 16),
        b1=torch.randn(16),
        W2=torch.randn(16, D),
        b2=torch.randn(D),
    )

    attn = ref_mha(X, weights.W_q, weights.W_k, weights.W_v, weights.W_o, H)
    X1 = F.layer_norm(X + attn, normalized_shape=(D,))
    ffn = torch.relu(X1 @ weights.W1 + weights.b1) @ weights.W2 + weights.b2
    expected = F.layer_norm(X1 + ffn, normalized_shape=(D,))

    actual = K.encoder_block_forward(X, weights, H)
    assert actual.shape == X.shape
    assert torch.allclose(actual, expected, atol=1e-6)


def test_cross_attention_uses_decoder_queries_and_encoder_keys_values():
    torch.manual_seed(12)
    B, T_dec, T_enc, D, H = 2, 3, 5, 8, 2
    dec = torch.randn(B, T_dec, D)
    enc = torch.randn(B, T_enc, D)
    Wq = torch.randn(D, D)
    Wk = torch.randn(D, D)
    Wv = torch.randn(D, D)
    Wo = torch.randn(D, D)

    expected, expected_weights = ref_cross(dec, enc, Wq, Wk, Wv, Wo, H)
    actual, weights = K.cross_attention(dec, enc, Wq, Wk, Wv, Wo, H)
    assert actual.shape == (B, T_dec, D)
    assert weights.shape == (B, H, T_dec, T_enc)
    assert torch.allclose(weights, expected_weights)
    assert torch.allclose(actual, expected)


def test_decoder_block_forward_matches_reference_formula():
    torch.manual_seed(13)
    B, T_dec, T_enc, D, H = 2, 4, 5, 8, 2
    Y = torch.randn(B, T_dec, D)
    enc = torch.randn(B, T_enc, D)
    weights = K.DecoderBlockWeights(
        self_W_q=torch.randn(D, D),
        self_W_k=torch.randn(D, D),
        self_W_v=torch.randn(D, D),
        self_W_o=torch.randn(D, D),
        cross_W_q=torch.randn(D, D),
        cross_W_k=torch.randn(D, D),
        cross_W_v=torch.randn(D, D),
        cross_W_o=torch.randn(D, D),
        W1=torch.randn(D, 16),
        b1=torch.randn(16),
        W2=torch.randn(16, D),
        b2=torch.randn(D),
    )
    mask = torch.triu(torch.ones(T_dec, T_dec, dtype=torch.bool), diagonal=1)

    self_attn = ref_mha(Y, weights.self_W_q, weights.self_W_k, weights.self_W_v, weights.self_W_o, H, mask=mask)
    Y1 = F.layer_norm(Y + self_attn, normalized_shape=(D,))
    cross_attn, _ = ref_cross(Y1, enc, weights.cross_W_q, weights.cross_W_k, weights.cross_W_v, weights.cross_W_o, H)
    Y2 = F.layer_norm(Y1 + cross_attn, normalized_shape=(D,))
    ffn = torch.relu(Y2 @ weights.W1 + weights.b1) @ weights.W2 + weights.b2
    expected = F.layer_norm(Y2 + ffn, normalized_shape=(D,))

    actual = K.decoder_block_forward(Y, enc, weights, H)
    assert actual.shape == Y.shape
    assert torch.allclose(actual, expected, atol=1e-6)
