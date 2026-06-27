import math

import torch

from attention_koans import koans as K


def test_causal_mask_blocks_only_future_positions():
    expected = torch.tensor([
        [False, True, True, True],
        [False, False, True, True],
        [False, False, False, True],
        [False, False, False, False],
    ])
    actual = K.causal_mask(4)
    assert actual.dtype == torch.bool
    assert torch.equal(actual, expected)


def test_apply_attention_mask_sets_masked_scores_to_negative_infinity():
    scores = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    mask = K.causal_mask(4)
    masked = K.apply_attention_mask(scores, mask)
    assert torch.isneginf(masked[0, 1])
    assert torch.isneginf(masked[1, 3])
    assert masked[2, 1] == scores[2, 1]


def test_masked_attention_weights_are_zero_for_future_tokens():
    torch.manual_seed(9)
    Q = torch.randn(4, 3)
    Kmat = torch.randn(4, 3)
    mask = K.causal_mask(4)
    weights = K.masked_attention_weights(Q, Kmat, mask)
    assert weights.shape == (4, 4)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(4))
    assert torch.all(weights[mask] == 0)


def test_masked_attention_weights_match_manual_formula():
    Q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    Kmat = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    mask = torch.tensor([[False, True], [False, False]])
    scores = (Q @ Kmat.T) / math.sqrt(2)
    expected = torch.softmax(scores.masked_fill(mask, float("-inf")), dim=-1)
    actual = K.masked_attention_weights(Q, Kmat, mask)
    assert torch.allclose(actual, expected)
