import math

import torch
import torch.nn.functional as F

from llm_koans import koans as K


def ref_split_heads(X, num_heads):
    B, T, D = X.shape
    Dh = D // num_heads
    return X.reshape(B, T, num_heads, Dh).permute(0, 2, 1, 3)


def ref_combine_heads(X):
    B, H, T, Dh = X.shape
    return X.permute(0, 2, 1, 3).contiguous().reshape(B, T, H * Dh)


def ref_mha(X, Wq, Wk, Wv, Wo, H):
    Q = ref_split_heads(X @ Wq, H)
    Kmat = ref_split_heads(X @ Wk, H)
    V = ref_split_heads(X @ Wv, H)
    scores = torch.matmul(Q, Kmat.transpose(-2, -1)) / math.sqrt(Q.shape[-1])
    weights = torch.softmax(scores, dim=-1)
    return ref_combine_heads(torch.matmul(weights, V)) @ Wo


def ref_routed_expert_ffn(X, expert_indices, W1, b1, W2, b2):
    output = torch.empty_like(X)
    for expert_id in range(W1.shape[0]):
        mask = expert_indices == expert_id
        if mask.any():
            tokens = X[mask]
            output[mask] = torch.relu(tokens @ W1[expert_id] + b1[expert_id]) @ W2[expert_id] + b2[expert_id]
    return output


def test_expert_router_logits_scores_each_token_against_each_expert():
    X = torch.tensor(
        [
            [[1.0, 2.0, -1.0], [0.0, 1.0, 3.0]],
            [[-2.0, 0.5, 1.0], [4.0, -1.0, 0.0]],
        ]
    )
    router_W = torch.tensor(
        [
            [1.0, -1.0, 0.5],
            [0.0, 2.0, -0.5],
            [-1.0, 0.25, 1.5],
        ]
    )
    router_b = torch.tensor([0.5, -0.25, 1.0])

    logits = K.expert_router_logits(X, router_W, router_b)

    assert logits.shape == (2, 2, 3)
    assert torch.allclose(logits, X @ router_W + router_b)


def test_top1_expert_routing_returns_chosen_expert_and_gate_probability():
    logits = torch.tensor(
        [
            [[1.0, 3.0, 2.0], [4.0, 0.0, -1.0]],
            [[0.5, 0.25, 2.0], [-1.0, 1.5, 1.0]],
        ]
    )

    expert_indices, gates = K.top1_expert_routing(logits)
    probabilities = torch.softmax(logits, dim=-1)
    expected_indices = torch.argmax(probabilities, dim=-1)
    expected_gates = torch.gather(probabilities, dim=-1, index=expected_indices.unsqueeze(-1)).squeeze(-1)

    assert expert_indices.shape == (2, 2)
    assert gates.shape == (2, 2)
    assert torch.equal(expert_indices, expected_indices)
    assert torch.allclose(gates, expected_gates)


def test_routed_expert_ffn_applies_the_selected_ffn_to_each_token():
    torch.manual_seed(14)
    B, T, D, D_ff, E = 2, 3, 4, 6, 3
    X = torch.randn(B, T, D)
    expert_indices = torch.tensor([[0, 2, 1], [2, 0, 1]])
    W1 = torch.randn(E, D, D_ff)
    b1 = torch.randn(E, D_ff)
    W2 = torch.randn(E, D_ff, D)
    b2 = torch.randn(E, D)

    actual = K.routed_expert_ffn(X, expert_indices, W1, b1, W2, b2)
    expected = ref_routed_expert_ffn(X, expert_indices, W1, b1, W2, b2)

    assert actual.shape == X.shape
    assert torch.allclose(actual, expected)


def test_moe_encoder_block_replaces_dense_ffn_with_top1_routed_experts():
    torch.manual_seed(15)
    B, T, D, H, E, D_ff = 2, 4, 8, 2, 3, 12
    X = torch.randn(B, T, D)
    weights = K.MoEEncoderBlockWeights(
        W_q=torch.randn(D, D),
        W_k=torch.randn(D, D),
        W_v=torch.randn(D, D),
        W_o=torch.randn(D, D),
        router_W=torch.randn(D, E),
        router_b=torch.randn(E),
        expert_W1=torch.randn(E, D, D_ff),
        expert_b1=torch.randn(E, D_ff),
        expert_W2=torch.randn(E, D_ff, D),
        expert_b2=torch.randn(E, D),
    )

    attn = ref_mha(X, weights.W_q, weights.W_k, weights.W_v, weights.W_o, H)
    X1 = F.layer_norm(X + attn, normalized_shape=(D,))
    router_logits = X1 @ weights.router_W + weights.router_b
    probabilities = torch.softmax(router_logits, dim=-1)
    expected_expert_indices = torch.argmax(probabilities, dim=-1)
    expected_gates = torch.gather(probabilities, dim=-1, index=expected_expert_indices.unsqueeze(-1)).squeeze(-1)
    expert_output = ref_routed_expert_ffn(
        X1,
        expected_expert_indices,
        weights.expert_W1,
        weights.expert_b1,
        weights.expert_W2,
        weights.expert_b2,
    )
    expected = F.layer_norm(X1 + expert_output * expected_gates.unsqueeze(-1), normalized_shape=(D,))

    actual, expert_indices, gates = K.moe_encoder_block_forward(X, weights, H)

    assert actual.shape == X.shape
    assert expert_indices.shape == (B, T)
    assert gates.shape == (B, T)
    assert torch.equal(expert_indices, expected_expert_indices)
    assert torch.allclose(gates, expected_gates)
    assert torch.allclose(actual, expected, atol=1e-6)
