import torch
from torch import nn

from llm_koans import koans as K


# ── Part 1: Naming axes ──────────────────────────────────────────────────────


def test_name_axes_maps_names_to_sizes():
    result = K.name_axes((2, 10, 64), ("B", "T", "D"))
    assert result == {"B": 2, "T": 10, "D": 64}


def test_name_axes_handles_arbitrary_names():
    result = K.name_axes((8, 4, 16, 32), ("batch", "heads", "tokens", "features"))
    assert result == {"batch": 8, "heads": 4, "tokens": 16, "features": 32}


def test_name_axes_works_with_two_dims():
    result = K.name_axes((100, 50), ("vocab", "dim"))
    assert result == {"vocab": 100, "dim": 50}


# ── Part 2: Matmul at every scale ────────────────────────────────────────────


def test_split_batch_and_matrix_dims():
    assert K.split_batch_and_matrix_dims((5, 7)) == ((), (5, 7))
    assert K.split_batch_and_matrix_dims((2, 5, 7)) == ((2,), (5, 7))
    assert K.split_batch_and_matrix_dims((2, 4, 5, 7)) == ((2, 4), (5, 7))


def test_matmul_vector_dot_scalar():
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])
    actual = K.matmul_vector_dot(a, b)
    assert actual.shape == torch.Size([])
    assert torch.allclose(actual, torch.tensor(32.0))


def test_matmul_matrix_vector():
    matrix = torch.tensor([[1.0, 0.0, 1.0], [0.0, 2.0, 0.0]])
    vector = torch.tensor([10.0, 20.0, 30.0])
    actual = K.matmul_matrix_vector(matrix, vector)
    assert actual.shape == (2,)
    assert torch.allclose(actual, torch.tensor([40.0, 40.0]))


def test_matmul_matrix_matrix():
    left = torch.tensor([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    right = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    actual = K.matmul_matrix_matrix(left, right)
    assert actual.shape == (2, 2)
    assert torch.allclose(actual, torch.tensor([[4.0, 5.0], [40.0, 50.0]]))


# ── Part 3: Linear layers act on the last dimension ──────────────────────────


def test_batched_linear_projection():
    tokens = torch.tensor(
        [[[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]],
         [[10.0, 11.0, 12.0, 13.0], [14.0, 15.0, 16.0, 17.0]]]
    )  # (B=2, T=2, D=4)
    weight = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]]
    )  # (D=4, E=2)
    actual = K.batched_linear_projection(tokens, weight)
    assert actual.shape == (2, 2, 2)
    assert torch.allclose(actual[0, 0], torch.tensor([8.0, 3.0]))
    assert torch.allclose(actual, torch.matmul(tokens, weight))


def test_batch_specific_linear_projection():
    tokens = torch.tensor([[[1.0, 2.0]], [[10.0, 20.0]]])  # (B=2, T=1, D=2)
    weights = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0]], [[2.0, 0.0], [0.0, 3.0]]]
    )  # (B=2, D=2, E=2)
    actual = K.batch_specific_linear_projection(tokens, weights)
    assert actual.shape == (2, 1, 2)
    assert torch.allclose(actual, torch.tensor([[[1.0, 2.0]], [[20.0, 60.0]]]))


# ── Part 4: Transpose brings axes together ────────────────────────────────────


def test_pairwise_dot_products():
    queries = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])  # (1, 1, 2, 2)
    keys = torch.tensor([[[[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]]]])  # (1, 1, 3, 2)
    actual = K.pairwise_dot_products(queries, keys)
    expected = torch.tensor([[[[1.0, 0.0, 3.0], [0.0, 2.0, 4.0]]]])
    assert actual.shape == (1, 1, 2, 3)
    assert torch.allclose(actual, expected)


# ── Part 5: Softmax over choices ─────────────────────────────────────────────


def test_softmax_over_choices_sums_to_one():
    scores = torch.tensor([
        [[2.0, 1.0, 0.0],
         [1.0, 2.0, 0.0],
         [0.0, 0.0, 3.0]]
    ])  # (B=1, T=3, T=3)
    weights = K.softmax_over_choices(scores)
    # Every row (last dim) should sum to 1
    assert weights.shape == scores.shape
    assert torch.allclose(weights.sum(dim=-1), torch.ones(1, 3))


def test_softmax_over_choices_respects_magnitude():
    scores = torch.tensor([[
        [100.0, 0.0, 0.0],   # first query strongly prefers key 0
    ]])
    weights = K.softmax_over_choices(scores)
    assert weights[0, 0, 0] > 0.99  # almost all mass on key 0


def test_softmax_over_choices_multihead():
    scores = torch.randn(2, 4, 8, 8)  # (B, H, T, T)
    weights = K.softmax_over_choices(scores)
    assert weights.shape == scores.shape
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 4, 8), atol=1e-6)


# ── Part 6: Reshape vs. transpose ────────────────────────────────────────────


def test_split_heads_for_attention_shape():
    x = torch.randn(2, 10, 64)  # (B, T, D=64)
    result = K.split_heads_for_attention(x, num_heads=4)  # d_head = 16
    assert result.shape == (2, 4, 10, 16)  # (B, H, T, d_head)


def test_split_heads_for_attention_values_preserved():
    # Create a tensor where each position has a unique value so we can trace it
    B, T, D = 2, 3, 8
    x = torch.arange(B * T * D, dtype=torch.float32).reshape(B, T, D)
    num_heads = 2  # d_head = 4
    result = K.split_heads_for_attention(x, num_heads)
    # For batch 0, token 0, the 8 features should be split into 2 heads of 4
    expected_head0_token0 = x[0, 0, :4]   # first 4 features → head 0
    expected_head1_token0 = x[0, 0, 4:]   # next 4 features  → head 1
    assert torch.equal(result[0, 0, 0, :], expected_head0_token0)
    assert torch.equal(result[0, 1, 0, :], expected_head1_token0)


def test_merge_heads_after_attention_shape():
    x = torch.randn(2, 4, 10, 16)  # (B, H, T, d_head)
    result = K.merge_heads_after_attention(x)
    assert result.shape == (2, 10, 64)  # (B, T, D=64)


def test_merge_heads_after_attention_roundtrips():
    B, T, D = 2, 5, 12
    H = 3
    original = torch.randn(B, T, D)
    split = K.split_heads_for_attention(original, H)
    merged = K.merge_heads_after_attention(split)
    assert torch.allclose(merged, original)


# ── Part 7: CrossEntropyLoss convention ──────────────────────────────────────


def test_prepare_for_cross_entropy_shapes():
    B, T, V = 4, 10, 100
    logits = torch.randn(B, T, V)
    targets = torch.randint(0, V, (B, T))
    logits_2d, targets_1d = K.prepare_for_cross_entropy(logits, targets)
    assert logits_2d.shape == (B * T, V)
    assert targets_1d.shape == (B * T,)


def test_prepare_for_cross_entropy_values_match():
    B, T, V = 2, 3, 5
    logits = torch.randn(B, T, V)
    targets = torch.randint(0, V, (B, T))
    logits_2d, targets_1d = K.prepare_for_cross_entropy(logits, targets)
    # Row 0 of the flattened logits should be logits[0, 0, :]
    assert torch.equal(logits_2d[0], logits[0, 0])
    # Row T of the flattened logits should be logits[0, 1, :]
    assert torch.equal(logits_2d[1], logits[0, 1])
    # Row 3 (B*T-1) should be logits[1, 2, :]
    assert torch.equal(logits_2d[5], logits[1, 2])
    # Targets should match
    assert targets_1d[0] == targets[0, 0]
    assert targets_1d[5] == targets[1, 2]


# ── Part 8: The training loop ────────────────────────────────────────────────


def test_training_step_returns_float():
    model = nn.Linear(10, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    x = torch.randn(32, 10)
    y = torch.randn(32, 1)
    loss = K.training_step(model, x, y, optimizer, loss_fn)
    assert isinstance(loss, float)


def test_training_step_modifies_parameters():
    model = nn.Linear(2, 1)
    # Capture initial parameters
    w_before = model.weight.data.clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    y = torch.tensor([[2.0], [3.0]])
    K.training_step(model, x, y, optimizer, loss_fn)
    w_after = model.weight.data.clone()
    # Weights must have changed
    assert not torch.equal(w_before, w_after)


def test_training_step_loss_decreases():
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()
    # Fixed data: y = 3x
    x = torch.tensor([[1.0], [2.0], [3.0]])
    y = torch.tensor([[3.0], [6.0], [9.0]])
    loss1 = K.training_step(model, x, y, optimizer, loss_fn)
    loss2 = K.training_step(model, x, y, optimizer, loss_fn)
    # After two steps on the same data, loss should decrease
    assert loss2 < loss1, f"Expected loss to decrease: {loss1} → {loss2}"
