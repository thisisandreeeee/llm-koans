import torch

from llm_koans import koans as K


def test_split_batch_and_matrix_dims_names_the_matmul_rule():
    assert K.split_batch_and_matrix_dims((5, 7)) == ((), (5, 7))
    assert K.split_batch_and_matrix_dims((2, 5, 7)) == ((2,), (5, 7))
    assert K.split_batch_and_matrix_dims((2, 4, 5, 7)) == ((2, 4), (5, 7))


def test_vector_at_vector_returns_one_scalar_dot_product():
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])

    actual = K.matmul_vector_dot(a, b)

    assert actual.shape == torch.Size([])
    assert torch.allclose(actual, torch.tensor(32.0))


def test_matrix_at_vector_consumes_features_and_keeps_rows():
    matrix = torch.tensor([
        [1.0, 0.0, 1.0],
        [0.0, 2.0, 0.0],
    ])
    vector = torch.tensor([10.0, 20.0, 30.0])

    actual = K.matmul_matrix_vector(matrix, vector)

    assert actual.shape == (2,)
    assert torch.allclose(actual, torch.tensor([40.0, 40.0]))


def test_matrix_at_matrix_contracts_only_the_shared_inner_dim():
    left = torch.tensor([
        [1.0, 2.0, 3.0],
        [10.0, 20.0, 30.0],
    ])
    right = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ])

    actual = K.matmul_matrix_matrix(left, right)

    assert actual.shape == (2, 2)
    assert torch.allclose(actual, torch.tensor([[4.0, 5.0], [40.0, 50.0]]))


def test_batched_linear_projection_uses_last_two_dims_as_matrix():
    tokens = torch.tensor([
        [
            [0.0, 1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0, 7.0],
        ],
        [
            [10.0, 11.0, 12.0, 13.0],
            [14.0, 15.0, 16.0, 17.0],
        ],
    ])  # (B=2, T=2, D=4)
    weight = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [2.0, 0.0],
    ])  # (D=4, E=2)

    actual = K.batched_linear_projection(tokens, weight)

    assert actual.shape == (2, 2, 2)
    assert torch.allclose(actual[0, 0], torch.tensor([8.0, 3.0]))
    assert torch.allclose(actual, torch.matmul(tokens, weight))


def test_batch_specific_projection_matches_leading_batch_dims():
    tokens = torch.tensor([
        [[1.0, 2.0]],
        [[10.0, 20.0]],
    ])  # (B=2, T=1, D=2)
    weights = torch.tensor([
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        [
            [2.0, 0.0],
            [0.0, 3.0],
        ],
    ])  # (B=2, D=2, E=2)

    actual = K.batch_specific_linear_projection(tokens, weights)

    assert actual.shape == (2, 1, 2)
    assert torch.allclose(actual, torch.tensor([[[1.0, 2.0]], [[20.0, 60.0]]]))


def test_pairwise_dot_products_are_attention_scores_shape_move():
    queries = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])  # (B=1, H=1, Tq=2, D=2)
    keys = torch.tensor([[[[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]]]])  # (B=1, H=1, Tk=3, D=2)

    actual = K.pairwise_dot_products(queries, keys)

    expected = torch.tensor([[[[1.0, 0.0, 3.0], [0.0, 2.0, 4.0]]]])
    assert actual.shape == (1, 1, 2, 3)
    assert torch.allclose(actual, expected)
