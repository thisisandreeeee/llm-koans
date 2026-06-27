import copy

import torch

from attention_koans import koans as K


def snapshot(model):
    return {name: param.detach().clone() for name, param in model.named_parameters()}


def test_train_one_step_returns_loss_and_updates_parameters():
    torch.manual_seed(14)
    model = K.TinyAttentionClassifier(vocab_size=10, d_model=8, num_heads=2, num_classes=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
    token_ids = torch.tensor([
        [1, 2, 3, 4],
        [4, 3, 2, 1],
        [1, 1, 2, 2],
    ])
    labels = torch.tensor([0, 1, 0])

    before = snapshot(model)
    loss = K.train_one_step(model, optimizer, token_ids, labels)
    after = snapshot(model)

    assert loss.ndim == 0
    assert loss.item() > 0
    assert K.parameter_delta_norm(before, after) > 0.0


def test_training_reduces_loss_on_tiny_repeated_batch():
    torch.manual_seed(15)
    model = K.TinyAttentionClassifier(vocab_size=12, d_model=12, num_heads=3, num_classes=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)
    token_ids = torch.tensor([
        [1, 2, 3, 4, 5],
        [5, 4, 3, 2, 1],
        [1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2],
    ])
    labels = torch.tensor([0, 1, 0, 1])

    losses = []
    for _ in range(25):
        loss = K.train_one_step(model, optimizer, token_ids, labels)
        losses.append(loss.item())

    assert losses[-1] < losses[0]


def test_parameter_delta_norm_is_zero_for_identical_snapshots():
    torch.manual_seed(16)
    model = K.TinyAttentionClassifier(vocab_size=7, d_model=8, num_heads=2, num_classes=3)
    state = snapshot(model)
    assert K.parameter_delta_norm(state, copy.deepcopy(state)) == 0.0
