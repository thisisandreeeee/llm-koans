import copy

import torch

from llm_koans import koans as K


def snapshot(model):
    return {name: param.detach().clone() for name, param in model.named_parameters()}


# ── Exercise A — TinyTransformer ──────────────────────────────────────────────


def test_A_causal_lm_is_masked_so_future_tokens_do_not_leak():
    torch.manual_seed(14)
    model = K.TinyTransformer(vocab_size=16, d_model=8, nhead=2, num_layers=1, max_seq_len=32)
    model.eval()
    ids = torch.tensor([[3, 1, 4, 1, 5]])

    logits = model(ids)

    assert logits.shape == (1, 5, 16)

    ids2 = ids.clone()
    ids2[:, 3] = 15
    logits2 = model(ids2)

    assert torch.allclose(logits[:, :3, :], logits2[:, :3, :])


def test_A_train_one_step_updates_parameters_and_returns_scalar_loss():
    torch.manual_seed(15)
    model = K.TinyTransformer(vocab_size=16, d_model=16, nhead=4, num_layers=1, max_seq_len=32)
    model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
    token_ids = torch.tensor([
        [1, 2, 3, 4],
        [4, 3, 2, 1],
        [1, 1, 2, 2],
    ])

    expected = torch.nn.functional.cross_entropy(
        model(token_ids)[:, :-1].reshape(-1, 16), token_ids[:, 1:].reshape(-1)
    )
    before = snapshot(model)
    loss = K.train_one_step(model, optimizer, token_ids)
    after = snapshot(model)

    assert loss.ndim == 0
    assert loss.item() > 0
    assert torch.allclose(loss, expected)
    assert K.parameter_delta_norm(before, after) > 0.0


def test_A_training_reduces_loss_over_repeated_steps():
    torch.manual_seed(16)
    model = K.TinyTransformer(vocab_size=16, d_model=16, nhead=4, num_layers=1, max_seq_len=32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)
    token_ids = torch.tensor([
        [1, 2, 3, 4, 5],
        [5, 4, 3, 2, 1],
        [1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2],
    ])

    losses = []
    for _ in range(25):
        loss = K.train_one_step(model, optimizer, token_ids)
        losses.append(loss.item())

    assert losses[-1] < losses[0]


# ── Exercise B — TinyEncoder ──────────────────────────────────────────────────


def test_B_encoder_is_bidirectional_so_future_tokens_do_leak():
    torch.manual_seed(18)
    model = K.TinyEncoder(vocab_size=16, d_model=8, nhead=2, num_layers=1, max_seq_len=32, num_classes=3)
    model.eval()
    ids = torch.tensor([[3, 1, 4, 1, 5]])

    out1 = model(ids)

    assert out1.shape == (1, 3)

    ids2 = ids.clone()
    ids2[:, 3] = 15
    out2 = model(ids2)

    assert not torch.allclose(out1, out2)


def test_B_classifier_training_updates_encoder():
    torch.manual_seed(19)
    model = K.TinyEncoder(vocab_size=16, d_model=16, nhead=4, num_layers=1, max_seq_len=32, num_classes=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
    token_ids = torch.tensor([
        [1, 2, 3, 4],
        [4, 3, 2, 1],
        [1, 1, 2, 2],
    ])
    labels = torch.tensor([0, 1, 0])

    before = snapshot(model)
    optimizer.zero_grad()
    loss = torch.nn.functional.cross_entropy(model(token_ids), labels)
    loss.backward()
    optimizer.step()
    after = snapshot(model)

    assert loss.item() > 0
    assert K.parameter_delta_norm(before, after) > 0.0


# ── Exercise C — TinyEncoderDecoder ───────────────────────────────────────────


def test_C_encoder_decoder_forward_shape_and_decoder_causality():
    torch.manual_seed(20)
    model = K.TinyEncoderDecoder(
        src_vocab_size=16, tgt_vocab_size=16, d_model=8, nhead=2, num_layers=1, max_seq_len=32
    )
    model.eval()
    src = torch.tensor([[3, 1, 4]])
    tgt = torch.tensor([[9, 8, 2, 7]])

    logits = model(src, tgt)

    assert logits.shape == (1, 4, 16)

    tgt2 = tgt.clone()
    tgt2[:, 2] = 15
    logits2 = model(src, tgt2)

    assert torch.allclose(logits[:, :2, :], logits2[:, :2, :])


def test_C_decoder_cross_attends_to_encoder_output():
    torch.manual_seed(21)
    model = K.TinyEncoderDecoder(
        src_vocab_size=16, tgt_vocab_size=16, d_model=8, nhead=2, num_layers=1, max_seq_len=32
    )
    model.eval()
    src = torch.tensor([[3, 1, 4]])
    tgt = torch.tensor([[9, 8, 2]])

    out1 = model(src, tgt)

    src2 = src.clone()
    src2[:, 0] = 15
    out2 = model(src2, tgt)

    assert not torch.allclose(out1, out2)


def test_C_train_one_step_works_for_encoder_decoder():
    torch.manual_seed(22)
    model = K.TinyEncoderDecoder(
        src_vocab_size=16, tgt_vocab_size=16, d_model=16, nhead=4, num_layers=1, max_seq_len=32
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
    src = torch.tensor([
        [3, 1, 4, 1],
        [4, 3, 2, 1],
    ])
    tgt = torch.tensor([
        [9, 8, 2, 7],
        [7, 2, 8, 9],
    ])

    before = snapshot(model)
    optimizer.zero_grad()
    logits = model(src, tgt)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_targets = tgt[:, 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]), shift_targets.view(-1)
    )
    loss.backward()
    optimizer.step()
    after = snapshot(model)

    assert loss.ndim == 0
    assert loss.item() > 0
    assert K.parameter_delta_norm(before, after) > 0.0


# ── Shared utility ────────────────────────────────────────────────────────────


def test_parameter_delta_norm_is_zero_for_identical_snapshots():
    torch.manual_seed(17)
    model = K.TinyTransformer(vocab_size=12, d_model=8, nhead=2, num_layers=1, max_seq_len=16)
    state = snapshot(model)
    assert K.parameter_delta_norm(state, copy.deepcopy(state)) == 0.0
